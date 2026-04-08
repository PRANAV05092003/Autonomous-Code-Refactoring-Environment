from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import multiprocessing as mp

import gymnasium as gym
import numpy as np

from acre.actions import transformations as tx
from acre.datasets.code_samples import CodeSample, CodeSampleDataset
from acre.tasks.task_registry import TaskRegistry
from acre.tasks.grader import grade_task

try:
    from radon.complexity import cc_visit
except Exception:  # pragma: no cover
    cc_visit = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _ExecResult:
    exit_code: int
    metrics: Dict[str, Any]
    error: Optional[str] = None


_BANNED_PATTERNS: Tuple[str, ...] = (
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+pathlib\b",
    r"\bimport\s+shutil\b",
    r"\bopen\s*\(",
    r"\bos\.(remove|unlink|rmdir|removedirs|rename|replace|system|popen)\b",
    r"\bshutil\.(rmtree|move|copy|copytree)\b",
    r"\bsubprocess\.(run|Popen|call|check_call|check_output)\b",
)


def _exec_worker(src: str, fname: str, out_q: "mp.Queue[dict]") -> None:
    start = time.perf_counter()
    try:
        if any(re.search(p, src) for p in _BANNED_PATTERNS):
            runtime_s = time.perf_counter() - start
            out_q.put({"exit_code": 2, "runtime_s": float(runtime_s), "error": "forbidden_operation"})
            return None

        compiled = compile(src, fname, "exec")
        exec_globals: Dict[str, Any] = {"__name__": "__main__"}
        exec(compiled, exec_globals, None)
        runtime_s = time.perf_counter() - start
        out_q.put({"exit_code": 0, "runtime_s": float(runtime_s), "error": None})
        return None
    except Exception as exc:
        runtime_s = time.perf_counter() - start
        out_q.put({"exit_code": 1, "runtime_s": float(runtime_s), "error": str(exc)})
        return None


class _InProcessExecutor:
    """
    Execute candidate code with a hard timeout to avoid hanging the server.

    This is critical for deployment: the agent can easily generate `while True: ...`
    or other long-running code. We treat timeout as an execution error.
    """

    def run(self, code: str, *, filename: str = "<acre>", timeout_s: float = 0.25) -> _ExecResult:
        q: "mp.Queue[dict]" = mp.Queue(maxsize=1)
        # NOTE: on Windows, Process target must be picklable (top-level function).
        proc = mp.Process(target=_exec_worker, args=(code, filename, q), daemon=True)
        proc.start()
        proc.join(timeout=max(0.01, float(timeout_s)))

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.1)
            return _ExecResult(exit_code=124, metrics={"runtime_s": float(timeout_s)}, error="timeout")

        payload: dict = {}
        try:
            payload = q.get_nowait()
        except Exception:
            payload = {"exit_code": 1, "runtime_s": 0.0, "error": "no result"}

        return _ExecResult(
            exit_code=int(payload.get("exit_code", 1)),
            metrics={"runtime_s": float(payload.get("runtime_s", 0.0) or 0.0)},
            error=payload.get("error"),
        )


class RefactorEnv(gym.Env):
    metadata = {"render_modes": []}

    MAX_STEPS = 5

    ACTION_MEANINGS: Dict[int, str] = {
        0: "rename_variable",
        1: "remove_dead_code",
        2: "simplify_loop",
        3: "optimize_condition",
        4: "inline_function",
    }

    def __init__(
        self,
        *,
        dataset: Optional[CodeSampleDataset] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e9, 1e9, 1e9, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.dataset: CodeSampleDataset = dataset or CodeSampleDataset(
            [
                CodeSample(
                    id="default",
                    language="python",
                    code="def f(x):\n    return x\n",
                )
            ]
        )
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        self.executor = _InProcessExecutor()
        self._registry = TaskRegistry()

        self._episode_steps = 0
        self._sample: Optional[CodeSample] = None
        self._code: str = ""
        self._expected_output: str = ""
        self._progress_score: float = 0.0
        self._last_runtime_s: float = 0.0
        self._last_error: bool = False
        self._last_complexity: float = 0.0

    def _compute_complexity(self, code: str) -> float:
        if cc_visit is None:
            return float(len(code.splitlines()))
        try:
            blocks = cc_visit(code)
            if not blocks:
                return 0.0
            return float(sum(getattr(b, "complexity", 0) for b in blocks))
        except Exception:
            return float(len(code.splitlines()))

    def _compute_runtime(self, code: str) -> Tuple[float, bool, bool]:
        res = self.executor.run(code, filename="env_exec.py", timeout_s=0.25)
        runtime_s = float(res.metrics.get("runtime_s", 0.0) or 0.0)
        is_timeout = bool(res.exit_code == 124)
        return runtime_s, bool(res.exit_code != 0), is_timeout

    def _observation(self) -> np.ndarray:
        return np.asarray(
            [
                float(len(self._code)),
                float(self._last_complexity),
                float(self._last_runtime_s),
                float(int(self._last_error)),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        samples = list(self.dataset)
        if not samples:
            samples = [CodeSample(id="empty", language="python", code="")]

        idx = int(self._np_random.integers(0, len(samples)))
        self._sample = samples[idx]
        self._code = str(self._sample.code)
        self._episode_steps = 0

        # Resolve expected output deterministically from task_registry based on sample_id.
        # sample ids are produced by openenv_interface as "{task_id}:{i}".
        self._expected_output = ""
        self._progress_score = 0.0
        sample_id = str(getattr(self._sample, "id", "") or "")
        if ":" in sample_id:
            task_id, raw_idx = sample_id.split(":", 1)
            task = self._registry.get_task(task_id)
            try:
                sample_idx = int(raw_idx)
            except Exception:
                sample_idx = 0
            if task is not None:
                self._expected_output = task.expected_output_for_index(sample_idx)
                self._progress_score = float(grade_task(self._code, self._expected_output))

        self._last_complexity = self._compute_complexity(self._code)
        self._last_runtime_s, self._last_error, _ = self._compute_runtime(self._code)

        info = {
            "sample_id": getattr(self._sample, "id", None),
            "language": getattr(self._sample, "language", None),
            "episode_steps": self._episode_steps,
            "progress_score": float(self._progress_score),
        }
        return self._observation(), info

    def step(self, action: int):
        action_i = int(action)
        if action_i not in self.ACTION_MEANINGS:
            raise ValueError(f"Invalid action {action_i}; expected 0..4")

        prev_complexity = float(self._last_complexity)
        prev_runtime = float(self._last_runtime_s)
        prev_error = bool(self._last_error)
        prev_score = float(self._progress_score)

        original = self._code
        if action_i == 0:
            transform = tx.rename_variable(original)
        elif action_i == 1:
            transform = tx.remove_dead_code(original)
        elif action_i == 2:
            transform = tx.simplify_loop(original)
        elif action_i == 3:
            transform = tx.optimize_condition(original)
        else:
            transform = tx.inline_function(original)

        self._code = transform.code
        self._episode_steps += 1

        self._last_complexity = self._compute_complexity(self._code)
        self._last_runtime_s, self._last_error, is_timeout = self._compute_runtime(self._code)

        # Deterministic task progress score toward expected output.
        score_now = prev_score
        if self._expected_output:
            score_now = float(grade_task(self._code, self._expected_output))
        self._progress_score = float(score_now)

        # ------------------------------------------------------------------
        # Step-wise reward (hackathon-friendly, deterministic)
        # ------------------------------------------------------------------
        # - better code (closer to expected_output) -> +0.3-ish
        # - reduced complexity -> +0.3-ish
        # - bug introduced -> -0.5
        # - infinite loop / timeout -> large penalty
        delta_score = float(score_now - prev_score)
        complexity_gain = (prev_complexity - float(self._last_complexity)) / max(prev_complexity, 1.0)
        runtime_gain = (prev_runtime - float(self._last_runtime_s)) / max(prev_runtime, 1e-6)

        better_code_reward = float(max(-1.0, min(1.0, delta_score)) * 0.6)
        complexity_reward = float(max(-1.0, min(1.0, complexity_gain)) * 0.3)
        runtime_reward = float(max(-1.0, min(1.0, runtime_gain)) * 0.1)

        bug_penalty = -0.5 if ((not prev_error) and self._last_error) else 0.0
        fixed_bonus = 0.2 if (prev_error and (not self._last_error)) else 0.0
        timeout_penalty = -1.0 if is_timeout else 0.0
        no_change_penalty = -0.05 if not transform.changed else 0.0

        raw_reward = float(
            better_code_reward
            + complexity_reward
            + runtime_reward
            + bug_penalty
            + fixed_bonus
            + timeout_penalty
            + no_change_penalty
        )

        # Normalize exactly as declared in openenv.yaml (clip to [0,1]).
        normalized_reward = float((raw_reward + 32.0) / 52.0)
        if normalized_reward < 0.0:
            normalized_reward = 0.0
        elif normalized_reward > 1.0:
            normalized_reward = 1.0

        terminated = bool(self._episode_steps >= int(self.MAX_STEPS))
        truncated = False

        info: Dict[str, Any] = {
            "action_name": self.ACTION_MEANINGS[action_i],
            "changed": bool(transform.changed),
            "transform": dict(transform.metadata),
            "reward_components": {
                "better_code_reward": float(better_code_reward),
                "complexity_gain": float(complexity_gain),
                "runtime_gain": float(runtime_gain),
                "complexity_reward": float(complexity_reward),
                "runtime_reward": float(runtime_reward),
                "bug_penalty": float(bug_penalty),
                "fixed_bonus": float(fixed_bonus),
                "timeout_penalty": float(timeout_penalty),
                "no_change_penalty": float(no_change_penalty),
            },
            "normalized_reward": normalized_reward,
            "episode_steps": int(self._episode_steps),
            "timeout": bool(is_timeout),
            "progress_score": float(score_now),
            "progress_delta": float(delta_score),
        }
        return self._observation(), raw_reward, terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        return {
            "current_code": self._code,
            "episode_steps": int(self._episode_steps),
            "max_steps": int(self.MAX_STEPS),
            "complexity": float(self._last_complexity),
            "last_runtime": float(self._last_runtime_s),
            "last_error": bool(self._last_error),
            "sample_id": getattr(self._sample, "id", None) if self._sample is not None else None,
            "language": getattr(self._sample, "language", None) if self._sample is not None else None,
            "observation": self._observation().tolist(),
            "action_meanings": dict(self.ACTION_MEANINGS),
            "progress_score": float(self._progress_score),
        }

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

