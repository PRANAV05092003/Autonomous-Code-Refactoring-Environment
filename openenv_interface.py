from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from openenv.env import Env as OpenEnvBase
except Exception:  # pragma: no cover
    class OpenEnvBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

from acre.datasets.code_samples import CodeSample, CodeSampleDataset
from acre.env.refactor_env import RefactorEnv
from acre.tasks.task_registry import TaskRegistry
from models import ActionModel, ObservationModel, RewardModel, StateResponse


class OpenEnvRefactorEnv(OpenEnvBase):
    """
    Canonical OpenEnv interface for ACRE.

    This wrapper keeps the strict hackathon contract:
    - reset() -> ObservationModel
    - step(action) -> (ObservationModel, RewardModel, done, info)
    - state() -> StateResponse
    """

    def __init__(
        self,
        *,
        env: Optional[RefactorEnv] = None,
        registry: Optional[TaskRegistry] = None,
    ) -> None:
        super().__init__(
            name="ACRE",
            state_space="ObservationModel",
            action_space="ActionModel",
            episode_max_length=RefactorEnv.MAX_STEPS,
        )
        self._env = env or RefactorEnv()
        self._registry = registry or TaskRegistry()
        self._task_id: Optional[str] = None
        self._last_reset_info: Dict[str, Any] = {}

    @property
    def action_meanings(self) -> Dict[int, str]:
        return self._env.ACTION_MEANINGS

    @property
    def last_reset_info(self) -> Dict[str, Any]:
        return dict(self._last_reset_info)

    def _load_episode_source(self, *, task_id: Optional[str], code: Optional[str]) -> None:
        initial_code = code
        if initial_code is None and task_id:
            task = self._registry.get_task(task_id)
            if task is None:
                raise ValueError(f"Task '{task_id}' not found")
            # Load a multi-sample dataset for this task. Sample selection is
            # deterministic given the `seed` passed to `reset()`.
            samples = list(getattr(task, "samples", []) or [])
            if not samples:
                initial_code = task.initial_code
            else:
                self._env.dataset = CodeSampleDataset(
                    [
                        CodeSample(
                            id=f"{task_id}:{i}",
                            language="python",
                            code=str(src),
                        )
                        for i, src in enumerate(samples)
                    ]
                )
                return None

        if initial_code is None:
            return None

        self._env.dataset = CodeSampleDataset(
            [
                CodeSample(
                    id=task_id or "custom",
                    language="python",
                    code=initial_code,
                )
            ]
        )
        return None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
        code: Optional[str] = None,
    ) -> ObservationModel:
        self._task_id = task_id
        self._load_episode_source(task_id=task_id, code=code)
        observation, info = self._env.reset(seed=seed)
        self._last_reset_info = dict(info)
        return ObservationModel.from_vector(observation.tolist())

    def step(self, action: int | ActionModel) -> Tuple[ObservationModel, RewardModel, bool, Dict[str, Any]]:
        action_value = action.action if isinstance(action, ActionModel) else int(action)
        observation, raw_reward, terminated, truncated, info = self._env.step(action_value)
        reward = RewardModel(
            raw=float(raw_reward),
            normalized=float(info.get("normalized_reward", 0.0)),
            components=dict(info.get("reward_components", {})),
        )
        done = bool(terminated or truncated)
        return ObservationModel.from_vector(observation.tolist()), reward, done, dict(info)

    def state(self) -> StateResponse:
        raw_state = self._env.state()
        observation_vector = list(raw_state.get("observation", [0.0, 0.0, 0.0, 0.0]))
        observation = ObservationModel.from_vector(observation_vector)
        return StateResponse(
            current_code=str(raw_state.get("current_code", "")),
            episode_steps=int(raw_state.get("episode_steps", 0)),
            max_steps=int(raw_state.get("max_steps", RefactorEnv.MAX_STEPS)),
            complexity=float(raw_state.get("complexity", 0.0)),
            last_runtime=float(raw_state.get("last_runtime", 0.0)),
            last_error=bool(raw_state.get("last_error", False)),
            sample_id=raw_state.get("sample_id"),
            language=raw_state.get("language"),
            task_id=self._task_id,
            observation=observation,
            observation_vector=observation.to_vector(),
            action_meanings=dict(raw_state.get("action_meanings", {})),
        )
