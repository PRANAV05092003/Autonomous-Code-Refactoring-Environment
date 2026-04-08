"""
OpenEnv / Hugging Face importable entrypoint.

OpenEnv validation expects an importable FastAPI app at:
  server.app:app
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
import uvicorn

# Ensure project root is importable when executed in Spaces/Docker.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None  # type: ignore[assignment]

from acre.tasks.task_registry import TaskRegistry
from models import (
    ActionModel,
    CompatibilityHealthResponse,
    GradeRequest,
    GradeResponse,
    HealthResponse,
    OptimizationStep,
    OptimizeRequest,
    OptimizeResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
    TaskInfo,
    TasksResponse,
)
from openenv_interface import OpenEnvRefactorEnv

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
DEFAULT_RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "acre_agent.zip")

app = FastAPI(
    title="ACRE — Autonomous Code Refactoring Environment",
    description="OpenEnv-compatible RL environment for Python code refactoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = TaskRegistry()
_env: Optional[OpenEnvRefactorEnv] = None
_rl_model_cache: dict[str, object] = {}


def get_env() -> OpenEnvRefactorEnv:
    global _env
    if _env is None:
        _env = OpenEnvRefactorEnv(registry=registry)
    return _env


def _state_response() -> StateResponse:
    return get_env().state()


def _choose_action_heuristic(code: str, task_id: Optional[str]) -> int:
    has_generic = re.search(r"\b(x|tmp|i)\b", code) is not None
    has_if_false = re.search(r"\bif\s+False\b", code) is not None
    has_if_true = re.search(r"\bif\s+True\b", code) is not None
    has_append_loop = ".append(" in code and "for " in code
    has_double_not = "not not" in code
    has_add_call = "add(" in code

    if task_id == "rename_variables":
        if has_generic:
            return 0
        if has_if_false or "unused" in code:
            return 1
        if has_append_loop:
            return 2
        if has_if_true or has_double_not:
            return 3
        return 4

    if task_id == "remove_dead_code":
        if has_if_false or "unused" in code:
            return 1
        if has_append_loop:
            return 2
        if has_if_true or has_double_not:
            return 3
        if has_generic:
            return 0
        return 4

    if has_generic:
        return 0
    if has_append_loop:
        return 2
    if has_if_false or has_if_true or has_double_not:
        return 3
    if has_add_call:
        return 4
    return 1


def _choose_action_llm(
    *,
    code: str,
    task_id: Optional[str],
    step_index: int,
    max_steps: int,
    api_base_url: str,
    model_name: str,
    api_token: str,
) -> tuple[int, str, str]:
    if not api_token.strip():
        return _choose_action_heuristic(code, task_id), "empty token -> heuristic", "heuristic"

    client = OpenAI(base_url=api_base_url, api_key=api_token)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a code-refactoring action selector. Return ONLY compact JSON: "
                '{"action": <0-4>, "reason": "..."}.\n'
                "Actions: 0=rename_variable,1=remove_dead_code,2=simplify_loop,3=optimize_condition,4=inline_function"
            ),
        },
        {
            "role": "user",
            "content": (
                f"task_id={task_id or 'auto'}\n"
                f"step={step_index}/{max_steps}\n"
                "Current code:\n"
                f"```python\n{code}\n```"
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        blob = m.group(0) if m else raw
        parsed = json.loads(blob)
        action = int(parsed.get("action", -1))
        reason = str(parsed.get("reason", "llm-selected action"))
        if 0 <= action <= 4:
            return action, reason, "llm"
    except Exception as exc:
        return _choose_action_heuristic(code, task_id), f"llm error -> heuristic: {exc}", "heuristic"

    return _choose_action_heuristic(code, task_id), "invalid llm output -> heuristic", "heuristic"


def _choose_action_rl(observation: list[float], model_path: str) -> tuple[Optional[int], str, str]:
    if PPO is None:
        return None, "stable-baselines3 unavailable", "rl"
    if not os.path.exists(model_path):
        return None, f"rl model not found: {model_path}", "rl"

    try:
        model = _rl_model_cache.get(model_path)
        if model is None:
            model = PPO.load(model_path)
            _rl_model_cache[model_path] = model

        obs = np.asarray(observation, dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
        action_i = int(action)
        if 0 <= action_i <= 4:
            return action_i, "rl policy action", "rl"
        return None, f"invalid rl action: {action_i}", "rl"
    except Exception as exc:
        return None, f"rl failure: {exc}", "rl"


def _demo_html() -> str:
    # Import the existing UI HTML from root server.py if present, else fallback.
    try:
        import server as legacy_server  # type: ignore
        return str(getattr(legacy_server, "_demo_html")())
    except Exception:
        return "<html><body><h1>ACRE</h1><p>UI unavailable.</p></body></html>"


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(content=_demo_html())


@app.get("/health", response_model=CompatibilityHealthResponse)
def health_compat() -> CompatibilityHealthResponse:
    return CompatibilityHealthResponse(status="healthy", service="acre-env")


@app.get("/demo")
def demo() -> JSONResponse:
    from inference import run_all_tasks

    return JSONResponse(content={"results": run_all_tasks()})


@app.get("/ui", response_class=HTMLResponse)
def demo_ui() -> HTMLResponse:
    return HTMLResponse(content=_demo_html())


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    env = get_env()
    try:
        obs = env.reset(seed=req.seed, task_id=req.task_id, code=req.code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ResetResponse(
        observation=obs,
        observation_vector=obs.to_vector(),
        info=env.last_reset_info,
        task_id=req.task_id,
        state=_state_response(),
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    env = get_env()
    if not (0 <= req.action <= 4):
        raise HTTPException(status_code=400, detail="action must be 0–4")

    obs, reward, done, info = env.step(req.action)
    action_name = str(info.get("action_name", env.action_meanings.get(req.action, "unknown")))
    return StepResponse(
        action=ActionModel(action=req.action, action_name=action_name),
        observation=obs,
        observation_vector=obs.to_vector(),
        reward=reward,
        done=done,
        terminated=done,
        truncated=False,
        info=info,
        state=_state_response(),
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return _state_response()


@app.get("/tasks", response_model=TasksResponse)
def list_tasks() -> TasksResponse:
    return TasksResponse(tasks=[TaskInfo.model_validate(t) for t in registry.list_tasks()])


@app.post("/tasks/{task_id}/grade", response_model=GradeResponse)
def grade(task_id: str, req: GradeRequest) -> GradeResponse:
    task = registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    score = task.grade_against_expected(req.code)
    return GradeResponse(task_id=task_id, score=round(score, 4), passed=score >= 0.8)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    code = req.code.strip("\n")
    if not code.strip():
        raise HTTPException(status_code=400, detail="code must be non-empty")

    env = get_env()
    try:
        env.reset(task_id=req.task_id, code=code)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    steps: list[OptimizationStep] = []
    cumulative_reward = 0.0

    for step_idx in range(1, req.max_steps + 1):
        state_now = env.state()
        current_code = state_now.current_code
        obs_list = [float(x) for x in state_now.observation_vector]

        action: int
        reason: str
        source: str

        if req.use_rl:
            rl_action, rl_reason, rl_source = _choose_action_rl(
                observation=obs_list,
                model_path=req.rl_model_path or DEFAULT_RL_MODEL_PATH,
            )
            if rl_action is not None:
                action, reason, source = rl_action, rl_reason, rl_source
            elif req.fallback_to_llm and req.use_llm:
                action, reason, source = _choose_action_llm(
                    code=current_code,
                    task_id=req.task_id,
                    step_index=step_idx,
                    max_steps=req.max_steps,
                    api_base_url=req.api_base_url or DEFAULT_API_BASE_URL,
                    model_name=req.model_name or DEFAULT_MODEL_NAME,
                    api_token=req.api_token or "",
                )
                reason = f"{rl_reason}; {reason}"
            else:
                action = _choose_action_heuristic(current_code, req.task_id)
                reason = f"{rl_reason}; heuristic fallback"
                source = "heuristic"
        elif req.use_llm:
            action, reason, source = _choose_action_llm(
                code=current_code,
                task_id=req.task_id,
                step_index=step_idx,
                max_steps=req.max_steps,
                api_base_url=req.api_base_url or DEFAULT_API_BASE_URL,
                model_name=req.model_name or DEFAULT_MODEL_NAME,
                api_token=req.api_token or "",
            )
        else:
            action = _choose_action_heuristic(current_code, req.task_id)
            reason = "heuristic policy"
            source = "heuristic"

        _, reward, done, info = env.step(action)
        state_now = env.state()
        cumulative_reward += float(reward.raw)
        steps.append(
            OptimizationStep(
                step=step_idx,
                action=action,
                action_name=info.get("action_name", "unknown"),
                reason=reason,
                source=source,
                reward=float(reward.raw),
                normalized_reward=float(reward.normalized),
                changed=bool(info.get("changed", False)),
                complexity=float(state_now.complexity),
            )
        )
        if done:
            break

    final_code = str(env.state().current_code)
    diff_lines = difflib.unified_diff(
        code.splitlines(),
        final_code.splitlines(),
        fromfile="original.py",
        tofile="optimized.py",
        lineterm="",
    )
    diff_text = "\n".join(diff_lines)

    task_score: Optional[float] = None
    if req.task_id:
        task = registry.get_task(req.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{req.task_id}' not found")
        task_score = round(task.grade(final_code), 4)

    return OptimizeResponse(
        original_code=code,
        optimized_code=final_code,
        diff=diff_text,
        steps=steps,
        cumulative_reward=round(cumulative_reward, 4),
        task_id=req.task_id,
        task_score=task_score,
    )


def main() -> None:
    """
    Entry point for OpenEnv multi-mode deployment.

    - API mode: OpenEnv imports `server.app:app`
    - CLI mode: OpenEnv / HF can run `server` script -> `server.app:main`
    """
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

