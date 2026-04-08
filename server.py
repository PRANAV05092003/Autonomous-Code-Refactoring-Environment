"""
ACRE OpenEnv HTTP server.

Endpoints (all required by OpenEnv spec):
  GET  /          — health check (must return HTTP 200)
  POST /reset     — reset environment, returns observation + info
  POST /step      — take one step, returns obs/reward/done/info
  GET  /state     — full current state snapshot
  GET  /tasks     — list all tasks with initial code
  POST /tasks/{task_id}/grade  — grade code for a specific task
"""
from __future__ import annotations

import difflib
import os
import re
import json
import sys
from typing import Optional

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

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

# Global singletons
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
    return """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>ACRE Refactor Demo</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        :root {
            --bg0: #0b1f2a;
            --bg1: #14344a;
            --ink: #eaf7ff;
            --muted: #a7c8db;
            --brand: #1ec28b;
            --warn: #ffcb47;
            --panel: rgba(8, 24, 36, 0.72);
            --stroke: rgba(140, 197, 225, 0.35);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
            background:
                radial-gradient(circle at 12% 18%, rgba(30, 194, 139, 0.28), transparent 35%),
                radial-gradient(circle at 88% 8%, rgba(255, 203, 71, 0.22), transparent 30%),
                linear-gradient(150deg, var(--bg0), var(--bg1));
            min-height: 100vh;
        }
        .wrap {
            max-width: 1200px;
            margin: 0 auto;
            padding: 28px 20px 40px;
        }
        h1 {
            margin: 0 0 6px;
            font-size: clamp(1.6rem, 2vw + 1rem, 2.6rem);
            letter-spacing: 0.2px;
        }
        .sub { margin: 0 0 20px; color: var(--muted); }
        .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 16px;
        }
        .panel {
            border: 1px solid var(--stroke);
            border-radius: 14px;
            background: var(--panel);
            backdrop-filter: blur(4px);
            padding: 14px;
        }
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 10px;
        }
        textarea, pre {
            width: 100%;
            min-height: 260px;
            border: 1px solid var(--stroke);
            border-radius: 10px;
            padding: 12px;
            background: rgba(1, 13, 24, 0.82);
            color: #dcf4ff;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            overflow: auto;
            white-space: pre;
        }
        button, select {
            border: 1px solid var(--stroke);
            border-radius: 10px;
            padding: 10px 12px;
            background: rgba(11, 36, 52, 0.9);
            color: var(--ink);
            font-weight: 600;
        }
        button.primary {
            background: linear-gradient(120deg, #19a7ff, #1ec28b);
            color: #032235;
            border: none;
        }
        .cols {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
        }
        .meta {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 8px;
        }
        .badge {
            color: #082b22;
            background: var(--brand);
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 12px;
            font-weight: 700;
        }
        .warn {
            color: #2a1c00;
            background: var(--warn);
        }
        @media (min-width: 900px) {
            .cols { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <h1>ACRE Live Refactor Arena</h1>
        <p class=\"sub\">Paste old code, run the agent, and compare before and after with a full diff and step-by-step rewards.</p>

        <div class=\"panel\">
            <div class=\"controls\">
                <button onclick=\"loadExample(1)\">Load Example 1</button>
                <button onclick=\"loadExample(2)\">Load Example 2</button>
                <select id=\"task\">
                    <option value=\"\">Auto strategy</option>
                    <option value=\"rename_variables\">rename_variables</option>
                    <option value=\"remove_dead_code\">remove_dead_code</option>
                    <option value=\"full_refactor\">full_refactor</option>
                </select>
                <button class=\"primary\" onclick=\"runOptimize()\">Run Optimization</button>
            </div>
            <div class=\"controls\" style=\"margin-bottom: 10px;\">
                <select id=\"mode\">
                    <option value=\"rl_then_llm\">RL First -> LLM Fallback</option>
                    <option value=\"heuristic\">Heuristic Agent (no API key)</option>
                    <option value=\"llm\">LLM Agent (OpenAI-compatible API)</option>
                </select>
                <input id=\"rlModelPath\" placeholder=\"RL model path\" value=\"acre_agent.zip\" style=\"border:1px solid var(--stroke);border-radius:10px;padding:10px 12px;background:rgba(1,13,24,0.82);color:#dcf4ff;\" />
                <input id=\"baseUrl\" placeholder=\"API base URL (optional)\" value=\"https://api.openai.com/v1\" style=\"border:1px solid var(--stroke);border-radius:10px;padding:10px 12px;background:rgba(1,13,24,0.82);color:#dcf4ff;\" />
                <input id=\"modelName\" placeholder=\"Model name (optional)\" value=\"gpt-4o-mini\" style=\"border:1px solid var(--stroke);border-radius:10px;padding:10px 12px;background:rgba(1,13,24,0.82);color:#dcf4ff;\" />
                <input id=\"apiToken\" type=\"password\" placeholder=\"Paste API token here for LLM mode\" style=\"border:1px solid var(--stroke);border-radius:10px;padding:10px 12px;background:rgba(1,13,24,0.82);color:#dcf4ff;\" />
            </div>
            <div class=\"controls\" style=\"margin-bottom: 10px;\">
                <label style=\"display:flex;align-items:center;gap:8px;padding:8px 10px;border:1px solid var(--stroke);border-radius:10px;\">
                    <input id=\"autoSuggest\" type=\"checkbox\" />
                    Auto suggest after typing pause
                </label>
            </div>
            <textarea id=\"input\" spellcheck=\"false\" placeholder=\"Paste your Python code here...\"></textarea>
            <p class=\"meta\" id=\"status\">Status: ready</p>
            <p class=\"meta\" id=\"liveResults\">Live results: loading...</p>
        </div>

        <div class=\"cols\" style=\"margin-top: 14px\">
            <div class=\"panel\">
                <h3>Original Code</h3>
                <pre id=\"original\"></pre>
            </div>
            <div class=\"panel\">
                <h3>Optimized Code</h3>
                <pre id=\"optimized\"></pre>
            </div>
        </div>

        <div class=\"panel\" style=\"margin-top: 14px\">
            <h3>Diff</h3>
            <pre id=\"diff\"></pre>
        </div>

        <div class=\"panel\" style=\"margin-top: 14px\">
            <h3>Step Logs</h3>
            <pre id=\"steps\"></pre>
        </div>
    </div>

    <script>
        const EX1 = `def compute(x, y, tmp):\n    tmp = x + y\n    x = tmp * 2\n    result = x\n    return result\n`;
        const EX2 = `def add(p, q):\n    return p + q\n\ndef compute(x, data, tmp):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    if False:\n        y = 999\n    if True:\n        val = add(x, tmp)\n    unused = 0\n    flag = not not True\n    return val\n    print(\"dead\")\n`;
        let autoTimer = null;

        function loadExample(i) {
            document.getElementById('input').value = i === 1 ? EX1 : EX2;
            document.getElementById('status').textContent = `Status: loaded example ${i}`;
        }

        async function runOptimize() {
            const code = document.getElementById('input').value;
            const task = document.getElementById('task').value || null;
            const mode = document.getElementById('mode').value;
            const useRl = mode === 'rl_then_llm';
            const useLlm = mode === 'llm' || mode === 'rl_then_llm';
            const fallbackToLlm = mode === 'rl_then_llm';
            const rlModelPath = document.getElementById('rlModelPath').value || null;
            const apiToken = document.getElementById('apiToken').value || null;
            const apiBaseUrl = document.getElementById('baseUrl').value || null;
            const modelName = document.getElementById('modelName').value || null;
            if (!code.trim()) {
                document.getElementById('status').innerHTML = 'Status: <span class=\"badge warn\">please paste code first</span>';
                return;
            }
            if (mode === 'llm' && (!apiToken || !apiToken.trim())) {
                document.getElementById('status').innerHTML = 'Status: <span class=\"badge warn\">paste API token for LLM mode</span>';
                return;
            }

            document.getElementById('status').textContent = 'Status: running optimization...';
            try {
                const res = await fetch('/optimize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        code,
                        task_id: task,
                        max_steps: 5,
                        use_rl: useRl,
                        use_llm: useLlm,
                        fallback_to_llm: fallbackToLlm,
                        rl_model_path: rlModelPath,
                        api_base_url: apiBaseUrl,
                        model_name: modelName,
                        api_token: apiToken,
                    })
                });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'request failed');
                }

                document.getElementById('original').textContent = data.original_code;
                document.getElementById('optimized').textContent = data.optimized_code;
                document.getElementById('diff').textContent = data.diff || '(no diff)';
                document.getElementById('steps').textContent = JSON.stringify(data.steps, null, 2);

                const scoreText = data.task_score === null ? 'n/a' : data.task_score;
                document.getElementById('status').innerHTML = `Status: <span class=\"badge\">done</span> cumulative_reward=${data.cumulative_reward.toFixed(2)} task_score=${scoreText}`;
            } catch (err) {
                document.getElementById('status').innerHTML = `Status: <span class=\"badge warn\">error</span> ${err.message}`;
            }
        }

        async function loadLiveResults() {
            const el = document.getElementById('liveResults');
            try {
                const res = await fetch('/demo');
                const data = await res.json();
                const r = (data && data.results) ? data.results : null;
                if (!res.ok || !r) {
                    throw new Error('demo request failed');
                }
                const easy = (r.easy ?? 0).toFixed(4);
                const medium = (r.medium ?? 0).toFixed(4);
                const hard = (r.hard ?? 0).toFixed(4);
                const final = (r.final ?? 0).toFixed(4);
                el.textContent = `Live results: Easy=${easy}  Medium=${medium}  Hard=${hard}  Final=${final}`;
            } catch (err) {
                el.textContent = `Live results: error (${err.message || err})`;
            }
        }

        loadExample(1);
        loadLiveResults();
        document.getElementById('input').addEventListener('input', () => {
            if (!document.getElementById('autoSuggest').checked) {
                return;
            }
            if (autoTimer) {
                clearTimeout(autoTimer);
            }
            autoTimer = setTimeout(() => {
                runOptimize();
            }, 1200);
        });
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """
    Hugging Face Space homepage.

    Serve the interactive UI so opening the Space shows a real demo page.
    The live JSON execution results remain available at `GET /demo`.
    """
    return HTMLResponse(content=_demo_html())


@app.get("/health", response_model=CompatibilityHealthResponse)
def health_compat() -> CompatibilityHealthResponse:
    """Compatibility health route used by some OpenEnv reference environments."""
    return CompatibilityHealthResponse(status="healthy", service="acre-env")


@app.get("/demo")
def demo() -> JSONResponse:
    """Run all tasks and return JSON results."""
    from inference import run_all_tasks

    return JSONResponse(content={"results": run_all_tasks()})


@app.get("/ui", response_class=HTMLResponse)
def demo_ui() -> HTMLResponse:
    """Alias for the interactive UI (same as `/`)."""
    return HTMLResponse(content=_demo_html())


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    """Reset the environment. Optionally load a task's initial code."""
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
    """Take one refactoring step."""
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
    """Return full current environment state (OpenEnv spec requirement)."""
    return _state_response()


@app.get("/tasks", response_model=TasksResponse)
def list_tasks() -> TasksResponse:
    """Enumerate all tasks (easy → medium → hard)."""
    return TasksResponse(tasks=[TaskInfo.model_validate(t) for t in registry.list_tasks()])


@app.post("/tasks/{task_id}/grade", response_model=GradeResponse)
def grade(task_id: str, req: GradeRequest) -> GradeResponse:
    """Grade submitted code against a task's grader (returns score 0.0–1.0)."""
    task = registry.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    # Use the deterministic expected-output grader for the public grade endpoint.
    score = task.grade_against_expected(req.code)
    return GradeResponse(
        task_id=task_id,
        score=round(score, 4),
        passed=score >= 0.8,
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Run a full optimization episode and return code comparison artifacts."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
