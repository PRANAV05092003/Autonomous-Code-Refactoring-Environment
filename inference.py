"""
ACRE inference script for OpenEnv submission evaluation.

Environment variables:
  - API_BASE_URL: LLM API endpoint (default allowed)
  - MODEL_NAME: model identifier (default allowed)
  - HF_TOKEN: API token for the OpenAI-compatible endpoint (NO default)
  - ENV_URL: running ACRE server base URL (required)
  - LOCAL_IMAGE_NAME: present for evaluator compatibility (optional)
  - USE_LLM: set to "1" to enable LLM action selection when HF_TOKEN is set

STRICT stdout format (do not change):
  START <task_id>
  STEP <action_int>
  END <score_float>
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
from openai import OpenAI

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
ENV_URL: str | None = os.getenv("ENV_URL")
LOCAL_IMAGE_NAME: str | None = os.getenv("LOCAL_IMAGE_NAME")

TASKS: List[str] = ["rename_variables", "remove_dead_code", "full_refactor"]

ACTION_MEANINGS: Dict[int, str] = {
    0: "rename_variable",
    1: "remove_dead_code",
    2: "simplify_loop",
    3: "optimize_condition",
    4: "inline_function",
}

SYSTEM_PROMPT = """\
You are an RL agent that refactors Python code. Choose one action per step.

Actions:
  0 rename_variable   - rename generic names (x, tmp, i) to descriptive ones
  1 remove_dead_code  - remove unreachable stmts, if False blocks, unused vars
  2 simplify_loop     - convert append-loops to list comprehensions
  3 optimize_condition- simplify 'not not x', 'if True/False', 'x==True'
  4 inline_function   - inline simple single-return module-level functions

Respond ONLY with valid JSON (no markdown):
{"action": <0-4>, "reason": "<one sentence>"}"""


def _env_url() -> str:
    if ENV_URL:
        return ENV_URL.rstrip("/")
    raise RuntimeError("ENV_URL must be set before running inference.py")


def _post(path: str, payload: dict | None = None) -> dict:
    response = requests.post(f"{_env_url()}{path}", json=payload or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def _get(path: str) -> dict:
    response = requests.get(f"{_env_url()}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def reset_env(task_id: str) -> dict:
    return _post("/reset", {"task_id": task_id})


def step_env(action: int) -> dict:
    return _post("/step", {"action": action})


def get_state() -> dict:
    return _get("/state")


def grade(task_id: str, code: str) -> float:
    response = requests.post(
        f"{_env_url()}/tasks/{task_id}/grade",
        json={"code": code},
        timeout=30,
    )
    response.raise_for_status()
    return float(response.json().get("score", 0.0))


def choose_action(client: Optional[OpenAI], state: dict, task_id: str) -> Tuple[int, str]:
    def heuristic_action() -> Tuple[int, str]:
        code = str(state.get("current_code", ""))
        step_i = int(state.get("episode_steps", 0))

        has_generic = re.search(r"\b(x|tmp|i)\b", code) is not None
        has_if_false = re.search(r"\bif\s+False\b", code) is not None
        has_if_true = re.search(r"\bif\s+True\b", code) is not None
        has_append_loop = ".append(" in code and "for " in code
        has_double_not = "not not" in code
        has_add_call = "add(" in code

        if task_id == "rename_variables":
            if has_generic:
                return 0, "heuristic: remove generic names first"
            if has_if_false or "unused" in code:
                return 1, "heuristic: remove dead code"
            if has_append_loop:
                return 2, "heuristic: simplify loop"
            if has_if_true or has_double_not:
                return 3, "heuristic: optimize conditions"
            return 4, "heuristic: inline simple function"

        if task_id == "remove_dead_code":
            if has_if_false or "unused" in code:
                return 1, "heuristic: remove dead code patterns"
            if has_append_loop:
                return 2, "heuristic: convert append-loop"
            if has_if_true or has_double_not:
                return 3, "heuristic: simplify conditions"
            if has_generic:
                return 0, "heuristic: clean generic names"
            return 4, "heuristic: inline helper"

        if has_generic:
            return 0, "heuristic: rename generic variables"
        if has_append_loop:
            return 2, "heuristic: simplify loop into listcomp"
        if has_if_false or has_if_true or has_double_not:
            return 3, "heuristic: optimize boolean branches"
        if has_add_call:
            return 4, "heuristic: inline add() call"
        if step_i >= 2:
            return 1, "heuristic: remove remaining dead code"
        return 3, "heuristic: condition optimization as safe default"

    use_llm = bool(HF_TOKEN) and os.getenv("USE_LLM", "0") == "1"
    if (not use_llm) or client is None:
        return heuristic_action()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {task_id}\n"
                f"Steps remaining: {state.get('max_steps', 5) - state.get('episode_steps', 0)}\n"
                f"Complexity: {state.get('complexity', 0)}\n\n"
                f"Current code:\n```python\n{state.get('current_code', '')}\n```\n\n"
                "Choose the best action."
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
        json_blob = raw

        if "{" not in json_blob or "}" not in json_blob:
            return heuristic_action()

        match = re.search(r"\{.*\}", json_blob, flags=re.DOTALL)
        if match:
            json_blob = match.group(0)

        parsed = json.loads(json_blob)
        action = int(parsed.get("action", -1))
        reason = str(parsed.get("reason", ""))
        if 0 <= action <= 4:
            return action, reason or "llm-selected action"
        return heuristic_action()
    except Exception:
        return heuristic_action()


def run_episode(client: Optional[OpenAI], task_id: str, episode_num: int) -> float:
    reset_env(task_id)
    state = get_state()

    # STRICT logging format required by evaluator.
    print(f"START {task_id}", flush=True)

    cumulative_reward = 0.0

    for step_num in range(1, 6):
        action, reason = choose_action(client, state, task_id)
        result = step_env(action)
        state = get_state()

        reward_payload = result.get("reward", {})
        raw_reward = float(reward_payload.get("raw", 0.0))
        norm_reward = float(reward_payload.get("normalized", (raw_reward + 32) / 52))
        cumulative_reward += raw_reward

        # STRICT logging format required by evaluator.
        print(f"STEP {int(action)}", flush=True)

        if result.get("done") or result.get("terminated") or result.get("truncated"):
            break

    final_state = get_state()
    task_score = grade(task_id, final_state.get("current_code", ""))

    # STRICT logging format required by evaluator.
    print(f"END {task_score:.4f}", flush=True)

    return task_score


def main() -> None:
    if not ENV_URL:
        raise SystemExit("ENV_URL is required. Example: ENV_URL=http://localhost:7860")

    client: Optional[OpenAI] = None
    if HF_TOKEN and os.getenv("USE_LLM", "0") == "1":
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    scores: List[float] = []
    for i, task_id in enumerate(TASKS, start=1):
        score = run_episode(client, task_id, i)
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    sys.exit(0 if avg_score >= 0.5 else 1)


if __name__ == "__main__":
    main()
