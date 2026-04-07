"""
ACRE pre-submission validator.

Checks the repository against the submission checklist and, when a server URL is
available, probes the HTTP API as well.

Run:
    python validate.py --url http://localhost:7860
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from typing import Any, Tuple

try:
    import requests
except ImportError:
    print("[ERROR] requests is required. Run: pip install requests")
    sys.exit(1)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    message = f"  {status}  {label}"
    if detail:
        message += f" - {detail}"
    print(message)
    return ok


def get(url: str, path: str, timeout: int = 15) -> Tuple[bool, Any]:
    try:
        response = requests.get(f"{url}{path}", timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def post(url: str, path: str, payload: dict, timeout: int = 15) -> Tuple[bool, Any]:
    try:
        response = requests.post(f"{url}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def run_validation(base_url: str) -> int:
    failures = 0

    print("\n" + "=" * 60)
    print("  ACRE Pre-Submission Validator")
    print("=" * 60)
    print(f"  Target: {base_url}\n")

    print("1. Static repository checks")
    try:
        interface_src = read_text("openenv_interface.py")
        tree = ast.parse(interface_src)
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        env_cls = classes.get("OpenEnvRefactorEnv")
        failures += 0 if check("openenv_interface.py exists", True) else 1
        failures += 0 if check("OpenEnvRefactorEnv is defined", env_cls is not None) else 1
        if env_cls is not None:
            methods = {node.name for node in env_cls.body if isinstance(node, ast.FunctionDef)}
            for method_name in ["reset", "step", "state"]:
                failures += 0 if check(
                    f"OpenEnvRefactorEnv implements {method_name}()",
                    method_name in methods,
                ) else 1
    except FileNotFoundError:
        failures += 1
        check("openenv_interface.py exists", False, "file not found")

    try:
        models_src = read_text("models.py")
        for name in ["ObservationModel", "ActionModel", "RewardModel"]:
            failures += 0 if check(
                f"{name} is defined in models.py",
                f"class {name}" in models_src,
            ) else 1
    except FileNotFoundError:
        failures += 1
        check("models.py exists", False, "file not found")

    print("\n2. Health check (GET /)")
    ok, data = get(base_url, "/")
    failures += 0 if check("GET / returns HTTP 200", ok) else 1
    if ok:
        failures += 0 if check(
            "Response has status field",
            isinstance(data, dict) and "status" in data,
            str(data),
        ) else 1

    print("\n3. Tasks (GET /tasks)")
    ok, data = get(base_url, "/tasks")
    failures += 0 if check("GET /tasks returns 200", ok) else 1
    if ok:
        tasks = data.get("tasks", []) if isinstance(data, dict) else []
        failures += 0 if check("At least 3 tasks defined", len(tasks) >= 3, f"found {len(tasks)}") else 1
        difficulties = [t.get("difficulty", "") for t in tasks]
        for diff in ["easy", "medium", "hard"]:
            failures += 0 if check(f"Task with difficulty '{diff}' exists", diff in difficulties) else 1
        for task in tasks:
            failures += 0 if check(
                f"Task '{task.get('id')}' has initial_code",
                bool(task.get("initial_code")),
            ) else 1

    print("\n4. Reset (POST /reset)")
    ok, data = post(base_url, "/reset", {})
    failures += 0 if check("POST /reset returns 200", ok) else 1
    if ok:
        observation = data.get("observation", {})
        failures += 0 if check("Response has observation field", isinstance(observation, dict)) else 1
        failures += 0 if check(
            "Observation is typed with 4 fields",
            {"code_length", "complexity_score", "runtime_s", "error_flag"}.issubset(observation),
            str(observation),
        ) else 1

    ok, _ = post(base_url, "/reset", {"task_id": "rename_variables"})
    failures += 0 if check("POST /reset with task_id works", ok) else 1

    print("\n5. State (GET /state)")
    ok, data = get(base_url, "/state")
    failures += 0 if check("GET /state returns 200", ok) else 1
    if ok:
        required_keys = [
            "current_code",
            "episode_steps",
            "max_steps",
            "complexity",
            "observation",
            "observation_vector",
            "action_meanings",
        ]
        for key in required_keys:
            failures += 0 if check(f"State has '{key}' field", key in data) else 1

    print("\n6. Step (POST /step)")
    post(base_url, "/reset", {"task_id": "rename_variables"})
    for action in range(5):
        ok, data = post(base_url, "/step", {"action": action})
        failures += 0 if check(
            f"Action {action} executes without error",
            ok and isinstance(data, dict) and "reward" in data and "done" in data,
        ) else 1
        if ok:
            reward_payload = data.get("reward", {})
            norm = reward_payload.get("normalized", -1)
            failures += 0 if check(
                f"Action {action} returns typed reward payload",
                {"raw", "normalized", "components"}.issubset(reward_payload),
                str(reward_payload),
            ) else 1
            failures += 0 if check(
                f"Action {action} normalized_reward in [0,1]",
                isinstance(norm, (int, float)) and 0.0 <= float(norm) <= 1.0,
                f"got {norm}",
            ) else 1
            if data.get("done"):
                break

    ok, data = post(base_url, "/step", {"action": 99})
    check("Invalid action returns error (not crash)", not ok or "detail" in str(data), "(expected 4xx)")

    print("\n7. Task graders (POST /tasks/{id}/grade)")
    for task_id in ["rename_variables", "remove_dead_code", "full_refactor"]:
        ok, data = post(base_url, f"/tasks/{task_id}/grade", {"code": "def f(): pass"})
        failures += 0 if check(f"Grade endpoint for '{task_id}' works", ok) else 1
        if ok:
            score = data.get("score", -1)
            failures += 0 if check(
                f"Score for '{task_id}' in [0.0, 1.0]",
                isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0,
                f"got {score}",
            ) else 1

    print("\n8. openenv.yaml")
    try:
        openenv_yaml = read_text("openenv.yaml")
        failures += 0 if check("openenv.yaml exists", True) else 1
        for field in ["tasks:", "action_space:", "observation_space:", "reward:", "entrypoint:", "validation:"]:
            failures += 0 if check(f"openenv.yaml has '{field}' section", field in openenv_yaml) else 1
    except FileNotFoundError:
        failures += 1
        check("openenv.yaml exists", False, "file not found")

    print("\n9. inference.py")
    try:
        inference_src = read_text("inference.py")
        failures += 0 if check("inference.py exists", True) else 1
        for marker in ['"event": "START"', '"event": "STEP"', '"event": "END"']:
            failures += 0 if check(f"inference.py emits {marker}", marker in inference_src) else 1
        failures += 0 if check(
            "Uses OpenAI client",
            "from openai import OpenAI" in inference_src,
        ) else 1
        for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN", "ENV_URL", "LOCAL_IMAGE_NAME"]:
            failures += 0 if check(f"inference.py reads {var} from env", var in inference_src) else 1
        failures += 0 if check(
            "API_BASE_URL has a default",
            'os.getenv("API_BASE_URL", "https://api.openai.com/v1")' in inference_src,
        ) else 1
        failures += 0 if check(
            "MODEL_NAME has a default",
            'os.getenv("MODEL_NAME", "gpt-4o-mini")' in inference_src,
        ) else 1
        failures += 0 if check(
            "HF_TOKEN has no default",
            re.search(r'HF_TOKEN\s*:\s*.*os\.getenv\("HF_TOKEN"\)', inference_src) is not None,
        ) else 1
    except FileNotFoundError:
        failures += 1
        check("inference.py exists", False, "file not found")

    print("\n10. Dockerfile")
    try:
        dockerfile = read_text("Dockerfile")
        failures += 0 if check("Dockerfile exists", True) else 1
        failures += 0 if check("Exposes port 7860", "7860" in dockerfile) else 1
        failures += 0 if check("Has CMD/ENTRYPOINT", "CMD" in dockerfile or "ENTRYPOINT" in dockerfile) else 1
        failures += 0 if check("Does not set a default HF_TOKEN", "ENV HF_TOKEN" not in dockerfile) else 1
    except FileNotFoundError:
        failures += 1
        check("Dockerfile exists", False, "file not found")

    print("\n11. README / Hugging Face metadata")
    try:
        readme = read_text("README.md")
        failures += 0 if check("README has docker SDK front matter", "sdk: docker" in readme) else 1
        failures += 0 if check("README includes openenv tag", "openenv" in readme) else 1
        for section in [
            "Environment Overview and Motivation",
            "Definitions of Action and Observation Spaces",
            "Task Descriptions with Expected Difficulty Levels",
            "Setup and Usage Instructions",
            "Baseline Performance Scores",
        ]:
            failures += 0 if check(f"README includes '{section}'", section in readme) else 1
    except FileNotFoundError:
        failures += 1
        check("README.md exists", False, "file not found")

    print("\n" + "=" * 60)
    if failures == 0:
        print(f"  {PASS}  All checks passed. Repository is submission-ready.")
    else:
        print(f"  {FAIL}  {failures} check(s) failed. Fix before submitting.")
    print("=" * 60 + "\n")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="ACRE pre-submission validator")
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL of the running ACRE server",
    )
    args = parser.parse_args()
    sys.exit(run_validation(args.url))


if __name__ == "__main__":
    main()
