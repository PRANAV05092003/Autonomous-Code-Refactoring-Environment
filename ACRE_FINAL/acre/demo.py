from __future__ import annotations

import os
import random
import sys
from typing import Any, Optional, Tuple

from acre.datasets.code_samples import CodeSample, CodeSampleDataset
from acre.env.refactor_env import RefactorEnv


def _load_model(path: str):
    """Load a Stable-Baselines3 PPO model if available; otherwise return None."""
    if not os.path.exists(path):
        return None
    try:
        from stable_baselines3 import PPO
    except Exception:
        return None
    try:
        return PPO.load(path)
    except Exception:
        return None


def _messy_sample_code() -> str:
    # Intentionally "messy" but valid Python for demo purposes.
    return (
        "def add(a,b):\n"
        "    x=0\n"
        "    for i in range(a):\n"
        "        x=x+1\n"
        "    if True:\n"
        "        x = x\n"
        "    if False:\n"
        "        y=123\n"
        "    else:\n"
        "        y=0\n"
        "    def f(p,q):\n"
        "        return p+q\n"
        "    r = f(x,y)\n"
        "    return r\n"
    )


def _format_code_block(code: str) -> str:
    return "\n".join(f"  {line}" for line in code.rstrip().splitlines()) + "\n"


def _safe_print(text: str) -> None:
    """
    Print text safely across Windows consoles (some default encodings can't print emojis).
    """
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        print(text, flush=True)
    except Exception:
        # Fall back to ASCII-friendly markers if emojis can't be encoded.
        safe = text.replace("✅", "[OK]").replace("⚠️", "[WARN]").replace("⚠", "[WARN]")
        print(safe, flush=True)


def _compute_runtime(executor: Any, code: str) -> float:
    """Best-effort runtime metric using the current executor contract."""
    try:
        res = executor.run(code, filename="demo.py")
        if getattr(res, "exit_code", 1) == 0 and isinstance(getattr(res, "metrics", None), dict):
            return float(res.metrics.get("runtime_s", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


def _choose_action(model: Any, obs, env: RefactorEnv, rng: random.Random) -> Tuple[int, str]:
    """Choose an action from the model, falling back to random."""
    n_actions = int(getattr(getattr(env, "action_space", None), "n", 5))
    if model is None:
        a = int(rng.randint(0, n_actions - 1))
        return a, "random"

    try:
        action, _state = model.predict(obs, deterministic=True)
        # SB3 may return scalar or 1-element array.
        if hasattr(action, "__len__"):
            a = int(action[0])
        else:
            a = int(action)
        return a, "ppo"
    except Exception:
        a = int(rng.randint(0, n_actions - 1))
        return a, "random"


def run_demo(*, model_path: str = "acre_agent.zip", seed: int = 0) -> None:
    rng = random.Random(seed)

    # Create a dataset with one messy sample so `reset()` loads it deterministically.
    dataset = CodeSampleDataset(
        [
            CodeSample(
                id="demo_sample",
                language="python",
                code=_messy_sample_code(),
            )
        ]
    )
    env = RefactorEnv(dataset=dataset, seed=seed)

    model = _load_model(model_path)
    model_status = "loaded" if model is not None else "not found (using random actions)"

    # Reset and capture the original code/metrics.
    obs, info = env.reset()
    original_code = getattr(env, "_code", "")
    original_complexity = float(getattr(env, "_compute_complexity")(original_code))
    original_runtime = _compute_runtime(env.executor, original_code)

    print("=" * 72)
    print("ACRE: Autonomous RL Code Refactoring Agent (5-step episode)")
    print(f"Model: {model_path} -> {model_status}")
    print(f"Sample: {info.get('sample_id')} ({info.get('language')})")
    print("=" * 72)
    print("\nORIGINAL CODE:\n")
    print(_format_code_block(original_code))

    total_reward = 0.0
    successful_transformations = 0
    steps_taken = 0

    for step_idx in range(1, 6):
        action, policy = _choose_action(model, obs, env, rng)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)
        steps_taken = step_idx

        action_name = step_info.get("action_name", "unknown")
        transform_meta = step_info.get("transform", {})
        if isinstance(transform_meta, dict) and bool(transform_meta.get("success", False)):
            successful_transformations += 1
        transformed_code = getattr(env, "_code", "")

        print("-" * 72)
        print(f"STEP {step_idx}/5")
        print(f"policy={policy} action={action} ({action_name})")
        print(f"transform={transform_meta}")
        print(f"reward={float(reward):.2f}  components={step_info.get('reward_components')}")
        print("\nUPDATED CODE:\n")
        print(_format_code_block(transformed_code))

        if terminated or truncated:
            break

    final_code = getattr(env, "_code", "")
    final_complexity = float(getattr(env, "_compute_complexity")(final_code))
    final_runtime = _compute_runtime(env.executor, final_code)

    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"total_reward: {total_reward:.2f}")
    print(f"complexity: {original_complexity:.0f} -> {final_complexity:.0f}")
    print(f"runtime_s:   {original_runtime:.4f} -> {final_runtime:.4f}")

    complexity_improvement = ((original_complexity - final_complexity) / max(original_complexity, 1.0)) * 100.0
    print(f"complexity improvement: {complexity_improvement:.2f}%")

    print("\nCHANGES APPLIED:")
    print(f"- Total steps: {steps_taken}")
    print(f"- Successful transformations: {successful_transformations}")

    if total_reward > 0:
        _safe_print("\n✅ Code improved successfully")
    else:
        _safe_print("\n⚠️ No significant improvement")

    print("\nFINAL CODE:\n")
    print(_format_code_block(final_code))

    env.close()


if __name__ == "__main__":
    run_demo()

