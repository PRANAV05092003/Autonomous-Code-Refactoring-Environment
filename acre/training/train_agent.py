from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from acre.env.refactor_env import RefactorEnv


@dataclass(frozen=True)
class TrainConfig:
    """Configuration stub for training."""

    total_steps: int = 5_000
    seed: Optional[int] = None
    model_path: str = "acre_agent.zip"


def train(*, env: Optional[RefactorEnv] = None, config: Optional[TrainConfig] = None) -> None:
    """
    Train a PPO agent on `RefactorEnv` using Stable-Baselines3.

    This is intentionally lightweight (hackathon-friendly) and focuses on a
    working demo: basic training loop, simple logging, and saving the model.
    """
    _config = config or TrainConfig()
    _env = env or RefactorEnv(seed=_config.seed)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:  # pragma: no cover
        print("Stable-Baselines3 is required for training. Install with `pip install -r requirements.txt`.")
        print(f"Import error: {e}")
        return None

    class EpisodeRewardPrinter(BaseCallback):
        """Print episode reward when an episode ends (via Monitor)."""

        def __init__(self) -> None:
            super().__init__()
            self.episode_count = 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                ep = info.get("episode") if isinstance(info, dict) else None
                if isinstance(ep, dict) and "r" in ep:
                    self.episode_count += 1
                    print(f"episode={self.episode_count} reward={ep['r']:.2f} length={int(ep.get('l', 0))}")
            return True

    # Wrap with Monitor so SB3 can compute episode stats and expose them in `info["episode"]`.
    def make_env() -> RefactorEnv:
        return Monitor(_env)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=0,
        seed=_config.seed,
        n_steps=64,
        batch_size=64,
    )

    print(f"Training PPO for {int(_config.total_steps)} timesteps...")
    model.learn(total_timesteps=int(_config.total_steps), callback=EpisodeRewardPrinter())

    model.save(_config.model_path)
    print(f"Saved model to {_config.model_path!r}")
    return None

