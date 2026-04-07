from __future__ import annotations

import argparse

from acre.training.train_agent import TrainConfig, train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acre", description="ACRE: Autonomous Code Refactoring Environment")
    sub = parser.add_subparsers(dest="command", required=False)

    train_p = sub.add_parser("train", help="Run training (stub)")
    train_p.add_argument("--total-steps", type=int, default=100, help="Total training steps (stub)")

    sub.add_parser("demo", help="Run a small demo (stub)")

    return parser


def run_demo() -> None:
    # Placeholder for a future interactive/demo flow.
    print("ACRE demo mode is not implemented yet.")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo()
        return

    total_steps = getattr(args, "total_steps", 100)
    train(config=TrainConfig(total_steps=total_steps))


if __name__ == "__main__":
    main()

