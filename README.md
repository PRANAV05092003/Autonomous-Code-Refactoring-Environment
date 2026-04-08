---
title: ACRE - Autonomous Code Refactoring Environment
colorFrom: blue
colorTo: green
sdk: docker
app_file: server.py
app_port: 7860
pinned: false
license: mit
tags:
  - openenv
---

# ACRE - Autonomous Code Refactoring Environment

ACRE is an OpenEnv-compatible environment for autonomous Python code refactoring. An agent receives real code-cleanup tasks and must improve the code through AST-based transformations while receiving dense reward feedback for correctness, simplification, and performance.

## Environment Overview and Motivation

This project simulates a realistic developer workflow: cleaning up messy Python code, removing dead logic, simplifying loops, and inlining trivial helpers. The canonical OpenEnv wrapper lives in `openenv_interface.py`, while the original Gymnasium-compatible environment remains available for RL training and demos.

## Definitions of Action and Observation Spaces

### Action Space - Discrete(5)

| Action | Name | Description |
|---|---|---|
| 0 | rename_variable | Rename generic variables like `x`, `tmp`, and `i` |
| 1 | remove_dead_code | Remove unreachable statements, `if False` branches, and unused assignments |
| 2 | simplify_loop | Convert append-loops into list comprehensions |
| 3 | optimize_condition | Simplify `not not x`, `if True`, `if False`, and boolean comparisons |
| 4 | inline_function | Inline simple single-return module-level functions |

### Observation Space - Box(4,)

The environment tracks:

- `code_length`
- `complexity_score`
- `runtime_s`
- `error_flag`

### Typed OpenEnv Models

The submission-facing interface uses Pydantic models in `models.py`:

- `ObservationModel`
- `ActionModel`
- `RewardModel`
- `StateResponse`

The canonical interface is:

```python
observation = env.reset(...)
observation, reward, done, info = env.step(action)
state = env.state()
```

## Task Descriptions with Expected Difficulty Levels

| Task ID | Difficulty | Objective |
|---|---|---|
| `rename_variables` | Easy | Remove generic variable names from the snippet |
| `remove_dead_code` | Medium | Eliminate dead branches, unreachable code, and unused assignments |
| `full_refactor` | Hard | Combine renaming, dead-code removal, loop simplification, condition optimization, and inlining |

Each task includes a deterministic AST-based grader returning a score in `[0.0, 1.0]`.

## Reward Design

Rewards are shaped throughout the trajectory instead of only at the end.

- Success reward for syntactically valid, executable output
- Complexity reward when control-flow complexity decreases
- Performance reward when runtime improves
- Error penalty for invalid or failing code
- No-change penalty to discourage loops and unproductive actions

Raw reward range is `[-32, 20]`, normalized to `[0.0, 1.0]` with `(raw + 32) / 52`.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Compatibility health check |
| POST | `/reset` | Reset environment and return typed observation/state |
| POST | `/step` | Apply one action and return typed observation/reward/done |
| GET | `/state` | Return the current typed state |
| GET | `/tasks` | List available tasks |
| POST | `/tasks/{task_id}/grade` | Grade submitted code |

## Setup and Usage Instructions

### Local setup

```bash
pip install -r requirements.txt
python server.py
```

### Baseline inference

Set environment variables before running:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key
export ENV_URL=http://localhost:7860
python inference.py
```

Notes:

- `API_BASE_URL` and `MODEL_NAME` have defaults in `inference.py`
- `HF_TOKEN` is optional because the script falls back to a deterministic heuristic baseline
- `LOCAL_IMAGE_NAME` is read for evaluator compatibility when using a local Docker image launcher

### Docker / Hugging Face Spaces

```bash
docker build -t acre .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_key \
  -e ENV_URL=http://localhost:7860 \
  acre
```

The repository is configured for a Docker-based Hugging Face Space and includes the `openenv` tag in the front matter.

## Validation

Run the repository validator:

```bash
python validate.py --url http://localhost:7860
```

When using the official hackathon tooling, also run:

```bash
openenv validate
```

## Interactive Demo

Start the server and open:

```text
http://localhost:7860/demo
```

The demo shows:

- Original code
- Optimized code
- Unified diff
- Per-step action and reward logs

## Baseline Performance Scores

The deterministic fallback policy used by `inference.py` produces the following reproducible task scores:

| Task | Score |
|---|---|
| `rename_variables` | 1.0000 |
| `remove_dead_code` | 0.2500 |
| `full_refactor` | 0.7143 |
| Average | 0.6548 |

These scores come from the built-in heuristic policy with `HF_TOKEN` unset, which keeps the baseline reproducible across runs.
