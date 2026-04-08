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

# 🚀 ACRE — Autonomous Code Refactoring Environment

> OpenEnv-powered AI system for real-world code optimization, refactoring, and evaluation.

![Status](https://img.shields.io/badge/Status-Running-success)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-green)

---

## 🔥 Overview

ACRE is an OpenEnv-compliant environment designed to simulate real-world software engineering workflows such as code cleanup, optimization, and refactoring using AI agents.

It enables agents to iteratively improve code through structured actions while receiving dense, step-wise reward feedback.

## Environment Overview and Motivation

ACRE models a realistic developer workflow where an agent incrementally improves Python code quality under a fixed action budget.
The environment is designed for OpenEnv Round 1 requirements: typed APIs, deterministic grading, multi-difficulty tasks, and reproducible inference behavior.

---

## 💡 Why This Matters

Modern software systems require automated code optimization and intelligent tooling.

ACRE enables:
- 🤖 AI coding assistants
- 🔍 Automated code review systems
- ⚡ Reinforcement learning-based optimization agents
- 🧠 Learning real developer workflows

---

## 🔄 How It Works

Code → Action → Refactor → Reward → Repeat

1. Load messy code
2. Apply transformation
3. Evaluate using grader
4. Compute reward
5. Iterate until optimal

---

## 🧠 Key Features

- ✅ Autonomous code refactoring
- ⚡ Step-wise reward feedback
- 🧪 OpenEnv compliant interface
- 📊 Deterministic grading system
- 🔁 Reproducible inference pipeline
- 🐳 Fully containerized (Docker + Hugging Face Spaces)

---

## 📂 Tasks

| Task ID | Difficulty | Objective |
|--------|----------|----------|
| `rename_variables` | Easy | Replace generic variable names |
| `remove_dead_code` | Medium | Remove unreachable logic |
| `full_refactor` | Hard | Combine multiple optimizations |

Each task uses AST-based transformations and deterministic grading.

## Task Descriptions with Expected Difficulty Levels

- Easy (`rename_variables`): rename generic names like `x`, `tmp`, `i` into descriptive identifiers.
- Medium (`remove_dead_code`): remove unreachable branches and unused assignments while preserving behavior.
- Hard (`full_refactor`): combine renaming, dead-code elimination, loop simplification, condition cleanup, and helper inlining.

---

## 🎯 Reward System

Rewards are computed at every step:

- ✅ Valid executable code → positive reward
- 📉 Reduced complexity → reward
- ⚡ Improved performance → reward
- ❌ Errors or invalid code → penalty
- 🔁 No progress → penalty

**Normalization:**

`(raw_reward + 32) / 52 → [0, 1]`

---

## 📊 Example Execution

```text
[START] task=rename_variables
[STEP] action=0
[END] task=rename_variables score=1.00

[START] task=remove_dead_code
[STEP] action=1
[END] task=remove_dead_code score=0.25

[START] task=full_refactor
[STEP] action=3
[END] task=full_refactor score=0.71

Final Score: 0.65
```

---

## 🏗️ Architecture

- `server/app.py` → FastAPI entry point used by OpenEnv + Docker
- `server.py` → legacy local runner / UI helper
- `openenv_interface.py` → OpenEnv wrapper
- `acre/env/` → Core environment logic
- `acre/tasks/` → Task definitions
- `acre/utils/` → Metrics and helpers
- `inference.py` → Evaluation pipeline

---

## ⚙️ OpenEnv Interface

```python
observation = env.reset()
observation, reward, done, info = env.step(action)
state = env.state()
```

Uses Pydantic models:

- `ObservationModel`
- `ActionModel`
- `RewardModel`

## Definitions of Action and Observation Spaces

- Observation space: Box(4) with fields `code_length`, `complexity_score`, `runtime_s`, `error_flag`.
- Action space: Discrete(5) with actions `rename_variable`, `remove_dead_code`, `simplify_loop`, `optimize_condition`, `inline_function`.

---

## 🌐 HTTP API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Compatibility check |
| POST | `/reset` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Get state |
| GET | `/tasks` | List tasks |
| POST | `/tasks/{task_id}/grade` | Grade code |

---

## 🚀 Run Locally

## Setup and Usage Instructions

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker / Hugging Face Spaces

```bash
docker build -t acre .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e API_KEY=your_key \
  -e ENV_URL=http://localhost:7860 \
  acre
```

---

## 🧪 Inference

Set environment variables:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export API_KEY=your_key
export ENV_URL=http://localhost:7860
```

Run:

```bash
python inference.py
```

Expected output:

```text
Easy: 1.00
Medium: 0.25
Hard: 0.71
Final: 0.65
```

---

## 📌 OpenEnv Compliance

- ✔ `step()` implemented
- ✔ `reset()` implemented
- ✔ `state()` implemented
- ✔ reward shaping
- ✔ deterministic grading
- ✔ structured logs

---

## 🧪 Validation

```bash
python validate.py --url http://localhost:7860
```

Or:

```bash
openenv validate
```

---

## 🌐 Live Demo

👉 Running on Hugging Face Spaces

---

## 📊 Baseline Performance

## Baseline Performance Scores

| Task | Score |
|---|---|
| `rename_variables` | 1.0000 |
| `remove_dead_code` | 0.2500 |
| `full_refactor` | 0.7143 |
| Average | 0.6548 |

---

## 🏆 Use Cases

- AI-powered code optimization
- Automated refactoring tools
- Reinforcement learning environments
- Developer productivity systems

---

## 📜 License

MIT License
