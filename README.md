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
START rename_variables
STEP 0
END 1.00

START remove_dead_code
STEP 1
END 0.25

START full_refactor
STEP 3
END 0.71

Final Score: 0.65
```

---

## 🏗️ Architecture

- `server.py` → FastAPI entry point
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

```bash
pip install -r requirements.txt
python server.py
```

---

## 🐳 Docker / Hugging Face Spaces

```bash
docker build -t acre .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_key \
  -e ENV_URL=http://localhost:7860 \
  acre
```

---

## 🧪 Inference

Set environment variables:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key
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
