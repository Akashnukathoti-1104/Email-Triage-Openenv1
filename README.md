---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 📧 Email Triage OpenEnv

A real-world OpenEnv environment where an AI agent triages emails — classifying each as `spam`, `important`, or `normal`. This is a task every knowledge worker performs daily.

## Environment Description

Email overload is a universal productivity problem. This environment simulates an inbox where an agent reads one email at a time and must classify it before seeing the next. A well-trained agent could save hours of human attention every week.

## Action & Observation Spaces

**Action**
| Field | Type | Description |
|---|---|---|
| `email_id` | string | Must match the current email's ID |
| `category` | enum | `spam` \| `important` \| `normal` |

**Observation**
| Field | Type | Description |
|---|---|---|
| `email_id` | string | Current email's unique ID |
| `sender` | string | Sender email address |
| `subject` | string | Subject line |
| `body` | string | Email body |
| `emails_remaining` | int | Emails left in queue |
| `total_emails` | int | Total emails this episode |
| `current_score` | float | Running accuracy |
| `reward` | float | Reward for last action |
| `done` | bool | Episode complete |

## Tasks

| Task | Emails | Difficulty | Notes |
|---|---|---|---|
| `easy` | 3 | ⭐ Easy | Clear obvious signals |
| `medium` | 5 | ⭐⭐ Medium | Some ambiguity |
| `hard` | 10 | ⭐⭐⭐ Hard | Nuanced judgment required |

## Reward Function

| Outcome | Reward | Description |
|---|---|---|
| Correct classification | `1.0` | Exact match with ground truth |
| Borderline confusion | `0.5` | `normal` ↔ `important` mix-up (ambiguous) |
| Wrong classification | `0.0` | Clearly incorrect category |
| Invalid category | `-0.1` | Unrecognized value — penalizes bad behavior |

- **Episode score**: `sum(rewards) / total_emails` ∈ [0.0, 1.0]
- Dense reward signal — feedback at every step, not just episode end
- Partial credit for borderline ambiguous cases (normal ↔ important)
- Penalty for undesirable behavior (invalid/garbage actions)

## Setup & Usage

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | Your Hugging Face API key |
| `OPENAI_API_KEY` | Alt | OpenAI API key (alternative to HF_TOKEN) |
| `API_BASE_URL` | No | LLM endpoint (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | No | Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `SERVER_URL` | No | Env server URL (default: `http://localhost:8000`) |

### Local

```bash
pip install "openenv-core[core]>=0.2.2" openai requests fastapi "uvicorn[standard]" pydantic
git clone https://huggingface.co/spaces/Nukathoti/email-triage-openenv
cd email-triage-openenv
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal (Linux/Mac):
export HF_TOKEN="hf_your_token_here"
export OPENAI_API_KEY="sk_your_key_here"   # alternative
export SERVER_URL="http://localhost:8000"
python inference.py

# Windows PowerShell:
# $env:HF_TOKEN="hf_your_token_here"
# $env:SERVER_URL="http://localhost:8000"
# python inference.py
```

### Against Live Space

```bash
export HF_TOKEN="hf_your_token_here"
export SERVER_URL="https://nukathoti-email-triage-openenv.hf.space"
python inference.py
```

### Docker

```bash
docker build -t email-triage .
docker run -p 7860:7860 -e HF_TOKEN=hf_your_token email-triage
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — returns `{"status": "healthy"}` |
| GET | `/metadata` | Environment name, description, tasks |
| GET | `/schema` | Action/observation/state JSON schemas |
| POST | `/reset` | Start episode. Body: `{"task": "easy\|medium\|hard"}` |
| POST | `/step` | Classify email. Body: `{"action": {"email_id": "...", "category": "..."}}` |
| GET | `/state` | Current episode state (safe to call before reset) |

## Baseline Scores

Model: `Qwen/Qwen2.5-72B-Instruct`

| Task | Score |
|---|---|
| easy | 1.000 |
| medium | 1.000 |
| hard | 0.900 |

## Project Structure

```
├── models.py                        # Action & Observation types + reward function
├── client.py                        # HTTP client
├── server/
│   ├── app.py                       # FastAPI server
│   ├── email_triage_environment.py  # Environment logic
│   └── Dockerfile                   # Container (alternative)
├── inference.py                     # Baseline script
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Package config
└── README.md
```
