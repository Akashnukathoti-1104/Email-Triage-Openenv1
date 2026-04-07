"""
inference.py — Email Triage OpenEnv Baseline Inference Script

BEFORE RUNNING, SET THESE:
  API_KEY        API key injected by validator (REQUIRED - do not bypass)
  API_BASE_URL   LLM proxy endpoint injected by validator (REQUIRED - do not bypass)
  MODEL_NAME     Model name (default: Qwen/Qwen2.5-72B-Instruct)
  SERVER_URL     Env server (default: http://localhost:8000)
  TASK           easy | medium | hard | all (default: all)

Windows PowerShell:
  $env:API_KEY="your_key"
  $env:API_BASE_URL="https://proxy.example.com/v1"
  $env:SERVER_URL="https://nukathoti-email-triage-openenv1.hf.space"
  python inference.py

Linux / Mac:
  export API_KEY="your_key"
  export API_BASE_URL="https://proxy.example.com/v1"
  export SERVER_URL="https://nukathoti-email-triage-openenv1.hf.space"
  python inference.py
"""

import json
import os
import sys
from typing import List, Optional

import requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
# Use exactly API_KEY and API_BASE_URL as injected by the validator
API_KEY      = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL   = os.environ.get("SERVER_URL",   "http://localhost:8000").rstrip("/")
TASK         = os.environ.get("TASK",         "all")
BENCHMARK    = "email-triage-openenv"
MAX_STEPS    = 15
TEMPERATURE  = 0.1

SYSTEM_PROMPT = """You are an expert email triage assistant. Classify each email into exactly one category:

- spam: Unwanted, phishing, lottery scams, unsolicited promotions, fake alerts
- important: Requires action — deadlines, client messages, security alerts, approvals, urgent issues
- normal: Routine, informational, newsletters, automated digests, FYI updates

Respond ONLY with a JSON object. No markdown, no explanation:
{"category": "spam"}

Valid values: spam, important, normal"""


# ── Mandatory log format ──────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment helpers ───────────────────────────────────────────────────────
def env_reset(task_name: str) -> dict:
    resp = requests.post(
        f"{SERVER_URL}/reset",
        json={"task": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(email_id: str, category: str) -> dict:
    resp = requests.post(
        f"{SERVER_URL}/step",
        json={"action": {"email_id": email_id, "category": category}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── LLM classification ────────────────────────────────────────────────────────
def classify_email(client: OpenAI, obs: dict) -> str:
    user_prompt = (
        f"Classify this email:\n\n"
        f"From: {obs.get('sender', '')}\n"
        f"Subject: {obs.get('subject', '')}\n"
        f"Body: {obs.get('body', '')}\n\n"
        f'Respond ONLY with JSON: {{"category": "spam"|"important"|"normal"}}'
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=50,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        cat = parsed.get("category", "normal").lower()
        return cat if cat in ("spam", "important", "normal") else "normal"
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return "normal"


# ── Episode runner ────────────────────────────────────────────────────────────
def run_task(task_name: str, client: OpenAI) -> None:
    rewards:     List[float] = []
    steps_taken: int   = 0
    score:       float = 0.0
    success:     bool  = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_name)
        obs    = result.get("observation", result)
        done   = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done or obs.get("email_id") == "done":
                break

            email_id   = obs.get("email_id", "")
            category   = classify_email(client, obs)
            action_str = f"classify(email_id={email_id!r},category={category!r})"

            result  = env_step(email_id, category)
            obs     = result.get("observation", result)
            reward  = float(result.get("reward", 0.0))
            done    = result.get("done", False)
            error   = result.get("error") or None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        total   = obs.get("total_emails") or max(steps_taken, 1)
        correct = sum(1 for r in rewards if r >= 1.0)
        score   = min(max(correct / total, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score   = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] No API key found!\n"
            "  Set API_KEY environment variable (injected by validator).\n"
            "  Fallback: OPENAI_API_KEY or HF_TOKEN",
            flush=True,
        )
        sys.exit(1)

    # Initialize OpenAI client with API_BASE_URL and API_KEY as injected
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    tasks = ["easy", "medium", "hard"] if TASK == "all" else [TASK]

    for task_name in tasks:
        run_task(task_name, client)
        print("", flush=True)


if __name__ == "__main__":
    main()
