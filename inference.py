import json
import os
import sys
from typing import List, Optional

import requests
from openai import OpenAI

# ── ✅ STRICT OpenEnv config (NO fallback allowed) ─────────────────────────────
# CRITICAL: These MUST come from the validator's injected environment
if not os.environ.get("API_KEY") or not os.environ.get("API_BASE_URL"):
    print("[FATAL] Missing API_KEY or API_BASE_URL in environment.", flush=True)
    sys.exit(1)

MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
SERVER_URL   = os.getenv("SERVER_URL", "http://localhost:8000").rstrip("/")
TASK         = os.getenv("TASK", "all")
BENCHMARK    = "email-triage-openenv"
MAX_STEPS    = 15
TEMPERATURE  = 0.1
MIN_SCORE_EPS = 1e-6


def build_client() -> OpenAI:
    # Keep this exact initialization pattern for validator compatibility.
    return OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )


def warmup_proxy_call(client: OpenAI) -> None:
    """Force a lightweight call through the injected LiteLLM proxy."""
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return exactly: ok"},
                {"role": "user", "content": "ok"},
            ],
            temperature=0,
            max_tokens=2,
        )
        print("[DEBUG] Proxy warmup call succeeded.", flush=True)
    except Exception as e:
        print(f"[DEBUG] Proxy warmup call failed: {e}", flush=True)

SYSTEM_PROMPT = """You are an expert email triage assistant. Classify each email into exactly one category:

- spam: Unwanted, phishing, lottery scams, unsolicited promotions, fake alerts
- important: Requires action — deadlines, client messages, security alerts, approvals, urgent issues
- normal: Routine, informational, newsletters, automated digests, FYI updates

Respond ONLY with a JSON object:
{"category": "spam"}"""


# ── ✅ LOGGING (required format) ─────────────────────────────────────────────
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


def strict_open_interval_score(value: float) -> float:
    """Clamp score to strict (0, 1) interval required by validator."""
    return min(max(value, MIN_SCORE_EPS), 1.0 - MIN_SCORE_EPS)


# ── ENV API ─────────────────────────────────────────────────────────────────
def env_reset(task_name: str) -> dict:
    resp = requests.post(f"{SERVER_URL}/reset", json={"task": task_name}, timeout=30)
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


# ── ✅ LLM CALL (THIS IS WHAT VALIDATOR CHECKS) ─────────────────────────────
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
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=50,
        )

        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw)
        category = parsed.get("category", "normal").lower()

        return category if category in ("spam", "important", "normal") else "normal"

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "normal"


# ── RUN TASK ────────────────────────────────────────────────────────────────
def run_task(task_name: str, client: OpenAI) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset(task_name)
        obs = result.get("observation", result)
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done or obs.get("email_id") == "done":
                break

            email_id = obs.get("email_id", "")
            category = classify_email(client, obs)

            action_str = f"classify(email_id={email_id!r},category={category!r})"

            result = env_step(email_id, category)
            obs = result.get("observation", result)
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            error = result.get("error") or None

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error)

            if done:
                break

        total = obs.get("total_emails") or max(steps_taken, 1)
        correct = sum(1 for r in rewards if r >= 1.0)
        score = strict_open_interval_score(correct / total)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        raw_score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        score = strict_open_interval_score(raw_score)
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    try:
        print(f"[DEBUG] Using BASE URL: {os.environ['API_BASE_URL']}", flush=True)
        print(f"[DEBUG] API_KEY set: {bool(os.environ['API_KEY'])}", flush=True)

        client = build_client()
        warmup_proxy_call(client)
    except Exception as e:
        print(f"[FATAL] Startup failed: {e}", flush=True)
        return

    tasks = ["easy", "medium", "hard"] if TASK == "all" else [TASK]

    for task in tasks:
        run_task(task, client)
        print()


if __name__ == "__main__":
    main()