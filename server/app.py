"""
FastAPI server for Email Triage OpenEnv.
Stateful singleton env — supports /health /metadata /schema /reset /step /state /openapi.json
All endpoints required by: openenv validate --url <your-space>
"""

import os
import sys

# Fix paths for HF Spaces (/app/env), Docker (/app), and local dev
_file_dir = os.path.dirname(os.path.abspath(__file__))          # .../server
_root_dir = os.path.dirname(_file_dir)                          # project root

for _p in ["/app/env", "/app", _root_dir, _file_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Any, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import with multiple fallbacks
try:
    from models import EmailTriageAction, EmailTriageObservation
    from server.email_triage_environment import EmailTriageEnvironment
except ImportError:
    try:
        from email_triage_environment import EmailTriageEnvironment
        from models import EmailTriageAction, EmailTriageObservation
    except ImportError:
        sys.path.insert(0, _file_dir)
        from email_triage_environment import EmailTriageEnvironment  # type: ignore
        sys.path.insert(0, _root_dir)
        from models import EmailTriageAction, EmailTriageObservation  # type: ignore

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment — classify emails as spam/important/normal.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton environment ─────────────────────────────────────────────────────
_env = EmailTriageEnvironment()


# ── Request models ────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: Optional[str] = "easy"
    seed: Optional[int] = None

    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    action: Dict[str, Any]

    class Config:
        extra = "allow"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "email_triage",
        "description": (
            "Real-world email triage OpenEnv environment. "
            "Agent classifies emails as spam, important, or normal. "
            "Tasks: easy (3 emails), medium (5), hard (10)."
        ),
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema():
    return {
        "action": EmailTriageAction.model_json_schema(),
        "observation": EmailTriageObservation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
            },
        },
    }


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    task = body.task if body.task in ("easy", "medium", "hard") else "easy"
    obs = _env.reset(task=task)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.post("/step")
def step(body: StepRequest):
    action = EmailTriageAction(**body.action)
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.get("/state")
def state():
    s = _env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
    }


@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema"],
        "tasks": ["easy", "medium", "hard"],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main()
