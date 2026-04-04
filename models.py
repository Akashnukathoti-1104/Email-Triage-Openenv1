"""
Data models for the Email Triage OpenEnv environment.
Classifies emails into: spam | important | normal

Reward design:
  1.0  — correct classification
  0.5  — borderline acceptable (e.g. normal classified as important)
  0.0  — wrong classification
 -0.1  — invalid/unrecognized category (penalizes bad behavior)
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EmailTriageAction(Action):
    """Classify the current email into a category."""
    email_id: str = Field(..., description="ID of the email being classified")
    category: str = Field(..., description="Classification: spam | important | normal")


class EmailTriageObservation(Observation):
    """Observation containing the current email to classify."""
    email_id:          str   = Field(default="",    description="Unique email ID")
    sender:            str   = Field(default="",    description="Sender email address")
    subject:           str   = Field(default="",    description="Email subject line")
    body:              str   = Field(default="",    description="Email body text")
    emails_remaining:  int   = Field(default=0,     description="Emails left to classify")
    total_emails:      int   = Field(default=0,     description="Total emails in episode")
    current_score:     float = Field(default=0.0,   description="Running accuracy score")
    task:              str   = Field(default="easy",description="Current task difficulty")


# ── Partial reward helper ─────────────────────────────────────────────────────
# Borderline pairs: classifying these as each other gets 0.5 instead of 0.0
# Rationale: normal vs important is ambiguous; spam vs important is never okay
BORDERLINE_PAIRS = {
    ("normal",    "important"),
    ("important", "normal"),
}

def compute_reward(predicted: str, ground_truth: str) -> float:
    """
    Compute reward for a single classification step.
      1.0  — exact match
      0.5  — borderline (normal <-> important confusion)
      0.0  — clearly wrong
     -0.1  — invalid category (caller should handle separately)
    """
    if predicted == ground_truth:
        return 1.0
    if (predicted, ground_truth) in BORDERLINE_PAIRS:
        return 0.5
    return 0.0
