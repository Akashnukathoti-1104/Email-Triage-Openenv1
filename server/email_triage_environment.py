"""
Email Triage Environment — real-world task of classifying emails.
Tasks: easy (3 emails), medium (5 emails), hard (10 emails).

Reward design:
  1.0  — correct classification
  0.5  — borderline (normal <-> important confusion)
  0.0  — clearly wrong
 -0.1  — invalid category (penalizes undesirable behavior)
"""

import random
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailTriageAction, EmailTriageObservation, compute_reward
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation, compute_reward


# ── Email datasets ────────────────────────────────────────────────────────────

EMAILS = {
    "easy": [
        {"id": "e1", "sender": "ceo@company.com", "subject": "Q4 Strategy Meeting - Attendance Required",
         "body": "Please join the mandatory all-hands meeting Friday at 2pm.", "label": "important"},
        {"id": "e2", "sender": "noreply@lottery-winner.xyz", "subject": "YOU WON $1,000,000!!!",
         "body": "Claim your prize now! Send bank details immediately!", "label": "spam"},
        {"id": "e3", "sender": "newsletter@techdigest.com", "subject": "This week in tech",
         "body": "Here are this week's top technology news stories.", "label": "normal"},
    ],
    "medium": [
        {"id": "m1", "sender": "hr@company.com", "subject": "Benefits enrollment deadline Friday",
         "body": "Annual benefits enrollment closes this Friday. Update your selections in the portal.", "label": "important"},
        {"id": "m2", "sender": "deals@shop.com", "subject": "50% off sale ends tonight",
         "body": "Don't miss our biggest sale! Shop now for massive discounts.", "label": "spam"},
        {"id": "m3", "sender": "colleague@company.com", "subject": "Re: Project update",
         "body": "Thanks for the update. I'll review the docs and get back to you.", "label": "normal"},
        {"id": "m4", "sender": "security@company.com", "subject": "Action required: Security alert",
         "body": "We detected unusual login from a new device. Please verify immediately.", "label": "important"},
        {"id": "m5", "sender": "noreply@freeprizes.net", "subject": "Your free gift is waiting",
         "body": "You've been selected! Claim your free iPhone by completing a survey.", "label": "spam"},
    ],
    "hard": [
        {"id": "h1", "sender": "client@bigcorp.com", "subject": "Following up on our conversation",
         "body": "Just wanted to follow up on our call last week. Let me know if you need anything.", "label": "important"},
        {"id": "h2", "sender": "alerts@monitoring.io", "subject": "Weekly system report",
         "body": "Weekly infrastructure report: CPU avg 45%, memory 60%, no critical issues.", "label": "normal"},
        {"id": "h3", "sender": "promo@legit-store.com", "subject": "Your order has shipped",
         "body": "Your order #45821 has shipped. Expected delivery: 3-5 business days.", "label": "normal"},
        {"id": "h4", "sender": "finance@company.com", "subject": "Invoice approval needed",
         "body": "Please approve the $12,400 invoice before the vendor deadline tomorrow.", "label": "important"},
        {"id": "h5", "sender": "info@webinar-free.com", "subject": "Exclusive webinar on wealth creation",
         "body": "Learn secrets the rich don't want you to know! Register free today!", "label": "spam"},
        {"id": "h6", "sender": "team@slack.com", "subject": "New message in #general",
         "body": "You have 12 unread messages in your Slack workspace. Click to view.", "label": "normal"},
        {"id": "h7", "sender": "cto@company.com", "subject": "Production outage - need your help NOW",
         "body": "Critical production outage affecting 10k users. Jump on the incident call immediately.", "label": "important"},
        {"id": "h8", "sender": "digest@medium.com", "subject": "Stories you might like",
         "body": "Based on your reading history, here are 5 curated stories for you.", "label": "normal"},
        {"id": "h9", "sender": "support@phishing-attempt.ru", "subject": "Your PayPal account limited",
         "body": "We limited your account. Click the link to restore access and verify identity.", "label": "spam"},
        {"id": "h10", "sender": "legal@company.com", "subject": "Contract review required by EOD",
         "body": "Client sent back NDA with revisions. Need your sign-off before 5pm today.", "label": "important"},
    ],
}

# Valid categories — anything outside this gets penalized
VALID_CATEGORIES = {"spam", "important", "normal"}


class EmailTriageEnvironment(Environment):
    """
    Real-world email triage environment.
    Agent reads emails one at a time and classifies each as spam/important/normal.

    Reward per step:
      1.0  correct classification
      0.5  borderline (normal <-> important)
      0.0  wrong classification
     -0.1  invalid/unrecognized category

    Episode score = sum(rewards) / total_emails
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task = "easy"
        self._queue: List[dict] = []
        self._all_emails: List[dict] = []
        self._current: Optional[dict] = None
        self._total_reward = 0.0
        self._total = 0
        self._done = False

    def reset(self, task: str = "easy") -> EmailTriageObservation:
        """Reset and start a new episode. task: easy | medium | hard"""
        if task not in EMAILS:
            task = "easy"
        self._task = task
        self._all_emails = list(EMAILS[task])
        self._queue = list(self._all_emails)
        random.shuffle(self._queue)
        self._total_reward = 0.0
        self._total = 0
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current = self._queue.pop(0)
        return self._make_obs(reward=0.0, done=False)

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:  # type: ignore[override]
        """Classify current email and advance to next."""
        if self._done or self._current is None:
            return self._make_obs(reward=0.0, done=True)

        self._state.step_count += 1

        # Penalize invalid category — undesirable behavior
        if action.category.lower() not in VALID_CATEGORIES:
            return self._make_obs(reward=-0.1, done=False)

        # Check email_id mismatch — no reward, no penalty, don't advance
        if action.email_id != self._current["id"]:
            return self._make_obs(reward=0.0, done=False)

        # Compute reward using partial credit function
        reward = compute_reward(
            predicted=action.category.lower(),
            ground_truth=self._current["label"],
        )
        self._total_reward += reward
        self._total += 1

        # Advance to next email
        if self._queue:
            self._current = self._queue.pop(0)
            done = False
        else:
            self._current = None
            done = True
        self._done = done

        return self._make_obs(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    def _make_obs(self, reward: float, done: bool) -> EmailTriageObservation:
        current_score = round(
            self._total_reward / max(self._total, 1), 4
        )
        if self._current is None or self._done:
            return EmailTriageObservation(
                email_id="done",
                sender="",
                subject="",
                body="",
                emails_remaining=0,
                total_emails=len(self._all_emails),
                current_score=current_score,
                task=self._task,
                done=True,
                reward=reward,
            )
        return EmailTriageObservation(
            email_id=self._current["id"],
            sender=self._current["sender"],
            subject=self._current["subject"],
            body=self._current["body"],
            emails_remaining=len(self._queue),
            total_emails=len(self._all_emails),
            current_score=current_score,
            task=self._task,
            done=done,
            reward=reward,
        )
