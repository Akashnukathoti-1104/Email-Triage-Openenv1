"""
HTTP client for the Email Triage OpenEnv environment.
"""

from openenv.core.env_client import HTTPEnvClient

try:
    from .models import EmailTriageAction, EmailTriageObservation
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnvClient(HTTPEnvClient[EmailTriageAction, EmailTriageObservation]):
    """Client for the Email Triage environment."""

    def _step_payload(self, action: EmailTriageAction) -> dict:
        return {"email_id": action.email_id, "category": action.category}

    def _parse_observation(self, payload: dict) -> EmailTriageObservation:
        return EmailTriageObservation(**payload)
