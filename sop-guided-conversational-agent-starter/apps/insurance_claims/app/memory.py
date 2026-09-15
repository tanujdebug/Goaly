"""Session slot memory. Deterministic state that the state machine reads and
writes; the LLM never mutates this directly. See app/state_machine.py."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

PII_FIELDS = ["full_name", "dob", "phone", "email", "ssn_last4"]

PHASES = (
    "VERIFY_ID",
    "RESOLVE_INTENT",
    "PROCESS_CASE",
    "POST_PROCESS",
    "HUMAN_ESCALATION",
    "ENDED",
)


@dataclass
class SessionMemory:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    phase: str = "VERIFY_ID"
    history: list[dict] = field(default_factory=list)

    # identity verification
    pii_slots: dict = field(default_factory=dict)
    verified: bool = False
    party_id: Optional[str] = None
    verification_attempts: int = 0

    # cross-phase memory: captured whenever the user says it, regardless of
    # current phase, and consumed once RESOLVE_INTENT is reached
    case_type_hint: Optional[str] = None
    status_hint: Optional[str] = None
    month_hint: Optional[str] = None
    free_text_hint: Optional[str] = None

    # intent / case resolution
    resolved_intent: Optional[str] = None
    active_case_id: Optional[str] = None
    active_document: Optional[str] = None
    pending_case_confirmation: Optional[str] = None

    # scope limiting + emotion
    off_topic_count: int = 0
    frustration_count: int = 0
    last_emotion: Optional[str] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None

    # post-process
    summary_text: Optional[str] = None
    email_offered: bool = False
    email_sent: bool = False

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    """In-memory session store. Fine for a single-process demo; not
    persisted across restarts and not safe across multiple worker
    processes."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: Optional[str]) -> SessionMemory:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        session = SessionMemory(session_id=session_id or uuid.uuid4().hex[:12])
        self._sessions[session.session_id] = session
        return session

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
