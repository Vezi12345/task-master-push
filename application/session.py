"""Persistent conversational sessions for the job-search agent.

The web chat layer used to run with a single in-memory agent whose
``pending_applications`` and ``last_query`` died on process restart, and
whose conversational context was re-derived from the disk tracker every
turn.  This module makes an agent's *intent state* durable so a "single
prompt → complete agent" flow can pause (the agent needs an answer, or is
awaiting approval) and later resume — even across server restarts.

Only minimal, JSON-serialisable state is stored on disk; heavy objects
(applications, ranked jobs) live in the tracker and profile store and are
rehydrated by identifier.  The session never stores secrets or answers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DATA_DIR

SESSIONS_DIR = DATA_DIR / "sessions"

# Modes an agent session can be paused on.  These drive what the UI offers
# on the next turn without the agent having to re-derive everything.
MODE_IDLE = "idle"                     # nothing in flight
MODE_AWAITING_APPROVAL = "awaiting_approval"   # apps ready for review/approve
MODE_NEEDS_INFORMATION = "needs_information"   # agent asked questions
MODE_AUTONOMOUS_PAUSED = "autonomous_paused"   # autonomous run awaiting input
MODE_SUBMITTING = "submitting"         # mid-run (non-blocking, informational)


@dataclass
class SessionState:
    session_id: str = ""
    mode: str = MODE_IDLE
    last_query: Optional[dict] = None
    pending_application_ids: list[str] = field(default_factory=list)
    # Tier 1.2: questions the agent is currently waiting on, keyed by
    # application id so answers can be routed back to the right app.
    pending_questions: dict[str, list] = field(default_factory=dict)
    # legacy alias used by the orchestrator for the same purpose
    questions_by_app: dict[str, list] = field(default_factory=dict)
    autonomous: Optional[dict] = None
    created_at: str = ""
    updated_at: str = ""
    # not persisted; remembered so a session saved outside the store keeps
    # writing to the directory it was created/loaded from
    _directory: Optional[Path] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

    def touch(self) -> None:
        self.updated_at = datetime.now().isoformat()

    def save(self) -> None:
        """Persist this session via its own store directory (convenience)."""
        SessionStore(directory=self._directory).save(self)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "last_query": self.last_query,
            "pending_application_ids": list(self.pending_application_ids),
            "pending_questions": self.pending_questions,
            "questions_by_app": self.questions_by_app,
            "autonomous": self.autonomous,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SessionStore:
    """Load/save SessionState records to disk keyed by session_id."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self._dir = directory or SESSIONS_DIR

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def create(self) -> SessionState:
        st = SessionState(_directory=self._dir)
        self.save(st)
        return st

    def load(self, session_id: str) -> Optional[SessionState]:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            st = SessionState(_directory=self._dir, **data)
            st.session_id = session_id
            return st
        except (json.JSONDecodeError, OSError, TypeError):
            return None

    def save(self, st: SessionState) -> None:
        st.touch()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(st.session_id).write_text(
            json.dumps(st.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_recent(self, limit: int = 50) -> list[SessionState]:
        if not self._dir.exists():
            return []
        sessions = []
        for path in self._dir.glob("*.json"):
            try:
                st = SessionState(**json.loads(path.read_text(encoding="utf-8")))
                sessions.append(st)
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]
