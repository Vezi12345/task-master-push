"""Outcome learning loop.

Turns post-submission outcomes (rejection, interview, offer, or a prolonged
silence after submission) into a *payoff signal* that tunes which job
sources, companies, and role families the search should favour next time.

The loop has two halves:

1. **Record** — ``OutcomeStore`` persists per-application outcome counts
   aggregated by *key* (source, company, role family). ``record_outcome``
   maps an application's status onto a countable band and updates the store.
   ``age_application`` marks long-silent submissions as ``no_response`` so a
   dead end is learned from (rather than silently re-prioritised).

2. **Feedback** — ``application_payoff_bonus`` converts a key's observed
   counts into a bounded bonus (default 0 when there is no history, so the
   existing scoring behaviour is untouched until real outcomes accumulate).

The payoff signal is deliberately additive and bounded: it never overrides
the profile/readiness scores, just nudges prioritisation toward surfaces that
have historically converted and away from those that ghost.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .models import Application, ApplicationStatus

# Outcome bands that count toward the payoff signal.
OUTCOME_OFFER = "offer"
OUTCOME_INTERVIEW = "interview"
OUTCOME_NO_RESPONSE = "no_response"
OUTCOME_REJECTED = "rejected"
OUTCOME_PENDING = "pending"
_VALID_OUTCOMES = {
    OUTCOME_OFFER,
    OUTCOME_INTERVIEW,
    OUTCOME_NO_RESPONSE,
    OUTCOME_REJECTED,
    OUTCOME_PENDING,
}

# Payoff weights per observed outcome.
_W_OFFER = 30.0
_W_INTERVIEW = 15.0
_W_NO_RESPONSE = -6.0
_W_REJECTED = 1.0  # a rejection is a real signal (not a ghost), mildly positive
_W_PENDING = 0.0

# Expected-value is normalised into a bounded bonus.
MAX_BONUS = 10
MIN_BONUS = -10
_SMOOTHING = 3  # additive smoothing on the positive-evidence rate
_DECAY_HALF_LIFE_DAYS = 30.0  # older outcomes weigh less

# After this many days with no response, a submitted/pending application is
# treated as a no-response dead end.
DEFAULT_SILENCE_DAYS = 21

# A coarse role bucket is derived from the title for cross-company learning.
_ROLE_HINTS = (
    ("finance_accounting", ("accountant", "bookkeeper", "finance", "payroll", "audit", "financial")),
    ("engineering", ("engineer", "developer", "software", "data", "devops", "qa")),
    ("admin", ("administrator", "admin", "reception", "filing", "clerk")),
    ("sales", ("sales", "account executive", "business development")),
    ("customer_service", ("support", "service", "helpdesk")),
    ("clinical", ("nurse", "clinical", "care", "health")),
    ("operations", ("operations", "logistics", "supply chain", "procurement")),
)


def role_family(title: str) -> str:
    """Coarse role bucket derived from a job title ('' when unknown)."""
    lowered = (title or "").strip().lower()
    for family, hints in _ROLE_HINTS:
        if any(hint in lowered for hint in hints):
            return family
    return ""


def _outcome_keys(app: Application) -> list[str]:
    """Aggregate keys (source, company, role family) for an application."""
    source = (app.submission_mode or "unknown").strip().lower()
    source = source if source in ("real", "mocked_test") else "unknown"
    keys = [f"source:{source}", f"company:{_norm(app.job_company)}"]
    family = role_family(app.job_title)
    if family:
        keys.append(f"role:{family}")
    return keys


def _norm(text: str) -> str:
    return (text or "").strip().lower().replace(" ", "_") or "unknown"


@dataclass
class KeyCounts:
    submitted: int = 0
    pending: int = 0
    no_response: int = 0
    rejected: int = 0
    interview: int = 0
    offer: int = 0

    def band(self, outcome: str) -> None:
        if outcome == OUTCOME_OFFER:
            self.offer += 1
        elif outcome == OUTCOME_INTERVIEW:
            self.interview += 1
        elif outcome == OUTCOME_NO_RESPONSE:
            self.no_response += 1
        elif outcome == OUTCOME_REJECTED:
            self.rejected += 1
        elif outcome == OUTCOME_PENDING:
            self.pending += 1
        else:
            raise ValueError(f"unknown outcome: {outcome!r}")

    def to_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "pending": self.pending,
            "no_response": self.no_response,
            "rejected": self.rejected,
            "interview": self.interview,
            "offer": self.offer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KeyCounts":
        return cls(
            submitted=int(data.get("submitted", 0) or 0),
            pending=int(data.get("pending", 0) or 0),
            no_response=int(data.get("no_response", 0) or 0),
            rejected=int(data.get("rejected", 0) or 0),
            interview=int(data.get("interview", 0) or 0),
            offer=int(data.get("offer", 0) or 0),
        )

    def known_weight(self) -> float:
        """Weighted evidence volume used for confidence."""
        return (
            self.offer * _W_OFFER
            + self.interview * _W_INTERVIEW
            + self.no_response * abs(_W_NO_RESPONSE)
            + self.rejected * abs(_W_REJECTED)
        )


def band_for_status(status: ApplicationStatus) -> Optional[str]:
    """Map an application status onto an outcome band, or None if it is not a
    learnable terminal/factual outcome."""
    if status == ApplicationStatus.OFFER:
        return OUTCOME_OFFER
    if status == ApplicationStatus.INTERVIEW:
        return OUTCOME_INTERVIEW
    if status == ApplicationStatus.REJECTED:
        return OUTCOME_REJECTED
    if status in (ApplicationStatus.SUBMITTED, ApplicationStatus.PENDING,
                  ApplicationStatus.AWAITING_CONFIRMATION,
                  ApplicationStatus.CONFIRMED):
        return OUTCOME_PENDING
    return None


class OutcomeStore:
    """Disk-backed aggregation of outcome counts by learning key.

    Uses optimistic read-modify-write under a lock; missing files start empty.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[str, KeyCounts] = {}
        self._dirty = False
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            self._data = {
                str(k): KeyCounts.from_dict(v)
                for k, v in (raw or {}).items()
                if isinstance(v, dict)
            }

    def get(self, key: str) -> KeyCounts:
        with self._lock:
            return self._data.get(key, KeyCounts())

    def record(self, key: str, outcome: str, *, persist: bool = True,
               persist_path=None) -> "OutcomeStore":
        if outcome not in _VALID_OUTCOMES:
            raise ValueError(f"unknown outcome: {outcome!r}")
        with self._lock:
            counts = self._data.setdefault(key, KeyCounts())
            counts.band(outcome)
            self._dirty = True
        if persist:
            self.save(persist_path or self.path)
        return self

    def record_application(self, app: Application, outcome: Optional[str] = None,
                           *, persist: bool = True) -> list[str]:
        """Record an application's outcome against all its aggregate keys."""
        band = outcome or band_for_status(app.status)
        if band is None:
            return []
        keys = _outcome_keys(app)
        for key in keys:
            self.record(key, band, persist=False)
        if persist:
            self.save(self.path)
        return keys

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def save(self, path=None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            return
        with self._lock:
            payload = {k: v.to_dict() for k, v in self._data.items()}
            self._dirty = False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            self._dirty = True


def compute_payoff_bonus(counts: KeyCounts) -> int:
    """Bounded payoff bonus in [-10, 10] for an aggregate key's counts.

    Uses a smoothed expected-value with a width-1 ``tanh``-style squash so the
    bonus cannot dominate the profile/readiness scoring. Returns 0 when there
    is no evidence.
    """
    evidence = counts.known_weight()
    if evidence <= 0:
        return 0
    positive = (
        counts.offer * _W_OFFER + counts.interview * _W_INTERVIEW
        + counts.rejected * _W_REJECTED * 0.2
    )
    negative = counts.no_response * abs(_W_NO_RESPONSE)
    expected = (positive - negative + _SMOOTHING) / (evidence + _SMOOTHING)
    # map expected in roughly [0.15, 1.0] to a bonus in [-MAX, +MAX]
    raw = (expected - 0.5) * 2 * MAX_BONUS
    raw = max(-MAX_BONUS, min(MAX_BONUS, raw))
    return round(raw)


def application_payoff_bonus(app: Application, store: OutcomeStore | None = None) -> int:
    """Best payoff bonus across an application's aggregate keys (0 if none)."""
    if store is None:
        return 0
    best = 0
    for key in _outcome_keys(app):
        bonus = compute_payoff_bonus(store.get(key))
        if abs(bonus) > abs(best):
            best = bonus
    return best


def adjust_priority_with_payoff(app: Application, store: OutcomeStore | None = None,
                                *,
                                clamp: bool = True) -> int:
    """Adjusted priority score that includes the payoff signal.

    ``clamp`` keeps the result within [0, 100]. With no store or no evidence
    this returns ``app.application_priority`` unchanged."""
    bonus = application_payoff_bonus(app, store)
    adjusted = app.application_priority + bonus
    if clamp:
        adjusted = max(0, min(100, round(adjusted)))
    return adjusted


def age_application(app: Application, store: OutcomeStore, *,
                    silence_days: int = DEFAULT_SILENCE_DAYS,
                    now: datetime | None = None) -> bool:
    """Mark a submitted-but-silent application as no_response and learn from it.

    Returns True if the application was aged (status moved to REJECTED-style
    terminal no-response and outcomes recorded). Applications still within the
    silence window, or ones already resolved, are left untouched.
    """
    if store is None:
        return False
    band = band_for_status(app.status)
    if band not in (OUTCOME_PENDING, OUTCOME_NO_RESPONSE):
        return False
    submitted_at = app.date_submitted or app.submitted_at or app.updated_at
    if not submitted_at:
        return False
    try:
        submitted_dt = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    now = now or datetime.now()
    if (now - submitted_dt).total_seconds() < silence_days * 86400:
        return False
    keys = _outcome_keys(app)
    for key in keys:
        store.record(key, OUTCOME_NO_RESPONSE, persist=False)
    store.save(store.path)
    app.date_responded = "no_response"
    return True


def summary(store: OutcomeStore | None) -> dict:
    """Human/tool friendly summary of what the model has learned."""
    if store is None:
        return {"keys": [], "bonus_by_key": {}}
    out: dict = {}
    for key in store.keys():
        counts = store.get(key)
        bonus = compute_payoff_bonus(counts)
        if counts.known_weight() > 0:
            out[key] = {"counts": counts.to_dict(), "payoff_bonus": bonus}
    return {"keys": list(out.keys()), "bonus_by_key": out}
