from __future__ import annotations

"""Application submission lifecycle.

Canonical forward path:

    DISCOVERED → SELECTED → APPLICATION_PAGE_FOUND → FORM_ANALYSIS
      → AUTO_FILLING → READY_FOR_REVIEW → USER_VERIFIED
      → SUBMITTED → AWAITING_CONFIRMATION → CONFIRMED

Problem states (reachable from most in-flight states):

    FAILED            — submission attempted and definitively failed
    BLOCKED           — employer-side block (login wall, region block…)
    REQUIRES_USER_ACTION — CAPTCHA / MFA / missing answers; resumable

SUBMITTED and CONFIRMED are deliberately separate: an application is
SUBMITTED when the browser actually confirms submission, and only becomes
CONFIRMED when a real employer/ATS confirmation email has been verified.
"""

from datetime import datetime
from typing import Iterable

from .models import Application, ApplicationStatus

S = ApplicationStatus

CANONICAL_PATH: tuple[ApplicationStatus, ...] = (
    S.DISCOVERED,
    S.SELECTED,
    S.APPLICATION_PAGE_FOUND,
    S.FORM_ANALYSIS,
    S.AUTO_FILLING,
    S.READY_FOR_REVIEW,
    S.USER_VERIFIED,
    S.SUBMITTED,
    S.AWAITING_CONFIRMATION,
    S.CONFIRMED,
)

_ALLOWED: dict[ApplicationStatus, frozenset[ApplicationStatus]] = {
    S.DISCOVERED: frozenset({S.SELECTED, S.FAILED, S.WITHDRAWN}),
    S.SELECTED: frozenset({
        S.APPLICATION_PAGE_FOUND, S.REQUIRES_USER_ACTION, S.FAILED, S.BLOCKED, S.WITHDRAWN,
    }),
    S.APPLICATION_PAGE_FOUND: frozenset({
        S.FORM_ANALYSIS, S.REQUIRES_USER_ACTION, S.FAILED, S.BLOCKED, S.WITHDRAWN,
    }),
    S.FORM_ANALYSIS: frozenset({
        S.AUTO_FILLING, S.REQUIRES_USER_ACTION, S.FAILED, S.BLOCKED, S.WITHDRAWN,
    }),
    S.AUTO_FILLING: frozenset({
        S.READY_FOR_REVIEW, S.REQUIRES_USER_ACTION, S.FAILED, S.WITHDRAWN,
    }),
    # legacy preparation states feed the same review gate
    S.DRAFT: frozenset({
        S.SELECTED, S.PREPARING, S.READY_FOR_REVIEW, S.NEEDS_INFORMATION,
        S.REQUIRES_USER_ACTION, S.FAILED, S.WITHDRAWN,
    }),
    S.PREPARING: frozenset({
        S.READY_FOR_REVIEW, S.NEEDS_INFORMATION, S.REQUIRES_USER_ACTION,
        S.FAILED, S.WITHDRAWN,
    }),
    S.NEEDS_INFORMATION: frozenset({
        S.READY_FOR_REVIEW, S.NEEDS_INFORMATION, S.REQUIRES_USER_ACTION,
        S.FAILED, S.WITHDRAWN,
    }),
    S.READY_FOR_REVIEW: frozenset({
        S.USER_VERIFIED, S.REQUIRES_USER_ACTION, S.FAILED, S.WITHDRAWN,
    }),
    # legacy approval state maps onto the same confirmation gate
    S.AWAITING_APPROVAL: frozenset({
        S.READY_FOR_REVIEW, S.USER_VERIFIED, S.SUBMITTING, S.SUBMITTED,
        S.REQUIRES_USER_ACTION, S.FAILED, S.WITHDRAWN,
    }),
    S.USER_VERIFIED: frozenset({
        S.SUBMITTING, S.SUBMITTED, S.REQUIRES_USER_ACTION, S.FAILED, S.BLOCKED,
    }),
    S.SUBMITTING: frozenset({
        S.SUBMITTED, S.FAILED, S.BLOCKED, S.REQUIRES_USER_ACTION,
    }),
    S.SUBMITTED: frozenset({
        S.AWAITING_CONFIRMATION, S.CONFIRMED, S.INTERVIEW, S.REJECTED,
        S.WITHDRAWN,
    }),
    S.AWAITING_CONFIRMATION: frozenset({
        S.CONFIRMED, S.EMAIL_REVIEW_REQUIRED, S.INTERVIEW, S.REJECTED, S.WITHDRAWN,
    }),
    S.EMAIL_REVIEW_REQUIRED: frozenset({
        S.CONFIRMED, S.INTERVIEW, S.REJECTED, S.WITHDRAWN,
    }),
    S.CONFIRMED: frozenset({S.INTERVIEW, S.REJECTED, S.OFFER, S.WITHDRAWN}),
    S.PENDING: frozenset({S.INTERVIEW, S.REJECTED, S.OFFER, S.WITHDRAWN}),
    # problem states are resumable
    S.FAILED: frozenset({
        S.FORM_ANALYSIS, S.AUTO_FILLING, S.READY_FOR_REVIEW, S.REQUIRES_USER_ACTION,
        S.WITHDRAWN,
    }),
    S.BLOCKED: frozenset({
        S.APPLICATION_PAGE_FOUND, S.FORM_ANALYSIS, S.REQUIRES_USER_ACTION, S.WITHDRAWN,
    }),
    S.REQUIRES_USER_ACTION: frozenset({
        S.FORM_ANALYSIS, S.AUTO_FILLING, S.READY_FOR_REVIEW, S.USER_VERIFIED,
        S.SUBMITTING, S.SUBMITTED, S.FAILED, S.WITHDRAWN,
    }),
    S.MANUAL_ACTION_REQUIRED: frozenset({
        S.READY_FOR_REVIEW, S.USER_VERIFIED, S.SUBMITTED, S.FAILED, S.WITHDRAWN,
    }),
    S.INTERVIEW: frozenset({S.OFFER, S.REJECTED, S.WITHDRAWN}),
    S.REJECTED: frozenset(set()),
    S.OFFER: frozenset(set()),
    S.WITHDRAWN: frozenset(set()),
}


class InvalidTransition(Exception):
    def __init__(self, current: ApplicationStatus, target: ApplicationStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid lifecycle transition: {current.value} → {target.value}"
        )


def can_transition(current: ApplicationStatus, target: ApplicationStatus) -> bool:
    return target in _ALLOWED.get(current, frozenset())


def allowed_transitions(current: ApplicationStatus) -> Iterable[ApplicationStatus]:
    return _ALLOWED.get(current, frozenset())


def transition(
    application: Application,
    target: ApplicationStatus,
    note: str = "",
) -> Application:
    """Move ``application`` to ``target``, enforcing the lifecycle rules.

    Stamps ``submitted_at`` when entering SUBMITTED and appends to the
    status history. Raises InvalidTransition for illegal moves."""
    if not can_transition(application.status, target):
        raise InvalidTransition(application.status, target)
    previous = application.status
    application.status = target
    application.updated_at = datetime.now().isoformat()
    if target == S.SUBMITTED and not application.submitted_at:
        application.submitted_at = application.updated_at
    if target == S.CONFIRMED and not application.confirmation_received_at:
        application.confirmation_received_at = application.updated_at
    application.status_history.append({
        "from": previous.value,
        "to": target.value,
        "at": application.updated_at,
        "note": note,
    })
    return application


def is_submission_state(status: ApplicationStatus) -> bool:
    """True once the browser has actually confirmed a real submission."""
    return status in (S.SUBMITTED, S.AWAITING_CONFIRMATION, S.CONFIRMED)


def needs_user(status: ApplicationStatus) -> bool:
    return status in (
        S.REQUIRES_USER_ACTION,
        S.MANUAL_ACTION_REQUIRED,
        S.NEEDS_INFORMATION,
    )
