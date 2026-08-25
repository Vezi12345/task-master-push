from __future__ import annotations

"""Phases 9-10 — confirmation-email integration and matching.

Uses a clearly-marked FakeMailboxConnector test double. The real
ImapMailboxConnector is tested only for configuration honesty — it must
never pretend to have read a mailbox without credentials.
"""

from datetime import datetime, timedelta, timezone

import pytest

from application.email_confirmation import (
    ConfirmationMatcher,
    EmailMessage,
    ImapMailboxConnector,
    MailboxNotConfigured,
    await_confirmation,
)
from application.lifecycle import transition
from application.models import Application, ApplicationStatus


def _app(status=ApplicationStatus.SUBMITTED, **kw) -> Application:
    defaults = dict(
        id="dvt-abc123",
        job_id="j1",
        job_title="Graduate Software Developer",
        job_company="DVT",
        job_url="https://careers.dvt.co.za/grad",
        application_url=APPLY_URL,
        application_platform="lever",
        application_reference="DVT-12345",
        submitted_at=datetime.now(timezone.utc).isoformat(),
        status=status,
    )
    defaults.update(kw)
    return Application(**defaults)


APPLY_URL = "https://jobs.lever.co/dvt/123"


def _email(**kw) -> EmailMessage:
    defaults = dict(
        id="1",
        from_addr="no-reply@lever.co",
        to_addr="lucky.vezi@example.com",
        subject="Thanks for applying — DVT Graduate Software Developer",
        body=(
            "Hi Lucky, thank you for applying to the Graduate Software "
            "Developer role at DVT. Your application has been received. "
            "Reference: DVT-12345"
        ),
        date=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return EmailMessage(**defaults)


class FakeMailboxConnector:
    """TEST DOUBLE mailbox (is_configured reports what tests set)."""

    def __init__(self, emails=None, configured: bool = True):
        self.emails = emails or []
        self.configured = configured

    def is_configured(self) -> bool:
        return self.configured

    def fetch_recent(self, since_days: int = 14, limit: int = 100):
        if not self.configured:
            raise MailboxNotConfigured("not configured")
        return list(self.emails)


class _Tracker:
    def __init__(self):
        self.saved = 0

    def update(self, app):
        self.saved += 1


# ---------------------------------------------------------------------------
# matcher unit tests (#12)
# ---------------------------------------------------------------------------

def test_strong_match_reference_company_title():
    result = ConfirmationMatcher().score_email(_email(), _app())
    assert result.strong
    assert "application_reference" in result.matched_on
    assert "company" in result.matched_on
    assert "job_title" in result.matched_on


def test_company_name_alone_is_ambiguous_not_confirmed():
    email = _email(
        id="2",
        from_addr="newsletter@dvt.co.za",
        subject="DVT monthly newsletter",
        body="News from DVT and other companies…",
    )
    result = ConfirmationMatcher().score_email(email, _app())
    assert not result.strong
    assert result.score < 0.75


def test_unrelated_email_scores_below_match_threshold():
    email = _email(
        id="3",
        from_addr="promo@onlineshopping.co.za",
        subject="50% off sneakers today!",
        body="Big sale on sneakers and laptops.",
    )
    result = ConfirmationMatcher().score_email(email, _app())
    assert not result.strong
    assert not result.ambiguous
    assert result.score < 0.40


def test_email_predating_submission_cannot_score_timing():
    old_email = _email(date=datetime.now(timezone.utc) - timedelta(days=3))
    result = ConfirmationMatcher().score_email(old_email, _app())
    assert "sent_after_submission" not in result.matched_on


def test_platform_sender_boosts_score():
    no_platform_ref = _app(application_reference="")
    email = _email(id="4", from_addr="no-reply@greenhouse.io", subject="Application received")
    result = ConfirmationMatcher().score_email(email, no_platform_ref)
    assert "platform_sender" in result.matched_on


def test_confirmation_language_detected():
    email = _email(
        id="5", from_addr="hr@dvt.co.za",
        subject="We received your application",
        body="Your application has been received and is under review.",
    )
    result = ConfirmationMatcher().score_email(email, _app(application_reference=""))
    assert "confirmation_language" in result.matched_on


# ---------------------------------------------------------------------------
# workflow (#15-#19)
# ---------------------------------------------------------------------------

def test_submitted_moves_to_awaiting_confirmation_first():
    app = _app()
    await_confirmation(app, _Tracker(), connector=FakeMailboxConnector([]))
    assert app.status == ApplicationStatus.AWAITING_CONFIRMATION


def test_no_emails_found_stays_awaiting():
    app = _app(status=ApplicationStatus.AWAITING_CONFIRMATION)
    await_confirmation(app, _Tracker(), connector=FakeMailboxConnector([]))
    assert app.status == ApplicationStatus.AWAITING_CONFIRMATION
    assert any("No confirmation email found yet" in n for n in app.notes)


def test_matching_email_confirms_application():
    app = _app()
    tracker = _Tracker()
    connector = FakeMailboxConnector([_email()])
    await_confirmation(app, tracker, connector=connector)
    assert app.status == ApplicationStatus.CONFIRMED
    assert app.confirmation_email_id == "1"
    assert app.confirmation_received_at is not None
    assert app.confirmation_confidence >= 0.75
    assert "DVT-12345" in app.confirmation_text or "DVT" in app.confirmation_text


def test_ambiguous_email_requires_manual_review():
    app = _app(application_reference="", application_platform="")
    ambiguous = _email(
        id="6",
        from_addr="careers@dvt.co.za",
        subject="DVT careers update",
        body="Thank you for applying to DVT.",
    )
    await_confirmation(app, _Tracker(), connector=FakeMailboxConnector([ambiguous]))
    assert app.status == ApplicationStatus.EMAIL_REVIEW_REQUIRED
    assert "Ambiguous" in app.error


def test_unconfigured_connector_stays_awaiting_with_note():
    app = _app()
    await_confirmation(app, _Tracker(), connector=FakeMailboxConnector([], configured=False))
    assert app.status == ApplicationStatus.AWAITING_CONFIRMATION
    assert any("not configured" in n for n in app.notes)


def test_no_connector_at_all_stays_awaiting():
    app = _app()
    await_confirmation(app, _Tracker(), connector=None)
    assert app.status == ApplicationStatus.AWAITING_CONFIRMATION


def test_old_email_ignored_entirely():
    stale = _email(
        id="7",
        date=datetime.now(timezone.utc) - timedelta(days=5),
        subject="Thanks for applying — DVT Graduate Software Developer",
        body="Reference: DVT-12345 …",
    )
    app = _app()
    await_confirmation(app, _Tracker(), connector=FakeMailboxConnector([stale]))
    # predates submission → cannot confirm; stays awaiting
    assert app.status == ApplicationStatus.AWAITING_CONFIRMATION


def test_review_required_can_later_become_confirmed():
    app = _app(status=ApplicationStatus.EMAIL_REVIEW_REQUIRED)
    strong = _email(id="8")
    await_confirmation(app, _Tracker(), connector=FakeMailboxConnector([strong]))
    assert app.status == ApplicationStatus.CONFIRMED


def test_status_persisted_through_tracker(tmp_path):
    from application.tracker import ApplicationTracker

    path = tmp_path / "applications.json"
    tracker = ApplicationTracker(path=path)
    app = _app()
    tracker.add(app)
    connector = FakeMailboxConnector([_email()])
    await_confirmation(app, tracker, connector=connector)

    reloaded = ApplicationTracker(path=path).get("dvt-abc123")
    assert reloaded.status == ApplicationStatus.CONFIRMED
    assert reloaded.confirmation_email_id == "1"


# ---------------------------------------------------------------------------
# real connector honesty
# ---------------------------------------------------------------------------

def test_real_connector_reports_unconfigured(monkeypatch):
    for var in ("TASK_MASTER_IMAP_HOST", "TASK_MASTER_EMAIL_ADDRESS", "TASK_MASTER_EMAIL_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    conn = ImapMailboxConnector()
    assert conn.is_configured() is False
    with pytest.raises(MailboxNotConfigured):
        conn.fetch_recent()


def test_real_connector_reads_env_credentials(monkeypatch):
    monkeypatch.setenv("TASK_MASTER_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("TASK_MASTER_EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setenv("TASK_MASTER_EMAIL_PASSWORD", "secret")
    conn = ImapMailboxConnector()
    assert conn.is_configured() is True
    assert conn.host == "imap.example.com"
