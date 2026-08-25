from __future__ import annotations

"""Employer confirmation-email verification.

Connects to the USER'S REAL mailbox through an authorised IMAP
integration (credentials come from the environment — never hard-coded)
and searches it for genuine employer / ATS confirmation emails after a
real submission.

An application stays AWAITING_CONFIRMATION until a real email is actually
found. Ambiguous matches become EMAIL_REVIEW_REQUIRED rather than being
silently treated as confirmed.
"""

import os
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from typing import Optional

from .lifecycle import transition
from .models import Application, ApplicationStatus


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class EmailMessage:
    id: str = ""
    from_addr: str = ""
    to_addr: str = ""
    subject: str = ""
    body: str = ""
    date: Optional[datetime] = None

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}"


# ---------------------------------------------------------------------------
# real mailbox connector (IMAP)
# ---------------------------------------------------------------------------

_PLATFORM_SENDER_DOMAINS = (
    "lever.co", "greenhouse.io", "myworkdayjobs.com", "myworkdaysite.com",
    "smartrecruiters.com", "taleo.net", "icims.com", "jobvite.com",
    "sapsf.com", "successfactors.com", "applytojob.com", "bamboohr.com",
    "workable.com", "personio.de", "recruitee.com", "breezy.hr",
    "recruiterbox.com",
)


class MailboxNotConfigured(Exception):
    pass


class MailboxUnavailable(Exception):
    """The mailbox cannot be reached (firewall blocks IMAP, network down,
    auth rejected). Distinct from not-configured so the UI can say exactly
    what happened instead of a raw 500."""


class ImapMailboxConnector:
    """Read-only IMAP client over the user's actual mailbox.

    Configuration (environment variables):
      TASK_MASTER_IMAP_HOST      e.g. imap.gmail.com
      TASK_MASTER_IMAP_PORT      default 993
      TASK_MASTER_EMAIL_ADDRESS  the candidate's email address
      TASK_MASTER_EMAIL_PASSWORD app password / OAuth token per provider
    """

    def __init__(
        self,
        host: Optional[str] = None,
        address: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 993,
        folder: str = "INBOX",
    ) -> None:
        self.host = host or os.environ.get("TASK_MASTER_IMAP_HOST", "")
        self.address = address or os.environ.get("TASK_MASTER_EMAIL_ADDRESS", "")
        self.password = password or os.environ.get("TASK_MASTER_EMAIL_PASSWORD", "")
        self.port = int(
            port if port != 993 else os.environ.get("TASK_MASTER_IMAP_PORT", 993)
        )
        self.folder = folder

    def is_configured(self) -> bool:
        return bool(self.host and self.address and self.password)

    def fetch_recent(self, since_days: int = 14, limit: int = 100) -> list[EmailMessage]:
        """Fetch recent messages (read-only). Raises MailboxNotConfigured
        when credentials are absent; propagates connection errors honestly."""
        if not self.is_configured():
            raise MailboxNotConfigured(
                "Email integration is not configured — set TASK_MASTER_IMAP_HOST, "
                "TASK_MASTER_EMAIL_ADDRESS and TASK_MASTER_EMAIL_PASSWORD"
            )
        import imaplib

        since_date = (datetime.now() - timedelta(days=since_days)).strftime("%d-%b-%Y")
        messages: list[EmailMessage] = []
        try:
            conn = imaplib.IMAP4_SSL(self.host, self.port)
        except (OSError, TimeoutError, imaplib.IMAP4.error) as exc:
            raise MailboxUnavailable(
                f"Cannot reach {self.host}:{self.port} — "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        try:
            try:
                conn.login(self.address, self.password)
            except imaplib.IMAP4.error as exc:
                raise MailboxUnavailable(
                    f"Gmail rejected the login for {self.address} — check "
                    "the app password"
                ) from exc
            conn.select(self.folder, readonly=True)
            status, data = conn.search(None, f'(SINCE "{since_date}")')
            if status != "OK":
                return []
            ids = data[0].split()[-limit:]
            for num in ids:
                status, msg_data = conn.fetch(num, "(RFC822)")
                if status != "OK" or not msg_data or msg_data[0] is None:
                    continue
                raw = msg_data[0][1]
                parsed = BytesParser(policy=policy.default).parsebytes(raw)
                body_part = parsed.get_body(preferencelist=("plain",))
                body = body_part.get_content() if body_part else ""
                date_raw = parsed.get("Date")
                try:
                    when = parsedate_to_datetime(date_raw) if date_raw else None
                except (TypeError, ValueError):
                    when = None
                messages.append(EmailMessage(
                    id=num.decode("ascii", "ignore"),
                    from_addr=parsed.get("From", ""),
                    to_addr=parsed.get("To", ""),
                    subject=parsed.get("Subject", ""),
                    body=body[:20000],
                    date=when,
                ))
        finally:
            try:
                conn.logout()
            except Exception:
                pass
        return messages


# ---------------------------------------------------------------------------
# confirmation matching (#12)
# ---------------------------------------------------------------------------

_CONFIRMATION_LANGUAGE = (
    "application received", "application submitted", "thank you for applying",
    "thanks for applying", "application confirmation", "we received your application",
    "your application has been", "confirmed your application",
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


def _sender_domain(from_addr: str) -> str:
    match = re.search(r"@([\w.\-]+)", from_addr or "")
    return match.group(1).lower() if match else ""


@dataclass
class MatchResult:
    email: EmailMessage
    score: float = 0.0
    matched_on: list[str] = dc_field(default_factory=list)

    @property
    def strong(self) -> bool:
        has_identity = (
            "application_reference" in self.matched_on
            or ("company" in self.matched_on and "job_title" in self.matched_on)
        )
        return self.score >= 0.75 and has_identity

    @property
    def ambiguous(self) -> bool:
        return 0.30 <= self.score < 0.75


class ConfirmationMatcher:
    """Scores how well an email confirms a specific application."""

    WEIGHTS = {
        "application_reference": 0.40,
        "company": 0.20,
        "job_title": 0.15,
        "platform_sender": 0.10,
        "confirmation_language": 0.10,
        "sent_after_submission": 0.05,
    }

    def score_email(self, email: EmailMessage, app: Application) -> MatchResult:
        result = MatchResult(email=email)
        hay = email.text.lower()
        sender = _sender_domain(email.from_addr)

        # timing: an email older than the submission cannot confirm it
        submitted = self._parse_dt(app.submitted_at)
        if submitted and email.date is not None:
            if email.date >= submitted - timedelta(minutes=5):
                result.score += self.WEIGHTS["sent_after_submission"]
                result.matched_on.append("sent_after_submission")

        ref = (app.application_reference or "").strip()
        if ref and ref.lower() in hay:
            result.score += self.WEIGHTS["application_reference"]
            result.matched_on.append("application_reference")

        company = (app.job_company or "").strip()
        if company:
            company_tokens = _tokens(company)
            hay_tokens = _tokens(hay)
            overlap = company_tokens & hay_tokens
            if overlap and len(overlap) >= max(1, len(company_tokens) // 2):
                result.score += self.WEIGHTS["company"]
                result.matched_on.append("company")

        title = (app.job_title or "").strip().lower()
        if title and len(title) > 3 and title in hay:
            result.score += self.WEIGHTS["job_title"]
            result.matched_on.append("job_title")

        platform = (app.application_platform or "").lower()
        if platform and platform in sender:
            result.score += self.WEIGHTS["platform_sender"]
            result.matched_on.append("platform_sender")
        elif any(d in sender for d in _PLATFORM_SENDER_DOMAINS):
            result.score += self.WEIGHTS["platform_sender"]
            result.matched_on.append("platform_sender")

        if any(phrase in hay for phrase in _CONFIRMATION_LANGUAGE):
            result.score += self.WEIGHTS["confirmation_language"]
            result.matched_on.append("confirmation_language")

        return result

    @staticmethod
    def _parse_dt(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


# ---------------------------------------------------------------------------
# awaiting / confirming workflow (#11, #15)
# ---------------------------------------------------------------------------

def await_confirmation(
    app: Application,
    tracker,
    connector=None,
    matcher: Optional[ConfirmationMatcher] = None,
) -> Application:
    """Look for a REAL confirmation email for a submitted application.

    No email yet → stays AWAITING_CONFIRMATION (never assumed confirmed).
    Strong match → CONFIRMED. Ambiguous → EMAIL_REVIEW_REQUIRED."""
    if app.status.value == "submitted":
        transition(app, ApplicationStatus.AWAITING_CONFIRMATION,
                   "Checking mailbox for employer confirmation")
    elif app.status not in (
        ApplicationStatus.AWAITING_CONFIRMATION,
        ApplicationStatus.EMAIL_REVIEW_REQUIRED,
    ):
        from .models import ApplicationStatus as S
        raise ValueError(
            f"Application is {app.status.value}; expected submitted / "
            "awaiting_confirmation / email_review_required"
        )

    matcher = matcher or ConfirmationMatcher()

    if connector is None or not connector.is_configured():
        app.notes.append(
            "Confirmation email not checked — email integration is not "
            "configured (set TASK_MASTER_IMAP_* environment variables)"
        )
        tracker.update(app)
        return app

    try:
        emails = connector.fetch_recent(since_days=14)
    except MailboxNotConfigured as exc:
        app.notes.append(str(exc))
        tracker.update(app)
        return app

    submitted = ConfirmationMatcher._parse_dt(app.submitted_at)
    candidates: list[MatchResult] = []
    for email in emails:
        if submitted and email.date is not None and email.date < submitted - timedelta(minutes=5):
            continue  # predates the submission — cannot confirm it
        result = matcher.score_email(email, app)
        if result.score >= 0.30:
            candidates.append(result)

    if not candidates:
        app.notes.append(
            "No confirmation email found yet — still awaiting confirmation"
        )
        tracker.update(app)
        return app

    candidates.sort(key=lambda r: r.score, reverse=True)
    best = candidates[0]

    if best.strong:
        transition(app, ApplicationStatus.CONFIRMED,
                   f"Confirmation email matched ({', '.join(best.matched_on)})")
        app.confirmation_email_id = best.email.id
        app.confirmation_text = best.email.subject or best.email.body[:120]
        app.confirmation_received_at = (
            best.email.date.isoformat() if best.email.date else app.updated_at
        )
        app.confirmation_confidence = round(best.score, 2)
    else:
        transition(app, ApplicationStatus.EMAIL_REVIEW_REQUIRED,
                   "Ambiguous confirmation email — manual review needed")
        app.error = (
            f"Ambiguous confirmation email (score {best.score:.2f}, matched: "
            f"{', '.join(best.matched_on)}). Subject: '{best.email.subject}' "
            "— please verify manually"
        )
        app.confirmation_confidence = round(best.score, 2)
    tracker.update(app)
    return app
