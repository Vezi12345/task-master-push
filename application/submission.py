from __future__ import annotations

"""Real application submission service.

Orchestrates the lifecycle for REAL employer application pages:

    start_application()   DISCOVERED → … → READY_FOR_REVIEW   (stops here)
    confirm_and_submit()  USER_VERIFIED → SUBMITTED           (explicit gate)
    await_email_confirmation()  SUBMITTED → AWAITING_CONFIRMATION → CONFIRMED

Safety rules enforced here:
  * The pipeline ALWAYS stops at READY_FOR_REVIEW and requires an explicit
    ``confirm_and_submit`` call before anything is sent.
  * SUBMITTED is only recorded when the browser actually confirms the
    submission (confirmation text / success URL / reference).
  * Records produced through non-real test drivers are tagged
    ``submission_mode="mocked_test"`` and must never be presented as real.
  * CAPTCHA / Cloudflare / MFA / login walls → REQUIRES_USER_ACTION.
"""

import re
import uuid
from dataclasses import dataclass, field as dc_field
from typing import Optional

from candidate.profile import CandidateProfile
from sources.base import Job

from .app_url import find_application_url, summarise_target
from .answer_engine import AnswerType
from .browser import BrowserDriver, BrowserError, BrowserUnavailable
from .documents import ApplicationDocuments
from .form_analyzer import FormAnalysis, analyze_application_page
from .form_filler import FillPlan, FormFiller, PlannedAnswer
from .lifecycle import transition
from .models import Application, ApplicationStatus


# ---------------------------------------------------------------------------
# confirmation detection
# ---------------------------------------------------------------------------

_CONFIRMATION_PATTERNS = (
    r"application (?:has been |was )?(?:successfully )?submitted",
    r"thank you for (?:applying|your application)",
    r"your application has been received",
    r"we(?:'ve| have)? received your application",
    r"application successfully submitted",
    r"thanks for applying",
)

_SUCCESS_URL_PATTERN = re.compile(
    r"/(confirmation|confirm|success|thanks|thank-you|received|complete)d?(?:[/?]|$)",
    re.IGNORECASE,
)

_REFERENCE_PATTERNS = (
    re.compile(r"(?:reference|ref(?:erence)?\s*(?:number|no\.?|#)?)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_/]{3,})", re.IGNORECASE),
    re.compile(r"\b([A-Z]{2,6}[-_]\d{3,8})\b"),
    re.compile(r"\bapplication\s*#?\s*(\d{4,12})\b", re.IGNORECASE),
)


def detect_submission_confirmation(page_text: str, url: str = "") -> tuple[bool, str, str]:
    """Return ``(confirmed, confirmation_text, application_reference)``.

    Only REAL signals count: explicit confirmation language on the page or
    a success/confirmation URL. A clicked button alone proves nothing."""
    snippet = ""
    for pattern in _CONFIRMATION_PATTERNS:
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(page_text), match.end() + 80)
            snippet = " ".join(page_text[start:end].split())
            break

    url_confirmed = bool(_SUCCESS_URL_PATTERN.search(url or ""))
    confirmed = bool(snippet) or url_confirmed

    reference = ""
    if confirmed:
        for pattern in _REFERENCE_PATTERNS:
            ref_match = pattern.search(page_text)
            if ref_match:
                reference = ref_match.group(1)
                break
    return confirmed, snippet, reference


# ---------------------------------------------------------------------------
# review preview (phase 6)
# ---------------------------------------------------------------------------

@dataclass
class ReviewPreview:
    """Everything the user must see BEFORE anything is submitted."""

    application_id: str = ""
    company: str = ""
    job_title: str = ""
    job_url: str = ""
    application_url: str = ""
    application_platform: str = ""

    candidate_name: str = ""
    candidate_email: str = ""
    candidate_phone: str = ""

    answers: list[dict] = dc_field(default_factory=list)
    documents: dict = dc_field(default_factory=dict)
    unanswered_required: list[str] = dc_field(default_factory=list)
    consent_questions: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.unanswered_required

    def to_dict(self) -> dict:
        return {
            "application_id": self.application_id,
            "company": self.company,
            "job_title": self.job_title,
            "job_url": self.job_url,
            "application_url": self.application_url,
            "application_platform": self.application_platform,
            "candidate_name": self.candidate_name,
            "candidate_email": self.candidate_email,
            "candidate_phone": self.candidate_phone,
            "answers": self.answers,
            "groups": _group_answers(self.answers),
            "documents": self.documents,
            "unanswered_required": self.unanswered_required,
            "consent_questions": self.consent_questions,
            "warnings": self.warnings,
            # the UI must keep submission disabled until the user explicitly
            # approves THIS review — no other path enables it
            "approval_required": True,
            "ready": self.ready,
        }


_REVIEW_GROUPS = (
    "user_verified",     # answered by you, remembered from previous sessions
    "auto_answered",     # filled automatically from your CV/profile data
    "evidence_based",    # derived or AI-drafted from evidence — review these
    "still_unanswered",  # missing — must be completed before submission
)

_USER_MEMORY_SOURCES = frozenset({"memory", "answer_store", "user"})


def _group_answers(answers: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {name: [] for name in _REVIEW_GROUPS}
    for entry in answers:
        if entry.get("needs_user") or not entry.get("value"):
            groups["still_unanswered"].append(entry)
        elif entry.get("answer_type") == "generated_from_evidence" \
                or entry.get("answer_type") == "derived":
            groups["evidence_based"].append(entry)
        elif entry.get("source") in _USER_MEMORY_SOURCES:
            groups["user_verified"].append(entry)
        else:
            groups["auto_answered"].append(entry)
    return groups


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------

class ApplicationAutomationError(Exception):
    pass


def _own_words_flag(app: Application) -> bool:
    """own_words_required survives as a plain dict after persistence."""
    fa = getattr(app, "form_analysis", None)
    if isinstance(fa, dict):
        return bool(fa.get("own_words_required"))
    return bool(getattr(fa, "own_words_required", False))


class ApplicationAutomationService:
    def __init__(self, cv_path=None, cover_letter_path=None, llm=None) -> None:
        self._cv_path = str(cv_path) if cv_path else ""
        self._cover_letter_path = str(cover_letter_path) if cover_letter_path else ""
        self._llm = llm

    # -- documents ----------------------------------------------------------
    def _ensure_documents(self, app: Application, profile: CandidateProfile, job: Job) -> None:
        docs = ApplicationDocuments(self._llm)
        docs.prepare_documents(app, profile, job)
        if self._cover_letter_path and app.documents.cover_letter_text:
            try:
                from pathlib import Path
                Path(self._cover_letter_path).write_text(
                    app.documents.cover_letter_text, encoding="utf-8"
                )
            except OSError:
                pass

    # -- phase 1-6: discovery → review ---------------------------------------
    def start_application(
        self,
        job: Job,
        profile: CandidateProfile,
        tracker,
        driver: Optional[BrowserDriver] = None,
        page_html: Optional[str] = None,
    ) -> Application:
        """Walk DISCOVERED → READY_FOR_REVIEW. Always stops at review."""
        existing = tracker.find_by_job_id(job.id)
        if existing is not None and existing.status not in (
            ApplicationStatus.WITHDRAWN, ApplicationStatus.REJECTED,
        ):
            return existing

        app = Application(
            job_id=job.id,
            job_title=job.title,
            job_company=job.company,
            job_location=job.location,
            job_url=job.url,
            job_description=job.description,
            job_salary_text=job.salary_text or "",
            job_remote=job.remote,
            status=ApplicationStatus.DISCOVERED,
            candidate_name=profile.name or "",
            candidate_email=profile.email or "",
        )
        app.id = f"{re.sub(r'[^a-z0-9]+', '-', (job.company or 'job').lower())}-{uuid.uuid4().hex[:8]}"[:60]
        tracker.add(app)

        transition(app, ApplicationStatus.SELECTED, "User selected this job")

        target = find_application_url(job, page_html=page_html)
        summary = summarise_target(target, job)
        app.application_url = summary["application_url"]
        app.application_platform = summary["application_platform"]

        if not target.found:
            if target.requires_user_action:
                transition(app, ApplicationStatus.REQUIRES_USER_ACTION, target.reason)
                app.error = target.reason
            else:
                transition(app, ApplicationStatus.FAILED, target.reason)
                app.error = target.reason
            tracker.update(app)
            return app

        transition(app, ApplicationStatus.APPLICATION_PAGE_FOUND, target.evidence)

        # load the real page
        owned_driver = driver is None
        if driver is None:
            from .browser import open_driver
            driver = open_driver()
        try:
            # test doubles expose ``started``; real drivers start lazily
            if getattr(driver, "started", None) is False or not hasattr(driver, "started"):
                driver.start()
            snapshot = driver.goto(target.application_url)
            if not snapshot.status_ok:
                transition(app, ApplicationStatus.BLOCKED, snapshot.error)
                app.error = f"Application page unreachable: {snapshot.error}"
                tracker.update(app)
                return app

            challenge = driver.check_for_challenge()
            if challenge is not None:
                transition(
                    app, ApplicationStatus.REQUIRES_USER_ACTION,
                    challenge.user_message(),
                )
                app.error = challenge.user_message()
                tracker.update(app)
                return app

            html = page_html if page_html is not None else driver.page_html()
            analysis = analyze_application_page(
                html, page_url=driver.current_url(),
                platform=app.application_platform,
            )
        except BrowserUnavailable as exc:
            transition(app, ApplicationStatus.REQUIRES_USER_ACTION, str(exc))
            app.error = str(exc)
            tracker.update(app)
            return app
        except BrowserError as exc:
            transition(app, ApplicationStatus.FAILED, str(exc))
            app.error = str(exc)
            tracker.update(app)
            return app
        finally:
            if owned_driver:
                driver.close()

        if analysis.challenge is not None:
            transition(app, ApplicationStatus.REQUIRES_USER_ACTION,
                       analysis.challenge.user_message())
            app.error = analysis.challenge.user_message()
            tracker.update(app)
            return app

        if not analysis.has_form:
            transition(app, ApplicationStatus.FAILED,
                       "No application form found on the application page")
            app.error = "No application form found on the application page"
            tracker.update(app)
            return app

        transition(app, ApplicationStatus.FORM_ANALYSIS,
                   f"{len(analysis.fields)} fields discovered")
        app.form_analysis = analysis.summary()
        if analysis.submit_button is not None:
            app.form_analysis["submit_selector"] = analysis.submit_button.selector

        self._ensure_documents(app, profile, job)

        filler = FormFiller(cv_path=self._cv_path, cover_letter_path=self._cover_letter_path)
        remembered = {
            getattr(m, "question", ""): getattr(m, "answer", "")
            for m in getattr(profile, "question_memory", [])
            if getattr(m, "question", "") and getattr(m, "answer", "")
        }
        job_context = {
            "title": job.title,
            "company": job.company,
            "description": job.description,
            "requirements": "",
        }
        plan = filler.build_plan(analysis, profile, job_context, remembered)

        transition(app, ApplicationStatus.AUTO_FILLING,
                   f"{sum(1 for e in plan.entries if e.value)} fields auto-filled")

        # apply the fillable entries to the real browser
        fill_errors = self._apply_plan(driver, plan)
        app.errors.extend(fill_errors)

        app.answers = {
            e.question: e.value for e in plan.entries
            if e.value is not None and e.upload_kind == ""
        }
        app.unanswered_required = plan.unanswered_required

        transition(app, ApplicationStatus.READY_FOR_REVIEW,
                   "Stopped for mandatory human review")
        app.fill_plan = [e.model_dump() for e in plan.entries]
        tracker.update(app)
        self._last_plan = plan
        return app

    # -- phase 6: review payload ---------------------------------------------
    def build_review(self, app: Application, plan: Optional[FillPlan] = None) -> ReviewPreview:
        plan = plan or getattr(self, "_last_plan", None)
        if plan is None and app.fill_plan:
            from .form_filler import PlannedAnswer
            plan = FillPlan(entries=[PlannedAnswer(**e) for e in app.fill_plan])
        preview = ReviewPreview(
            application_id=app.id,
            company=app.job_company,
            job_title=app.job_title,
            job_url=app.job_url,
            application_url=app.application_url,
            application_platform=app.application_platform,
            candidate_name=app.candidate_name,
            candidate_email=app.candidate_email,
        )
        if plan is not None:
            preview.answers = [
                {
                    "question": e.question,
                    "value": e.value,
                    "answer_type": e.answer_type,
                    "source": e.source,
                    "needs_user": e.needs_user,
                    "reason": e.reason,
                    "required": e.required,
                    "conflict": e.conflict,
                }
                for e in plan.entries
            ]
            preview.unanswered_required = plan.unanswered_required
            if _own_words_flag(app):
                preview.warnings.append(
                    "This employer requires answers in your own words — "
                    "AI-generated drafts will NOT be submitted until you "
                    "rewrite them"
                )
            preview.consent_questions = [e.question for e in plan.consent_entries]
            preview.documents = {
                "cv_path": self._cv_path,
                "cover_letter_path": self._cover_letter_path,
            }
        preview.warnings = list(app.warnings)
        return preview

    # -- phase 7+8: explicit confirmation and real submission -----------------
    def confirm_and_submit(
        self,
        app: Application,
        tracker,
        driver: BrowserDriver,
        consent_granted: bool = False,
        user_answers: Optional[dict[str, str]] = None,
        plan: Optional[FillPlan] = None,
    ) -> Application:
        """Submit ONLY after explicit user confirmation. Never silently."""
        if app.status != ApplicationStatus.READY_FOR_REVIEW:
            raise ApplicationAutomationError(
                f"Application is {app.status.value}, not READY_FOR_REVIEW — "
                "nothing may be submitted"
            )

        plan = plan or getattr(self, "_last_plan", None)
        if plan is None and app.fill_plan:
            from .form_filler import PlannedAnswer
            plan = FillPlan(entries=[PlannedAnswer(**e) for e in app.fill_plan])
        if plan is None:
            raise ApplicationAutomationError("No fill plan available for this application")

        # outstanding questions answered by the user just now?
        user_answers = user_answers or {}
        for entry in plan.entries:
            if entry.needs_user and entry.question in user_answers:
                entry.value = user_answers[entry.question]
                entry.needs_user = False
                entry.answer_type = "verified"
                entry.reason = "Provided by you during review"

        # employer forbids AI-generated content: any answer that is still a
        # generated draft blocks submission — the user must rewrite it
        own_words_required = _own_words_flag(app)
        if own_words_required:
            generated = [
                e.question for e in plan.entries
                if e.answer_type == AnswerType.GENERATED_FROM_EVIDENCE.value
                and not e.is_consent and not e.is_terms
                and e.upload_kind == ""
            ]
            if generated:
                transition(app, ApplicationStatus.REQUIRES_USER_ACTION,
                           "Employer requires your own words")
                app.error = (
                    "Cannot submit — this employer asks for answers in your "
                    "own words. Rewrite these AI-generated drafts yourself: "
                    + "; ".join(generated)
                )
                tracker.update(app)
                return app

        # consent items are gated separately below, not as "missing answers"
        consent_questions = {e.question for e in plan.consent_entries}
        still_missing = [
            q for q in plan.unanswered_required if q not in consent_questions
        ]
        if still_missing:
            transition(app, ApplicationStatus.REQUIRES_USER_ACTION,
                       "Required information missing")
            app.unanswered_required = still_missing
            app.error = "Cannot submit — missing required answers: " + "; ".join(still_missing)
            tracker.update(app)
            return app

        consents = plan.consent_entries
        if consents and not consent_granted:
            transition(app, ApplicationStatus.REQUIRES_USER_ACTION,
                       "Explicit consent required")
            app.error = (
                "Cannot submit — you must explicitly agree to: "
                + "; ".join(e.question for e in consents)
            )
            tracker.update(app)
            return app

        transition(app, ApplicationStatus.USER_VERIFIED,
                   "User reviewed and confirmed the application")

        # consent checkboxes can now be ticked — the user explicitly agreed
        for entry in consents:
            try:
                driver.set_checkbox(entry.selector, True)
            except BrowserError as exc:
                app.errors.append(f"consent checkbox: {exc}")

        transition(app, ApplicationStatus.SUBMITTING, "Submitting via real browser")

        submit_selector = (app.form_analysis or {}).get("submit_selector", "")
        try:
            if submit_selector:
                driver.click(submit_selector)
            else:
                raise BrowserError("No submit button was identified on the form")
        except BrowserError as exc:
            transition(app, ApplicationStatus.FAILED, str(exc))
            app.error = f"Submission failed: {exc}"
            tracker.update(app)
            return app

        # verify what ACTUALLY happened — never infer success from the click
        try:
            page_text = driver.page_text()
            url_now = driver.current_url()
        except BrowserError as exc:
            transition(app, ApplicationStatus.FAILED, str(exc))
            app.error = f"Could not verify submission outcome: {exc}"
            tracker.update(app)
            return app

        validation_errors = driver.validation_errors()
        if validation_errors:
            transition(app, ApplicationStatus.FAILED,
                       "The form reported validation errors")
            app.error = "Submission rejected by the form: " + "; ".join(validation_errors[:5])
            tracker.update(app)
            return app

        challenge = driver.check_for_challenge()
        if challenge is not None:
            transition(app, ApplicationStatus.REQUIRES_USER_ACTION,
                       challenge.user_message())
            app.error = challenge.user_message()
            tracker.update(app)
            return app

        confirmed, snippet, reference = detect_submission_confirmation(page_text, url_now)
        if not confirmed:
            transition(app, ApplicationStatus.FAILED,
                       "No submission confirmation was detected after submitting")
            app.error = (
                "The browser could not confirm that the application was "
                "submitted — no confirmation message or success page appeared"
            )
            tracker.update(app)
            return app

        transition(app, ApplicationStatus.SUBMITTED,
                   f"Browser confirmed submission: {snippet[:80]}")
        app.submitted = True
        app.confirmation_text = snippet
        app.confirmation_url = url_now
        app.application_reference = reference
        app.submission_url = url_now
        app.submission_time = app.submitted_at
        app.date_submitted = app.submitted_at or ""
        # MOCKED vs REAL separation
        app.submission_mode = "real" if driver.is_real else "mocked_test"
        if not driver.is_real:
            app.warnings.append(
                "TEST RECORD — produced with a mocked browser; NOT a real submission"
            )
        tracker.update(app)
        return app

    # -- helpers ---------------------------------------------------------------
    def reprepare(
        self,
        app: Application,
        profile: CandidateProfile,
        driver: BrowserDriver,
    ) -> FillPlan:
        """Re-navigate to the real application page and re-fill the form so a
        submission can be confirmed in a LATER request (fresh browser
        session). Raises BrowserError/BrowserUnavailable on failure."""
        driver.start()
        snapshot = driver.goto(app.application_url)
        if not snapshot.status_ok:
            raise BrowserError(f"Application page unreachable: {snapshot.error}")
        challenge = driver.check_for_challenge()
        if challenge is not None:
            raise BrowserError(challenge.user_message())
        analysis = analyze_application_page(
            driver.page_html(), page_url=driver.current_url(),
            platform=app.application_platform,
        )
        if analysis.challenge is not None:
            raise BrowserError(analysis.challenge.user_message())
        filler = FormFiller(cv_path=self._cv_path, cover_letter_path=self._cover_letter_path)
        remembered = {
            getattr(m, "question", ""): getattr(m, "answer", "")
            for m in getattr(profile, "question_memory", [])
            if getattr(m, "question", "") and getattr(m, "answer", "")
        }
        job_context = {
            "title": app.job_title,
            "company": app.job_company,
            "description": app.job_description,
            "requirements": "",
        }
        plan = filler.build_plan(analysis, profile, job_context, remembered)
        self._apply_plan(driver, plan)
        return plan

    def _apply_plan(self, driver: BrowserDriver, plan: FillPlan) -> list[str]:
        errors: list[str] = []
        for entry in plan.entries:
            if entry.needs_user or entry.value is None:
                continue
            try:
                if entry.upload_kind:
                    driver.upload_file(entry.selector, entry.value)
                elif entry.field_type in ("select",):
                    driver.select_option(entry.selector, entry.value)
                elif entry.options:  # radio group
                    driver.click(f'{entry.selector}[value="{entry.value}"]')
                elif entry.field_type == "checkbox":
                    driver.set_checkbox(entry.selector, True)
                else:
                    driver.fill(entry.selector, entry.value)
            except BrowserError as exc:
                errors.append(f"{entry.question}: {exc}")
        return errors
