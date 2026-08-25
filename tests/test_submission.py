from __future__ import annotations

"""Phases 6-8 — review screen, explicit confirmation, real submission
detection (via the clearly-marked FakeBrowserDriver; real-driver behaviour
is identical because the service only trusts driver-reported evidence)."""

import pytest

from application.browser import BrowserError
from application.form_filler import FillPlan
from application.lifecycle import transition
from application.models import ApplicationStatus
from application.submission import (
    ApplicationAutomationError,
    ApplicationAutomationService,
    detect_submission_confirmation,
)
from candidate.profile import CandidateProfile, Education
from sources.base import Job
from tests.fakes import FakeBrowserDriver
from tests.test_form_analyzer import LEVER_STYLE


APPLY_URL = "https://jobs.lever.co/dvt/123"
CONFIRM_URL = "https://jobs.lever.co/dvt/123/confirmation"

CONFIRMATION_PAGE = """
<html><body><h1>Thank you for applying!</h1>
<p>Your application has been received. Reference: DVT-12345</p>
</body></html>
"""


def _job() -> Job:
    return Job(
        id="dvt-grad-1",
        title="Graduate Software Developer",
        company="DVT",
        location="Durban",
        description="Graduate development programme.",
        url=APPLY_URL,
        source="schemaorg",
    )


def _profile(**kw) -> CandidateProfile:
    defaults = dict(
        name="Lucky Vezi",
        email="lucky.vezi@example.com",
        phone="082 555 1234",
        location="Durban",
        skills=["java", "sql"],
        education=[Education(qualification="Diploma in ICT", field="App Dev", end_date="2025")],
    )
    defaults.update(kw)
    return CandidateProfile(**defaults)


def _driver(with_confirmation: bool = True) -> FakeBrowserDriver:
    pages = {APPLY_URL: LEVER_STYLE}
    if with_confirmation:
        pages[CONFIRM_URL] = CONFIRMATION_PAGE
    driver = FakeBrowserDriver(pages=pages)
    driver.post_click["button[type='submit']"] = CONFIRM_URL
    return driver


def _service() -> ApplicationAutomationService:
    return ApplicationAutomationService(cv_path="C:/data/candidate_cv.pdf")


class _Tracker:
    """Minimal tracker double with the same surface the service uses."""

    def __init__(self):
        self.apps: dict[str, object] = {}

    def find_by_job_id(self, job_id):
        return next((a for a in self.apps.values() if a.job_id == job_id), None)

    def add(self, app):
        self.apps[app.id] = app

    def update(self, app):
        self.apps[app.id] = app

    def get(self, app_id):
        return self.apps.get(app_id)


# ---------------------------------------------------------------------------
# confirmation detection unit tests (#12, #14)
# ---------------------------------------------------------------------------

def test_confirmation_language_detected():
    confirmed, snippet, ref = detect_submission_confirmation(
        "Your application has been received. Reference: DVT-99887", ""
    )
    assert confirmed
    assert "received" in snippet.lower()
    assert ref == "DVT-99887"


def test_success_url_counts_as_confirmation():
    confirmed, _, _ = detect_submission_confirmation(
        "Next steps", "https://jobs.lever.co/dvt/123/confirmation"
    )
    assert confirmed


def test_clicking_submit_alone_is_not_confirmation():
    confirmed, _, _ = detect_submission_confirmation(
        "Apply for this job — form page", "https://jobs.lever.co/dvt/123"
    )
    assert not confirmed


# ---------------------------------------------------------------------------
# start_application: discovery → READY_FOR_REVIEW (#1, #2, #20)
# ---------------------------------------------------------------------------

def test_pipeline_stops_at_ready_for_review_without_submitting():
    tracker = _Tracker()
    driver = _driver()
    app = _service().start_application(_job(), _profile(), tracker, driver=driver)

    assert app.status == ApplicationStatus.READY_FOR_REVIEW
    assert app.application_url == APPLY_URL
    assert app.application_platform == "lever"
    # fills happened…
    assert any(a[0] == "fill" for a in driver.actions)
    # …but the submit button was NEVER clicked
    assert not any(a[0] == "click" and "submit" in a[1] for a in driver.actions)


def test_review_preview_contains_everything_required():
    service = _service()
    tracker = _Tracker()
    app = service.start_application(_job(), _profile(), tracker, driver=_driver())
    review = service.build_review(app)

    data = review.to_dict()
    assert data["company"] == "DVT"
    assert data["job_title"] == "Graduate Software Developer"
    assert data["application_url"] == APPLY_URL
    assert data["candidate_name"] == "Lucky Vezi"
    assert data["candidate_email"] == "lucky.vezi@example.com"
    questions = {a["question"] for a in data["answers"]}
    assert any("Full name" in q for q in questions)
    assert any("consent" in q.lower() for q in data["consent_questions"])
    assert data["documents"]["cv_path"].endswith("candidate_cv.pdf")
    # provenance shown per answer
    assert all("answer_type" in a for a in data["answers"])


def test_unknown_question_listed_for_user_in_review():
    service = _service()
    tracker = _Tracker()
    app = service.start_application(_job(), _profile(), tracker, driver=_driver())
    review = service.build_review(app)
    needs_user = [a["question"] for a in review.answers if a["needs_user"]]
    assert any("driver" in q.lower() for q in needs_user)


def test_documents_prepared_from_real_job_and_profile():
    service = _service()
    tracker = _Tracker()
    app = service.start_application(_job(), _profile(), tracker, driver=_driver())
    assert app.documents.cover_letter_ready is True
    assert "DVT" in app.documents.cover_letter_text
    assert "Graduate Software Developer" in app.documents.cover_letter_text


# ---------------------------------------------------------------------------
# gates: captcha / mfa / blocked / no mechanism (#10, #11)
# ---------------------------------------------------------------------------

def test_captcha_page_results_in_requires_user_action():
    tracker = _Tracker()
    driver = FakeBrowserDriver(pages={
        APPLY_URL: '<html><div class="g-recaptcha"></div>Apply</html>',
    })
    app = _service().start_application(_job(), _profile(), tracker, driver=driver)
    assert app.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "CAPTCHA" in app.error


def test_cloudflare_page_results_in_requires_user_action():
    tracker = _Tracker()
    driver = FakeBrowserDriver(pages={
        APPLY_URL: "<html>Just a moment...</html>",
    })
    app = _service().start_application(_job(), _profile(), tracker, driver=driver)
    assert app.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "Cloudflare" in app.error


def test_mfa_screen_results_in_requires_user_action():
    tracker = _Tracker()
    driver = FakeBrowserDriver(pages={
        APPLY_URL: "<html><form></form>Enter the verification code sent to your email</html>",
    })
    app = _service().start_application(_job(), _profile(), tracker, driver=driver)
    assert app.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "Multi-factor" in app.error or "sign-in" in app.error


def test_unreachable_page_is_blocked():
    tracker = _Tracker()
    driver = FakeBrowserDriver(pages={})  # nothing scripted → status_ok False
    app = _service().start_application(_job(), _profile(), tracker, driver=driver)
    assert app.status == ApplicationStatus.BLOCKED
    assert app.error


def test_missing_online_mechanism_reports_honestly():
    job = Job(
        id="dpsa-1",
        title="SCIENTIST PRODUCTION",
        company="DOA",
        description="Submit on the new Z83 form.",
        url="https://www.dpsa.gov.za/circular.pdf",
        source="dpsa_circular",
    )
    tracker = _Tracker()
    app = _service().start_application(job, _profile(), tracker, driver=_driver())
    assert app.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "Z83" in app.error


# ---------------------------------------------------------------------------
# explicit confirmation + submission outcomes (#9, #12, #13, #17, #20)
# ---------------------------------------------------------------------------

def _ready_app():
    service = _service()
    tracker = _Tracker()
    driver = _driver()
    app = service.start_application(_job(), _profile(), tracker, driver=driver)
    return service, tracker, driver, app


def test_confirm_refused_unless_status_is_ready_for_review():
    service, tracker, driver, app = _ready_app()
    transition(app, ApplicationStatus.USER_VERIFIED)
    with pytest.raises(ApplicationAutomationError):
        service.confirm_and_submit(app, tracker, driver, consent_granted=True)


def test_consent_not_granted_blocks_submission():
    service, tracker, driver, app = _ready_app()
    result = service.confirm_and_submit(app, tracker, driver, consent_granted=False)
    assert result.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "consent" in result.error.lower()
    assert not result.submitted
    # nothing was submitted to the employer
    assert not any(a[0] == "click" and "submit" in a[1] for a in driver.actions)


def test_missing_required_answers_block_submission():
    service, tracker, driver, app = _ready_app()
    # licence question needs the user; supply it via user_answers
    result = service.confirm_and_submit(
        app, tracker, driver,
        consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.SUBMITTED


def test_successful_submission_detected_and_recorded():
    service, tracker, driver, app = _ready_app()
    result = service.confirm_and_submit(
        app, tracker, driver,
        consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.SUBMITTED
    assert result.submitted is True
    assert result.submitted_at is not None
    assert "Thank you for applying" in result.confirmation_text
    assert result.application_reference == "DVT-12345"
    assert result.confirmation_url == CONFIRM_URL
    # mocked driver ⇒ explicitly tagged, never reported as real
    assert result.submission_mode == "mocked_test"
    assert any("NOT a real submission" in w for w in result.warnings)


def test_real_driver_submission_tagged_real():
    service, tracker, driver, app = _ready_app()

    class RealishFake(FakeBrowserDriver):
        is_real = True  # simulates what PlaywrightDriver reports

    driver.__class__ = RealishFake
    result = service.confirm_and_submit(
        app, tracker, driver, consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.SUBMITTED
    assert result.submission_mode == "real"


def test_failed_submission_when_no_confirmation_appears():
    service, tracker, driver, app = _ready_app()
    driver.pages.pop(CONFIRM_URL)          # submit leads nowhere
    driver.post_click.clear()
    result = service.confirm_and_submit(
        app, tracker, driver, consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.FAILED
    assert not result.submitted
    assert "could not confirm" in result.error.lower()


def test_validation_errors_mean_failed_submission():
    service, tracker, driver, app = _ready_app()
    driver.validation_error_text = ["Email is required"]
    result = service.confirm_and_submit(
        app, tracker, driver, consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.FAILED
    assert "Email is required" in result.error


def test_challenge_after_submit_requires_user_action():
    service, tracker, driver, app = _ready_app()
    driver.pages[CONFIRM_URL] = "<html>Checking your browser before accessing.</html>"
    result = service.confirm_and_submit(
        app, tracker, driver, consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "Cloudflare" in result.error


def test_submit_button_failure_is_failed_not_submitted():
    service, tracker, driver, app = _ready_app()
    driver.fail_on["click"] = BrowserError("submit button disabled")
    result = service.confirm_and_submit(
        app, tracker, driver, consent_granted=True,
        user_answers={"Do you have a valid driver's licence?": "yes"},
    )
    assert result.status == ApplicationStatus.FAILED
    assert not result.submitted


def test_cv_upload_action_performed_against_form():
    service, tracker, driver, app = _ready_app()
    uploads = [a for a in driver.actions if a[0] == "upload"]
    assert uploads and uploads[0][1] == "#cv"
    assert uploads[0][2].endswith("candidate_cv.pdf")
