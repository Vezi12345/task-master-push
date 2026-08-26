"""Tests that all submission paths converge on ApplicationAutomationService.

Verifies:
- orchestrator._submit_application() routes through the centralized service
- app.py legacy /approve endpoint routes through the centralized service
- Safety checks (status validation, confirmation detection) are enforced
- No path bypasses the centralized submission gate
"""

from __future__ import annotations

import re

import pytest

from application.browser import BrowserError
from application.form_filler import FillPlan, PlannedAnswer
from application.lifecycle import transition
from application.models import Application, ApplicationStatus
from application.submission import (
    ApplicationAutomationError,
    ApplicationAutomationService,
    detect_submission_confirmation,
)
from application.tracker import ApplicationTracker
from candidate.profile import CandidateProfile, Education
from sources.base import Job
from tests.fakes import FakeBrowserDriver
from tests.test_form_analyzer import LEVER_STYLE


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

APPLY_URL = "https://jobs.lever.co/dvt/123"
CONFIRM_URL = "https://jobs.lever.co/dvt/123/confirmation"

CONFIRMATION_PAGE = """
<html><body><h1>Thank you for applying!</h1>
<p>Your application has been received. Reference: DVT-99999</p>
</body></html>
"""


def _job() -> Job:
    return Job(
        id="cent-test-1",
        title="Test Developer",
        company="CentTest",
        location="Durban",
        description="Test role.",
        url=APPLY_URL,
        source="test",
    )


def _profile() -> CandidateProfile:
    return CandidateProfile(
        name="Test User",
        email="test@example.com",
        phone="082 555 0000",
        location="Durban",
        skills=["python"],
        education=[Education(qualification="Diploma", field="IT")],
    )


def _driver(with_confirmation: bool = True) -> FakeBrowserDriver:
    pages = {APPLY_URL: LEVER_STYLE}
    if with_confirmation:
        pages[CONFIRM_URL] = CONFIRMATION_PAGE
    driver = FakeBrowserDriver(pages=pages)
    driver.post_click["button[type='submit']"] = CONFIRM_URL
    return driver


# ---------------------------------------------------------------------------
# orchestrator._submit_application routes through centralized service
# ---------------------------------------------------------------------------

class TestOrchestratorSubmitCentralization:
    """Verify that _submit_application uses ApplicationAutomationService."""

    def _make_agent(self, tmp_path, monkeypatch):
        """Create a JobApplicationAgent with isolated tracker."""
        from agent.orchestrator import JobApplicationAgent
        import config
        import application.tracker as tracker_mod
        monkeypatch.setattr(tracker_mod, "TRACKER_FILE", tmp_path / "apps.json")
        monkeypatch.setattr(config, "ANSWERS_FILE", tmp_path / "answers.json")
        monkeypatch.setattr(config, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")
        region = config.load_region("za")
        agent = JobApplicationAgent(region)
        return agent

    def test_submit_uses_centralized_service(self, tmp_path, monkeypatch):
        """_submit_application must call confirm_and_submit, not fill_and_submit."""
        agent = self._make_agent(tmp_path, monkeypatch)

        app = Application(
            id="test-sub-1",
            job_id="j-1",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.AWAITING_APPROVAL,
            answers={"Full name": "Test User", "Email": "test@example.com"},
        )
        agent.tracker.add(app)

        called = {"service": False}

        class FakeService:
            def confirm_and_submit(self, app, tracker, driver, *,
                                   consent_granted, user_answers, plan):
                called["service"] = True
                assert consent_granted is True, "must pass consent"
                assert plan is not None, "must pass a fill plan"
                app.status = ApplicationStatus.SUBMITTED
                app.submitted = True
                app.confirmation_text = "Received"
                app.submission_mode = "mocked_test"
                tracker.update(app)
                return app

        # Patch at application.submission since that's where the local import resolves
        import application.submission as sub_mod
        monkeypatch.setattr(sub_mod, "ApplicationAutomationService", lambda: FakeService())
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda: FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE}),
        )

        result = agent._submit_application(app)

        assert result is True
        assert called["service"], "confirm_and_submit must have been called"
        assert app.status == ApplicationStatus.SUBMITTED
        assert app.submitted is True

    def test_submit_rejects_wrong_status(self, tmp_path, monkeypatch):
        """Application in DISCOVERED status must not be submitted."""
        agent = self._make_agent(tmp_path, monkeypatch)

        app = Application(
            id="test-sub-2",
            job_id="j-2",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.DISCOVERED,
        )
        agent.tracker.add(app)

        result = agent._submit_application(app)

        assert result is False
        assert app.status == ApplicationStatus.FAILED
        assert "expected ready_for_review or awaiting_approval" in app.errors[-1].lower()

    def test_submit_transitions_awaiting_to_ready(self, tmp_path, monkeypatch):
        """Legacy AWAITING_APPROVAL status is transitioned to READY_FOR_REVIEW
        before calling the centralized service."""
        agent = self._make_agent(tmp_path, monkeypatch)

        app = Application(
            id="test-sub-3",
            job_id="j-3",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.AWAITING_APPROVAL,
            answers={"Name": "Test"},
        )
        agent.tracker.add(app)

        transitions_seen = []

        class TrackService:
            def confirm_and_submit(self, app, tracker, driver, **kwargs):
                transitions_seen.append(app.status.value)
                app.status = ApplicationStatus.SUBMITTED
                app.submitted = True
                app.confirmation_text = "Done"
                app.submission_mode = "mocked_test"
                tracker.update(app)
                return app

        import application.submission as sub_mod
        monkeypatch.setattr(sub_mod, "ApplicationAutomationService", lambda: TrackService())
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda: FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE}),
        )

        result = agent._submit_application(app)

        assert result is True
        # The app should have been READY_FOR_REVIEW when confirm was called
        assert "ready_for_review" in transitions_seen

    def test_submit_handles_service_exception(self, tmp_path, monkeypatch):
        """Exceptions from the service are caught and recorded."""
        agent = self._make_agent(tmp_path, monkeypatch)

        app = Application(
            id="test-sub-4",
            job_id="j-4",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.AWAITING_APPROVAL,
        )
        agent.tracker.add(app)

        def boom(*a, **kw):
            raise RuntimeError("browser exploded")

        import application.submission as sub_mod
        monkeypatch.setattr(
            sub_mod, "ApplicationAutomationService",
            lambda: type("S", (), {"confirm_and_submit": staticmethod(boom)})(),
        )
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda: FakeBrowserDriver(),
        )

        result = agent._submit_application(app)

        assert result is False
        assert app.status == ApplicationStatus.FAILED
        assert "browser exploded" in app.errors[-1]

    def test_submit_uses_reprepare_for_legacy_apps(self, tmp_path, monkeypatch):
        """Legacy apps without form_analysis trigger reprepare() to analyse
        the real page and discover the submit button selector."""
        agent = self._make_agent(tmp_path, monkeypatch)

        app = Application(
            id="test-sub-5",
            job_id="j-5",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            application_url=APPLY_URL,
            status=ApplicationStatus.AWAITING_APPROVAL,
            answers={"Full name": "Test User"},
        )
        # No form_analysis — simulates legacy _prepare_applications() path
        assert not app.form_analysis
        agent.tracker.add(app)

        from candidate.profile import CandidateProfile, Education
        profile = CandidateProfile(
            name="Test", email="t@t.com",
            education=[Education(qualification="Diploma", field="IT")],
        )
        from candidate import storage as storage_mod
        monkeypatch.setattr(storage_mod, "load_profile", lambda: profile)

        reprepare_called = {"called": False}
        submit_called = {"called": False}

        class FakeService:
            def reprepare(self, app, profile, driver):
                reprepare_called["called"] = True
                # Simulate what reprepare does: set form_analysis and return a plan
                app.form_analysis = {"submit_selector": "button[type='submit']"}
                from application.form_filler import FillPlan, PlannedAnswer
                return FillPlan(entries=[
                    PlannedAnswer(question="Full name", value="Test User",
                                  answer_type="verified", source="user"),
                ])

            def confirm_and_submit(self, app, tracker, driver, **kwargs):
                submit_called["called"] = True
                # Verify reprepare populated the submit selector
                assert app.form_analysis.get("submit_selector"), \
                    "reprepare should have set submit_selector"
                app.status = ApplicationStatus.SUBMITTED
                app.submitted = True
                app.confirmation_text = "OK"
                app.submission_mode = "mocked_test"
                tracker.update(app)
                return app

        import application.submission as sub_mod
        monkeypatch.setattr(sub_mod, "ApplicationAutomationService", lambda: FakeService())
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda: FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE}),
        )

        result = agent._submit_application(app)

        assert result is True
        assert reprepare_called["called"], "reprepare must have been called for legacy app"
        assert submit_called["called"], "confirm_and_submit must have been called"
        assert app.form_analysis.get("submit_selector") == "button[type='submit']"

    def test_submit_skips_reprepare_when_form_analysis_exists(self, tmp_path, monkeypatch):
        """Apps that already have form_analysis skip reprepare()."""
        agent = self._make_agent(tmp_path, monkeypatch)

        app = Application(
            id="test-sub-6",
            job_id="j-6",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            application_url=APPLY_URL,
            status=ApplicationStatus.AWAITING_APPROVAL,
            answers={"Full name": "Test User"},
            form_analysis={"submit_selector": "#submit-btn", "fields": []},
        )
        agent.tracker.add(app)

        reprepare_called = {"called": False}

        class FakeService:
            def reprepare(self, app, profile, driver):
                reprepare_called["called"] = True
                raise AssertionError("reprepare should NOT be called when form_analysis exists")

            def confirm_and_submit(self, app, tracker, driver, **kwargs):
                app.status = ApplicationStatus.SUBMITTED
                app.submitted = True
                app.confirmation_text = "OK"
                app.submission_mode = "mocked_test"
                tracker.update(app)
                return app

        import application.submission as sub_mod
        monkeypatch.setattr(sub_mod, "ApplicationAutomationService", lambda: FakeService())
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda: FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE}),
        )

        result = agent._submit_application(app)

        assert result is True
        assert not reprepare_called["called"], "reprepare should be skipped"


# ---------------------------------------------------------------------------
# Legacy /approve endpoint centralization
# ---------------------------------------------------------------------------

class TestLegacyApproveEndpointCentralization:
    """Verify that /api/applications/<id>/approve uses centralized service."""

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        from candidate import storage as storage_mod
        import application.tracker as tracker_mod
        import config as config_mod

        monkeypatch.setattr(storage_mod, "PROFILE_FILE", tmp_path / "profile.json")
        monkeypatch.setattr(storage_mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tracker_mod, "TRACKER_FILE", tmp_path / "apps.json")
        monkeypatch.setattr(config_mod, "ANSWERS_FILE", tmp_path / "answers.json")
        monkeypatch.setattr(config_mod, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")
        monkeypatch.setattr(config_mod, "CV_FILE", tmp_path / "cv.pdf")
        monkeypatch.setattr(config_mod, "COVER_LETTER_FILE", tmp_path / "letter.txt")

        profile = _profile()
        storage_mod.save_profile(profile)

        from app import app as flask_app
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c

    def test_legacy_approve_uses_centralized_service(self, client, tmp_path, monkeypatch):
        """The legacy /approve endpoint must route through confirm_and_submit."""
        tracker = ApplicationTracker(path=tmp_path / "apps.json")
        app_obj = Application(
            id="legacy-1",
            job_id="j-leg-1",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.AWAITING_APPROVAL,
            answers={"Full name": "Test User"},
        )
        tracker.add(app_obj)

        service_called = {"submit": False}

        class FakeSubService:
            def confirm_and_submit(self, app, tracker, driver, **kwargs):
                service_called["submit"] = True
                assert kwargs.get("consent_granted") is True
                app.status = ApplicationStatus.SUBMITTED
                app.submitted = True
                app.confirmation_text = "Received"
                app.submission_mode = "mocked_test"
                tracker.update(app)
                return app

        import app as app_module
        monkeypatch.setattr(
            app_module,
            "_submission_service",
            lambda: FakeSubService(),
        )
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda **kw: FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE}),
        )

        resp = client.post("/api/applications/legacy-1/approve")
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["ok"] is True
        assert service_called["submit"], "confirm_and_submit must have been called"

    def test_legacy_approve_rejects_wrong_status(self, client, tmp_path):
        """Applications not in AWAITING_APPROVAL or READY_FOR_REVIEW are rejected."""
        tracker = ApplicationTracker(path=tmp_path / "apps.json")
        app_obj = Application(
            id="legacy-2",
            job_id="j-leg-2",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.DRAFT,
        )
        tracker.add(app_obj)

        resp = client.post("/api/applications/legacy-2/approve")
        data = resp.get_json()

        assert resp.status_code == 400
        assert "error" in data
        assert "draft" in data["error"].lower()

    def test_legacy_approve_accepts_ready_for_review(self, client, tmp_path, monkeypatch):
        """READY_FOR_REVIEW status should be accepted by the legacy endpoint."""
        tracker = ApplicationTracker(path=tmp_path / "apps.json")
        app_obj = Application(
            id="legacy-3",
            job_id="j-leg-3",
            job_title="Dev",
            job_company="Co",
            job_url=APPLY_URL,
            status=ApplicationStatus.READY_FOR_REVIEW,
            answers={"Full name": "Test User"},
        )
        tracker.add(app_obj)

        class FakeSubService:
            def confirm_and_submit(self, app, tracker, driver, **kwargs):
                app.status = ApplicationStatus.SUBMITTED
                app.submitted = True
                app.confirmation_text = "OK"
                app.submission_mode = "mocked_test"
                tracker.update(app)
                return app

        import app as app_module
        monkeypatch.setattr(app_module, "_submission_service", lambda: FakeSubService())
        monkeypatch.setattr(
            "application.browser.open_driver",
            lambda **kw: FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE}),
        )

        resp = client.post("/api/applications/legacy-3/approve")
        data = resp.get_json()

        assert resp.status_code == 200
        assert data["ok"] is True


# ---------------------------------------------------------------------------
# Confirmation detection still works in all paths
# ---------------------------------------------------------------------------

class TestConfirmationDetectionInAllPaths:
    """Ensure detect_submission_confirmation is used everywhere."""

    def test_confirmation_required_for_submitted_status(self):
        """A submission without confirmation text must NOT be marked submitted."""
        confirmed, snippet, ref = detect_submission_confirmation(
            "Apply for this job - form", APPLY_URL
        )
        assert not confirmed
        assert not snippet
        assert not ref

    def test_confirmation_detected_on_success_page(self):
        confirmed, snippet, ref = detect_submission_confirmation(
            CONFIRMATION_PAGE, CONFIRM_URL
        )
        assert confirmed
        assert "DVT-99999" in ref

    def test_confirmation_via_url_pattern(self):
        confirmed, _, _ = detect_submission_confirmation(
            "Next steps", "https://example.com/thank-you"
        )
        assert confirmed


# ---------------------------------------------------------------------------
# No direct platform.fill_and_submit calls in orchestrator or app.py
# ---------------------------------------------------------------------------

class TestNoDirectPlatformCalls:
    """Static analysis: verify no production code calls fill_and_submit directly."""

    def test_orchestrator_no_direct_fill_and_submit(self):
        """orchestrator.py must not contain direct fill_and_submit calls."""
        import inspect
        from agent import orchestrator
        source = inspect.getsource(orchestrator)
        # Strip comments before checking for dangerous calls
        code_lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Strip inline comments
            code_part = line.split("#")[0]
            code_lines.append(code_part)
        code_only = "\n".join(code_lines)
        assert "platform.fill_and_submit" not in code_only, (
            "orchestrator.py still contains direct platform.fill_and_submit() call"
        )

    def test_app_py_no_direct_fill_and_submit_in_approve(self):
        """The /approve endpoint must not call fill_and_submit directly."""
        import inspect
        from app import api_approve_application
        source = inspect.getsource(api_approve_application)
        # Strip comments before checking
        code_lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            code_part = line.split("#")[0]
            code_lines.append(code_part)
        code_only = "\n".join(code_lines)
        assert "platform.fill_and_submit" not in code_only, (
            "api_approve_application still contains direct platform.fill_and_submit() call"
        )
