"""Dashboard API tests for the real-submission pipeline endpoints.

These use Flask's test client. The browser driver and automation service are
replaced with fakes — the endpoints themselves must enforce the safety gates
(explicit confirmation, status checks, honest email-confirmation states).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import application.tracker as tracker_module
from application.form_filler import FillPlan, PlannedAnswer
from application.lifecycle import transition
from application.models import Application, ApplicationStatus


# ---------------------------------------------------------------- fakes ----

class FakeDriver:
    is_real = True

    def start(self): ...
    def goto(self, url):
        from application.browser import PageSnapshot
        return PageSnapshot(url=url, title="Apply")
    def check_for_challenge(self):
        return None
    def page_html(self):
        return "<html><body></body></html>"
    def current_url(self):
        return "https://careers.dvt.co.za/apply/grad"
    def close(self): ...


class FakeService:
    def __init__(self):
        self.calls = []

    def build_review(self, app_obj):
        from application.submission import ReviewPreview
        return ReviewPreview(
            application_id=app_obj.id,
            company=app_obj.job_company,
            job_title=app_obj.job_title,
            application_url=app_obj.application_url,
        )

    def reprepare(self, app_obj, profile, driver):
        self.calls.append("reprepare")
        return FillPlan(entries=[])

    def confirm_and_submit(self, app_obj, tracker, driver, *, consent_granted,
                           user_answers, plan):
        self.calls.append(("confirm", consent_granted, dict(user_answers)))
        if not consent_granted:
            return app_obj
        transition(app_obj, ApplicationStatus.USER_VERIFIED, "user verified")
        transition(app_obj, ApplicationStatus.SUBMITTING, "submitting")
        transition(app_obj, ApplicationStatus.SUBMITTED, "submitted")
        app_obj.submission_mode = "real"
        return app_obj


# --------------------------------------------------------------- helpers ---

def _ready_app(**overrides) -> Application:
    kwargs = dict(
        job_id="j1",
        job_title="Graduate Software Developer",
        job_company="DVT",
        status=ApplicationStatus.READY_FOR_REVIEW,
    )
    kwargs.update(overrides)
    app = Application(**kwargs)
    app.application_url = "https://careers.dvt.co.za/apply/grad"
    app.application_platform = "greenhouse"
    app.fill_plan = [
        PlannedAnswer(
            name="first_name", question="First name", value="Thabo",
            answer_type="verified", required=True,
        ).model_dump()
    ]
    return app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker_module, "TRACKER_FILE", tmp_path / "applications.json")
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def _seed(tmp_path, app_obj) -> None:
    tracker_module.ApplicationTracker(path=tmp_path / "applications.json").add(app_obj)


# ----------------------------------------------------------------- tests ---

def test_dashboard_page_renders(client):
    resp = client.get("/applications")
    assert resp.status_code == 200
    assert b"Applications" in resp.data


def test_review_endpoint_returns_persisted_plan(client, tmp_path):
    app_obj = _ready_app()
    _seed(tmp_path, app_obj)
    resp = client.post(f"/api/applications/{app_obj.id}/review")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["application_id"] == app_obj.id
    assert data["company"] == "DVT"
    assert data["job_title"] == "Graduate Software Developer"


def test_review_unknown_application_returns_404(client):
    resp = client.post("/api/applications/nonexistent/review")
    assert resp.status_code == 404


def test_confirm_requires_ready_for_review_status(client, tmp_path):
    app_obj = _ready_app(status=ApplicationStatus.SUBMITTED)
    _seed(tmp_path, app_obj)
    resp = client.post(
        f"/api/applications/{app_obj.id}/confirm",
        json={"consent_granted": True},
    )
    assert resp.status_code == 400
    assert "ready_for_review" in resp.get_json()["error"]


def test_confirm_without_consent_refuses_submission(client, tmp_path, monkeypatch):
    app_obj = _ready_app()
    _seed(tmp_path, app_obj)
    service = FakeService()
    monkeypatch.setattr("app._submission_service", lambda: service)
    monkeypatch.setattr("application.browser.open_driver", lambda **kw: FakeDriver())

    resp = client.post(
        f"/api/applications/{app_obj.id}/confirm",
        json={"consent_granted": False},
    )
    assert resp.status_code == 200
    data = resp.get_json()["application"]
    # refused: still waiting for review, never submitted
    assert data["status"] == "ready_for_review"
    assert ("confirm", False, {}) in service.calls


def test_confirm_with_consent_submits_and_marks_real(client, tmp_path, monkeypatch):
    app_obj = _ready_app()
    _seed(tmp_path, app_obj)
    service = FakeService()
    monkeypatch.setattr("app._submission_service", lambda: service)
    monkeypatch.setattr("application.browser.open_driver", lambda **kw: FakeDriver())

    resp = client.post(
        f"/api/applications/{app_obj.id}/confirm",
        json={"consent_granted": True, "answers": {"Notice period": "Immediately"}},
    )
    assert resp.status_code == 200
    data = resp.get_json()["application"]
    assert data["status"] == "submitted"
    assert data["submission_mode"] == "real"
    assert ("confirm", True, {"Notice period": "Immediately"}) in service.calls

    persisted = tracker_module.ApplicationTracker(
        path=tmp_path / "applications.json").get(app_obj.id)
    assert persisted.status == ApplicationStatus.SUBMITTED


def test_check_confirmation_without_email_config_stays_awaiting(client, tmp_path, monkeypatch):
    for var in ("TASK_MASTER_IMAP_HOST", "TASK_MASTER_EMAIL_ADDRESS",
                "TASK_MASTER_EMAIL_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    app_obj = _ready_app(status=ApplicationStatus.SUBMITTED)
    _seed(tmp_path, app_obj)

    resp = client.post(f"/api/applications/{app_obj.id}/check-confirmation")
    assert resp.status_code == 200
    data = resp.get_json()["application"]
    assert data["status"] == "awaiting_confirmation"

    persisted = tracker_module.ApplicationTracker(
        path=tmp_path / "applications.json").get(app_obj.id)
    assert persisted.status == ApplicationStatus.AWAITING_CONFIRMATION
