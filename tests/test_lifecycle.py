from __future__ import annotations

"""Phase 1 — application lifecycle model and persistence."""

import json

import pytest

from application.lifecycle import (
    CANONICAL_PATH,
    InvalidTransition,
    can_transition,
    is_submission_state,
    needs_user,
    transition,
)
from application.models import Application, ApplicationStatus
from application.tracker import ApplicationTracker


def _app(**kwargs) -> Application:
    defaults = dict(
        id="dvt-grad-001",
        job_id="job-1",
        job_title="Graduate Software Developer",
        job_company="DVT",
        job_url="https://careers.dvt.co.za/jobs/graduate-software-developer",
        # new-pipeline records start at DISCOVERED (legacy default stays DRAFT)
        status=ApplicationStatus.DISCOVERED,
    )
    defaults.update(kwargs)
    return Application(**defaults)


def test_canonical_path_covers_full_lifecycle():
    assert CANONICAL_PATH[0] == ApplicationStatus.DISCOVERED
    assert CANONICAL_PATH[-1] == ApplicationStatus.CONFIRMED
    for current, target in zip(CANONICAL_PATH, CANONICAL_PATH[1:]):
        assert can_transition(current, target), f"{current} → {target}"


def test_happy_path_transition_walk():
    app = _app()
    for target in CANONICAL_PATH[1:]:
        transition(app, target)
    assert app.status == ApplicationStatus.CONFIRMED
    assert len(app.status_history) == len(CANONICAL_PATH) - 1


def test_illegal_jump_is_rejected():
    app = _app()
    with pytest.raises(InvalidTransition):
        transition(app, ApplicationStatus.SUBMITTED)
    # state unchanged after rejection
    assert app.status == ApplicationStatus.DISCOVERED


def test_submitted_and_confirmed_are_distinct_statuses():
    assert ApplicationStatus.SUBMITTED != ApplicationStatus.CONFIRMED
    app = _app()
    transition(app, ApplicationStatus.SELECTED)
    transition(app, ApplicationStatus.APPLICATION_PAGE_FOUND)
    transition(app, ApplicationStatus.FORM_ANALYSIS)
    transition(app, ApplicationStatus.AUTO_FILLING)
    transition(app, ApplicationStatus.READY_FOR_REVIEW)
    transition(app, ApplicationStatus.USER_VERIFIED)
    transition(app, ApplicationStatus.SUBMITTED)
    assert app.status == ApplicationStatus.SUBMITTED
    assert not is_submission_state(ApplicationStatus.READY_FOR_REVIEW)
    assert is_submission_state(app.status)


def test_submitted_stamps_submitted_at():
    app = _app()
    for target in (
        ApplicationStatus.SELECTED,
        ApplicationStatus.APPLICATION_PAGE_FOUND,
        ApplicationStatus.FORM_ANALYSIS,
        ApplicationStatus.AUTO_FILLING,
        ApplicationStatus.READY_FOR_REVIEW,
        ApplicationStatus.USER_VERIFIED,
        ApplicationStatus.SUBMITTED,
    ):
        transition(app, target)
    assert app.submitted_at
    assert app.confirmation_received_at is None
    transition(app, ApplicationStatus.AWAITING_CONFIRMATION)
    transition(app, ApplicationStatus.CONFIRMED)
    assert app.confirmation_received_at


def test_requires_user_action_is_resumable():
    app = _app(status=ApplicationStatus.FORM_ANALYSIS)
    transition(
        app, ApplicationStatus.REQUIRES_USER_ACTION,
        note="Cloudflare verification required",
    )
    assert needs_user(app.status)
    transition(app, ApplicationStatus.AUTO_FILLING, note="user completed challenge")
    assert app.status == ApplicationStatus.AUTO_FILLING


def test_failed_can_retry_form_analysis():
    app = _app(status=ApplicationStatus.USER_VERIFIED)
    transition(app, ApplicationStatus.FAILED, note="form submit button missing")
    transition(app, ApplicationStatus.FORM_ANALYSIS, note="retry")
    assert app.status == ApplicationStatus.FORM_ANALYSIS


def test_status_persists_across_tracker_reload(tmp_path):
    path = tmp_path / "applications.json"
    tracker = ApplicationTracker(path=path)
    app = _app()
    tracker.add(app)
    transition(app, ApplicationStatus.SELECTED)
    transition(app, ApplicationStatus.APPLICATION_PAGE_FOUND)
    app.application_url = "https://boards.greenhouse.io/dvt/jobs/123"
    app.application_platform = "greenhouse"
    app.error = ""
    tracker.update(app)

    reloaded = ApplicationTracker(path=path)
    stored = reloaded.get("dvt-grad-001")
    assert stored is not None
    assert stored.status == ApplicationStatus.APPLICATION_PAGE_FOUND
    assert stored.application_url.endswith("/123")
    assert stored.application_platform == "greenhouse"
    assert stored.employer == "DVT"
    assert stored.status_history[-1]["to"] == "application_page_found"


def test_legacy_record_without_new_fields_still_loads(tmp_path):
    path = tmp_path / "applications.json"
    legacy = {
        "id": "old-1",
        "job_id": "j9",
        "job_title": "Developer",
        "job_company": "OldCo",
        "status": "draft",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    path.write_text(json.dumps([legacy]), encoding="utf-8")
    tracker = ApplicationTracker(path=path)
    app = tracker.get("old-1")
    assert app is not None
    assert app.status == ApplicationStatus.DRAFT
    assert app.application_url == ""
    assert app.submitted_at is None
    assert app.submission_mode == ""


def test_new_fields_round_trip_through_json(tmp_path):
    app = _app(
        application_url="https://dvt.applytojob.com/apply/abc",
        application_platform="lever",
        application_reference="DVT-12345",
        confirmation_text="Thank you for applying",
        submission_mode="real",
    )
    data = json.loads(app.model_dump_json())
    restored = Application(**data)
    assert restored.application_reference == "DVT-12345"
    assert restored.confirmation_text == "Thank you for applying"
    assert restored.submission_mode == "real"


def test_preview_includes_application_pipeline_fields():
    app = _app(application_url="https://x/apply", application_platform="workday")
    preview = app.to_preview()
    assert preview["application_url"] == "https://x/apply"
    assert preview["application_platform"] == "workday"
    assert preview["employer"] == "DVT"
