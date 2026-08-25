from __future__ import annotations

"""Review-screen provenance grouping + explicit approval gate tests.

The review payload must separate answers into:
  user_verified / auto_answered / evidence_based / still_unanswered
and must always declare ``approval_required`` so the UI keeps the final
submission button disabled until the user explicitly approves.

No browser, no network, no submissions — pure payload testing.
"""

import json

import pytest

import config
from application.models import Application, ApplicationStatus
from application.form_filler import FillPlan, PlannedAnswer
from application.submission import (
    ApplicationAutomationService,
    ReviewPreview,
    _group_answers,
)


def _entry(question, value=None, answer_type="verified", source="profile",
           needs_user=False, required=False, **kw) -> dict:
    d = {
        "selector": "#f", "name": "f", "question": question,
        "field_type": "text", "category": "other",
        "required": required, "value": value,
        "answer_type": answer_type, "source": source,
        "needs_user": needs_user, "reason": "", "conflict": {},
    }
    d.update(kw)
    return d


def _app_with_plan(entries: list[dict], tmp_path) -> Application:
    app = Application(
        id="acme-review-1", job_id="j1",
        job_title="Developer", job_company="Acme",
        job_url="https://jobs.lever.co/acme/1",
        application_url="https://jobs.lever.co/acme/1/apply",
        application_platform="lever",
        status=ApplicationStatus.READY_FOR_REVIEW,
    )
    app.fill_plan = entries
    return app


# ---------------------------------------------------------------------------
# grouping logic (unit)
# ---------------------------------------------------------------------------

def test_group_answers_four_buckets():
    answers = [
        # remembered from an earlier session → user_verified
        _entry("Phone number", "082 000 0000", source="memory"),
        # typed by the user just now in the plan → user_verified
        _entry("Notice period", "30 days", source="answer_store"),
        # structured CV data → auto_answered
        _entry("Email", "lucky@example.com", source="profile"),
        # derived fact → evidence_based
        _entry("Years of experience", "1.5 years", "derived", source="derived"),
        # AI draft → evidence_based
        _entry("Why this role?", "As a ... graduate...",
               "generated_from_evidence", source="generated"),
        # missing required → still_unanswered
        _entry("Are you willing to relocate?", None, needs_user=True),
    ]
    groups = _group_answers(answers)
    assert [a["question"] for a in groups["user_verified"]] == [
        "Phone number", "Notice period"]
    assert [a["question"] for a in groups["auto_answered"]] == ["Email"]
    assert len(groups["evidence_based"]) == 2
    assert [a["question"] for a in groups["still_unanswered"]] == [
        "Are you willing to relocate?"]
    total = sum(len(v) for v in groups.values())
    assert total == len(answers)


def test_conflicting_entry_counts_as_unanswered():
    answers = [_entry(
        "Phone number", None, answer_type="unknown", source="",
        needs_user=True, conflict={"profile_value": "083", "remembered_value": "082"},
    )]
    assert _group_answers(answers)["still_unanswered"]


def test_empty_values_never_look_answered():
    answers = [_entry("Salary", "", source="memory")]
    assert _group_answers(answers)["still_unanswered"]


# ---------------------------------------------------------------------------
# ReviewPreview payload
# ---------------------------------------------------------------------------

def test_review_payload_has_groups_and_approval_flag():
    preview = ReviewPreview(application_id="x")
    preview.answers = [
        _entry("Phone", "082", source="memory"),
        _entry("Relocate?", None, needs_user=True),
    ]
    data = preview.to_dict()
    assert data["approval_required"] is True          # gate always declared
    assert set(data["groups"]) == {
        "user_verified", "auto_answered", "evidence_based", "still_unanswered"}
    assert len(data["groups"]["user_verified"]) == 1
    assert len(data["groups"]["still_unanswered"]) == 1


def test_build_review_groups_from_stored_fill_plan(tmp_path):
    """A persisted application (fill_plan as dicts) regroups correctly."""
    app = _app_with_plan([
        {"question": "Phone number", "value": "082 111 2222",
         "answer_type": "verified", "source": "memory"},
        {"question": "Expected salary", "value": "R25000",
         "answer_type": "verified"},  # legacy entry: no source field
        {"question": "Recent graduate?", "value": "Yes",
         "answer_type": "derived", "source": "derived"},
        {"question": "Portfolio URL", "value": None,
         "answer_type": "unknown", "needs_user": True},
    ], tmp_path)
    service = ApplicationAutomationService()
    preview = service.build_review(app)
    data = preview.to_dict()
    groups = data["groups"]

    assert [a["question"] for a in groups["user_verified"]] == ["Phone number"]
    # legacy entry without a source is treated conservatively: auto-filled
    assert [a["question"] for a in groups["auto_answered"]] == [
        "Expected salary"]
    assert [a["question"] for a in groups["evidence_based"]] == [
        "Recent graduate?"]
    assert [a["question"] for a in groups["still_unanswered"]] == [
        "Portfolio URL"]


def test_planned_answer_source_survives_roundtrip():
    plan = FillPlan(entries=[
        PlannedAnswer(question="Phone", value="082", answer_type="verified",
                      source="memory"),
    ])
    restored = FillPlan(entries=[PlannedAnswer(**e.model_dump())
                                 for e in plan.entries])
    assert restored.entries[0].source == "memory"


def test_legacy_fill_plan_dicts_still_load():
    plan = FillPlan(entries=[PlannedAnswer(**{
        "question": "Q", "value": "A", "answer_type": "verified"})])
    assert plan.entries[0].source == ""


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "APPLICATIONS_FILE", tmp_path / "applications.json")

    from application.tracker import ApplicationTracker

    app = _app_with_plan([
        {"question": "Phone number", "value": "082 111 2222",
         "answer_type": "verified", "source": "memory", "required": True},
        {"question": "Notice period", "value": None,
         "answer_type": "unknown", "needs_user": True, "required": True},
    ], tmp_path)
    ApplicationTracker().add(app)

    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_api_review_returns_groups_and_approval(client):
    res = client.post("/api/applications/acme-review-1/review")
    assert res.status_code == 200
    data = res.get_json()
    assert data["approval_required"] is True
    groups = data["groups"]
    assert [a["question"] for a in groups["user_verified"]] == ["Phone number"]
    assert [a["question"] for a in groups["still_unanswered"]] == [
        "Notice period"]
    # flat answers list preserved for backward compatibility
    assert len(data["answers"]) == 2
    assert data["ready"] is False   # required question unanswered


def test_confirm_service_blocked_without_consent_even_after_review(tmp_path):
    """The backend gate stays shut: with a required answer missing,
    confirm_and_submit must stop at REQUIRES_USER_ACTION and must not
    touch the browser at all."""
    from application.submission import (
        ApplicationAutomationService as Svc,  # local alias for clarity
    )

    app_obj = _app_with_plan([
        {"question": "Phone number", "value": "082 111 2222",
         "answer_type": "verified", "source": "memory", "required": True},
        {"question": "Notice period", "value": None,
         "answer_type": "unknown", "needs_user": True, "required": True},
    ], tmp_path)

    class BoomDriver:
        def __getattr__(self, name):
            raise AssertionError(f"browser.{name} must not be touched")

    class Tracker:
        def update(self, app):
            pass

    result = Svc().confirm_and_submit(
        app_obj, Tracker(), BoomDriver(), consent_granted=True,
    )
    assert result.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "missing required answers" in (result.error or "").lower()
    assert result.status != ApplicationStatus.SUBMITTED
