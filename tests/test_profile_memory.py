from __future__ import annotations

"""Persistent candidate profile / answer-memory system tests.

Every test runs against isolated tmp_path stores — the developer's real
profile and answers are never touched. No network, no submissions, no
emails: this is pure state-machine testing of the memory layer.
"""

import json
from datetime import datetime, timedelta

import pytest

import config
from candidate.profile import CandidateProfile, Education
from candidate import storage as storage_module

from application.profile_memory import (
    CATEGORY_ORDER,
    ProfileMemoryService,
    REGISTRY,
    REGISTRY_BY_KEY,
    answers_equivalent,
    field_for_question,
)
from application.question_engine import AnswerStore


def _service(tmp_path, profile=None, **kw) -> ProfileMemoryService:
    return ProfileMemoryService(
        profile=profile if profile is not None else CandidateProfile(),
        store=AnswerStore(tmp_path / "answers.json"),
        conflicts_path=tmp_path / "conflicts.json",
        persist_profile=False,
        **kw,
    )


# ---------------------------------------------------------------------------
# registry coverage — every recurring question the task lists has a home
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    # Personal
    "first_name", "last_name", "email", "phone", "country_of_residence",
    "citizenship", "south_african_citizen", "work_authorisation",
    # Education
    "highest_qualification", "university", "graduation_year",
    "recent_graduate",
    # Work & Experience
    "years_experience", "drivers_licence", "relocation", "notice_period",
    "availability", "expected_salary", "travel_preference",
    # Preferences / links / skills
    "work_preference", "online_linkedin", "skills",
}

SENSITIVE_MUST_BE = {"date_of_birth", "citizenship", "south_african_citizen",
                     "race", "gender", "disability"}


def test_registry_covers_all_required_fields():
    assert REQUIRED_KEYS <= set(REGISTRY_BY_KEY)


def test_registry_sensitive_flags():
    flagged = {f.key for f in REGISTRY if f.sensitive}
    assert SENSITIVE_MUST_BE <= flagged
    assert not ({f.key for f in REGISTRY} - SENSITIVE_MUST_BE) & flagged & {
        "first_name", "email", "phone", "highest_qualification",
    }


def test_categories_match_dashboard_groups():
    cats = {f.category for f in REGISTRY}
    assert cats <= set(CATEGORY_ORDER)


# ---------------------------------------------------------------------------
# question normalisation / semantic matching
# ---------------------------------------------------------------------------

def test_relocation_phrasings_map_to_same_field():
    for phrasing in (
        "Are you willing to relocate?",
        "Would you consider relocation?",
        "Are you open to relocating?",
        "If offered a job in another city, would you relocate?",
    ):
        field = field_for_question(phrasing)
        assert field is not None, phrasing
        assert field.key == "relocation"


def test_equivalent_answers_normalise_equal():
    assert answers_equivalent("Yes", "yes")
    assert answers_equivalent("Yes!", " yes ")
    assert answers_equivalent("Cape Town", "cape town")
    assert not answers_equivalent("Yes", "No")
    assert not answers_equivalent("", "Yes")


def test_uncertain_question_maps_to_no_field():
    """Ambiguous questions are NOT blindly matched — they get asked."""
    assert field_for_question("What is your greatest weakness?") is None
    assert field_for_question("Describe a difficult situation.") is None


# ---------------------------------------------------------------------------
# verified-answer reuse
# ---------------------------------------------------------------------------

def test_saved_answer_is_verified_and_reused(tmp_path):
    svc = _service(tmp_path)
    outcome = svc.save_user_answer("relocation", "Yes")
    assert outcome["ok"] and outcome["saved"]
    snap = svc.snapshot()
    entry = next(f for c in snap["categories"] for f in c["fields"]
                 if f["key"] == "relocation")
    assert entry["value"] == "Yes"
    assert entry["status"] == "verified"
    assert entry["verified"] is True
    assert entry["source"] == "user"


def test_reuse_survives_new_service_instance(tmp_path):
    """Persistence across application sessions."""
    first = _service(tmp_path)
    first.save_user_answer("phone", "082 111 2222")

    second = ProfileMemoryService(
        profile=CandidateProfile(),
        store=AnswerStore(tmp_path / "answers.json"),
        conflicts_path=tmp_path / "conflicts.json",
        persist_profile=False,
    )
    snap = second.snapshot()
    entry = next(f for c in snap["categories"] for f in c["fields"]
                 if f["key"] == "phone")
    assert entry["value"] == "082 111 2222"
    assert entry["verified"] is True


def test_equivalent_future_question_reuses_saved_answer(tmp_path):
    """A differently-phrased employer question hits the memory."""
    svc = _service(tmp_path)
    svc.save_user_answer("relocation", "Yes")
    result = svc.engine.answer("Would you consider relocation?",
                               svc.profile)
    assert result.answer == "Yes"
    assert not result.needs_user


def test_legacy_string_store_loads_as_user_verified(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"expected_salary": "R25000"}), encoding="utf-8")
    svc = ProfileMemoryService(
        profile=CandidateProfile(),
        store=AnswerStore(path),
        conflicts_path=tmp_path / "conflicts.json",
        persist_profile=False,
    )
    entry = next(f for c in svc.snapshot()["categories"]
                 for f in c["fields"] if f["key"] == "expected_salary")
    assert entry["value"] == "R25000"
    assert entry["verified"] is True
    assert entry["source"] == "user"


# ---------------------------------------------------------------------------
# unknown-question handling — never fabricate
# ---------------------------------------------------------------------------

def test_unknown_personal_facts_are_asked_not_invented(tmp_path):
    svc = _service(tmp_path)
    missing = svc.missing_questions()
    keys = {q["key"] for q in missing}
    assert {"notice_period", "availability", "expected_salary"} <= keys


def test_missing_questions_exposed_with_canonical_question_text(tmp_path):
    svc = _service(tmp_path)
    notice = next(q for q in svc.missing_questions() if q["key"] == "notice_period")
    assert "notice period" in notice["question"].lower()


def test_empty_answer_rejected(tmp_path):
    svc = _service(tmp_path)
    assert svc.save_user_answer("phone", "   ")["ok"] is False
    assert svc.save_user_answer("", "x")["ok"] is False


# ---------------------------------------------------------------------------
# sensitive-question handling — USER_REQUIRED, never inferred
# ---------------------------------------------------------------------------

def test_sensitive_unanswered_marked_user_required(tmp_path):
    svc = _service(tmp_path)
    fields = {f["key"]: f for c in svc.snapshot()["categories"]
              for f in c["fields"]}
    for key in ("race", "gender", "disability", "date_of_birth"):
        assert fields[key]["status"] == "user_required", key


def test_sensitive_never_taken_from_cv_like_data(tmp_path):
    """Even a fully populated CV profile must not produce demographic
    answers — only explicit user entry counts."""
    profile = CandidateProfile(
        name="Lucky Vezi",
        email="lucky@example.com",
        phone="082 000 0000",
        professional_summary="Team player. Female. African.",
        location="Durban",
    )
    svc = _service(tmp_path, profile=profile)
    fields = {f["key"]: f for c in svc.snapshot()["categories"]
              for f in c["fields"]}
    for key in ("race", "gender", "disability"):
        assert fields[key]["status"] == "user_required"
        assert fields[key]["value"] in (None, "")


def test_sensitive_answer_once_then_reused(tmp_path):
    svc = _service(tmp_path)
    outcome = svc.save_user_answer("gender", "Female")
    assert outcome["ok"]
    fields = {f["key"]: f for c in svc.snapshot()["categories"]
              for f in c["fields"]}
    assert fields["gender"]["status"] == "verified"
    assert fields["gender"]["value"] == "Female"

    result = svc.engine.answer("What is your gender?", svc.profile)
    assert result.answer == "Female"
    assert not result.needs_user


# ---------------------------------------------------------------------------
# conflicting-answer handling — verified answers are never overwritten
# ---------------------------------------------------------------------------

def test_conflicting_answer_creates_pending_conflict_not_overwrite(tmp_path):
    svc = _service(tmp_path)
    svc.save_user_answer("phone", "082 OLD 1111")
    outcome = svc.save_user_answer("phone", "083 NEW 2222")
    assert outcome.get("conflict") is True
    assert outcome["existing_value"] == "082 OLD 1111"
    assert outcome["proposed_value"] == "083 NEW 2222"
    # nothing changed yet
    assert svc.store.get("phone") == "082 OLD 1111"
    assert len(svc.pending_conflicts()) == 1


def test_conflict_persisted_across_instances(tmp_path):
    _service(tmp_path).save_user_answer("phone", "082 OLD 1111")
    _service(tmp_path).save_user_answer("phone", "083 NEW 2222")
    reopened = _service(tmp_path)
    assert len(reopened.pending_conflicts()) == 1


def test_resolve_conflict_keep_existing(tmp_path):
    svc = _service(tmp_path)
    first = svc.save_user_answer("phone", "082 KEEP 1111")
    conflict = svc.save_user_answer("phone", "083 DROP 2222")
    outcome = svc.resolve_conflict(conflict["conflict_id"], "existing")
    assert outcome["ok"]
    assert svc.store.get("phone") == "082 KEEP 1111"
    rec = svc.store.record("phone")
    assert rec.verified and rec.source == "user"
    assert svc.pending_conflicts() == []
    assert first["ok"]


def test_resolve_conflict_use_new_overwrites_after_explicit_choice(tmp_path):
    svc = _service(tmp_path)
    svc.save_user_answer("phone", "082 OLD 1111")
    conflict = svc.save_user_answer("phone", "083 NEW 2222")
    outcome = svc.resolve_conflict(conflict["conflict_id"], "new")
    assert outcome["ok"]
    assert svc.store.get("phone") == "083 NEW 2222"
    assert svc.store.record("phone").verified is True
    assert svc.pending_conflicts() == []


def test_resolve_invalid_choice_and_missing_id(tmp_path):
    svc = _service(tmp_path)
    svc.save_user_answer("phone", "082 X 1111")
    conflict = svc.save_user_answer("phone", "083 Y 2222")
    assert svc.resolve_conflict(conflict["conflict_id"], "maybe")["ok"] is False
    assert svc.resolve_conflict("nope", "new")["ok"] is False
    # original untouched
    assert svc.store.get("phone") == "082 X 1111"


def test_same_value_resave_does_not_conflict(tmp_path):
    svc = _service(tmp_path)
    svc.save_user_answer("relocation", "Yes")
    again = svc.save_user_answer("relocation", "yes!")
    assert again.get("conflict") is None
    assert again.get("unchanged") is True
    assert svc.pending_conflicts() == []


# ---------------------------------------------------------------------------
# evidence-based / derived answers clearly marked
# ---------------------------------------------------------------------------

def test_derived_graduate_status_marked_needs_confirmation(tmp_path):
    year = datetime.now().year - 1
    profile = CandidateProfile(education=[Education(
        institution="DUT", qualification="Diploma", field="ICT",
        start_date=f"{year - 3}-01", end_date=f"{year}-06",
        is_highest=True,
    )])
    svc = _service(tmp_path, profile=profile)
    entry = next(f for c in svc.snapshot()["categories"]
                 for f in c["fields"] if f["key"] == "recent_graduate")
    assert entry["value"] == "Yes"          # derived, shown…
    assert entry["status"] == "needs_confirmation"  # …but flagged
    assert entry["verified"] is False       # …never as user-verified


# ---------------------------------------------------------------------------
# custom (free-form) application questions
# ---------------------------------------------------------------------------

def test_custom_application_question_listed_and_reused(tmp_path):
    profile = CandidateProfile()
    profile.remember_answer(
        "How many A4 sheets fit in your car?", "400",
        field_key="q_abc123", source="user", confidence="high",
    )
    svc = _service(tmp_path, profile=profile)
    snap = svc.snapshot()
    cat = next(c for c in snap["categories"]
               if c["name"] == "Application Questions")
    assert any(f["label"].startswith("How many A4") for f in cat["fields"])
    assert all(f["category"] == "Application Questions" for f in cat["fields"])


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """Flask test client wired to tmp stores."""
    monkeypatch.setattr(config, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(config, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")
    monkeypatch.setattr(storage_module, "PROFILE_FILE", tmp_path / "profile.json")

    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_memory_snapshot_shape(api_client):
    res = api_client.get("/api/profile/memory")
    assert res.status_code == 200
    data = res.get_json()
    names = [c["name"] for c in data["categories"]]
    assert "Personal" in names and "Contact" in names
    assert set(data["counts"]) >= {"verified", "unknown", "user_required"}
    assert isinstance(data["pending_conflicts"], list)


def test_api_memory_save_and_conflict_flow(api_client):
    res = api_client.post("/api/profile/memory/answer",
                          json={"key": "notice_period", "answer": "30 days"})
    assert res.status_code == 200
    assert res.get_json()["saved"] is True

    # conflicting answer → 409 + conflict payload, value untouched
    res = api_client.post("/api/profile/memory/answer",
                          json={"key": "notice_period", "answer": "Immediate"})
    assert res.status_code == 409
    body = res.get_json()
    conflict_id = body["conflict_id"]
    assert body["existing_value"] == "30 days"

    snap = api_client.get("/api/profile/memory").get_json()
    assert len(snap["pending_conflicts"]) == 1

    # resolve → keep current
    res = api_client.post(
        f"/api/profile/memory/conflict/{conflict_id}/resolve",
        json={"choice": "existing"})
    assert res.status_code == 200
    snap = api_client.get("/api/profile/memory").get_json()
    assert snap["pending_conflicts"] == []
    entry = next(f for c in snap["categories"] for f in c["fields"]
                 if f["key"] == "notice_period")
    assert entry["value"] == "30 days"


def test_api_memory_missing_endpoint_lists_gaps(api_client):
    res = api_client.get("/api/profile/memory/missing")
    assert res.status_code == 200
    questions = res.get_json()["questions"]
    assert questions, "an empty profile must have something to ask"
    assert all({"key", "question", "priority"} <= set(q) for q in questions)
    assert any(q["priority"] == "high" for q in questions)  # sensitive ones


def test_api_bad_requests(api_client):
    assert api_client.post("/api/profile/memory/answer",
                           json={"key": "", "answer": ""}).status_code == 400
    res = api_client.post("/api/profile/memory/conflict/x/resolve",
                          json={"choice": "sideways"})
    assert res.status_code == 400
