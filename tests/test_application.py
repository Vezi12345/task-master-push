from __future__ import annotations

import json
import uuid

import pytest

from application.models import Application, ApplicationStatus, DocumentStatus, MissingInfo
from application.question_engine import AnswerStore, QuestionEngine
from application.cover_letter import CoverLetterGenerator, _build_candidate_summary
from application.form_filler import (
    ApplicationPlatform,
    FormField,
    FieldType,
    GenericApplicationAdapter,
    PlaywrightApplicationAdapter,
    PlatformRegistry,
    SubmissionResult,
    get_platform,
    register_platform,
)
from application.documents import ApplicationDocuments
from application.tracker import ApplicationTracker
from application.scoring import application_priority_score, compute_all_scores, rank_applications
from candidate.profile import CandidateProfile, Education, Experience, Certification, Project, KnownField, KnowledgeStatus
from sources.base import Job


# ---------------------------------------------------------------------------
# Application model tests
# ---------------------------------------------------------------------------

def test_application_defaults():
    app = Application(id="test-1", job_id="job-1", job_title="Dev")
    assert app.status == ApplicationStatus.DRAFT
    assert app.submitted is False
    assert app.application_priority == 0
    assert app.created_at != ""
    assert isinstance(app.documents, DocumentStatus)


def test_application_update_status():
    app = Application(id="test-1")
    app.update_status(ApplicationStatus.PREPARING)
    assert app.status == ApplicationStatus.PREPARING
    assert app.updated_at >= app.created_at


def test_application_is_submittable():
    app = Application(
        id="test-1",
        status=ApplicationStatus.AWAITING_APPROVAL,
        submitted=False,
    )
    assert app.is_submittable is True


def test_application_not_submittable_when_missing_info():
    app = Application(
        id="test-1",
        status=ApplicationStatus.AWAITING_APPROVAL,
        missing_information=[MissingInfo(question="Do you have a licence?")],
    )
    assert app.is_submittable is False


def test_application_not_submittable_when_already_submitted():
    app = Application(
        id="test-1",
        status=ApplicationStatus.AWAITING_APPROVAL,
        submitted=True,
    )
    assert app.is_submittable is False


def test_application_has_missing_info():
    app = Application(
        id="test-1",
        missing_information=[MissingInfo(question="What is your salary?")],
    )
    assert app.has_missing_info is True


def test_application_no_missing_info():
    app = Application(id="test-1")
    assert app.has_missing_info is False


def test_application_to_preview():
    app = Application(
        id="test-1",
        job_title="Developer",
        job_company="ACME",
        job_location="Durban",
        job_preference_score=80,
        candidate_match_score=75,
        readiness_score=90,
        application_priority=79,
    )
    preview = app.to_preview()
    assert preview["company"] == "ACME"
    assert preview["role"] == "Developer"
    assert preview["location"] == "Durban"
    assert preview["application_priority"] == 79


def test_application_serialization():
    app = Application(
        id="test-1",
        job_title="Dev",
        job_company="Co",
        status=ApplicationStatus.AWAITING_APPROVAL,
    )
    data = app.model_dump()
    restored = Application(**data)
    assert restored.id == "test-1"
    assert restored.status == ApplicationStatus.AWAITING_APPROVAL


def test_application_status_enum():
    assert ApplicationStatus.DRAFT.value == "draft"
    assert ApplicationStatus.SUBMITTED.value == "submitted"
    assert ApplicationStatus.NEEDS_INFORMATION.value == "needs_information"
    assert ApplicationStatus.AWAITING_APPROVAL.value == "awaiting_approval"


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

def test_application_priority_score_basic():
    score = application_priority_score(job_preference=80, candidate_match=70, readiness=90)
    expected = round(80 * 0.40 + 70 * 0.40 + 90 * 0.20)
    assert score == expected


def test_application_priority_score_capped():
    score = application_priority_score(100, 100, 100)
    assert score == 100


def test_application_priority_score_zero():
    score = application_priority_score(0, 0, 0)
    assert score == 0


def test_compute_all_scores():
    app = Application(
        id="test-1",
        job_preference_score=80,
        candidate_match_score=70,
        readiness_score=90,
    )
    compute_all_scores(app)
    assert app.application_priority > 0
    assert app.application_priority <= 100


def test_rank_applications():
    app1 = Application(id="a", application_priority=90)
    app2 = Application(id="b", application_priority=70)
    app3 = Application(id="c", application_priority=95)
    ranked = rank_applications([app1, app2, app3])
    assert ranked[0].id == "c"
    assert ranked[1].id == "a"
    assert ranked[2].id == "b"


# ---------------------------------------------------------------------------
# Question engine tests
# ---------------------------------------------------------------------------

def test_answer_store_basic():
    store = AnswerStore()
    assert store.has("test") is False
    store.set("test", "yes")
    assert store.has("test") is True
    assert store.get("test") == "yes"


def test_answer_store_bulk_set():
    store = AnswerStore()
    store.bulk_set({"a": "1", "b": "2"})
    assert store.get("a") == "1"
    assert store.get("b") == "2"


def test_answer_store_persistence(tmp_path):
    path = tmp_path / "answers.json"
    store1 = AnswerStore(path)
    store1.set("key", "value")
    store2 = AnswerStore(path)
    assert store2.get("key") == "value"


def test_question_engine_profile_lookup_qualification():
    engine = QuestionEngine()
    profile = CandidateProfile(
        education=[Education(qualification="BSc", field="Computer Science")]
    )
    entry = {"question": "Highest qualification?", "field_key": "highest_qualification", "profile_lookup": "qualification"}
    answer, needs_input = engine.answer_question(entry, profile)
    assert answer == "BSc Computer Science"
    assert needs_input is False


def test_question_engine_profile_lookup_experience():
    engine = QuestionEngine()
    profile = CandidateProfile(
        experience=[Experience(title="Dev", company="Co")]
    )
    entry = {"question": "Years of experience?", "field_key": "years_experience", "profile_lookup": "years_experience"}
    answer, needs_input = engine.answer_question(entry, profile)
    assert answer == "1"
    assert needs_input is False


def test_question_engine_profile_lookup_unknown():
    engine = QuestionEngine()
    profile = CandidateProfile()
    entry = {"question": "Expected salary?", "field_key": "expected_salary", "profile_lookup": None}
    answer, needs_input = engine.answer_question(entry, profile)
    assert answer is None
    assert needs_input is True


def test_question_engine_uses_answer_store():
    store = AnswerStore()
    store.set("expected_salary", "R25000")
    engine = QuestionEngine(store)
    profile = CandidateProfile()
    entry = {"question": "Expected salary?", "field_key": "expected_salary", "profile_lookup": None}
    answer, needs_input = engine.answer_question(entry, profile)
    assert answer == "R25000"
    assert needs_input is False


def test_resolve_common_questions():
    engine = QuestionEngine()
    profile = CandidateProfile(
        education=[Education(qualification="Diploma", field="IT")],
        experience=[Experience(title="Dev", company="Co")],
    )
    answered, missing = engine.resolve_common_questions(profile)
    assert "highest_qualification" in answered
    assert "years_experience" in answered
    assert any(m.field_key == "expected_salary" for m in missing)


def test_resolve_common_questions_no_profile():
    engine = QuestionEngine()
    answered, missing = engine.resolve_common_questions(None)
    assert len(answered) == 0
    assert len(missing) > 0


# ---------------------------------------------------------------------------
# Cover letter tests
# ---------------------------------------------------------------------------

def test_cover_letter_generation():
    gen = CoverLetterGenerator()
    profile = CandidateProfile(
        name="Jane Doe",
        skills=["Python", "JavaScript"],
        education=[Education(qualification="BSc", field="CS")],
    )
    job = Job(title="Software Developer", company="ACME", description="Build things.")
    letter = gen.generate(profile, job)
    assert "Jane Doe" in letter
    assert "ACME" in letter
    assert "Software Developer" in letter
    assert len(letter) > 100


def test_cover_letter_no_skills():
    gen = CoverLetterGenerator()
    profile = CandidateProfile(name="Test")
    job = Job(title="Dev", company="Co")
    letter = gen.generate(profile, job)
    assert "Test" in letter
    assert "Co" in letter


def test_cover_letter_summary_used():
    gen = CoverLetterGenerator()
    profile = CandidateProfile(
        name="X",
        professional_summary="Experienced developer with 5 years in Python.",
    )
    job = Job(title="Dev", company="Co", description="Python dev.")
    letter = gen.generate(profile, job)
    assert "5 years in Python" in letter


def test_build_candidate_summary():
    profile = CandidateProfile(
        name="John",
        email="j@test.com",
        location="Durban",
        skills=["Python", "Java"],
    )
    summary = _build_candidate_summary(profile)
    assert "John" in summary
    assert "j@test.com" in summary
    assert "Durban" in summary
    assert "Python" in summary


# ---------------------------------------------------------------------------
# Form filler tests
# ---------------------------------------------------------------------------

def test_form_field_model():
    field = FormField(name="email", label="Email", field_type=FieldType.EMAIL, required=True)
    assert field.required is True
    assert field.field_type == FieldType.EMAIL


def test_generic_adapter_can_handle_any():
    adapter = GenericApplicationAdapter()
    assert adapter.can_handle("https://example.com/jobs/1") is True


def test_generic_adapter_inspect_form():
    adapter = GenericApplicationAdapter()
    fields = adapter.inspect_form("https://example.com")
    assert fields == []


def test_generic_adapter_fill_and_submit():
    adapter = GenericApplicationAdapter()
    result = adapter.fill_and_submit("https://example.com", {"name": "Test"})
    assert result.success is False
    assert result.requires_human_input is True


def test_submission_result_model():
    result = SubmissionResult(success=True, application_url="https://example.com/applied")
    assert result.success is True


def test_platform_registry():
    registry = PlatformRegistry()
    platform = registry.get_platform("https://example.com/jobs/1")
    assert isinstance(platform, GenericApplicationAdapter)


def test_get_platform_global():
    platform = get_platform("https://any-url.com/jobs/1")
    assert isinstance(platform, PlaywrightApplicationAdapter)


# ---------------------------------------------------------------------------
# Document generation tests
# ---------------------------------------------------------------------------

def test_application_documents_prepare():
    docs = ApplicationDocuments()
    profile = CandidateProfile(
        name="Test",
        email="test@test.com",
        phone="+27 82 123 4567",
        skills=["Python"],
    )
    job = Job(title="Dev", company="Co", description="Python dev.")
    app = Application(id="test-1")
    result = docs.prepare_documents(app, profile, job)
    assert result.documents.cv_ready is True
    assert result.documents.cover_letter_ready is True
    assert len(result.documents.cover_letter_text) > 0


def test_application_documents_no_contact():
    docs = ApplicationDocuments()
    profile = CandidateProfile(name="Test", skills=["Python"])
    job = Job(title="Dev", company="Co")
    app = Application(id="test-1")
    result = docs.prepare_documents(app, profile, job)
    assert result.documents.cv_ready is False
    assert any("incomplete" in w.lower() or "contact" in w.lower() for w in result.warnings)


def test_application_documents_tailored_summary():
    docs = ApplicationDocuments()
    profile = CandidateProfile(
        professional_summary="Senior Python developer.",
    )
    job = Job(title="Dev", company="Co")
    app = Application(id="test-1")
    result = docs.prepare_documents(app, profile, job)
    assert result.documents.tailored_summary == "Senior Python developer."


def test_application_documents_fallback_summary():
    docs = ApplicationDocuments()
    profile = CandidateProfile(skills=["Python", "Java"])
    job = Job(title="Dev", company="ACME")
    app = Application(id="test-1")
    result = docs.prepare_documents(app, profile, job)
    assert "Python" in result.documents.tailored_summary
    assert "ACME" in result.documents.tailored_summary


# ---------------------------------------------------------------------------
# Tracker tests
# ---------------------------------------------------------------------------

def test_tracker_add_and_get(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    app = Application(id="a-1", job_title="Dev", job_company="Co")
    tracker.add(app)
    retrieved = tracker.get("a-1")
    assert retrieved is not None
    assert retrieved.job_title == "Dev"


def test_tracker_update(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    app = Application(id="a-1", job_title="Dev")
    tracker.add(app)
    app.update_status(ApplicationStatus.SUBMITTED)
    tracker.update(app)
    retrieved = tracker.get("a-1")
    assert retrieved.status == ApplicationStatus.SUBMITTED


def test_tracker_remove(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    app = Application(id="a-1")
    tracker.add(app)
    assert tracker.remove("a-1") is True
    assert tracker.get("a-1") is None


def test_tracker_all(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="a-1"))
    tracker.add(Application(id="a-2"))
    assert len(tracker.all()) == 2


def test_tracker_by_status(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="a-1", status=ApplicationStatus.SUBMITTED))
    tracker.add(Application(id="a-2", status=ApplicationStatus.DRAFT))
    tracker.add(Application(id="a-3", status=ApplicationStatus.SUBMITTED))
    submitted = tracker.by_status(ApplicationStatus.SUBMITTED)
    assert len(submitted) == 2


def test_tracker_find_by_job_id(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    app = Application(id="a-1", job_id="j-1", job_title="Dev")
    tracker.add(app)
    found = tracker.find_by_job_id("j-1")
    assert found is not None
    assert found.id == "a-1"


def test_tracker_is_duplicate(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="a-1", job_id="j-1", submitted=True))
    assert tracker.is_duplicate("j-1") is True
    assert tracker.is_duplicate("j-2") is False


def test_tracker_is_duplicate_draft_not_counted(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="a-1", job_id="j-1", status=ApplicationStatus.DRAFT))
    assert tracker.is_duplicate("j-1") is False


def test_tracker_get_summary(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="a-1", status=ApplicationStatus.SUBMITTED))
    tracker.add(Application(id="a-2", status=ApplicationStatus.FAILED))
    summary = tracker.get_summary()
    assert summary["total"] == 2
    assert summary["by_status"]["submitted"] == 1
    assert summary["by_status"]["failed"] == 1
    assert len(summary["needs_attention"]) == 1


def test_tracker_persistence(tmp_path):
    path = tmp_path / "apps.json"
    tracker1 = ApplicationTracker(path)
    tracker1.add(Application(id="a-1", job_title="Dev"))
    tracker2 = ApplicationTracker(path)
    assert tracker2.get("a-1") is not None
