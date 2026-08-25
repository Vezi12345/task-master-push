from __future__ import annotations

import pytest

from sources.base import (
    ApplicationPlatformType,
    Job,
    detect_platform_type,
)
from candidate.matching import (
    CandidateMatch,
    DetailedMatch,
    MatchDimension,
    match_candidate_to_job,
    match_candidate_to_job_detailed,
)
from candidate.profile import CandidateProfile, Education, Experience
from application.form_filler import (
    GenericApplicationAdapter,
    PlaywrightApplicationAdapter,
    PlatformRegistry,
    SubmissionResult,
    get_platform,
    get_playwright_adapter,
)
from application.models import Application, ApplicationStatus
from application.question_engine import (
    QuestionEngine,
    AnswerStore,
    _ELIGIBILITY_KEYWORDS,
)
from application.tracker import ApplicationTracker
from agent.parse_intent import UserIntent, parse_user_intent
import config


@pytest.fixture(autouse=True)
def _isolated_answer_store(tmp_path, monkeypatch):
    """Never read/write the real data/answers.json from tests: live usage of
    the app persists genuine user answers there, which would otherwise flip
    'missing' expectations."""
    monkeypatch.setattr(config, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(config, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")


def _region():
    return config.load_region("za")


# ---------------------------------------------------------------------------
# Platform detection tests
# ---------------------------------------------------------------------------

def test_detect_platform_workday():
    assert detect_platform_type("https://company.wd5.myworkdayjobs.com/jobs") == ApplicationPlatformType.WORKDAY


def test_detect_platform_greenhouse():
    assert detect_platform_type("https://boards.greenhouse.io/company/jobs/123") == ApplicationPlatformType.GREENHOUSE


def test_detect_platform_lever():
    assert detect_platform_type("https://jobs.lever.co/company/abc123") == ApplicationPlatformType.LEVER


def test_detect_platform_smartrecruiters():
    assert detect_platform_type("https://careers.smartrecruiters.com/company/job/123") == ApplicationPlatformType.SMARTRECRUITERS


def test_detect_platform_email():
    assert detect_platform_type("mailto:hr@company.com") == ApplicationPlatformType.EMAIL


def test_detect_platform_generic():
    assert detect_platform_type("https://company.com/careers/123") == ApplicationPlatformType.GENERIC_WEB


def test_detect_platform_empty():
    assert detect_platform_type("") == ApplicationPlatformType.UNKNOWN


def test_job_auto_detects_platform():
    job = Job(title="Dev", company="Co", url="https://company.wd5.myworkdayjobs.com/jobs")
    assert job.platform == ApplicationPlatformType.WORKDAY


def test_job_greenhouse_platform():
    job = Job(title="Dev", company="Co", url="https://boards.greenhouse.io/company/jobs/123")
    assert job.platform == ApplicationPlatformType.GREENHOUSE


def test_job_lever_platform():
    job = Job(title="Dev", company="Co", url="https://jobs.lever.co/company/abc")
    assert job.platform == ApplicationPlatformType.LEVER


def test_job_smartrecruiters_platform():
    job = Job(title="Dev", company="Co", url="https://careers.smartrecruiters.com/company/job")
    assert job.platform == ApplicationPlatformType.SMARTRECRUITERS


def test_application_platform_type_enum():
    assert ApplicationPlatformType.GENERIC_WEB.value == "generic_web"
    assert ApplicationPlatformType.WORKDAY.value == "workday"
    assert ApplicationPlatformType.GREENHOUSE.value == "greenhouse"
    assert ApplicationPlatformType.LEVER.value == "lever"
    assert ApplicationPlatformType.SMARTRECRUITERS.value == "smartrecruiters"
    assert ApplicationPlatformType.EMAIL.value == "email"
    assert ApplicationPlatformType.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Multi-dimensional matching tests
# ---------------------------------------------------------------------------

def _profile(**kwargs) -> CandidateProfile:
    defaults = {"skills": [], "education": [], "experience": [], "certifications": [], "projects": [], "location": ""}
    defaults.update(kwargs)
    return CandidateProfile(**defaults)


def _job(**kwargs) -> Job:
    defaults = {"title": "Software Developer", "company": "Test Co", "location": "", "remote": False, "description": ""}
    defaults.update(kwargs)
    return Job(**defaults)


def test_detailed_match_returns_dimensions():
    profile = _profile(skills=["Python", "Flask"])
    job = _job(description="Python and Flask developer.")
    detailed = match_candidate_to_job_detailed(profile, job)
    assert isinstance(detailed, DetailedMatch)
    assert len(detailed.dimensions) > 0
    assert detailed.overall_score >= 0
    assert detailed.overall_score <= 100


def test_detailed_match_skills_dimension():
    profile = _profile(skills=["Python", "Flask", "SQL"])
    job = _job(description="python flask sql required")
    detailed = match_candidate_to_job_detailed(profile, job)
    skills_dim = next(d for d in detailed.dimensions if d.name == "skills")
    assert skills_dim.score >= 75
    assert "3/" in skills_dim.reason


def test_detailed_match_experience_dimension():
    exp = Experience(title="Senior Developer", description="senior developer with 5 years")
    profile = _profile(experience=[exp])
    job = _job(description="Senior developer, 5+ years experience required.")
    detailed = match_candidate_to_job_detailed(profile, job)
    exp_dim = next(d for d in detailed.dimensions if d.name == "experience")
    assert exp_dim.score > 50


def test_detailed_match_salary_dimension():
    profile = _profile(skills=["Python"])
    profile.set_known("expected_salary", "R20000")
    job = _job(description="Python developer.", salary_min=25000)
    detailed = match_candidate_to_job_detailed(profile, job)
    salary_dim = next(d for d in detailed.dimensions if d.name == "salary")
    assert salary_dim.score > 0


def test_detailed_match_strengths_concerns():
    profile = _profile(skills=["Python"])
    job = _job(description="Python developer.")
    detailed = match_candidate_to_job_detailed(profile, job)
    assert len(detailed.strengths) > 0
    assert isinstance(detailed.concerns, list)


def test_detailed_match_graduate_eligible():
    profile = _profile(education=[Education(qualification="BSc", field="CS")])
    job = _job(description="Recent graduate developer position.")
    detailed = match_candidate_to_job_detailed(profile, job)
    assert detailed.graduate_eligible is True
    assert any("graduate" in s.lower() for s in detailed.strengths)


def test_detailed_match_empty_profile():
    profile = _profile()
    job = _job(description="Python developer.")
    detailed = match_candidate_to_job_detailed(profile, job)
    assert isinstance(detailed, DetailedMatch)
    assert detailed.overall_score >= 0


def test_match_dimension_model():
    dim = MatchDimension(name="skills", score=85, weight=0.35, reason="Good match")
    data = dim.model_dump()
    assert data["name"] == "skills"
    assert data["score"] == 85


def test_detailed_match_serialization():
    profile = _profile(skills=["Python"])
    job = _job(description="Python developer.")
    detailed = match_candidate_to_job_detailed(profile, job)
    data = detailed.model_dump()
    restored = DetailedMatch(**data)
    assert restored.overall_score == detailed.overall_score


# ---------------------------------------------------------------------------
# Playwright adapter tests
# ---------------------------------------------------------------------------

def test_playwright_adapter_can_handle():
    adapter = PlaywrightApplicationAdapter()
    assert adapter.can_handle("https://any-url.com/jobs/1") is True


def test_playwright_adapter_inspect_form():
    adapter = PlaywrightApplicationAdapter()
    fields = adapter.inspect_form("https://example.com")
    assert fields == []


def test_playwright_adapter_fill_returns_error():
    adapter = PlaywrightApplicationAdapter()
    result = adapter.fill_and_submit("https://example.com", {"name": "Test"})
    assert result.success is False
    assert result.requires_human_input is True


def test_get_playwright_adapter():
    adapter = get_playwright_adapter("https://example.com/jobs/1")
    assert isinstance(adapter, PlaywrightApplicationAdapter)


def test_platform_registry_get():
    registry = PlatformRegistry()
    platform = registry.get_platform("https://example.com/jobs/1")
    assert isinstance(platform, GenericApplicationAdapter)


# ---------------------------------------------------------------------------
# Question engine taxonomy tests
# ---------------------------------------------------------------------------

def test_eligibility_keywords_coverage():
    assert "work_authorisation" in _ELIGIBILITY_KEYWORDS
    assert "disability" in _ELIGIBILITY_KEYWORDS
    assert "race" in _ELIGIBILITY_KEYWORDS
    assert "gender" in _ELIGIBILITY_KEYWORDS
    assert "recent_graduate" in _ELIGIBILITY_KEYWORDS
    assert "south_african_citizen" in _ELIGIBILITY_KEYWORDS
    assert "drivers_licence" in _ELIGIBILITY_KEYWORDS
    assert "relocation" in _ELIGIBILITY_KEYWORDS


def test_detect_job_requirements_work_authorisation():
    engine = QuestionEngine()
    profile = CandidateProfile(email="test@test.com")
    desc = "Must be authorised to work in South Africa."
    answered, missing = engine.detect_job_requirements(desc, profile)
    assert any(m.field_key == "work_authorisation" for m in missing)


def test_detect_job_requirements_disability():
    engine = QuestionEngine()
    profile = CandidateProfile(email="test@test.com")
    desc = "Employment equity and disability confirmation required."
    answered, missing = engine.detect_job_requirements(desc, profile)
    assert any(m.field_key == "disability" for m in missing)


def test_detect_job_requirements_with_answer_store(tmp_path):
    store = AnswerStore(tmp_path / "answers.json")
    store.set("work_authorisation", "Yes")
    engine = QuestionEngine(store)
    profile = CandidateProfile()
    desc = "Must be authorised to work in South Africa."
    answered, missing = engine.detect_job_requirements(desc, profile)
    assert answered.get("work_authorisation") == "Yes"


def test_detect_job_requirements_no_match():
    engine = QuestionEngine()
    profile = CandidateProfile()
    desc = "Build great software for our team."
    answered, missing = engine.detect_job_requirements(desc, profile)
    assert len(missing) == 0


def test_question_engine_full_taxonomy(tmp_path):
    engine = QuestionEngine(AnswerStore(tmp_path / "answers.json"))
    profile = CandidateProfile(
        education=[Education(qualification="BSc", field="CS")],
        experience=[
            Experience(
                title="Dev", company="Co",
                start_date="2025-01-01", end_date="2025-07-01",
            ),
        ],
    )
    answered, missing = engine.resolve_common_questions(profile)
    assert "highest_qualification" in answered
    assert "years_experience" in answered  # derived from real employment dates
    assert any(m.field_key == "expected_salary" for m in missing)


# ---------------------------------------------------------------------------
# Application model new fields tests
# ---------------------------------------------------------------------------

def test_application_status_discovered():
    assert ApplicationStatus.DISCOVERED.value == "discovered"


def test_application_status_selected():
    assert ApplicationStatus.SELECTED.value == "selected"


def test_application_status_ready_for_review():
    assert ApplicationStatus.READY_FOR_REVIEW.value == "ready_for_review"


def test_application_status_manual_action_required():
    assert ApplicationStatus.MANUAL_ACTION_REQUIRED.value == "manual_action_required"


def test_application_platform_field():
    app = Application(id="test-1", job_platform="workday")
    preview = app.to_preview()
    assert preview["platform"] == "workday"


def test_application_submission_tracking():
    app = Application(
        id="test-1",
        submission_platform="playwright",
        confirmation_id="CONF-123",
        date_submitted="2025-01-01",
    )
    preview = app.to_preview()
    assert preview["submission_platform"] == "playwright"
    assert preview["confirmation_id"] == "CONF-123"
    assert preview["date_submitted"] == "2025-01-01"


# ---------------------------------------------------------------------------
# Tracker new methods tests
# ---------------------------------------------------------------------------

def test_tracker_find_by_partial_id(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    app = Application(id="abc123def456", job_title="Dev")
    tracker.add(app)
    found = tracker.find_by_partial_id("abc123")
    assert found is not None
    assert found.id == "abc123def456"


def test_tracker_find_by_partial_id_not_found(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="abc123", job_title="Dev"))
    found = tracker.find_by_partial_id("xyz")
    assert found is None


def test_tracker_get_submittable(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(
        id="a-1",
        status=ApplicationStatus.AWAITING_APPROVAL,
        submitted=False,
    ))
    tracker.add(Application(
        id="a-2",
        status=ApplicationStatus.DRAFT,
    ))
    submittable = tracker.get_submittable()
    assert len(submittable) == 1
    assert submittable[0].id == "a-1"


def test_tracker_get_by_status_group(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(id="a-1", status=ApplicationStatus.SUBMITTED))
    tracker.add(Application(id="a-2", status=ApplicationStatus.DRAFT))
    tracker.add(Application(id="a-3", status=ApplicationStatus.SUBMITTED))
    groups = tracker.get_by_status_group()
    assert len(groups["submitted"]) == 2
    assert len(groups["draft"]) == 1


def test_tracker_summary_includes_submittable(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(
        id="a-1",
        status=ApplicationStatus.AWAITING_APPROVAL,
        submitted=False,
    ))
    summary = tracker.get_summary()
    assert summary["submittable_count"] == 1


def test_tracker_needs_attention_includes_manual_action(tmp_path):
    tracker = ApplicationTracker(tmp_path / "apps.json")
    tracker.add(Application(
        id="a-1",
        status=ApplicationStatus.MANUAL_ACTION_REQUIRED,
    ))
    needs = tracker.needs_attention()
    assert len(needs) == 1


# ---------------------------------------------------------------------------
# Parse intent target_id tests
# ---------------------------------------------------------------------------

def test_approve_application_by_id():
    region = _region()
    intent = parse_user_intent("approve application abc123", region)
    assert intent.intent_type == "approve"
    assert intent.target_id == "abc123"


def test_approve_all_no_target():
    region = _region()
    intent = parse_user_intent("approve all applications", region)
    assert intent.intent_type == "approve"
    assert intent.target_id is None


def test_cancel_application_by_id():
    region = _region()
    intent = parse_user_intent("cancel application xyz789", region)
    assert intent.intent_type == "cancel"
    assert intent.target_id == "xyz789"


def test_cancel_all_no_target():
    region = _region()
    intent = parse_user_intent("cancel all", region)
    assert intent.intent_type == "cancel"
    assert intent.target_id is None


def test_submit_application_by_id():
    region = _region()
    intent = parse_user_intent("submit application abc123", region)
    assert intent.intent_type == "approve"
    assert intent.target_id == "abc123"


def test_approve_yes_submit():
    region = _region()
    intent = parse_user_intent("yes, submit", region)
    assert intent.intent_type == "approve"
    assert intent.target_id is None


def test_cancel_stop():
    region = _region()
    intent = parse_user_intent("stop", region)
    assert intent.intent_type == "cancel"
    assert intent.target_id is None
