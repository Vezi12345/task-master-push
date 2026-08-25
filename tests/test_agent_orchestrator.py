from __future__ import annotations

import pytest

from agent.parse_intent import (
    JobQuery,
    UserIntent,
    parse_user_intent,
    parse_intent,
    _APPLY_PATTERNS,
    _MATCH_SCORE_PATTERNS,
    _SHOW_APPLICATIONS_PATTERNS,
    _APPROVE_PATTERNS,
    _CANCEL_PATTERNS,
)
from agent.orchestrator import (
    AgentState,
    AgentMessage,
    AgentResult,
    JobApplicationAgent,
    PipelineResult,
)
from application.models import ApplicationStatus
from application.question_engine import QuestionEngine
from application.answer_engine import AnswerType
from candidate.profile import CandidateProfile, Education, Experience, KnownField, KnowledgeStatus
from sources.base import Job
from conftest import make_valid_job
import config


def _region():
    return config.load_region("za")


@pytest.fixture(autouse=True)
def _isolated_answer_store(tmp_path, monkeypatch):
    """QuestionEngine() must never read the real data/answers.json: live app
    usage persists genuine verified user answers there."""
    monkeypatch.setattr(config, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(config, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")


# ---------------------------------------------------------------------------
# UserIntent parsing tests
# ---------------------------------------------------------------------------

def test_search_intent():
    region = _region()
    intent = parse_user_intent("find me developer jobs in Durban", region)
    assert intent.intent_type == "search"
    assert intent.search_query is not None
    assert any("developer" in r.lower() or "software" in r.lower() for r in intent.search_query.roles)


def test_apply_with_count():
    region = _region()
    intent = parse_user_intent("apply to the best 5 jobs", region)
    assert intent.intent_type == "apply"
    assert intent.apply_count == 5


def test_apply_with_match_score():
    region = _region()
    intent = parse_user_intent("apply to jobs where I have at least 80% match", region)
    assert intent.intent_type == "apply"
    assert intent.min_match_score == 80


def test_apply_with_count_and_search():
    region = _region()
    intent = parse_user_intent("apply to the best 3 software developer jobs", region)
    assert intent.intent_type == "apply"
    assert intent.apply_count == 3
    assert intent.search_query is not None


def test_show_applications():
    region = _region()
    intent = parse_user_intent("show my applications", region)
    assert intent.intent_type == "show_applications"


def test_show_applications_status():
    region = _region()
    intent = parse_user_intent("application status", region)
    assert intent.intent_type == "show_applications"


def test_needs_attention():
    region = _region()
    intent = parse_user_intent("show applications that need my attention", region)
    assert intent.intent_type == "needs_attention"


def test_approve():
    region = _region()
    intent = parse_user_intent("approve all applications", region)
    assert intent.intent_type == "approve"


def test_approve_submit():
    region = _region()
    intent = parse_user_intent("yes, submit", region)
    assert intent.intent_type == "approve"


def test_cancel():
    region = _region()
    intent = parse_user_intent("cancel all applications", region)
    assert intent.intent_type == "cancel"


def test_cancel_stop():
    region = _region()
    intent = parse_user_intent("stop", region)
    assert intent.intent_type == "cancel"


def test_apply_regex_patterns():
    assert _APPLY_PATTERNS[0].search("apply to the best 5 jobs").group(1) == "5"
    assert _APPLY_PATTERNS[0].search("Apply to best 3 jobs").group(1) == "3"
    assert _APPLY_PATTERNS[1].search("apply to best 10 applications").group(1) == "10"


def test_match_score_regex():
    assert _MATCH_SCORE_PATTERNS[0].search("where I have at least 80%").group(1) == "80"
    assert _MATCH_SCORE_PATTERNS[1].search("85% match").group(1) == "85"


def test_show_applications_regex():
    assert _SHOW_APPLICATIONS_PATTERNS[0].search("show applications") is not None
    assert _SHOW_APPLICATIONS_PATTERNS[1].search("application history") is not None


def test_approve_regex():
    assert _APPROVE_PATTERNS[1].search("approve all applications") is not None
    assert _APPROVE_PATTERNS[4].search("yes, submit") is not None


def test_cancel_regex():
    assert _CANCEL_PATTERNS[1].search("cancel all applications") is not None
    assert _CANCEL_PATTERNS[3].search("stop") is not None


# ---------------------------------------------------------------------------
# Agent state tests
# ---------------------------------------------------------------------------

def test_agent_state_enum():
    assert AgentState.IDLE.value == "idle"
    assert AgentState.SEARCHING.value == "searching"
    assert AgentState.AWAITING_APPROVAL.value == "awaiting_approval"
    assert AgentState.SUBMITTED.value == "submitted"
    assert AgentState.COMPLETED.value == "completed"


def test_agent_message_defaults():
    msg = AgentMessage(content="hello")
    assert msg.role == "agent"
    assert msg.message_type == "text"


def test_agent_result_defaults():
    result = AgentResult()
    assert result.state == AgentState.IDLE
    assert result.messages == []
    assert result.error is None


# ---------------------------------------------------------------------------
# JobApplicationAgent tests
# ---------------------------------------------------------------------------

def test_agent_search(monkeypatch):
    from sources.dpsa_circular import DpsaCircularSource
    monkeypatch.setattr(
        DpsaCircularSource, "search",
        lambda self, query: [make_valid_job()],
    )

    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("find me developer jobs")
    assert result.state == AgentState.COMPLETED
    assert result.query is not None
    assert len(result.ranked) > 0
    assert result.error is None


def test_agent_search_with_cv(monkeypatch, tmp_path):
    from sources.dpsa_circular import DpsaCircularSource
    from candidate import storage

    monkeypatch.setattr(
        DpsaCircularSource, "search",
        lambda self, query: [make_valid_job()],
    )
    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    profile = CandidateProfile(
        name="Test User",
        email="test@test.com",
        skills=["Python", "JavaScript"],
        education=[Education(qualification="BSc", field="CS")],
    )
    storage.save_profile(profile)

    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("find me developer jobs")
    assert result.state == AgentState.COMPLETED
    assert len(result.matched_jobs) > 0
    assert result.profile is not None


def test_agent_apply(monkeypatch, tmp_path):
    from sources.dpsa_circular import DpsaCircularSource
    from candidate import storage
    from application import tracker as tracker_module

    monkeypatch.setattr(
        DpsaCircularSource, "search",
        lambda self, query: [make_valid_job()],
    )
    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tracker_module, "TRACKER_FILE", tmp_path / "applications.json")
    monkeypatch.setattr(config, "ANSWERS_FILE", tmp_path / "answers.json")

    profile = CandidateProfile(
        name="Test User",
        email="test@test.com",
        skills=["Python"],
    )
    storage.save_profile(profile)

    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("apply to the best 2 software developer jobs")
    assert result.state == AgentState.AWAITING_APPROVAL
    assert len(result.applications) > 0
    assert result.applications[0].status in (
        ApplicationStatus.AWAITING_APPROVAL,
        ApplicationStatus.NEEDS_INFORMATION,
    )


def test_agent_apply_no_cv(monkeypatch, tmp_path):
    from sources.dpsa_circular import DpsaCircularSource
    from candidate import storage

    monkeypatch.setattr(DpsaCircularSource, "search", lambda self, query: [])
    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("apply to the best 5 jobs")
    assert result.error is not None
    assert "CV" in result.error or "cv" in result.error.lower()


def test_agent_show_applications(tmp_path):
    from application.tracker import ApplicationTracker
    from application.models import Application, ApplicationStatus

    tracker_path = tmp_path / "apps.json"
    tracker = ApplicationTracker(tracker_path)
    tracker.add(Application(id="test-1", job_title="Dev", job_company="Co"))

    region = _region()
    agent = JobApplicationAgent(region)
    agent.tracker = tracker
    result = agent.process_input("show my applications")
    assert result.state == AgentState.COMPLETED
    assert len(result.application_summaries) == 1


def test_agent_needs_attention(tmp_path):
    from application.tracker import ApplicationTracker
    from application.models import Application, ApplicationStatus

    tracker_path = tmp_path / "apps.json"
    tracker = ApplicationTracker(tracker_path)
    tracker.add(Application(
        id="test-1", job_title="Dev", job_company="Co",
        status=ApplicationStatus.NEEDS_INFORMATION,
    ))

    region = _region()
    agent = JobApplicationAgent(region)
    agent.tracker = tracker
    result = agent.process_input("show applications that need my attention")
    assert result.state == AgentState.COMPLETED
    assert len(result.application_summaries) == 1


def test_agent_approve_empty():
    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("approve all applications")
    assert result.state == AgentState.COMPLETED


def test_agent_cancel_empty():
    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("cancel all")
    assert result.state == AgentState.COMPLETED


def test_agent_provide_answers():
    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.provide_answers({"expected_salary": "R25000"})
    assert result.state is not None


def test_agent_unknown_intent(monkeypatch):
    from sources.dpsa_circular import DpsaCircularSource

    searched = []
    monkeypatch.setattr(
        DpsaCircularSource, "search",
        lambda self, query: searched.append(query) or [],
    )

    region = _region()
    agent = JobApplicationAgent(region)
    result = agent.process_input("hello there")
    # greetings must NOT trigger a source crawl - agent asks for clarity instead
    assert result.error is not None
    assert result.ranked == []
    assert searched == []


# ---------------------------------------------------------------------------
# Candidate profile knowledge tracking tests
# ---------------------------------------------------------------------------

def test_candidate_profile_known_fields():
    profile = CandidateProfile(
        name="Test",
        email="test@test.com",
        skills=["Python"],
    )
    profile.populate_known_fields()
    assert profile.is_known("name") is True
    assert profile.is_known("email") is True
    assert profile.is_known("skills") is True
    assert profile.is_known("expected_salary") is False


def test_candidate_profile_set_known():
    profile = CandidateProfile()
    profile.set_known("expected_salary", "R25000", "user")
    assert profile.get_known_value("expected_salary") == "R25000"


def test_candidate_profile_set_unknown():
    profile = CandidateProfile()
    profile.set_unknown("drivers_licence")
    assert profile.is_known("drivers_licence") is False


def test_candidate_profile_summary():
    profile = CandidateProfile(
        name="John",
        email="j@test.com",
        skills=["Python", "Java"],
        education=[Education(qualification="BSc", field="CS")],
    )
    summary = profile.summary
    assert "John" in summary
    assert "Python" in summary
    assert "BSc" in summary


def test_candidate_profile_knowledge_from_education():
    profile = CandidateProfile(
        education=[Education(qualification="Diploma", field="IT")],
    )
    profile.populate_known_fields()
    assert profile.get_known_value("highest_qualification") == "Diploma IT"


def test_candidate_profile_knowledge_from_experience():
    """Experience length must never be fabricated by counting CV entries —
    it is derived from actual employment dates (see answer engine)."""
    profile = CandidateProfile(
        experience=[
            Experience(title="Dev1", company="Co1"),
            Experience(title="Dev2", company="Co2"),
        ],
    )
    profile.populate_known_fields()
    assert profile.get_known_value("years_experience") is None

    dated = CandidateProfile(
        experience=[
            Experience(
                title="Dev1", company="Co1",
                start_date="2025-01-01", end_date="2025-07-01",
                experience_type="employment",
            ),
        ],
    )
    engine = QuestionEngine()
    result = engine.answer("How many years of experience do you have?", dated)
    assert result.is_answered
    assert result.answer_type == AnswerType.DERIVED
    assert "6" in result.answer  # months calculated from the real date range


def test_candidate_profile_always_unknown_fields():
    profile = CandidateProfile()
    profile.populate_known_fields()
    assert profile.is_known("expected_salary") is False
    assert profile.is_known("relocation") is False
    assert profile.is_known("drivers_licence") is False
    assert profile.is_known("notice_period") is False
    assert profile.is_known("work_authorisation") is False
    assert profile.is_known("availability") is False
    assert profile.is_known("citizenship") is False


# ---------------------------------------------------------------------------
# Conversational turn isolation
# ---------------------------------------------------------------------------

def test_each_turn_gets_fresh_message_buffer(monkeypatch):
    from sources.dpsa_circular import DpsaCircularSource

    region = _region()
    agent = JobApplicationAgent(region)

    # turn 1: an unknown-intent turn that leaves messages behind
    first = agent.process_input("hello there")
    assert first.messages, "turn must contain its own messages"

    # turn 2: the buffer must NOT replay turn 1
    second = agent.process_input("show my applications")
    contents = " ".join(m.content for m in second.messages)
    assert "hello there" not in contents
    assert len(second.messages) < len(first.messages) + 3


def test_turn_isolation_across_searches(monkeypatch):
    from sources.dpsa_circular import DpsaCircularSource

    calls = {"n": 0}

    def fake_search(self, query):
        calls["n"] += 1
        return []

    monkeypatch.setattr(DpsaCircularSource, "search", fake_search)
    monkeypatch.setattr("agent.orchestrator.search_jobs", lambda q, r: ([], ["msg-a"]))

    region = _region()
    agent = JobApplicationAgent(region)
    agent.process_input("find barista jobs in Durban")
    assert calls["n"] == 0  # dpsa untouched: search_jobs is patched at orchestrator

    result = agent.process_input("show my applications")
    blob = " ".join(m.content for m in result.messages)
    assert "msg-a" not in blob, "previous search chatter leaked into next turn"
