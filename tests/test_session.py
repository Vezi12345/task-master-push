from __future__ import annotations

"""Tests for durable conversational sessions (application/session).

A session lets a "single prompt → complete agent" flow pause (awaiting
approval, or needs information) and resume later — even after the process
restarts — by persisting minimal intent state to disk.
"""

import pytest

from agent.orchestrator import JobApplicationAgent
from agent.parse_intent import JobQuery
from application.models import Application, ApplicationStatus
from application.session import (
    MODE_AWAITING_APPROVAL,
    MODE_NEEDS_INFORMATION,
    SessionState,
    SessionStore,
)


@pytest.fixture()
def region():
    import config
    return config.load_region("za")


@pytest.fixture()
def make_agent(monkeypatch, tmp_path, region):
    from candidate import storage as storage_mod
    from candidate.profile import CandidateProfile
    import application.tracker as tracker_mod
    import config as config_mod

    monkeypatch.setattr(storage_mod, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tracker_mod, "TRACKER_FILE", tmp_path / "apps.json")
    monkeypatch.setattr(config_mod, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(config_mod, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")

    profile = CandidateProfile(
        name="Test User",
        email="test@test.com",
        skills=["Python"],
        location="Durban",
    )
    storage_mod.save_profile(profile)

    from agent import orchestrator as orch
    monkeypatch.setattr(orch, "search_jobs", lambda *a, **k: ([], ["none"]))
    monkeypatch.setattr(orch, "rank_jobs", lambda *a, **k: [])
    return JobApplicationAgent, region, tmp_path


class TestSessionStore:
    def test_create_load_roundtrip(self, tmp_path):
        store = SessionStore(directory=tmp_path / "s")
        st = store.create()
        assert st.session_id
        loaded = store.load(st.session_id)
        assert loaded is not None
        assert loaded.session_id == st.session_id

    def test_load_missing_returns_none(self, tmp_path):
        store = SessionStore(directory=tmp_path / "s")
        assert store.load("nope") is None

    def test_save_persists_mode_and_questions(self, tmp_path):
        store = SessionStore(directory=tmp_path / "s")
        st = store.create()
        st.mode = MODE_NEEDS_INFORMATION
        st.pending_questions = {"abc123": [{"field_key": "licence", "question": "Do you drive?"}]}
        store.save(st)
        loaded = store.load(st.session_id)
        assert loaded.mode == MODE_NEEDS_INFORMATION
        assert loaded.pending_questions["abc123"][0]["field_key"] == "licence"


class TestAgentSessionPersistence:
    def test_last_query_persists_across_agent_instances(self, make_agent):
        JobApplicationAgent, region, tmp_path = make_agent
        store = SessionStore(directory=tmp_path / "sessions")
        state = store.create()
        state.last_query = JobQuery(roles=["developer"], seniority="junior").model_dump()
        store.save(state)

        # a brand-new agent (new process) restores the query
        agent = JobApplicationAgent(region, None, session=store.load(state.session_id))
        assert agent.last_query is not None
        assert agent.last_query.roles == ["developer"]

    def test_pending_applications_restored_from_tracker(self, make_agent):
        JobApplicationAgent, region, tmp_path = make_agent
        from application.tracker import ApplicationTracker
        tracker = ApplicationTracker()
        app = Application(
            id="sess-app-1",
            job_id="j1",
            job_title="Dev",
            job_company="Co",
            job_url="https://example.com/job",
            status=ApplicationStatus.AWAITING_APPROVAL,
        )
        tracker.add(app)

        store = SessionStore(directory=tmp_path / "sessions")
        state = store.create()
        state.pending_application_ids = ["sess-app-1"]
        state.mode = MODE_AWAITING_APPROVAL
        store.save(state)

        agent = JobApplicationAgent(region, None, session=store.load(state.session_id))
        assert len(agent.pending_applications) == 1
        assert agent.pending_applications[0].id == "sess-app-1"

    def test_finish_turn_snapshots_mode_and_questions(self, make_agent):
        JobApplicationAgent, region, tmp_path = make_agent
        store = SessionStore(directory=tmp_path / "sessions")
        state = store.create()
        agent = JobApplicationAgent(region, None, session=state)

        # simulate a pending application needing information
        from application.tracker import ApplicationTracker
        from application.models import MissingInfo
        tracker = ApplicationTracker()
        app = Application(
            id="sess-app-2",
            job_id="j2",
            job_title="Dev",
            job_company="Co",
            status=ApplicationStatus.NEEDS_INFORMATION,
            missing_information=[
                MissingInfo(question="Do you have a licence?", field_key="licence")
            ],
        )
        tracker.add(app)
        agent.pending_applications = [app]

        from agent.orchestrator import AgentResult, AgentState
        result = agent._finish_agent_result(AgentResult(state=AgentState.AWAITING_APPROVAL))
        assert state.mode == MODE_AWAITING_APPROVAL
        store.save(state)

        loaded = store.load(state.session_id)
        assert loaded.pending_application_ids == ["sess-app-2"]
