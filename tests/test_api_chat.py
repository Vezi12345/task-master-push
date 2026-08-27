from __future__ import annotations

"""Tests for the conversational /api/chat endpoints.

The orchestrator's search path is faked at the module boundary
(agent.orchestrator.search_jobs / rank_jobs) exactly like the other
API-level suites, so no network or LLM is touched.
"""

import pytest

from agent import orchestrator as orch_module
from agent.orchestrator import AgentState, PipelineResult, RankedJob
from agent.parse_intent import JobQuery
from application.models import Application, ApplicationStatus
from conftest import make_valid_job
from sources.base import Job


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from candidate import storage as storage_mod
    from candidate.profile import CandidateProfile
    import application.tracker as tracker_mod
    import application.session as session_mod
    import config as config_mod

    monkeypatch.setattr(storage_mod, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tracker_mod, "TRACKER_FILE", tmp_path / "applications.json")
    # chat agent writes user answers - keep them out of the real store
    monkeypatch.setattr(config_mod, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(config_mod, "ANSWER_CONFLICTS_FILE", tmp_path / "conflicts.json")
    # keep sessions out of the real store and give each test a clean slate
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path / "sessions")

    profile = CandidateProfile(
        name="Test User",
        email="test@test.com",
        skills=["Python"],
        location="Durban",
    )
    storage_mod.save_profile(profile)

    from app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _fake_search(monkeypatch, jobs):
    query = JobQuery(roles=["developer"], seniority="junior")

    def fake_search(query_in, region, stats=None):
        return jobs, ["Found %d jobs." % len(jobs)]

    def fake_rank(jobs_in, query_in, llm=None):
        return [RankedJob(job=j, score=80, reasons=["good"], summary="fit") for j in jobs_in]

    monkeypatch.setattr(orch_module, "search_jobs", fake_search)
    monkeypatch.setattr(orch_module, "rank_jobs", fake_rank)
    return query


def test_chat_greeting_returns_help(client):
    resp = client.post("/api/chat", json={"message": "hello there"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["state"] in ("idle", "received", "completed")
    assert "job-search agent" in data["reply"]
    assert "Search" in data["reply"]


def test_chat_empty_message_rejected(client):
    resp = client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_chat_search_returns_ranked_payload(client, monkeypatch):
    job = make_valid_job()
    _fake_search(monkeypatch, [job])
    resp = client.post("/api/chat", json={"message": "find junior developer jobs"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["jobs_found"] == 1
    assert len(data["ranked"]) == 1
    entry = data["ranked"][0]
    for key in ("title", "company", "score", "candidate_match", "readiness"):
        assert key in entry
    assert "reply" in data and data["reply"]


def test_chat_search_zero_matches_offers_related(client, monkeypatch):
    """Listings exist but none pass role gates -> related buckets included."""
    accountant = make_valid_job(title="Accountant", description="Finance degree and bookkeeping required.")
    _fake_search(monkeypatch, [accountant])

    real_rank = orch_module.rank_jobs
    monkeypatch.setattr(orch_module, "rank_jobs", lambda j, q, llm=None: [])

    resp = client.post("/api/chat", json={"message": "find software developer jobs"})
    data = resp.get_json()
    assert data["jobs_found"] == 1
    assert data["ranked"] == []
    assert data["related"], "expected related buckets when ranked is empty"
    labels = {r["label"] for r in data["related"]}
    assert any("ccountant" in lbl or "dministrator" in lbl for lbl in labels)


def test_chat_show_applications_empty(client):
    resp = client.post("/api/chat", json={"message": "show my applications"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["application_summaries"] == []


def test_chat_approve_with_nothing_pending_is_friendly(client):
    resp = client.post("/api/chat", json={"message": "approve all applications"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "No applications pending approval" in data["reply"]


def test_chat_cancel_pending_clears_them(client, monkeypatch, tmp_path):
    from application.tracker import ApplicationTracker

    tracker = ApplicationTracker()
    app_obj = Application(
        id="chatcancel01",
        job_id="j1",
        job_title="Dev",
        job_company="Co",
        candidate_name="Test User",
        candidate_email="test@test.com",
    )
    app_obj.update_status(ApplicationStatus.AWAITING_APPROVAL)
    tracker.add(app_obj)

    resp = client.post("/api/chat", json={"message": "cancel all applications"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "cancelled" in data["reply"].lower()


def test_chat_answers_endpoint_requires_dict(client):
    resp = client.post("/api/chat/answers", json={"answers": "nope"})
    assert resp.status_code == 400
    resp = client.post("/api/chat/answers", json={})
    assert resp.status_code == 400


def test_chat_answers_endpoint_updates_store(client, monkeypatch, tmp_path):
    # trigger agent creation through a cheap turn first
    first = client.post("/api/chat", json={"message": "show my applications"})
    sid = first.get_json()["session_id"]
    assert sid

    resp = client.post(
        "/api/chat/answers",
        json={"session_id": sid, "answers": {"right_to_work_sa": "Yes"}},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["error"] is None
    # the answered turn persist the same, stable session
    assert data["session_id"] == sid


def test_chat_apply_without_cv_is_helpful(client, monkeypatch, tmp_path):
    """apply intent with no profile -> friendly guidance, no network."""
    import candidate.storage as cs
    import agent.orchestrator as orch

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(cs, "PROFILE_FILE", empty / "profile.json")
    monkeypatch.setattr(cs, "DATA_DIR", empty)
    # orchestrator binds load_profile directly at import time
    monkeypatch.setattr(orch, "load_profile", lambda: None)

    resp = client.post("/api/chat", json={"message": "apply to the best 3 matching jobs"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "CV" in data["reply"]
