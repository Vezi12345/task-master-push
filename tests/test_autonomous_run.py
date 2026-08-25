from __future__ import annotations

"""End-to-end autonomous run orchestration tests.

The FULL pipeline runs: search → score → decide → limits → prepare →
answer → humanise → validate → submit → record → confirmation check.
Browser and application service are test doubles — NOTHING touches a real
site, and no real submission can occur.
"""

import json
from types import SimpleNamespace as NS

import pytest

import config
from application import autonomy
from application.autonomy import (
    AutonomyPolicy,
    run_autonomous_job_search,
)
from application.models import Application, ApplicationStatus
from application.tracker import ApplicationTracker
from candidate.profile import CandidateProfile, Education
from sources.base import Job


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _profile(**kw) -> CandidateProfile:
    defaults = {
        "name": "Test Candidate", "email": "test@example.com",
        "skills": ["python", "sql", "git", "rest api"],
        "education": [Education(qualification="National Diploma",
                                field="ICT Application Development",
                                institution="DUT")],
        "experience": [], "certifications": [], "location": "Durban",
    }
    defaults.update(kw)
    return CandidateProfile(**defaults)


def _ranked(title="Junior Python Developer", company="TestCo", pref=90,
            desc="Entry-level graduate developer role. Requirements: Python, "
                 "SQL, Git, REST APIs. Junior/graduate candidates welcome. "
                 "Durban based. No experience required."):
    return NS(job=Job(title=title, company=company, location="Durban",
                      description=desc), score=pref, reasons=[])


def _plan_entry(**kw) -> dict:
    d = {"selector": "#f", "name": "f", "question": "Q", "field_type": "text",
         "category": "other", "required": False, "value": "A",
         "answer_type": "verified", "source": "memory", "needs_user": False,
         "reason": "", "conflict": {}}
    d.update(kw)
    return d


class StubService:
    """Replaces ApplicationAutomationService. Records every call."""

    def __init__(self, plan_builder=None):
        self.calls = []
        self.plan_builder = plan_builder or self._default_plan
        self.submit_result_status = ApplicationStatus.SUBMITTED

    def _default_plan(self, job):
        return [
            _plan_entry(question="Phone number", value="082 000 0000",
                        required=True),
            _plan_entry(
                question="Why this role?",
                value="As a highly motivated individual, I am excited to "
                      "leverage my skills in python.",
                answer_type="generated_from_evidence", source="generated"),
        ]

    def start_application(self, job, profile, tracker, driver=None,
                          page_html=None):
        app = Application(job_id=job.id, job_title=job.title,
                          job_company=job.company, status=ApplicationStatus.READY_FOR_REVIEW)
        app.application_url = f"https://apply.example.com/{job.company}"
        app.form_analysis = {"submit_selector": "#submit"}
        app.fill_plan = self.plan_builder(job)
        tracker.add(app)
        self.calls.append(("start", job.title))
        return app

    def confirm_and_submit(self, app, tracker, driver, consent_granted=False,
                           user_answers=None, plan=None):
        assert consent_granted is True, "run must pass explicit consent"
        assert plan is not None and isinstance(plan.entries, list), \
            "must submit the freshly validated plan"
        self.calls.append(("confirm", app.job_title))
        app.status = self.submit_result_status
        app.confirmation_text = "We received your application"
        app.submitted = True
        app.submitted_at = __import__("datetime").datetime.now().isoformat()
        tracker.update(app)
        return app


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated data dir + profile store so nothing real is touched."""
    monkeypatch.setattr(config, "APPLICATIONS_FILE", tmp_path / "apps.json")
    monkeypatch.setattr(config, "AUTONOMOUS_RUNS_DIR", tmp_path / "runs")
    profile = _profile()

    def fake_load(path=None):
        return profile

    from candidate import storage
    monkeypatch.setattr(storage, "load_profile", fake_load)
    # autonomy imports load_profile inside the function from candidate.storage
    import application.autonomy as aut
    monkeypatch.setattr(aut, "_default_tracker",
                        lambda: ApplicationTracker(path=tmp_path / "apps.json"))
    return NS(tmp=tmp_path, profile=profile, tracker_path=tmp_path / "apps.json")


# ---------------------------------------------------------------------------
# happy path: search → decide → apply → submit → confirm-check
# ---------------------------------------------------------------------------

def test_full_run_applies_to_suitable_jobs_only(env, monkeypatch):
    tr = ApplicationTracker(path=env.tracker_path)
    service = StubService()
    ranked = [_ranked("Job A", "Alpha", 92), _ranked("Job B", "Beta", 85),
              _ranked("Job C", "Gamma", 60,
                      desc="Completely unrelated senior architect role")]

    report = run_autonomous_job_search(
        query_text="python durban",
        search_fn=lambda q, p: NS(ranked=ranked),
        service_factory=lambda: service,
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(min_score=75, max_per_run=5, max_per_day=10),
    )

    assert report["error"] == ""
    assert report["jobs_discovered"] == 3
    assert report["suitable_jobs"] == 2
    outcomes = {a["title"]: a["outcome"] for a in report["applications"]}
    assert outcomes == {"Job A": "submitted", "Job B": "submitted"}
    skips = {s["title"]: s["reason"] for s in report["skipped"]}
    # unrelated/senior role is hard-gated before scoring even matters
    assert "seniority above candidate level" in skips["Job C"]
    # both applications went through prepare AND confirm
    assert ("start", "Job A") in service.calls and ("confirm", "Job A") in service.calls
    # generated draft was humanised before submission
    confirmed_app = tr.get(report["applications"][0]["application_id"])
    why = [e for e in confirmed_app.fill_plan if e["question"] == "Why this role?"][0]
    assert "leverage" not in why["value"].lower()
    assert "highly motivated" not in why["value"].lower()
    assert "(humanised)" in why["reason"]
    # confirmation emails checked for each submitted application
    assert set(report["confirmation_check"]) == {
        report["applications"][0]["application_id"],
        report["applications"][1]["application_id"]}
    assert all(v == "awaiting_confirmation" for v in report["confirmation_check"].values())
    # report persisted to disk
    saved = list((env.tmp / "runs").glob("run_*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["summary"].startswith(
        "AUTONOMOUS RUN COMPLETE")
    assert "AUTONOMOUS RUN COMPLETE" in report["summary"]


def test_confirmation_emails_flip_status_to_confirmed(env, monkeypatch):
    def fake_await(app, tracker, connector=None, matcher=None):
        from application.lifecycle import transition
        transition(app, ApplicationStatus.CONFIRMED, "matched stub email")
        tracker.update(app)
        return app

    monkeypatch.setattr("application.email_confirmation.await_confirmation",
                        fake_await)
    tr = ApplicationTracker(path=env.tracker_path)
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=[_ranked()]),
        service_factory=lambda: StubService(),
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(),
    )
    assert report["confirmation_check"]
    assert list(report["confirmation_check"].values()) == ["confirmed"]
    app_id = report["applications"][0]["application_id"]
    assert tr.get(app_id).status == ApplicationStatus.CONFIRMED


# ---------------------------------------------------------------------------
# safety: critical unknowns stop the application — never guessed
# ---------------------------------------------------------------------------

def test_critical_unknown_question_blocks_submission(env):
    def critical_plan(job):
        return [
            _plan_entry(question="Phone number", value="082", required=True),
            _plan_entry(question="Are you a South African citizen?",
                        value="", required=True),
        ]
    tr = ApplicationTracker(path=env.tracker_path)
    service = StubService(plan_builder=critical_plan)

    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=[_ranked()]),
        service_factory=lambda: service,
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(),
    )
    entry = report["applications"][0]
    assert entry["outcome"] == "REQUIRES_USER_INPUT"
    assert ("confirm",) != (entry["outcome"],)  # never submitted
    assert service.calls == [("start", entry["title"])]
    app_obj = tr.get(entry["application_id"])
    assert app_obj.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "South African citizen" in app_obj.error


def test_failed_validation_blocks_submission(env):
    def lying_plan(job):
        return [
            _plan_entry(question="Phone number", value="082", required=True),
            _plan_entry(question="Tell us about yourself",
                        value="I have 15 years of experience leading enterprise teams.",
                        answer_type="generated_from_evidence",
                        source="generated"),
        ]
    tr = ApplicationTracker(path=env.tracker_path)
    service = StubService(plan_builder=lying_plan)
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=[_ranked()]),
        service_factory=lambda: service,
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(),
    )
    assert report["applications"][0]["outcome"] == "REQUIRES_USER_INPUT"
    assert service.calls == [("start", report["applications"][0]["title"])]


# ---------------------------------------------------------------------------
# limits & duplicates across the whole run
# ---------------------------------------------------------------------------

def test_run_enforces_per_run_limit(env):
    tr = ApplicationTracker(path=env.tracker_path)
    ranked = [_ranked(f"Job {i}", f"Co{i}", 90) for i in range(7)]
    service = StubService()
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=ranked),
        service_factory=lambda: service,
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(min_score=75, max_per_run=2, max_per_day=100),
    )
    submitted = [a for a in report["applications"] if a["outcome"] == "submitted"]
    limited = [s for s in report["skipped"] if "run limit" in s["reason"]]
    assert len(submitted) == 2
    assert len(limited) == 5


def test_run_skips_jobs_already_tracked(env):
    tr = ApplicationTracker(path=env.tracker_path)
    seen_job = _ranked("Seen", "SeenCo", 90).job
    tr.add(Application(job_id=seen_job.id, job_title=seen_job.title,
                       job_company=seen_job.company,
                       status=ApplicationStatus.SUBMITTED))
    service = StubService()
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(
            ranked=[NS(job=seen_job, score=90, reasons=[]),
                    _ranked("Fresh", "FreshCo", 88)]),
        service_factory=lambda: service,
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(),
    )
    dup = [s for s in report["skipped"] if s["title"] == "Seen"]
    assert dup and "duplicate" in dup[0]["reason"]
    assert len([a for a in report["applications"] if a["outcome"] == "submitted"]) == 1


def test_dry_run_never_prepares_or_submits(env):
    tr = ApplicationTracker(path=env.tracker_path)
    service = StubService()
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=[_ranked()]),
        service_factory=lambda: service,
        driver_factory=lambda: NS(),
        tracker=tr, dry_run=True,
    )
    assert service.calls == []                    # no browser work at all
    assert report["dry_run"] is True
    assert report["applications"][0]["outcome"] == "DRY_RUN"


def test_missing_profile_aborts_cleanly(env, monkeypatch):
    from candidate import storage
    monkeypatch.setattr(storage, "load_profile", lambda path=None: None)
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=[]),
        service_factory=lambda: StubService(),
        driver_factory=lambda: NS(),
        tracker=ApplicationTracker(path=env.tracker_path),
    )
    assert "profile" in report["error"].lower()
    assert report["summary"].startswith("AUTONOMOUS RUN ABORTED")


def test_one_bad_site_does_not_kill_the_run(env):
    class ExplodingService(StubService):
        def start_application(self, job, profile, tracker, driver=None,
                              page_html=None):
            if job.company == "BadCo":
                raise RuntimeError("site timeout")
            return super().start_application(job, profile, tracker, driver)

    tr = ApplicationTracker(path=env.tracker_path)
    report = run_autonomous_job_search(
        search_fn=lambda q, p: NS(ranked=[
            _ranked("Good", "GoodCo", 92), _ranked("Bad", "BadCo", 91)]),
        service_factory=lambda: ExplodingService(),
        driver_factory=lambda: NS(),
        tracker=tr,
        policy=AutonomyPolicy(),
    )
    by_title = {a["title"]: a for a in report["applications"]}
    assert by_title["Bad"]["outcome"] == "FAILED"
    assert "site timeout" in by_title["Bad"]["detail"]
    assert by_title["Good"]["outcome"] == "submitted"


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "APPLICATIONS_FILE", tmp_path / "apps.json")
    monkeypatch.setattr(config, "AUTONOMOUS_RUNS_DIR", tmp_path / "runs")

    def fake_run(query_text="", *, dry_run=False, **kw):
        return {"started_at": "t", "query": query_text, "threshold": 75,
                "jobs_discovered": 1, "jobs_evaluated": 1, "suitable_jobs": 1,
                "limits": {"per_run": 5, "per_day": 10},
                "applications": [{"company": "C", "title": "T", "score": 90,
                                  "outcome": "DRY_RUN" if dry_run else "submitted"}],
                "skipped": [], "error": "", "dry_run": dry_run}

    monkeypatch.setattr("application.autonomy.run_autonomous_job_search",
                        fake_run)
    from app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_api_autonomous_run_dry(client):
    res = client.post("/api/autonomous/run", json={"dry_run": True})
    assert res.status_code == 200
    body = res.get_json()
    assert body["dry_run"] is True
    assert body["applications"][0]["outcome"] == "DRY_RUN"


def test_api_autonomous_report_empty_then_present(client, tmp_path):
    assert client.get("/api/autonomous/report").get_json() == {}
    (tmp_path / "runs").mkdir(parents=True)
    (tmp_path / "runs" / "run_x.json").write_text(json.dumps({"ok": 1}),
                                                  encoding="utf-8")
    assert client.get("/api/autonomous/report").get_json() == {"ok": 1}
