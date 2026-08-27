from __future__ import annotations

"""Adaptive end-to-end pipeline tests.

For several professions the FULL autonomous pipeline runs with an injected
search function returning a mixed job pool (relevant + unrelated + hard-
gated).  Proves: candidate profile → dynamic decisions, pollution filtered,
hard gates enforced, explanations produced — with ZERO code changes between
professions.  Nothing touches a real site; dry-run only.
"""

from types import SimpleNamespace as NS

import pytest

import config
from application import autonomy
from application.autonomy import (
    AutonomyPolicy,
    run_autonomous_job_search,
)
from application.tracker import ApplicationTracker
from candidate.profile import CandidateProfile, Education, Experience
from sources.base import Job


# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------

def _profile(**kw) -> CandidateProfile:
    defaults = {"skills": [], "education": [], "experience": [],
                "certifications": [], "location": "Durban"}
    defaults.update(kw)
    return CandidateProfile(name="T", email="t@x.com", **defaults)


NURSE = _profile(
    skills=["patient care", "wound care", "dispensing",
            "sanc registered nurse"],
    education=[Education(qualification="Diploma in Nursing",
                         field="General Nursing", institution="NETCARE")],
    experience=[Experience(title="Staff Nurse", company="Hospital",
                           start_date="2022-01", end_date="present")],
    drivers_licence="Code 8")

ACCOUNTANT = _profile(
    skills=["pastel evolution", "vat reconciliations"],
    education=[Education(qualification="Diploma in Accounting",
                         field="Financial Management", institution="DUT")],
    experience=[Experience(title="Bookkeeper", company="Firm",
                           start_date="2024-01", end_date="present")],
    drivers_licence="Code 8")

ADMIN = _profile(
    skills=["ms office", "excel", "filing", "diary management"],
    education=[Education(qualification="Diploma in Business Management",
                         field="Business Management", institution="C")],
    experience=[Experience(title="Office Administrator", company="Co",
                           start_date="2023-01", end_date="present")],
    drivers_licence="Code 8")

DRIVER = _profile(
    skills=["deliveries", "vehicle checks"],
    education=[],   # no formal qualification on purpose
    experience=[Experience(title="Delivery Driver", company="Logistics",
                           start_date="2021-06", end_date="present")],
    drivers_licence="Code 10, PDP")

WEAK_CV = _profile(skills=[], education=[], experience=[])


def _job(title, desc, company="Co", location="Durban"):
    return Job(title=title, description=desc, company=company,
               location=location)


def _pool() -> list:
    """Mixed pool every candidate sees — decisions must differ by profile."""
    return [
        NS(job=_job("Staff Nurse",
                    "Requirements: SANC registered. Patient care in ward. "
                    "Durban hospital."), score=90),
        NS(job=_job("Junior Bookkeeper",
                    "Requirements: diploma in accounting, pastel, VAT "
                    "reconciliations. Durban."), score=88),
        NS(job=_job("Office Administrator",
                    "Requirements: ms office, excel, filing, diaries. "
                    "Durban."), score=87),
        NS(job=_job("Delivery Driver",
                    "Requirements: code 10 licence essential, PDP. "
                    "Deliveries around Durban."), score=86),
        NS(job=_job("Senior Financial Controller",
                    "Requirements: minimum of 10 years experience, CA(SA). "
                    "Senior accounting leadership role."), score=95),
    ]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "APPLICATIONS_FILE", tmp_path / "apps.json")
    monkeypatch.setattr(config, "AUTONOMOUS_RUNS_DIR", tmp_path / "runs")
    from candidate import storage
    import application.autonomy as aut

    holder = {}

    def fake_load(path=None):
        return holder.get("profile")

    monkeypatch.setattr(storage, "load_profile", fake_load)
    monkeypatch.setattr(aut, "_default_tracker",
                        lambda: ApplicationTracker(path=tmp_path / "apps.json"))
    return NS(tmp=tmp_path, holder=holder)


def _run(env, profile, ranked):
    env.holder["profile"] = profile
    return run_autonomous_job_search(
        query_text=None,
        search_fn=lambda q, p: NS(ranked=ranked),
        service_factory=lambda: NS(),
        driver_factory=lambda: NS(),
        tracker=ApplicationTracker(path=env.tmp / "apps.json"),
        policy=AutonomyPolicy(min_score=75, max_per_run=5, max_per_day=10),
        dry_run=True,
    )


def test_autonomous_run_ages_stale_pending(tmp_path, env):
    """Passing an OutcomeStore makes an autonomous run learn from stale
    submissions (silence-after-submit is recorded as no_response)."""
    from datetime import datetime, timedelta

    from application.models import Application, ApplicationStatus
    from application.outcome_learning import OutcomeStore

    env.holder["profile"] = _profile()
    tracker = ApplicationTracker(path=tmp_path / "apps.json")
    tracker.add(Application(
        id="stale1",
        job_title="Junior Bookkeeper",
        job_company="Acme (Pty) Ltd",
        status=ApplicationStatus.SUBMITTED,
        submission_mode="real",
        date_submitted=(datetime.now() - timedelta(days=40)).isoformat(),
    ))

    store = OutcomeStore(tmp_path / "outcomes.json")
    run_autonomous_job_search(
        query_text=None,
        search_fn=lambda q, p: NS(ranked=[]),
        service_factory=lambda: NS(),
        driver_factory=lambda: NS(),
        tracker=tracker,
        policy=AutonomyPolicy(min_score=75, max_per_run=1, max_per_day=1),
        dry_run=True,
        outcome_store=store,
    )
    assert store.get("company:acme_(pty)_ltd").no_response == 1


def test_autonomous_run_without_store_skips_aging(tmp_path, env):
    """No store → the run does not touch or learn from pending apps."""
    from datetime import datetime, timedelta

    from application.models import Application, ApplicationStatus
    from application.outcome_learning import OutcomeStore

    env.holder["profile"] = _profile()
    tracker = ApplicationTracker(path=tmp_path / "apps.json")
    tracker.add(Application(
        id="fresh1",
        job_title="Junior Bookkeeper",
        job_company="Acme (Pty) Ltd",
        status=ApplicationStatus.SUBMITTED,
        submission_mode="real",
        date_submitted=(datetime.now() - timedelta(days=40)).isoformat(),
    ))

    store = OutcomeStore(tmp_path / "outcomes2.json")
    run_autonomous_job_search(
        query_text=None,
        search_fn=lambda q, p: NS(ranked=[]),
        service_factory=lambda: NS(),
        driver_factory=lambda: NS(),
        tracker=tracker,
        policy=AutonomyPolicy(min_score=75, max_per_run=1, max_per_day=1),
        dry_run=True,
    )
    assert store.get("company:acme_(pty)_ltd").no_response == 0


# ---------------------------------------------------------------------------
# per-profession outcomes over the SAME pool — no code changes anywhere
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile,expected_apply,forbidden_titles", [
    (NURSE, ["Staff Nurse"], ["Junior Bookkeeper", "Office Administrator"]),
    (ACCOUNTANT, ["Junior Bookkeeper"], ["Staff Nurse"]),
    (ADMIN, ["Office Administrator"], ["Junior Bookkeeper"]),
])
def test_profile_determines_which_jobs_are_actionable(
        env, profile, expected_apply, forbidden_titles):
    report = _run(env, profile, _pool())
    assert report["error"] == ""
    applied = [a["title"] for a in report["applications"]]
    skipped_titles = {s["title"] for s in report["skipped"]}
    for title in expected_apply:
        assert title in applied
    for title in forbidden_titles:
        assert title in skipped_titles
    assert "Senior Financial Controller" in skipped_titles  # seniority gate


def test_same_pool_flips_decision_when_profile_changes(env):
    """The acceptance criterion: profile is the source of truth."""
    admin_report = _run(env, ADMIN, _pool())
    driver_report = _run(env, DRIVER, _pool())
    admin_applied = {a["title"] for a in admin_report["applications"]}
    driver_applied = {a["title"] for a in driver_report["applications"]}
    assert "Delivery Driver" not in admin_applied      # code 8 vs code 10
    assert "Delivery Driver" in driver_applied         # exact fit
    assert "Office Administrator" in admin_applied
    assert "Office Administrator" not in driver_applied


def test_explanations_present_for_applications_and_skips(env):
    report = _run(env, ACCOUNTANT, _pool())
    for entry in report["applications"]:
        assert entry.get("explanation"), entry["title"]
        assert "why:" in entry["explanation"].lower()
    for skip in report["skipped"]:
        assert skip.get("reason"), skip["title"]


def test_hard_gate_reasons_are_explicit(env):
    report = _run(env, NURSE, _pool())
    reasons = {s["title"]: s["reason"] for s in report["skipped"]}
    assert "seniority above candidate level" in \
        reasons["Senior Financial Controller"]
    bookkeeper_reasons = reasons["Junior Bookkeeper"]
    assert ("qualification" in bookkeeper_reasons.lower()
            or "below threshold" in bookkeeper_reasons.lower())


def test_registration_unknown_on_aligned_role_asks_user(env):
    nurse_no_sanc = _profile(
        skills=["patient care"],
        education=[Education(qualification="Diploma in Nursing",
                             field="General Nursing", institution="X")],
        experience=[Experience(title="Ward Nurse", company="H",
                               start_date="2023-01", end_date="present")])
    report = _run(env, nurse_no_sanc, _pool())
    reasons = {s["title"]: s["reason"] for s in report["skipped"]}
    assert any("REQUIRES_USER_INPUT" in r and "registered" in r.lower()
               for t, r in reasons.items() if t == "Staff Nurse")


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_empty_search_results_is_a_clean_zero_run(env):
    report = _run(env, ACCOUNTANT, [])
    assert report["error"] == ""
    assert report["jobs_discovered"] == 0
    assert report["suitable_jobs"] == 0
    assert report["applications"] == []


def test_duplicates_across_sources_counted_once(env):
    first = _pool()[0]
    dup_pool = [first, NS(job=first.job, score=first.score)]
    report = _run(env, NURSE, dup_pool)
    assert report["duplicates_dropped"] >= 1
    titles = [a["title"] for a in report["applications"]]
    assert titles.count("Staff Nurse") <= 1


def test_weak_cv_selects_nothing_and_never_crashes(env):
    report = _run(env, WEAK_CV, _pool())
    assert report["error"] == ""
    assert report["suitable_jobs"] == 0
    for skip in report["skipped"]:
        assert skip["reason"]


def test_broad_pollution_fully_filtered(env):
    """208-style flood of unrelated government posts → zero applications."""
    flood = [
        NS(job=_job(f"POST 27/{i}: ARTISAN/ELECTRICIAN/ARTISAN PRODUCTION",
                    "Government advert. Requirements vary. Various "
                    "departments across the country."), score=i % 40)
        for i in range(30)
    ]
    report = _run(env, ACCOUNTANT, flood)
    assert report["jobs_evaluated"] == len(flood)
    assert report["suitable_jobs"] == 0


def test_candidate_search_result_metadata_recorded(env):
    result = NS(ranked=_pool(),
                queries_used=("junior accountant durban",
                              "bookkeeper durban"),
                expanded_queries=("accounts clerk",),
                strategy=NS(career_level="junior",
                            occupations=[{"label": "Accountant"}],
                            inference="evidence",
                            locations=["Durban"]))
    env.holder["profile"] = ACCOUNTANT
    report = run_autonomous_job_search(
        query_text=None,
        search_fn=lambda q, p: result,
        service_factory=lambda: NS(),
        driver_factory=lambda: NS(),
        tracker=ApplicationTracker(path=env.tmp / "apps.json"),
        policy=AutonomyPolicy(min_score=75, max_per_run=5, max_per_day=10),
        dry_run=True,
    )
    assert report["queries_used"][0] == "junior accountant durban"
    assert report["expanded_queries"] == ["accounts clerk"]
    assert report["search_strategy"]["occupations"] == ["Accountant"]
