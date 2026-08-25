from __future__ import annotations

"""Autonomy engine tests — scoring, selection, thresholds, limits,
duplicates, question criticality, plan readiness, humanisation, validation.

All pure unit tests: fake jobs/profiles, tmp trackers. No network/browser.
"""

from datetime import datetime
from types import SimpleNamespace as NS

import pytest

import config
from application.autonomy import (
    AutonomyPolicy,
    classify_question_criticality,
    decide_job,
    evaluate_plan_readiness,
    select_jobs,
    submissions_today,
    suitability_score,
)
from application.humanise import humanise, validate_against_profile
from application.models import Application, ApplicationStatus
from application.tracker import ApplicationTracker
from candidate.profile import CandidateProfile, Education, Experience
from sources.base import Job


def _profile(**kw) -> CandidateProfile:
    defaults = {
        "skills": ["python", "django", "sql", "git"],
        "education": [Education(
            qualification="National Diploma",
            field="ICT Application Development",
            institution="DUT")],
        "experience": [],
        "certifications": [],
        "location": "Durban",
    }
    defaults.update(kw)
    return CandidateProfile(**defaults)


def _ranked(title="Junior Python Developer", company="TestCo",
            desc="Python Django REST API entry-level graduate role in Durban",
            pref=80) -> NS:
    job = Job(title=title, company=company, location="Durban",
              description=desc)
    return NS(job=job, score=pref, reasons=[])


# ---------------------------------------------------------------------------
# 1. suitability scoring from real profile evidence
# ---------------------------------------------------------------------------

def test_suitability_score_bounds_and_reasons():
    p = _profile()
    score, reasons, concerns = suitability_score(p, _ranked().job, 80)
    assert 0 <= score <= 100
    assert reasons and all(r.startswith("✓") for r in reasons)


def test_suitability_rewards_matching_skills():
    strong = _profile()
    none = _profile(skills=[])
    s_strong, _, _ = suitability_score(strong, _ranked().job)
    s_none, _, _ = suitability_score(none, _ranked().job)
    assert s_strong > s_none


# ---------------------------------------------------------------------------
# 2–3. selection + threshold
# ---------------------------------------------------------------------------

def _tracker(tmp_path) -> ApplicationTracker:
    return ApplicationTracker(path=tmp_path / "apps.json")


import tempfile
from pathlib import Path


def _tracker_tmp() -> ApplicationTracker:
    return ApplicationTracker(path=Path(tempfile.mkdtemp()) / "a.json")


def test_select_applies_multiple_and_skips_below_threshold():
    p = _profile()
    items = [
        _ranked("Job A", "Alpha", pref=92),
        _ranked("Job B", "Beta", pref=87),
        _ranked("Job C", "Gamma", "some other role entirely", pref=71),
        _ranked("Job D", "Delta", pref=54),
    ]
    selected, skipped = select_jobs(items, p, tracker=_tracker_tmp())
    titles = [d.job.title for d in selected]
    assert set(titles) == {"Job A", "Job B"}
    skip_reasons = {d.job.title: d.reason for d in skipped}
    # 50-64% is the transferable review band; below 50% stays plain reject
    assert "review recommended" in skip_reasons["Job C"]
    assert "below threshold" in skip_reasons["Job D"]


def test_decision_includes_positive_evidence_lines():
    d = decide_job(_ranked(), _profile(), tracker=_tracker_tmp(), policy=AutonomyPolicy())
    assert d.decision == "apply"
    assert any("skill" in r.lower() or "qualification" in r.lower() for r in d.reasons)


def _tracker_tmp():
    import tempfile
    from pathlib import Path
    return ApplicationTracker(path=Path(tempfile.mkdtemp()) / "a.json")


def test_senior_jobs_never_selected():
    senior = _ranked("Senior Python Developer", "BigCo",
                     desc="Senior role, 10+ years experience required", pref=95)
    d = decide_job(senior, _profile(), tracker=_tracker_tmp())
    assert d.decision == "skip"
    assert "seniority" in d.reason


# ---------------------------------------------------------------------------
# 4–5. limits + duplicates
# ---------------------------------------------------------------------------

def test_per_run_limit_enforced():
    p = _profile()
    items = [_ranked(f"Job {i}", f"Co{i}", pref=90) for i in range(8)]
    policy = AutonomyPolicy(max_per_run=2, max_per_day=100)
    selected, skipped = select_jobs(items, p, tracker=_tracker_tmp(), policy=policy)
    assert len(selected) == 2
    assert all("run limit" in d.reason for d in skipped[2:])


def test_per_day_limit_counts_prior_submissions(tmp_path):
    tr = _tracker(tmp_path)
    done = Application(job_id="old", job_title="Old", job_company="OldCo",
                       status=ApplicationStatus.SUBMITTED, submitted=True,
                       submitted_at=datetime.now().isoformat())
    tr.add(done)
    assert submissions_today(tr) == 1

    p = _profile()
    items = [_ranked(f"Job {i}", f"Co{i}", pref=90) for i in range(4)]
    policy = AutonomyPolicy(min_score=75, max_per_run=10, max_per_day=3)
    selected, skipped = select_jobs(items, p, tracker=tr, policy=policy)
    assert len(selected) == 2          # 3 - 1 already submitted today
    assert all("daily limit" in d.reason for d in skipped[-1:])


def test_duplicate_tracked_job_skipped(tmp_path):
    tr = _tracker(tmp_path)
    ranked = _ranked()
    existing = Application(
        job_id=ranked.job.id, job_title=ranked.job.title,
        job_company=ranked.job.company, status=ApplicationStatus.SUBMITTED)
    tr.add(existing)
    d = decide_job(ranked, _profile(), tracker=tr)
    assert d.decision == "skip"
    assert "duplicate" in d.reason


def test_same_job_twice_in_one_run_deduped():
    p = _profile()
    item = _ranked(pref=90)
    selected, skipped = select_jobs([item, _ranked(pref=85)], p,
                                    tracker=_tracker_tmp())
    assert len(selected) == 1
    assert any("duplicate" in d.reason for d in skipped)


# ---------------------------------------------------------------------------
# 6–7. question classification + plan readiness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("Are you a South African citizen?", "critical"),
    ("Do you have the legal right to work in South Africa?", "critical"),
    ("Do you have a disability?", "critical"),
    ("Do you have a criminal record?", "critical"),
    ("Which driver's licence do you hold?", "critical"),
    ("What is your ID number?", "critical"),
    ("Please state your expected salary", "critical"),   # mandatory salary
    ("Portfolio URL (optional)", "noncritical"),
    ("What is your favourite programming language?", "noncritical"),
])
def test_criticality_classification(question, expected):
    required = question.startswith("Please state")
    assert classify_question_criticality(question, required) == expected


def test_salary_optional_is_noncritical():
    assert classify_question_criticality("Expected salary", required=False) == "noncritical"


def test_plan_readiness_blocks_on_required_unknowns():
    plan = [
        {"question": "Phone number", "value": "082", "required": True},
        {"question": "Are you a SA citizen?", "value": "", "required": True},
        {"question": "Notice period", "value": "", "required": True},
        {"question": "Portfolio URL", "value": "", "required": False},
        {"question": "Upload CV", "value": "/tmp/cv.pdf", "upload_kind": "cv"},
    ]
    r = evaluate_plan_readiness(plan)
    assert r["ready"] is False
    assert "Are you a SA citizen?" in r["critical_missing"]
    assert "Notice period" in r["noncritical_missing"]
    assert r["answered"] == 2   # phone + CV upload


def test_plan_ready_when_all_required_answered():
    plan = [{"question": "Phone number", "value": "082", "required": True}]
    assert evaluate_plan_readiness(plan)["ready"] is True


# ---------------------------------------------------------------------------
# 8. humanisation
# ---------------------------------------------------------------------------

def test_humanise_strips_corporate_language():
    robotic = ("As a highly motivated and passionate individual, I am excited "
               "to leverage my comprehensive technical skillset to add value.")
    out = humanise(robotic)
    low = out.lower()
    assert "highly motivated" not in low
    assert "leverage" not in low
    assert "passionate individual" not in low
    assert out.strip().endswith(".")


def test_humanise_keeps_natural_text_readable():
    natural = ("I recently completed my diploma in ICT Application Development, "
               "and I'm looking for an opportunity where I can build my software "
               "development skills in a real working environment")
    out = humanise(natural)
    assert "recently completed" in out
    assert out.endswith(".")


# ---------------------------------------------------------------------------
# 9. validation against profile — never invent facts
# ---------------------------------------------------------------------------

def test_validation_rejects_inflated_experience():
    fresh_grad = _profile(experience=[])
    ok, issues = validate_against_profile(
        "I have 8 years of experience building enterprise systems.", fresh_grad)
    assert not ok and issues


def test_validation_allows_truthful_graduate_language():
    p = _profile(experience=[Experience(
        company="X", title="Intern", start_date="2025-01", end_date="2025-06",
        skills=[])])
    ok, issues = validate_against_profile(
        "I recently completed my National Diploma in ICT Application Development "
        "and completed a six-month internship.", p)
    assert ok, issues


def test_validation_rejects_unsupported_qualification_claim():
    no_edu = _profile(education=[])
    ok, issues = validate_against_profile(
        "I hold a bachelor's degree in medicine.", no_edu)
    assert not ok and issues


def test_validation_rejects_uncertified_claims():
    p = _profile(certifications=[{"name": "CompTIA A+"}])
    ok, issues = validate_against_profile(
        "I am certified in AWS Solutions Architecture.", p)
    assert not ok and issues
