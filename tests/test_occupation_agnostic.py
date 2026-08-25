from __future__ import annotations

"""Occupation-agnosticism and candidate-adaptivity regression suite.

Proves the pipeline derives occupations from candidate evidence or explicit
request only, that search intent never leaks into claimed skills, that
hard requirements stay hard, that transferable matches are labelled and
never auto-submitted, and that multi-source discovery degrades gracefully.
"""

from types import SimpleNamespace as NS

import pytest

import config
from agent.parse_intent import parse_intent
from agent.search import SOURCE_REGISTRY, search_jobs
from agent.candidate_search import search_for_candidate
from application.autonomy import AutonomyPolicy, decide_job
from candidate.profile import (
    CandidateProfile,
    Education,
    Experience,
    Certification,
)
from candidate.suitability import evaluate as evaluate_suitability
from agent.relevance import is_relevant_job
from sources.base import Job, JobSourceError


def _region():
    return config.load_region("za")


def _profile(**kw) -> CandidateProfile:
    defaults = {"skills": [], "education": [], "experience": [],
                "certifications": [], "location": ""}
    defaults.update(kw)
    return CandidateProfile(name="T", email="t@x.com", **defaults)


NURSE = _profile(
    skills=["patient care", "wound care"],
    education=[Education(qualification="Diploma in Nursing",
                         field="General Nursing", institution="Hospital")],
    experience=[Experience(title="Staff Nurse", company="Clinic",
                           start_date="2021-01", end_date="present")])

ACCOUNTANT = _profile(
    skills=["pastel", "reconciliations", "vat"],
    education=[Education(qualification="Diploma in Accounting",
                         field="Financial Management", institution="CPT")],
    experience=[Experience(title="Accounts Clerk", company="Firm",
                           start_date="2023-02", end_date="present")])

TEACHER = _profile(
    skills=["lesson planning", "classroom management"],
    education=[Education(qualification="Bachelor of Education",
                         field="Intermediate Phase", institution="UJ")],
    experience=[Experience(title="Educator", company="School",
                           start_date="2022-01", end_date="present")])

DRIVER = _profile(
    skills=["deliveries", "vehicle checks"],
    experience=[Experience(title="Delivery Driver", company="Logistics Co",
                           start_date="2022-06", end_date="present")],
    drivers_licence="Code 10")

ADMIN = _profile(
    skills=["ms office", "filing", "data entry"],
    education=[Education(qualification="Diploma in Business Management",
                         institution="College")],
    experience=[Experience(title="Office Administrator", company="Co",
                           start_date="2023-01", end_date="present")])

ICT_GRAD = _profile(
    skills=["java", "c#", "sql", "css", "linux", "communication"],
    education=[Education(
        qualification=("Diploma in Information and Communication "
                       "Technology - Application Development"),
        institution="DUT")],
    experience=[])

FINANCE_GRAD = _profile(
    skills=["pastel", "reconciliations", "vat", "journals"],
    education=[Education(qualification="Bachelor of Commerce in Accounting",
                         field="Accounting", institution="UWC")],
    experience=[Experience(title="Finance Intern", company="Auditors",
                           start_date="2025-01", end_date="2025-12")])

WEAK_CV = _profile(skills=[])

_SOFTWARE_MARKERS = ("software", "developer", "programmer", "backend",
                     "frontend", "full stack", "qa tester")


def _sp(profile):
    from candidate.search_profile import build_search_profile
    return build_search_profile(profile)


def _queries(profile):
    from candidate.search_profile import generate_queries
    return generate_queries(_sp(profile))


# ---------------------------------------------------------------------------
# 1-6: occupation detection is driven by evidence, not IT defaults
# ---------------------------------------------------------------------------

def test_1_nurse_profile_does_not_generate_software_queries():
    qs = _queries(NURSE)
    assert qs, "nurse evidence must produce queries"
    assert not any(m in " ".join(qs).lower() for m in _SOFTWARE_MARKERS)


def test_2_accountant_profile_does_not_generate_software_queries():
    qs = _queries(ACCOUNTANT)
    joined = " ".join(qs).lower()
    assert not any(m in joined for m in _SOFTWARE_MARKERS)
    assert any(word in joined for word in ("accountant", "bookkeeper",
                                           "accounts", "payroll"))


def test_3_teacher_profile_does_not_generate_software_queries():
    joined = " ".join(_queries(TEACHER)).lower()
    assert not any(m in joined for m in _SOFTWARE_MARKERS)
    assert any(word in joined for word in ("teacher", "educator", "tutor",
                                           "teaching assistant"))


def test_4_driver_profile_generates_driver_logistics_roles():
    sp = _sp(DRIVER)
    assert sp.occupations and sp.occupations[0]["key"] == "driver"
    joined = " ".join(_queries(DRIVER)).lower()
    assert "driver" in joined


def test_5_admin_profile_generates_administration_roles():
    sp = _sp(ADMIN)
    assert sp.occupations and sp.occupations[0]["key"] == "administrator"
    joined = " ".join(_queries(ADMIN)).lower()
    assert any(w in joined for w in ("administrator", "admin", "clerk",
                                     "data capturer"))


def test_6_ict_profile_generates_ict_roles():
    sp = _sp(ICT_GRAD)
    assert sp.occupations and sp.occupations[0]["key"] == "software_developer"
    joined = " ".join(_queries(ICT_GRAD)).lower()
    assert "developer" in joined or "software" in joined


# ---------------------------------------------------------------------------
# 7-10: explicit intent overrides profiles; keywords are never skills
# ---------------------------------------------------------------------------

def test_7_explicit_finance_query_overrides_ict_profile_guessing():
    seen: list[str] = []
    res = search_for_candidate(
        profile=ICT_GRAD, region=_region(), llm=None,
        query_text="finance jobs",
        pipeline_fn=lambda q: (seen.append(q), NS(ranked=[]))[1],
    )
    assert seen == ["finance jobs"], "profile guesses must not be prepended"
    assert not any("developer" in q for q in seen)


def test_8_explicit_nursing_query_targets_nursing_for_any_candidate():
    q = parse_intent("Find me nursing jobs", _region())
    assert q.roles == ["Nurse"]
    assert q.skills == []
    nurse_job = Job(title="PROFESSIONAL NURSE GRADE 1", company="Health",
                    location="Durban",
                    description="Provide patient care in a clinic.")
    other_job = Job(title="STATE ACCOUNTANT: REVENUE", company="Treasury",
                    location="Durban", description="Capture journals.")
    assert is_relevant_job(nurse_job, q)[0] is True
    assert is_relevant_job(other_job, q)[0] is False


@pytest.mark.parametrize("prompt", [
    "Find me nursing jobs",
    "Find me electrician jobs",
    "entry level accounting clerk jobs",
])
def test_9_search_keywords_never_become_candidate_skills(prompt):
    q = parse_intent(prompt, _region())
    sector_words = {"nursing", "electrician", "accounting", "clerk"}
    leaked = {s for s in q.skills if s in sector_words}
    assert leaked == set(), f"search target leaked into skills: {leaked}"
    # and the profile itself was never mutated by parsing
    assert WEAK_CV.skills == []


def test_10_finance_query_grants_ict_candidate_no_finance_skill_match():
    job = Job(title="POST 27/57 : FINANCE CLERK REF NO: 3/1/1/1/2026/197",
              company="Department", location="Durban",
              description=(
                  "Requirements: Minimum requirements: Applicants must be in "
                  "possession of a Grade 12 Certificate. Duties: capture "
                  "salaries, bonuses, salary adjustments, deductions, "
                  "reconcile accounts, handle cash and banking."))
    result = evaluate_suitability(job, ICT_GRAD)
    blob = " ".join(result.matched).lower()
    assert "finance skill" not in blob
    assert not any(
        word in blob for word in ("you have finance", "your finance skill"))
    assert any("Transferable match" in m for m in result.matched), \
        "cross-domain match must be labelled transferable"


# ---------------------------------------------------------------------------
# 11-12: honesty about weak CVs; explainable expansion
# ---------------------------------------------------------------------------

def test_11_weak_cv_invents_no_occupation():
    sp = _sp(WEAK_CV)
    assert sp.occupations == []
    assert sp.inference == "minimal"
    qs = _queries(WEAK_CV)
    joined = " ".join(qs).lower()
    assert not any(m in joined for m in _SOFTWARE_MARKERS)
    assert not any(w in joined for w in ("nurse", "accountant", "teacher"))


def test_12_adjacent_occupations_are_explainable():
    sp = _sp(ACCOUNTANT)
    reasons = sp.expansion_reasons
    direct_reason = next((v for t, v in reasons.items()
                          if t == "junior accountant"), "")
    adjacent_reason = next((v for t, v in reasons.items()
                            if t == "audit clerk"), "")
    assert "Direct role" in direct_reason and "Accountant" in direct_reason
    assert "Adjacent" in adjacent_reason and "Accountant" in adjacent_reason


# ---------------------------------------------------------------------------
# 13-15: hard gates, REQUIRES_USER_INPUT, review-band behaviour
# ---------------------------------------------------------------------------

def test_13_mandatory_requirements_remain_hard_gates():
    job = Job(title="Driver", company="Co", location="Durban",
              description=(
                  "Requirements: A valid Code 10 driving licence is "
                  "required. Must have 3 years driving experience."))
    # profile explicitly says NO licence -> clear lack, hard reject
    no_licence = _profile(
        skills=["deliveries"],
        experience=[Experience(title="Driver", company="X",
                               start_date="2020-01", end_date="present")],
        drivers_licence="No")
    result = evaluate_suitability(job, no_licence)
    assert result.decision == "reject"
    assert result.blockers


def test_14_unknown_mandatory_facts_require_user_input():
    cases = [
        # nurse evidence, but nowhere does the profile claim SANC registration
        (Job(title="Ward Nurse", company="Hospital", location="Durban",
             description=(
                 "Requirements: Applicants must be registered with SANC.")),
         NURSE),
        (Job(title="Messenger", company="Dept", location="Durban",
             description="Requirements: A valid driver's licence is needed."),
         _profile(skills=["filing"])),
        (Job(title="Registry Clerk", company="Dept", location="Durban",
             description="Note: South African citizens only may apply."),
         _profile(skills=["filing"])),
    ]
    for job, profile in cases:
        result = evaluate_suitability(job, profile)
        assert result.decision == "requires_user_input", job.title
        assert result.unknowns
        assert not result.blockers, job.title


def _finance_clerk_item():
    job = Job(title="POST 27/57 : FINANCE CLERK REF NO: 3/1/1/1/2026/197",
              company="Department of Land Reform", location="Durban",
              description=(
                  "Requirements: Minimum requirements: Applicants must be in "
                  "possession of a Grade 12 Certificate. Duties: check "
                  "advices for correctness, capture salaries, bonuses, "
                  "salary adjustments, capture all deductions, reconcile "
                  "accounts, handle cash, banking and filing."))
    return NS(job=job, score=55, reasons=[], summary="")


def test_15_transferable_band_is_never_auto_submitted():
    item = _finance_clerk_item()
    engine = evaluate_suitability(item.job, ICT_GRAD)
    assert 50 <= engine.score < 65, \
        f"test setup expects 50-64 band, got {engine.score}"
    d = decide_job(item, ICT_GRAD, tracker=None,
                   policy=AutonomyPolicy(min_score=75))
    assert d.decision != "apply"
    assert "review recommended" in d.reason


# ---------------------------------------------------------------------------
# 16-18: multi-source merging, dedup, graceful degradation
# ---------------------------------------------------------------------------

_DPSA_DOC = ("https://www.dpsa.gov.za/dpsa2g/documents/vacancies/2026/"
             "PSV%20CIRCULAR.pdf")


class _FakeSourceA:
    name = "fake_a"

    def __init__(self, cfg):
        pass

    def search(self, query):
        return [Job(title="Accounts Clerk", company="Firm A",
                    location="Durban", url=_DPSA_DOC,
                    description=(
                        "Requirements: Grade 12 Certificate. Duties: capture "
                        "invoices, reconcile accounts and handle filing."),
                    source="fake_a")]


class _FakeSourceB:
    name = "fake_b"

    def __init__(self, cfg, explode=False):
        self._explode = explode

    def search(self, query):
        if self._explode:
            raise JobSourceError("source B is down")
        return [Job(title="Bookkeeper", company="Firm B",
                    location="Durban", url=_DPSA_DOC,
                    description=(
                        "Requirements: Grade 12 Certificate. Duties: process "
                        "journals, reconciliations, vat and payroll."),
                    source="fake_b")]


def _multi_source_region(names):
    return {
        "name": "Testland", "currency": "ZAR",
        "locations": {}, "skills_dictionary": {},
        "sources": [{"name": n, "enabled": True} for n in names],
    }


def test_16_multiple_sources_can_contribute_jobs(monkeypatch):
    monkeypatch.setitem(SOURCE_REGISTRY, "fake_a", _FakeSourceA)
    monkeypatch.setitem(SOURCE_REGISTRY, "fake_b", _FakeSourceB)
    jobs, messages = search_jobs(parse_intent("accounting jobs", _region()),
                                 _multi_source_region(["fake_a", "fake_b"]))
    titles = {j.title for j in jobs}
    assert {"Accounts Clerk", "Bookkeeper"} <= titles
    assert any("fake_a: 1 jobs" in m for m in messages)
    assert any("fake_b: 1 jobs" in m for m in messages)


def test_17_duplicate_jobs_across_sources_removed(monkeypatch):
    class DupB:
        name = "fake_b"

        def __init__(self, cfg):
            pass

        def search(self, query):
            return [Job(title="accounts clerk", company="firm a",
                        url="https://b/dup", source="fake_b")]

    monkeypatch.setitem(SOURCE_REGISTRY, "fake_a", _FakeSourceA)
    monkeypatch.setitem(SOURCE_REGISTRY, "fake_b", DupB)
    jobs, _ = search_jobs(parse_intent("accounting jobs", _region()),
                          _multi_source_region(["fake_a", "fake_b"]))
    keys = [(j.title.lower(), j.company.lower()) for j in jobs]
    assert len(keys) == len(set(keys))


def test_18_one_failed_source_does_not_terminate_search(monkeypatch):
    class BadB(_FakeSourceB):
        def __init__(self, cfg):
            super().__init__(cfg, explode=True)

    monkeypatch.setitem(SOURCE_REGISTRY, "fake_a", _FakeSourceA)
    monkeypatch.setitem(SOURCE_REGISTRY, "fake_b", BadB)
    jobs, messages = search_jobs(parse_intent("accounting jobs", _region()),
                                 _multi_source_region(["fake_a", "fake_b"]))
    assert any(j.title == "Accounts Clerk" for j in jobs)
    assert any("fake_b" in m and ("down" in m or "error" in m)
               for m in messages)


# ---------------------------------------------------------------------------
# 19-20: strategy and decisions are adaptive, not fixed
# ---------------------------------------------------------------------------

def test_19_same_candidate_different_queries_produce_different_strategies():
    seen_nursing: list[str] = []
    seen_finance: list[str] = []

    def recorder(seen):
        return lambda q: (seen.append(q), NS(ranked=[]))[1]

    search_for_candidate(profile=ACCOUNTANT, region=_region(), llm=None,
                         query_text="nursing jobs",
                         pipeline_fn=recorder(seen_nursing))
    search_for_candidate(profile=ACCOUNTANT, region=_region(), llm=None,
                         query_text="finance jobs",
                         pipeline_fn=recorder(seen_finance))
    assert seen_nursing == ["nursing jobs"]
    assert seen_finance == ["finance jobs"]
    assert seen_nursing != seen_finance


def test_20_same_job_pool_different_candidates_different_decisions():
    item = _finance_clerk_item()
    finance_view = evaluate_suitability(item.job, FINANCE_GRAD)
    ict_view = evaluate_suitability(item.job, ICT_GRAD)
    assert finance_view.score > ict_view.score
    fin_d = decide_job(item, FINANCE_GRAD, tracker=None,
                       policy=AutonomyPolicy(min_score=75))
    ict_d = decide_job(item, ICT_GRAD, tracker=None,
                       policy=AutonomyPolicy(min_score=75))
    assert fin_d.suitability > ict_d.suitability
    assert any("Commerce" in m or "commerce" in m
               for m in finance_view.matched) or \
        any("Qualification matches" in m or "Matric-level" in m
            for m in finance_view.matched)
    assert any("Transferable match" in m for m in ict_view.matched)
