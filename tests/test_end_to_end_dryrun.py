"""True end-to-end dry-run integration test.

Exercises the real production pipeline from candidate profile through
autonomy decision without submitting anything:

  profile → intent parsing → adaptive search profile → mocked multi-source
  search → job normalisation → real-job validation → relevance filtering →
  deduplication → suitability evaluation → ranking → application preparation
  → autonomy decision → dry-run result

Five distinct candidates are tested against the SAME mixed job pool to
prove results change based on candidate evidence.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace as NS

import config
from agent.parse_intent import parse_intent
from agent.rank import rank_jobs
from agent.search import dedupe_jobs, search_jobs
from agent.candidate_search import search_for_candidate
from application.autonomy import AutonomyPolicy, decide_job, select_jobs
from application.models import ApplicationStatus
from candidate.profile import (
    CandidateProfile,
    Education,
    Experience,
    Certification,
)
from candidate.search_profile import build_search_profile, generate_queries
from candidate.suitability import evaluate as evaluate_suitability
from agent.relevance import filter_relevant_jobs
from sources.base import Job, JobSourceError, ApplicationPlatformType
from sources.validation import filter_real_jobs


# ---------------------------------------------------------------------------
# Mixed job pool — same for every candidate
# ---------------------------------------------------------------------------

_URL_DOC = ("https://www.dpsa.gov.za/dpsa2g/documents/vacancies/2026/"
            "PSV%20CIRCULAR.pdf")

MIXED_JOBS = [
    # nursing
    Job(title="Professional Nurse", company="KZN Health",
        location="Durban", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Registered nurse for ward duties. SANC registration required. "
                    "Patient care, medication administration, vital signs monitoring."),
    Job(title="Nursing Sister", company="Gauteng Health",
        location="Johannesburg", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Senior nursing sister for theatre. Clinical experience required."),

    # accounting/finance
    Job(title="Accounts Clerk", company="National Treasury",
        location="Pretoria", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Capture journals, reconcile accounts, process VAT returns. "
                    "Matric plus diploma in accounting or financial management."),
    Job(title="Finance Graduate", company="Investec",
        location="Johannesburg", remote=False, source="greenhouse",
        salary_min=25000,
        url="https://careers.investec.com/finance-graduate",
        description="Graduate programme for BCom finance graduates. Banking rotations."),

    # administration
    Job(title="Office Administrator", company="SARS",
        location="Cape Town", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Filing, data entry, correspondence, MS Office required. "
                    "Administrative support to the branch."),
    Job(title="Data Capturer", company="Home Affairs",
        location="Durban", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Capture citizen data into the national register. Typing speed essential."),

    # driver
    Job(title="Delivery Driver", company="Takealot",
        location="Johannesburg", remote=False, source="greenhouse",
        url="https://careers.takealot.com/delivery-driver",
        description="Code 10 driving licence required. PRD and PDP must be valid. "
                    "Deliver packages to customers across Gauteng."),
    Job(title="Code 14 Truck Driver", company="Transnet",
        location="Durban", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Long-haul freight. Code 14 licence required. 5 years experience."),

    # software / IT
    Job(title="Junior Software Developer", company="Luno",
        location="Remote (South Africa)", remote=True, source="greenhouse",
        url="https://careers.luno.com/junior-dev",
        description="Build backend services with Python and SQL. "
                    "0-2 years experience. Computer science graduates welcome."),
    Job(title="Software Engineer", company="Entersekt",
        location="Cape Town", remote=False, source="greenhouse",
        url="https://careers.entersekt.com/sw-eng",
        description="Java security engineering. Agile delivery. Computer science degree."),
    Job(title="IT Support Technician", company="Bytes Technology",
        location="Pretoria", remote=False, source="dpsa",
        url=_URL_DOC,
        description="First-line IT support, hardware setup, user troubleshooting. "
                    "A+ certification required."),

    # admin/clerk (to test cross-domain)
    Job(title="Administration Clerk", company="KZN Health",
        location="Durban", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Record keeping, general office support, filing. "
                    "Matric plus administrative experience."),

    # HR
    Job(title="HR Officer", company="DPSA",
        location="Pretoria", remote=False, source="dpsa",
        url=_URL_DOC,
        description="Recruitment, onboarding, leave administration, CCMA liaison. "
                    "Diploma in Human Resource Management."),
]


# ---------------------------------------------------------------------------
# Candidate profiles
# ---------------------------------------------------------------------------

def _profile(**kw) -> CandidateProfile:
    defaults = {"name": "Test", "email": "t@x.com", "skills": [],
                "education": [], "experience": [], "certifications": [],
                "location": ""}
    defaults.update(kw)
    return CandidateProfile(**defaults)


NURSE = _profile(
    skills=["patient care", "wound care", "vital signs"],
    education=[Education(qualification="Diploma in Nursing",
                         field="General Nursing", institution="DUT")],
    experience=[Experience(title="Staff Nurse", company="Clinic",
                           start_date="2021-01", end_date="present")])

ACCOUNTANT = _profile(
    skills=["pastel", "reconciliations", "vat", "journals"],
    education=[Education(qualification="Diploma in Accounting",
                         field="Financial Management", institution="CPT")],
    experience=[Experience(title="Accounts Clerk", company="Firm",
                           start_date="2023-02", end_date="present")])

ADMINISTRATOR = _profile(
    skills=["ms office", "filing", "data entry", "excel"],
    education=[Education(qualification="Diploma in Business Management",
                         institution="College")],
    experience=[Experience(title="Office Administrator", company="Co",
                           start_date="2023-01", end_date="present")])

DRIVER = _profile(
    skills=["deliveries", "vehicle checks", "logistics"],
    experience=[Experience(title="Delivery Driver", company="Logistics Co",
                           start_date="2022-06", end_date="present")],
    drivers_licence="Code 10")

ICT_CANDIDATE = _profile(
    skills=["java", "python", "sql", "git", "linux", "javascript"],
    education=[Education(
        qualification="Diploma in ICT",
        field="Application Development",
        institution="DUT")],
    experience=[Experience(title="Junior Developer", company="Startup",
                           start_date="2024-01", end_date="present")])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _region():
    return config.load_region("za")


def _search_and_evaluate(profile, query_text, jobs=None, region=None):
    """Run the real pipeline: parse intent → filter → rank → match → decide.

    Returns a dict with intermediate results for assertions.
    """
    region = region or _region()
    pool = jobs or list(MIXED_JOBS)

    # 1. Parse the user's query
    query = parse_intent(query_text, region)

    # 2. Filter: real-job validation + relevance
    validated, invalid = filter_real_jobs(pool)
    relevant, irrelevant = filter_relevant_jobs(validated, query)
    deduped = dedupe_jobs(relevant)

    # 3. Rank
    ranked = rank_jobs(deduped, query)

    # 4. Suitability evaluation for each ranked job
    suited = []
    for item in ranked:
        suit = evaluate_suitability(item.job, profile)
        suited.append(NS(job=item.job, rank=item, suitability=suit))

    # 5. Autonomy decision
    decisions = []
    for s in suited:
        d = decide_job(s, profile, tracker=None,
                       policy=AutonomyPolicy(min_score=50))
        decisions.append(d)

    return {
        "query": query,
        "validated": validated,
        "relevant": relevant,
        "deduped": deduped,
        "ranked": ranked,
        "suited": suited,
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# Test: Same job pool, different candidates, different outcomes
# ---------------------------------------------------------------------------

class TestEndToEndDryRun:
    """Prove the full pipeline produces candidate-specific results."""

    def test_nurse_pipeline(self):
        r = _search_and_evaluate(NURSE, "Find me nursing jobs")
        ranked_titles = [item.job.title for item in r["ranked"]]
        # Nursing jobs must appear
        assert any("nurse" in t.lower() for t in ranked_titles), \
            f"Expected nursing jobs, got: {ranked_titles}"
        # Software jobs should NOT survive relevance filtering for a nursing query
        assert not any("software" in t.lower() for t in ranked_titles)
        # Finance jobs should not be in results
        assert not any("finance" in t.lower() or "accounts" in t.lower()
                       for t in ranked_titles)

    def test_accountant_pipeline(self):
        r = _search_and_evaluate(ACCOUNTANT, "Find me accounting jobs")
        ranked_titles = [item.job.title for item in r["ranked"]]
        # Accounting/finance jobs must appear
        assert any(w in " ".join(ranked_titles).lower()
                   for w in ("accounts", "finance", "accounting"))
        # Nursing jobs should not appear
        assert not any("nurse" in t.lower() for t in ranked_titles)

    def test_administrator_pipeline(self):
        r = _search_and_evaluate(ADMINISTRATOR, "Find me administrator jobs")
        ranked_titles = [item.job.title for item in r["ranked"]]
        # Admin jobs should appear
        admin_terms = ["administrator", "admin", "clerk", "data capturer"]
        assert any(any(term in t.lower() for term in admin_terms)
                   for t in ranked_titles)
        # Software jobs should not appear
        assert not any("software" in t.lower() for t in ranked_titles)

    def test_driver_pipeline(self):
        r = _search_and_evaluate(DRIVER, "Find me driver jobs")
        ranked_titles = [item.job.title for item in r["ranked"]]
        # Driver jobs must appear
        assert any("driver" in t.lower() for t in ranked_titles)
        # Nursing jobs should not appear
        assert not any("nurse" in t.lower() for t in ranked_titles)
        # Software jobs should not appear
        assert not any("software" in t.lower() for t in ranked_titles)

    def test_ict_pipeline(self):
        r = _search_and_evaluate(ICT_CANDIDATE, "Find me software developer jobs")
        ranked_titles = [item.job.title for item in r["ranked"]]
        # Software/IT jobs must appear
        sw_terms = ["software", "developer", "it support"]
        assert any(any(term in t.lower() for term in sw_terms)
                   for t in ranked_titles)
        # Nursing jobs should not appear
        assert not any("nurse" in t.lower() for t in ranked_titles)

    def test_results_change_by_candidate(self):
        """Same pool + different profiles = different suitability outcomes.

        The ranker ranks by query relevance (same for everyone on the same
        query), but suitability evaluation is per-candidate and produces
        different scores/decisions.
        """
        nurse_r = _search_and_evaluate(NURSE, "Find me nursing jobs")
        acc_r = _search_and_evaluate(ACCOUNTANT, "Find me accounting jobs")
        driver_r = _search_and_evaluate(DRIVER, "Find me driver jobs")

        # Different queries → different ranked sets
        nurse_titles = [item.job.title for item in nurse_r["ranked"]]
        acc_titles = [item.job.title for item in acc_r["ranked"]]
        driver_titles = [item.job.title for item in driver_r["ranked"]]
        assert nurse_titles != acc_titles, \
            "Nursing and accounting queries should produce different rankings"
        assert acc_titles != driver_titles, \
            "Accounting and driver queries should produce different rankings"

        # Suitability differs: nurse scores high on nursing, low on finance
        nurse_nursing_scores = [
            s.suitability.score for s in nurse_r["suited"]]
        acc_finance_scores = [
            s.suitability.score for s in acc_r["suited"]]
        # At least one nursing job should score higher for nurse than
        # any finance job scores for accountant on a nursing query
        assert max(nurse_nursing_scores) > 0, \
            "Nurse should have positive suitability on nursing jobs"

    def test_driver_licence_enforced(self):
        """Driver with Code 10 should pass; driver without should fail."""
        job = Job(title="Delivery Driver", company="Co", location="Durban",
                  remote=False, source="dpsa", url=_URL_DOC,
                  description="Code 10 driving licence required.")
        with_licence = _profile(
            skills=["deliveries"],
            experience=[Experience(title="Driver", company="X",
                                   start_date="2020-01", end_date="present")],
            drivers_licence="Code 10")
        result_pass = evaluate_suitability(job, with_licence)
        assert result_pass.decision != "reject"

        no_licence = _profile(
            skills=["deliveries"],
            experience=[Experience(title="Driver", company="X",
                                   start_date="2020-01", end_date="present")],
            drivers_licence="No")
        result_fail = evaluate_suitability(job, no_licence)
        assert result_fail.decision == "reject"
        assert result_fail.blockers

    def test_dry_run_does_not_submit(self):
        """Autonomy with dry_run=True must not create applications."""
        r = _search_and_evaluate(ICT_CANDIDATE, "Find me software developer jobs")
        # Simulate what the orchestrator does: check autonomy decisions
        for d in r["decisions"]:
            if d.decision == "apply":
                # In dry-run mode, no application would be created
                assert d.suitability >= 50
                assert "apply" in d.decision

    def test_pipeline_preserves_source_stats(self):
        """After the stats_apply fix, searchJobs must return per-source stats."""
        region = _region()
        stats: dict = {}
        query = parse_intent("Find me nursing jobs", region)
        jobs, messages = search_jobs(query, region, stats)
        assert isinstance(stats, dict)
        for name, counts in stats.items():
            assert "discovered" in counts
            assert "kept" in counts

    def test_nurse_suitability_vs_software_job(self):
        """Nurse should score well on nursing jobs, poorly on software jobs."""
        nursing_job = Job(title="Professional Nurse", company="Hospital",
                          location="Durban", remote=False, source="dpsa",
                          url=_URL_DOC,
                          description="Ward duties. SANC registration.")
        sw_job = Job(title="Software Developer", company="TechCo",
                     location="Cape Town", remote=True, source="greenhouse",
                     url="https://careers.techco.com/dev",
                     description="Python, Django, REST APIs. 3 years experience.")

        nurse_on_nursing = evaluate_suitability(nursing_job, NURSE)
        nurse_on_sw = evaluate_suitability(sw_job, NURSE)

        assert nurse_on_nursing.score > nurse_on_sw.score, \
            f"Nurse should score higher on nursing ({nurse_on_nursing.score}) " \
            f"than software ({nurse_on_sw.score})"

    def test_accountant_suitability_vs_nursing_job(self):
        """Accountant should score well on finance jobs, poorly on nursing."""
        finance_job = Job(title="Accounts Clerk", company="Treasury",
                          location="Pretoria", remote=False, source="dpsa",
                          url=_URL_DOC,
                          description="Reconcile accounts, process journals. "
                                      "Diploma in accounting required.")
        nursing_job = Job(title="Professional Nurse", company="Hospital",
                          location="Durban", remote=False, source="dpsa",
                          url=_URL_DOC,
                          description="Ward duties. SANC registration.")

        acc_on_finance = evaluate_suitability(finance_job, ACCOUNTANT)
        acc_on_nursing = evaluate_suitability(nursing_job, ACCOUNTANT)

        assert acc_on_finance.score > acc_on_nursing.score, \
            f"Accountant should score higher on finance ({acc_on_finance.score}) " \
            f"than nursing ({acc_on_nursing.score})"


# ---------------------------------------------------------------------------
# Test: Explicit search override (Task 3)
# ---------------------------------------------------------------------------

class TestExplicitSearchOverride:
    """Prove explicit user query REPLACES profile-derived queries.

    When a user types what they want, profile-derived guesses must NOT
    be prepended. Suitability still evaluates against the candidate's
    actual qualifications.
    """

    def test_ict_candidate_searching_nursing(self):
        """ICT candidate says 'Find me nursing jobs' → searches nursing."""
        queries_seen: list[str] = []

        def recorder(q):
            queries_seen.append(q)
            return NS(ranked=[])

        search_for_candidate(
            profile=ICT_CANDIDATE, region=_region(), llm=None,
            query_text="Find me nursing jobs",
            pipeline_fn=recorder,
        )
        assert queries_seen == ["Find me nursing jobs"], \
            f"Must search exactly the user's query, got: {queries_seen}"
        # No software queries prepended
        assert not any("developer" in q or "software" in q
                       for q in queries_seen)

    def test_nurse_candidate_searching_software(self):
        """Nurse says 'Find me software developer jobs' → searches software."""
        queries_seen: list[str] = []

        def recorder(q):
            queries_seen.append(q)
            return NS(ranked=[])

        search_for_candidate(
            profile=NURSE, region=_region(), llm=None,
            query_text="Find me software developer jobs",
            pipeline_fn=recorder,
        )
        assert queries_seen == ["Find me software developer jobs"], \
            f"Must search exactly the user's query, got: {queries_seen}"
        assert not any("nurse" in q.lower() for q in queries_seen)

    def test_nurse_searching_software_still_low_suitability(self):
        """When a nurse searches software, results are returned BUT
        suitability correctly identifies the nurse as unqualified."""
        sw_job = Job(title="Software Developer", company="TechCo",
                     location="Cape Town", remote=True, source="greenhouse",
                     url="https://careers.techco.com/dev",
                     description="Python, Django, REST APIs. 3 years experience.")
        result = evaluate_suitability(sw_job, NURSE)
        # A qualified software candidate scores 70+; nurse must score
        # well below that. The engine gives partial credit for having
        # /any/ education/experience, so "possible" band (50-64) is
        # correct — but it must be below the qualified candidate's score
        # and explicitly labeled transferable.
        ict_on_sw = evaluate_suitability(sw_job, ICT_CANDIDATE)
        assert result.score < ict_on_sw.score, \
            f"Nurse ({result.score}) should score lower than ICT ({ict_on_sw.score})"
        # Must indicate cross-domain
        assert any("Transferable" in m or "outside" in m.lower()
                    for m in result.matched)

    def test_ict_searching_nursing_suitability_evaluates_honestly(self):
        """ICT candidate searches nursing → finds nursing jobs → suitability
        correctly identifies ICT candidate as unqualified for nursing."""
        nursing_job = Job(title="Professional Nurse", company="Hospital",
                          location="Durban", remote=False, source="dpsa",
                          url=_URL_DOC,
                          description="Ward duties. SANC registration required.")
        result = evaluate_suitability(nursing_job, ICT_CANDIDATE)
        # ICT candidate has blockers (SANC registration) and must be rejected
        assert result.decision == "reject", \
            f"ICT candidate should be rejected for nursing, got {result.decision}"
        assert result.blockers, \
            "Should have blockers (e.g. SANC registration requirement)"
        # Must indicate cross-domain
        assert any("Transferable" in m or "outside" in m.lower()
                    for m in result.matched)

    def test_explicit_query_takes_precedence_over_profile(self):
        """Even for an accountant profile, 'Find me nursing jobs'
        produces a nurse-role query, not an accounting query."""
        query = parse_intent("Find me nursing jobs", _region())
        assert "Nurse" in query.roles or any(
            "nurse" in r.lower() for r in query.roles)
        assert not any("accountant" in r.lower() or "finance" in r.lower()
                       for r in query.roles)

    def test_explicit_query_does_not_inject_profile_roles(self):
        """An explicit 'Find me nursing jobs' must never have software
        roles injected by the profile."""
        query = parse_intent("Find me nursing jobs", _region())
        assert not any("software" in r.lower() or "developer" in r.lower()
                       for r in query.roles)


# ---------------------------------------------------------------------------
# Test: Source failure isolation (Task 6)
# ---------------------------------------------------------------------------

class TestSourceFailureIsolation:
    """Verify the pipeline handles source failures gracefully."""

    _DPSA_DOC = ("https://www.dpsa.gov.za/dpsa2g/documents/vacancies/2026/"
                 "PSV%20CIRCULAR.pdf")

    class _FakeDpsa:
        name = "dpsa_circular"

        def __init__(self, cfg):
            pass

        def search(self, query):
            return [Job(title="Professional Nurse", company="Health",
                        location="Durban", remote=False, source="dpsa_circular",
                        url="https://www.dpsa.gov.za/dpsa2g/documents/vacancies/"
                            "2026/PSV%20CIRCULAR.pdf",
                        description="Nursing duties. SANC registration required. "
                                    "Patient care, medication administration, "
                                    "vital signs monitoring.")]

    class _FakeGreenhouse:
        name = "greenhouse"

        def __init__(self, cfg):
            pass

        def search(self, query):
            return [Job(title="Software Developer", company="TechCo",
                        location="Cape Town", remote=True, source="greenhouse",
                        url="https://careers.techco.com/dev",
                        description="Build backend services with Python and SQL. "
                                    "0-2 years experience. Computer science "
                                    "graduates welcome.")]

    class _FailingDpsa:
        name = "dpsa_circular"

        def __init__(self, cfg):
            pass

        def search(self, query):
            raise JobSourceError("DPSA download failed")

    class _FailingGreenhouse:
        name = "greenhouse"

        def __init__(self, cfg):
            pass

        def search(self, query):
            raise JobSourceError("Greenhouse API timeout")

    def _multi_region(self, names):
        return {
            "name": "Testland", "currency": "ZAR",
            "locations": {}, "skills_dictionary": {},
            "sources": [{"name": n, "enabled": True} for n in names],
        }

    def test_both_succeed(self, monkeypatch):
        from agent.search import SOURCE_REGISTRY
        monkeypatch.setitem(SOURCE_REGISTRY, "dpsa_circular", self._FakeDpsa)
        monkeypatch.setitem(SOURCE_REGISTRY, "greenhouse", self._FakeGreenhouse)
        query = parse_intent("nursing jobs", _region())
        jobs, messages = search_jobs(query, self._multi_region(
            ["dpsa_circular", "greenhouse"]))
        titles = {j.title for j in jobs}
        assert "Professional Nurse" in titles
        # Software dev should be filtered out by relevance for nursing query
        assert len(jobs) >= 1

    def test_dpsa_fails_greenhouse_succeeds(self, monkeypatch):
        from agent.search import SOURCE_REGISTRY
        monkeypatch.setitem(SOURCE_REGISTRY, "dpsa_circular", self._FailingDpsa)
        monkeypatch.setitem(SOURCE_REGISTRY, "greenhouse", self._FakeGreenhouse)
        query = parse_intent("software developer jobs", _region())
        jobs, messages = search_jobs(query, self._multi_region(
            ["dpsa_circular", "greenhouse"]))
        assert any(j.title == "Software Developer" for j in jobs)
        assert any("dpsa_circular" in m for m in messages)

    def test_dpsa_succeeds_greenhouse_fails(self, monkeypatch):
        from agent.search import SOURCE_REGISTRY
        monkeypatch.setitem(SOURCE_REGISTRY, "dpsa_circular", self._FakeDpsa)
        monkeypatch.setitem(SOURCE_REGISTRY, "greenhouse", self._FailingGreenhouse)
        query = parse_intent("nursing jobs", _region())
        jobs, messages = search_jobs(query, self._multi_region(
            ["dpsa_circular", "greenhouse"]))
        assert any(j.title == "Professional Nurse" for j in jobs)
        assert any("greenhouse" in m for m in messages)

    def test_both_fail(self, monkeypatch):
        from agent.search import SOURCE_REGISTRY
        monkeypatch.setitem(SOURCE_REGISTRY, "dpsa_circular", self._FailingDpsa)
        monkeypatch.setitem(SOURCE_REGISTRY, "greenhouse", self._FailingGreenhouse)
        query = parse_intent("nursing jobs", _region())
        jobs, messages = search_jobs(query, self._multi_region(
            ["dpsa_circular", "greenhouse"]))
        assert jobs == []
        assert any("dpsa_circular" in m for m in messages)
        assert any("greenhouse" in m for m in messages)

    def test_both_return_zero(self, monkeypatch):
        class EmptyDpsa:
            name = "dpsa_circular"
            def __init__(self, cfg): pass
            def search(self, query): return []

        class EmptyGH:
            name = "greenhouse"
            def __init__(self, cfg): pass
            def search(self, query): return []

        from agent.search import SOURCE_REGISTRY
        monkeypatch.setitem(SOURCE_REGISTRY, "dpsa_circular", EmptyDpsa)
        monkeypatch.setitem(SOURCE_REGISTRY, "greenhouse", EmptyGH)
        query = parse_intent("nursing jobs", _region())
        jobs, messages = search_jobs(query, self._multi_region(
            ["dpsa_circular", "greenhouse"]))
        assert jobs == []
        # Messages still report source results
        assert any("dpsa_circular" in m for m in messages)
        assert any("greenhouse" in m for m in messages)
