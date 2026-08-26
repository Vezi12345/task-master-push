from __future__ import annotations

"""Relevance-filtering and pipeline-order tests.

Real-job validation ("is this a legitimate vacancy?") and relevance
("does this role match the search?") are separate checks. A legitimate
Scientist vacancy must still be rejected for a software-developer search,
and legitimate DPSA circular vacancies sharing one PDF URL must survive.
"""

import config
from agent.parse_intent import parse_intent
from agent.relevance import filter_relevant_jobs, is_relevant_job, normalise_title
from agent.search import SOURCE_REGISTRY, dedupe_jobs, search_jobs
from conftest import make_valid_job
from sources.base import Job


def _query(prompt: str):
    return parse_intent(prompt, config.load_region("za"))


def _dpsa_style_job(title: str, company: str = "DEPARTMENT OF AGRICULTURE (DOA)") -> Job:
    """A legitimate circular vacancy: real content, shared PDF source URL."""
    return Job(
        title=title,
        company=company,
        location="Pretoria",
        description=(
            "Requirements: Applicant must be in possession of a Grade 12 "
            "Certificate and an appropriate recognised qualification. "
            "Duties as per the post circular."
        ),
        url="https://www.dpsa.gov.za/dpsa2g/documents/vacancies/2026/PSV%20CIRCULAR%2027%20of%202026.pdf",
        source="dpsa_circular",
    )


# ---------------------------------------------------------------------------
# Relevance decisions on job titles
# ---------------------------------------------------------------------------

def test_real_scientist_job_rejected_for_software_developer_search():
    query = _query("entry level software developer jobs")
    job = _dpsa_style_job("POST 27/01 : SCIENTIST PRODUCTION (GRADE A-C) REF NO: 3/3/1/64/2026")
    relevant, reason = is_relevant_job(job, query)
    assert relevant is False
    assert reason


def test_real_software_developer_job_accepted():
    query = _query("entry level software developer jobs")
    job = _dpsa_style_job("POST 27/40 : SOFTWARE DEVELOPER REF NO: DPSA 12/2026")
    assert is_relevant_job(job, query)[0] is True


def test_real_software_engineer_job_accepted():
    query = _query("software engineer jobs")
    job = Job(
        title="Software Engineer",
        company="Entersekt",
        description="Build security software with Java.",
        url="https://www.entersekt.com/careers/software-engineer",
        source="schemaorg",
    )
    assert is_relevant_job(job, query)[0] is True


def test_graduate_software_developer_job_accepted():
    query = _query("graduate software developer jobs")
    job = _dpsa_style_job(
        "Graduate Software Developer", company="DVT"
    )
    assert is_relevant_job(job, query)[0] is True


def test_accountant_and_hr_jobs_rejected_for_developer_search():
    query = _query("entry level software developer jobs")
    for title in (
        "ACCOUNTANT REF NO: 3/3/1/70/2026",
        "HR OFFICER REF NO: HR5/1/2/3/18",
        "HUMAN RESOURCE OFFICER",
    ):
        assert is_relevant_job(_dpsa_style_job(title), query)[0] is False, title


def test_other_irrelevant_roles_rejected_for_developer_search():
    query = _query("entry level software developer jobs")
    for title in (
        "ADMINISTRATIVE CLERK",
        "NURSE CLINICAL",
        "DRIVER",
        "LEGAL ADVISOR",
        "MARKETING OFFICER",
        "AGRICULTURAL SCIENTIST",
        "DEPUTY DIRECTOR: ORGANISATIONAL DEVELOPMENT",
        "PROPERTY DEVELOPER",
        "BUSINESS DEVELOPMENT MANAGER",
    ):
        assert is_relevant_job(_dpsa_style_job(title), query)[0] is False, title


def test_developer_title_variants_accepted():
    query = _query("entry level software developer jobs")
    for title in (
        "Junior Software Developer",
        "Software Development Intern",
        "Graduate Developer",
        "Full Stack Developer",
        "Backend Developer",
        "Frontend Developer",
        "Application Developer",
        "Systems Developer",
        "Web Developer",
        "Programmer",
        "SOFTWARE DEVELOPERS (X2 POSTS)",
    ):
        assert is_relevant_job(_dpsa_style_job(title), query)[0] is True, title


def test_it_technician_plural_still_matches_it_support_query():
    query = _query("it support jobs")
    job = _dpsa_style_job("POST 27/36 : IT TECHNICIANS REF NO: EEC-ITP-04-02/2026")
    assert is_relevant_job(job, query)[0] is True


def test_normalise_title_strips_circular_noise():
    assert normalise_title(
        "POST 27/01 : SCIENTIST PRODUCTION (GRADE A-C) REF NO: 3/3/1/64/2026"
    ) == "scientist production"


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

def test_multiple_circular_vacancies_share_url_but_all_survive():
    """A circular PDF is ONE document with MANY vacancies — dedupe must not
    collapse them."""
    jobs = [
        _dpsa_style_job("SCIENTIST PRODUCTION"),
        _dpsa_style_job("RESOURCE CONSERVATION INSPECTOR", "DEPARTMENT OF AGRICULTURE"),
        _dpsa_style_job("IT TECHNICIANS", "Ekurhuleni East TVET College"),
        _dpsa_style_job("SOFTWARE DEVELOPER", "DEPARTMENT OF BASIC EDUCATION"),
    ]
    unique = dedupe_jobs(jobs)
    assert len(unique) == 4


def test_duplicates_are_still_removed():
    jobs = [
        make_valid_job(),
        make_valid_job(),  # same title+company+url
        make_valid_job(title="Graduate Software Developer ", url="https://www.dvt.co.za/opportunities/graduate-software-developer?utm=x"),
    ]
    unique = dedupe_jobs(jobs)
    assert len(unique) == 1


def test_pipeline_returns_only_relevant_valid_jobs(monkeypatch):
    from sources.dpsa_circular import DpsaCircularSource

    mixed = [
        _dpsa_style_job("SCIENTIST PRODUCTION"),           # valid but irrelevant
        _dpsa_style_job("SOFTWARE DEVELOPER"),             # valid + relevant
        make_valid_job(),                                  # valid + relevant
        Job(title="", company="", description="", url=""), # invalid record
    ]
    monkeypatch.setattr(DpsaCircularSource, "search", lambda self, query: mixed)

    region = config.load_region("za")
    query = _query("entry level software developer jobs")
    jobs, messages = search_jobs(query, region)

    titles = [j.title for j in jobs]
    assert "SOFTWARE DEVELOPER" in titles
    assert "Graduate Software Developer" in titles
    assert all("SCIENTIST" not in t.upper() for t in titles)
    assert any("not role-relevant" in m for m in messages)


def test_empty_relevant_result_is_allowed(monkeypatch):
    from sources.dpsa_circular import DpsaCircularSource

    only_scientists = [
        _dpsa_style_job("SCIENTIST PRODUCTION"),
        _dpsa_style_job("ADMINISTRATIVE CLERK", "DEPARTMENT OF HEALTH"),
    ]
    monkeypatch.setattr(DpsaCircularSource, "search", lambda self, query: only_scientists)

    region = {
        "name": "Testland", "currency": "ZAR",
        "locations": {}, "skills_dictionary": {},
        "sources": [{"name": "dpsa_circular", "enabled": True}],
    }
    query = _query("entry level software developer jobs")
    jobs, messages = search_jobs(query, region)
    assert jobs == []
    assert any("not role-relevant" in m for m in messages)


def test_demo_source_remains_unregistered_and_fake_jobs_rejected():
    assert "demo" not in SOURCE_REGISTRY
    from sources.validation import validate_real_job

    fake = Job(
        title="Demo Software Developer",
        company="Acme Corp",
        description="This is a sample job placeholder for testing purposes only.",
        url="https://example.com/jobs/1",
        source="demo",
    )
    checked, reason = validate_real_job(fake)
    assert checked is None
    assert reason
