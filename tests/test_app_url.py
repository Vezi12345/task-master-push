from __future__ import annotations

"""Phase 2 — real application URL / ATS platform detection."""

from application.app_url import (
    ApplicationTarget,
    detect_platform,
    find_application_url,
    summarise_target,
)
from sources.base import Job


def _job(url: str = "https://www.dvt.co.za/careers/graduate-software-developer", **kw) -> Job:
    defaults = dict(
        title="Graduate Software Developer",
        company="DVT",
        location="Durban",
        description="Graduate development programme. Apply online.",
        url=url,
        source="schemaorg",
    )
    defaults.update(kw)
    return Job(**defaults)


# ---------------------------------------------------------------------------
# platform fingerprinting
# ---------------------------------------------------------------------------

def test_detect_platform_for_major_ats_systems():
    cases = {
        "https://boards.greenhouse.io/dvt/jobs/4012345": "greenhouse",
        "https://job-boards.greenhouse.io/dvt/jobs/4012345": "greenhouse",
        "https://jobs.lever.co/acme/8f2b": "lever",
        "https://acme.wd3.myworkdayjobs.com/en-US/careers/job/Dev": "workday",
        "https://jobs.smartrecruiters.com/Acme/743999": "smartrecruiters",
        "https://acme.taleo.net/careersection/2/jobdetail.ftl?job=123": "taleo",
        "https://acme.icims.com/jobs/1234/login": "icims",
        "https://jobs.jobvite.com/acme/job/abc": "jobvite",
        "https://career8.sapsf.com/career?company=acme": "successfactors",
        "https://acme.applytojob.com/apply/xyz": "applytojob",
    }
    for url, expected in cases.items():
        assert detect_platform(url) == expected, url


def test_detect_platform_unknown_or_non_http():
    assert detect_platform("https://careers.dvt.co.za/jobs/1") == ""
    assert detect_platform("not a url") == ""


# ---------------------------------------------------------------------------
# discovery from job URL
# ---------------------------------------------------------------------------

def test_job_url_on_ats_is_the_application_page():
    job = _job(url="https://boards.greenhouse.io/dvt/jobs/4012345")
    target = find_application_url(job)
    assert target.ok
    assert target.application_url == job.url
    assert target.platform == "greenhouse"
    assert target.confidence >= 0.9


# ---------------------------------------------------------------------------
# discovery from page HTML
# ---------------------------------------------------------------------------

GREENHOUSE_PAGE = """
<html><body>
  <a href="/careers">Back to careers</a>
  <a href="https://boards.greenhouse.io/dvt/jobs/4012345?gh_src=menu" class="apply">Apply for this job</a>
</body></html>
"""


def test_apply_link_to_ats_extracted_from_page_html():
    job = _job()
    target = find_application_url(job, page_html=GREENHOUSE_PAGE)
    assert target.ok
    assert target.platform == "greenhouse"
    assert "greenhouse.io" in target.application_url


def test_relative_explicit_apply_link_resolved_and_accepted():
    html = '<html><body><a href="/careers/graduate-software-developer/apply">Apply now</a></body></html>'
    job = _job(url="https://www.dvt.co.za/careers/graduate-software-developer")
    target = find_application_url(job, page_html=html)
    assert target.ok
    assert target.application_url == (
        "https://www.dvt.co.za/careers/graduate-software-developer/apply"
    )
    assert target.platform in ("custom", "")


def test_generic_careers_homepage_is_not_proof_of_application():
    html = '<html><body><a href="/careers">Careers at DVT</a></body></html>'
    job = _job()
    target = find_application_url(job, page_html=html)
    assert not target.found
    assert target.reason


def test_fetch_function_used_when_no_html_given():
    calls = []

    def fake_fetch(url: str) -> str:
        calls.append(url)
        return GREENHOUSE_PAGE

    job = _job()
    target = find_application_url(job, fetch=fake_fetch)
    assert calls == [job.url]
    assert target.ok
    assert target.platform == "greenhouse"


def test_fetch_failure_falls_back_to_honest_failure():
    def broken_fetch(url: str) -> str:
        raise ConnectionError("network down")

    job = _job(url="https://careers.example.com/jobs/9")
    target = find_application_url(job, fetch=broken_fetch)
    assert not target.found
    assert target.reason


# ---------------------------------------------------------------------------
# DPSA circular offline process
# ---------------------------------------------------------------------------

def test_dpsa_circular_reports_offline_z83_process():
    job = Job(
        title="POST 27/01 : SCIENTIST PRODUCTION",
        company="DEPARTMENT OF AGRICULTURE",
        description=(
            "Applications must be submitted on the new Z83 form, "
            "obtainable from any Public Service department."
        ),
        url="https://www.dpsa.gov.za/dpsa2g/documents/vacancies/2026/PSV%20CIRCULAR%2027%20of%202026.pdf",
        source="dpsa_circular",
    )
    target = find_application_url(job)
    assert not target.found
    assert target.requires_user_action
    assert "Z83" in target.reason


# ---------------------------------------------------------------------------
# recordable summary
# ---------------------------------------------------------------------------

def test_summary_stores_required_fields():
    job = _job(url="https://jobs.lever.co/dvt/abcd")
    target = find_application_url(job)
    summary = summarise_target(target, job)
    assert summary["application_url"] == job.url
    assert summary["application_platform"] == "lever"
    assert summary["employer"] == "DVT"
    assert summary["job_title"] == "Graduate Software Developer"
    assert summary["source"] == "schemaorg"
