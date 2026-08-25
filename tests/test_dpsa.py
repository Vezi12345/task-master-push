from pathlib import Path

import pytest
import requests

from agent.search import dedupe_jobs
from sources.base import JobSource, JobSourceError
from sources.dpsa_circular import DpsaCircularSource, parse_circular

FIXTURES = Path(__file__).parent / "fixtures"

REALISTIC = (FIXTURES / "dpsa_circular.txt").read_text(encoding="utf-8")
MALFORMED = (FIXTURES / "dpsa_circular_malformed.txt").read_text(encoding="utf-8")
ANNEXURE = (FIXTURES / "dpsa_circular_annexure.txt").read_text(encoding="utf-8")

CIRCULAR_URL = "https://example.invalid/circular-14-2026.pdf"


def test_parse_circular_extracts_multiple_vacancies():
    jobs = parse_circular(REALISTIC)
    assert len(jobs) == 5


def test_title_keeps_post_reference():
    jobs = parse_circular(REALISTIC)
    assert jobs[0].title == "POST 14/05/01 : DEPUTY DIRECTOR: RISK MANAGEMENT"


def test_department_is_company():
    jobs = parse_circular(REALISTIC)
    agriculture = [j for j in jobs if "AGRICULTURE" in j.title or j.id == "14/05/01"][0]
    health = [j for j in jobs if j.id == "14/04/01"][0]
    assert agriculture.company == "DEPARTMENT OF AGRICULTURE, LAND REFORM AND RURAL DEVELOPMENT"
    assert health.company == "KWAZULU-NATAL DEPARTMENT OF HEALTH"


def test_location_extracted_from_centre():
    jobs = parse_circular(REALISTIC)
    clerk = [j for j in jobs if j.id == "14/05/02"][0]
    assert clerk.location == "Durban"


def test_salary_extracted_as_monthly():
    jobs = parse_circular(REALISTIC)
    clerk = [j for j in jobs if j.id == "14/05/02"][0]
    assert clerk.salary_min == round(261372 / 12)
    assert clerk.salary_max == round(301608 / 12)
    assert clerk.salary_text == "R 261 372 - R 301 608 per annum (Level 5)"


def test_closing_date_inherited_from_department_block():
    jobs = parse_circular(REALISTIC)
    by_id = {j.id: j for j in jobs}
    assert by_id["14/05/01"].posted_date == "28 August 2026"
    assert by_id["14/05/02"].posted_date == "27 August 2026"
    assert by_id["14/05/03"].posted_date == "28 August 2026"


def test_source_identifier_and_url_are_set():
    jobs = parse_circular(REALISTIC, source_url=CIRCULAR_URL)
    for job in jobs:
        assert job.source == "dpsa_circular"
        assert job.url == CIRCULAR_URL
        assert job.id == job.title.split(" : ")[0].replace("POST ", "")
    assert jobs[0].id == "14/05/01"


def test_requirements_and_duties_in_description():
    jobs = parse_circular(REALISTIC)
    deputy = [j for j in jobs if j.id == "14/05/01"][0]
    assert "Requirements:" in deputy.description
    assert "Duties:" in deputy.description
    assert "Financial Accounting" in deputy.description


def test_no_false_entries_from_section_headings():
    jobs = parse_circular(REALISTIC)
    assert all(job.title.startswith("POST ") for job in jobs)
    assert all("SECTION" not in job.title for job in jobs)
    assert all("CIRCULAR" not in job.title for job in jobs)


def test_default_company_used_when_no_department_heading():
    text = (
        "POST 1/1 : GENERAL WORKER\n"
        "CENTRE : Pretoria\n"
        "SALARY : R 200 000 per annum\n"
    )
    job = parse_circular(text, default_company="DPSA / Government")[0]
    assert job.company == "DPSA / Government"


def test_malformed_entries_do_not_crash_and_keep_valid_ones():
    jobs = parse_circular(MALFORMED)
    assert len(jobs) == 4


def test_empty_salary_is_conservative():
    jobs = parse_circular(MALFORMED)
    driver = [j for j in jobs if j.id == "20/01/01"][0]
    assert driver.salary_min is None
    assert driver.salary_max is None
    assert driver.salary_text is None


def test_missing_centre_and_closing_date():
    jobs = parse_circular(MALFORMED)
    capturer = [j for j in jobs if j.id == "20/01/02"][0]
    assert capturer.location == ""
    assert capturer.posted_date == "30 August 2026"
    driver = [j for j in jobs if j.id == "20/01/01"][0]
    assert driver.posted_date is None


def test_malformed_post_headers_do_not_create_jobs():
    jobs = parse_circular(MALFORMED)
    titles = {j.id for j in jobs}
    assert "20/01/03" not in titles
    assert "20/01/04" not in titles


def test_duplicate_posts_are_deduped():
    jobs = parse_circular(MALFORMED)
    foremen = [j for j in jobs if j.id == "20/02/01"]
    assert len(foremen) == 1
    assert foremen[0].salary_min == round(200000 / 12)
    assert foremen[0].posted_date == "31 August 2026"


def test_empty_text_returns_no_jobs():
    assert parse_circular("") == []


def test_subsection_headings_do_not_clobber_department_or_closing_date():
    jobs = parse_circular(ANNEXURE)
    assert len(jobs) == 3
    for job in jobs:
        assert job.company in (
            "DEPARTMENT OF AGRICULTURE (DOA)",
            "DEPARTMENT OF BASIC EDUCATION",
        )
    scientist = [j for j in jobs if j.id == "27/01"][0]
    director = [j for j in jobs if j.id == "27/03"][0]
    deputy = [j for j in jobs if j.id == "27/05"][0]
    assert scientist.company == "DEPARTMENT OF AGRICULTURE (DOA)"
    assert director.company == "DEPARTMENT OF AGRICULTURE (DOA)"
    assert deputy.company == "DEPARTMENT OF BASIC EDUCATION"
    assert scientist.posted_date == "17 August 2026 at 16:00"
    assert director.posted_date == "17 August 2026 at 16:00"
    assert deputy.posted_date == "18 August 2026"
    assert scientist.salary_min == round(791604 / 12)


def test_wrapped_title_continuation_is_not_mistaken_for_department():
    jobs = parse_circular(ANNEXURE)
    director = [j for j in jobs if j.id == "27/03"][0]
    assert director.title.endswith("DBE/59/2026")
    assert director.company == "DEPARTMENT OF AGRICULTURE (DOA)"
    assert director.salary_min == round(1554696 / 12)
    assert director.location == "Pretoria"


def test_dedupe_via_search_pipeline():
    jobs = parse_circular(REALISTIC) * 2
    unique = dedupe_jobs(jobs)
    assert len(unique) == 5


def test_source_works_through_jobsource_abstraction(monkeypatch):
    source = DpsaCircularSource(
        {
            "name": "dpsa_circular",
            "url": CIRCULAR_URL,
            "default_company": "DPSA / Government",
        }
    )
    monkeypatch.setattr(source, "fetch_text", lambda url: REALISTIC)
    assert isinstance(source, JobSource)
    jobs = source.search(None)
    assert len(jobs) == 5
    assert all(job.source == "dpsa_circular" for job in jobs)
    assert all(job.url == CIRCULAR_URL for job in jobs)


def test_search_without_url_returns_empty():
    assert DpsaCircularSource({"name": "dpsa_circular"}).search(None) == []


class _FakeResponse:
    def __init__(self, content: bytes, ok: bool = True, status: int = 200):
        self.content = content
        self.status_code = status
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Not Found for url: {CIRCULAR_URL}"
            )


class _FakePdf:
    def __init__(self, *args, **kwargs):
        self.pages = [_FakePage()]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakePage:
    def extract_text(self):
        return REALISTIC


def test_fetch_timeout_raises_job_source_error(monkeypatch):
    def _timeout(*args, **kwargs):
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr("requests.get", _timeout)
    source = DpsaCircularSource({"name": "dpsa_circular", "url": CIRCULAR_URL})
    with pytest.raises(JobSourceError, match="could not download circular"):
        source.fetch_text(CIRCULAR_URL)


def test_fetch_http_error_raises_job_source_error(monkeypatch):
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: _FakeResponse(b"", ok=False, status=404)
    )
    source = DpsaCircularSource({"name": "dpsa_circular", "url": CIRCULAR_URL})
    with pytest.raises(JobSourceError, match="could not download circular"):
        source.fetch_text(CIRCULAR_URL)


def test_fetch_malformed_pdf_raises_job_source_error(monkeypatch):
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: _FakeResponse(b"definitely not a pdf")
    )
    source = DpsaCircularSource({"name": "dpsa_circular", "url": CIRCULAR_URL})
    with pytest.raises(JobSourceError, match="could not parse circular PDF"):
        source.fetch_text(CIRCULAR_URL)


def test_fetch_success_returns_extracted_text(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse(b"<pdf bytes>"))
    monkeypatch.setattr("pdfplumber.open", _FakePdf)
    source = DpsaCircularSource({"name": "dpsa_circular", "url": CIRCULAR_URL})
    assert source.fetch_text(CIRCULAR_URL) == REALISTIC
