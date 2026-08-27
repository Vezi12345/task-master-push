"""Tests for the public ATS job-board sources (Lever, Recruitee, Workable).

Each source mirrors the Greenhouse design: it is query-agnostic (role
matching is owned centrally by ``agent.relevance``), South-Africa-filtered by
default, and keeps original employer URLs. Fakes use genuine-looking URLs so
records survive ``sources.validation`` exactly like live traffic.
"""
from __future__ import annotations

import pytest
import requests

import sources.lever as lever_module
import sources.recruitee as recruitee_module
import sources.workable as workable_module
from agent.parse_intent import JobQuery, parse_intent
from agent.search import SOURCE_REGISTRY, search_jobs
from sources.base import ApplicationPlatformType, JobSourceError
from sources.lever import LeverSource
from sources.recruitee import RecruiteeSource
from sources.workable import WorkableSource


class _FakeResponse:
    def __init__(self, status):
        self.status_code = status


# ------------------------------------------------------------------ Lever

LEVER_URL = "https://jobs.lever.co/acme/5ac21346-8e0c-4494-8e7a-3eb92ff77902"


def lever_entry(title="Junior Bookkeeper", jid="abc-1",
                location="Cape Town", description_plain=None,
                hosted=LEVER_URL, workplace="on-site",
                country="ZA", created=1778622524938):
    e = {"id": jid, "text": title, "country": country, "workplaceType": workplace,
         "createdAt": created, "hostedUrl": hosted,
         "applyUrl": hosted + "/apply",
         "categories": {"location": location, "team": "Finance",
                        "commitment": "Full-time",
                        "allLocations": [location]}}
    if description_plain is not None:
        e["descriptionPlain"] = description_plain
    return e


class TestLever:
    def test_registered(self):
        assert "lever" in SOURCE_REGISTRY

    def test_maps_fields(self, monkeypatch):
        jobs = self._search(monkeypatch, [lever_entry(
            description_plain="Bookkeeping with pastel and vat in cape town.")])
        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Junior Bookkeeper"
        assert job.company == "Acme Ltd"
        assert job.location == "Cape Town"
        assert job.url == LEVER_URL
        assert job.source == "lever"
        assert job.id == "lever-abc-1"
        assert job.platform is ApplicationPlatformType.LEVER
        assert job.posted_date == "2026-05-12"

    def test_remote_from_workplace_type(self, monkeypatch):
        jobs = self._search(monkeypatch, [lever_entry(
            title="Accountant (Remote)", workplace="remote",
            description_plain="Fully remote role.")])
        assert jobs[0].remote is True

    def test_remote_from_location_text(self, monkeypatch):
        jobs = self._search(monkeypatch, [lever_entry(
            location="Remote", workplace="unspecified",
            description_plain="Some role working from anywhere in south africa.")])
        assert jobs[0].remote is True

    def test_salary_range_extracted(self, monkeypatch):
        entry = lever_entry()
        entry["salaryRange"] = {"currency": "ZAR", "interval": "monthly",
                                "min": 15000, "max": 25000}
        job = self._search(monkeypatch, [entry])[0]
        assert job.salary_min == 15000 and job.salary_max == 25000

    def test_missing_title_dropped(self, monkeypatch):
        jobs = self._search(monkeypatch, [
            lever_entry(title="", jid="6"),
            lever_entry(jid="7"),
        ])
        assert [j.title for j in jobs] == ["Junior Bookkeeper"]

    def test_sa_filter_drops_overseas(self, monkeypatch):
        jobs = self._search(monkeypatch, [lever_entry(
            title="Accountant", location="London, UK",
            description_plain="A role in london for british candidates.")])
        assert jobs == []

    def test_sa_marker_in_description_kept(self, monkeypatch):
        jobs = self._search(monkeypatch, [lever_entry(
            location="Remote - Global",
            description_plain="You may work from Johannesburg.")])
        assert len(jobs) == 1

    def test_sa_filter_disabled(self, monkeypatch):
        cfg = {"companies": [{"site": "acme", "name": "Acme Ltd"}],
               "require_south_africa": False}
        jobs = self._search(monkeypatch, [lever_entry(
            location="London, UK",
            description_plain="A british role.")], cfg=cfg)
        assert len(jobs) == 1

    def test_eu_fallback_on_404(self, monkeypatch):
        calls = []
        base = {}

        def fake_fetch(url, params):
            calls.append(url)
            if "/api.eu.lever.co/" in url:
                return [lever_entry()]
            raise requests.HTTPError("404", response=_FakeResponse(404))

        monkeypatch.setattr(lever_module, "fetch", fake_fetch)
        jobs = LeverSource({"companies": [{"site": "acme", "name": "Acme Ltd"}]}).search(JobQuery())
        assert len(jobs) == 1
        assert any("/api.eu.lever.co/" in u for u in calls)

    def test_one_failed_site_does_not_kill(self, monkeypatch):
        def fake_fetch(url, params):
            if "bad" in url:
                raise requests.HTTPError("500", response=_FakeResponse(500))
            return [lever_entry(jid="zz")]

        monkeypatch.setattr(lever_module, "fetch", fake_fetch)
        cfg = {"companies": [{"site": "bad", "name": "Bad"},
                             {"site": "good", "name": "Good"}],
               "require_south_africa": True}
        jobs = LeverSource(cfg).search(JobQuery())
        assert len(jobs) == 1 and jobs[0].company == "Good"

    def test_all_failing_raises(self, monkeypatch):
        def fake_fetch(url, params):
            raise requests.HTTPError("500", response=_FakeResponse(500))

        monkeypatch.setattr(lever_module, "fetch", fake_fetch)
        with pytest.raises(JobSourceError):
            LeverSource({"companies": ["acme"]}).search(JobQuery())

    def test_no_companies_empty(self):
        assert LeverSource({}).search(JobQuery()) == []

    def _search(self, monkeypatch, entries, cfg=None):
        cfg = cfg or {"companies": [{"site": "acme", "name": "Acme Ltd"}],
                      "require_south_africa": True}

        def fake_fetch(url, params):
            return entries

        monkeypatch.setattr(lever_module, "fetch", fake_fetch)
        return LeverSource(cfg).search(JobQuery())


# --------------------------------------------------------------- Recruitee

RECRUITEE_URL = "https://acme.recruitee.com/o/accountant"


def recruitee_entry(title="Accounts Clerk", jid="42", location="Durban",
                    description="<p>Pastel and vat in durban.</p>",
                    url=RECRUITEE_URL, published="2026-08-01T08:00:00Z"):
    return {"id": jid, "title": title, "location": location,
            "description": description, "careers_url": url,
            "published_at": published}


class TestRecruitee:
    def test_registered(self):
        assert "recruitee" in SOURCE_REGISTRY

    def test_maps_fields(self, monkeypatch):
        cfg = {"companies": [{"subdomain": "acme", "name": "Acme (Pty) Ltd"}],
               "require_south_africa": True}
        jobs = self._search(monkeypatch, cfg, [recruitee_entry()])
        job = jobs[0]
        assert job.title == "Accounts Clerk"
        assert job.company == "Acme (Pty) Ltd"
        assert job.location == "Durban"
        assert job.url == RECRUITEE_URL
        assert job.source == "recruitee"
        assert job.id == "recruitee-42"
        assert job.posted_date == "2026-08-01T08:00:00Z"
        assert job.platform is ApplicationPlatformType.GENERIC_WEB

    def test_strips_html(self, monkeypatch):
        cfg = self._cfg()
        job = self._search(monkeypatch, cfg, [recruitee_entry(
            description="<div><p>Bookkeep for <b>us</b>.</p>")])[0]
        assert "<" not in job.description and "evil" not in job.description

    def test_remote_flagged(self, monkeypatch):
        job = self._search(monkeypatch, self._cfg(), [recruitee_entry(
            title="Accountant (Remote)", location="Remote")])[0]
        assert job.remote is True

    def test_sa_filter(self, monkeypatch):
        jobs = self._search(monkeypatch, self._cfg(), [recruitee_entry(
            location="London, UK",
            description="<p>a completely british role.</p>")])
        assert jobs == []

    def test_one_failed_board_not_fatal(self, monkeypatch):
        def fake_fetch(url, params=None):
            if "bad" in url:
                raise requests.HTTPError("404", response=_FakeResponse(404))
            return {"offers": [recruitee_entry(jid="5")]}

        monkeypatch.setattr(recruitee_module, "fetch_json", fake_fetch)
        cfg = {"companies": [{"subdomain": "bad", "name": "Bad"},
                             {"subdomain": "good", "name": "Good"}],
               "require_south_africa": True}
        jobs = RecruiteeSource(cfg).search(JobQuery())
        assert len(jobs) == 1 and jobs[0].company == "Good"

    def test_no_companies_empty(self):
        assert RecruiteeSource({"companies": []}).search(JobQuery()) == []

    def _cfg(self):
        return {"companies": [{"subdomain": "acme", "name": "Acme Ltd"}],
                "require_south_africa": True}

    def _search(self, monkeypatch, cfg, offers):
        def fake_fetch(url, params=None):
            return {"offers": offers}

        monkeypatch.setattr(recruitee_module, "fetch_json", fake_fetch)
        return RecruiteeSource(cfg).search(JobQuery())


# --------------------------------------------------------------- Workable

WORKABLE_URL = "https://apply.workable.com/acme/j/B7E55A124A/apply"


def workable_entry(title="Junior Accountant", jid="b7e55a",
                   location="Cape Town, South Africa", url=WORKABLE_URL,
                   description="Bookkeeping and vat reconciliations in cape town.",
                   published="2026-08-02T09:00:00Z", remote=None):
    loc = location if isinstance(location, str) else location
    return {"id": jid, "title": title, "url": url, "location": loc,
            "description": description, "published_on": published,
            "remote": remote}


class TestWorkable:
    def test_registered(self):
        assert "workable" in SOURCE_REGISTRY

    def test_maps_fields(self, monkeypatch):
        cfg = {"companies": [{"account": "acme", "name": "Acme Ltd"}],
               "require_south_africa": True}
        job = self._search(monkeypatch, cfg, [workable_entry()])[0]
        assert job.title == "Junior Accountant"
        assert job.company == "Acme Ltd"
        assert job.location == "Cape Town, South Africa"
        assert job.url == WORKABLE_URL
        assert job.source == "workable"
        assert job.id == "workable-b7e55a"
        assert job.posted_date == "2026-08-02T09:00:00Z"

    def test_location_object(self, monkeypatch):
        entry = workable_entry(location={
            "city": "Durban", "state_code": "KZN",
            "country_name": "South Africa"})
        job = self._search(monkeypatch, self._cfg(), [entry])[0]
        assert job.location == "Durban, KZN, South Africa"

    def test_remote_flag(self, monkeypatch):
        entry = workable_entry()
        entry["remote"] = True
        job = self._search(monkeypatch, self._cfg(), [entry])[0]
        assert job.remote is True

    def test_sa_filter(self, monkeypatch):
        jobs = self._search(monkeypatch, self._cfg(), [workable_entry(
            location="London, UK",
            description="a completely british role in london.")])
        assert jobs == []

    def test_one_failed_board_not_fatal(self, monkeypatch):
        def fake_fetch(url, params=None):
            if "bad" in url:
                raise requests.HTTPError("404", response=_FakeResponse(404))
            return {"jobs": [workable_entry()]}

        monkeypatch.setattr(workable_module, "fetch_json", fake_fetch)
        cfg = {"companies": [{"account": "bad", "name": "Bad"},
                             {"account": "good", "name": "Good"}],
               "require_south_africa": True}
        jobs = WorkableSource(cfg).search(JobQuery())
        assert len(jobs) == 1 and jobs[0].company == "Good"

    def test_no_companies_empty(self):
        assert WorkableSource({}).search(JobQuery()) == []

    def _cfg(self):
        return {"companies": [{"account": "acme", "name": "Acme Ltd"}],
                "require_south_africa": True}

    def _search(self, monkeypatch, cfg, entries):
        def fake_fetch(url, params=None):
            return {"jobs": entries}

        monkeypatch.setattr(workable_module, "fetch_json", fake_fetch)
        return WorkableSource(cfg).search(JobQuery())


# ------------------------------------------------- pipeline integration

def test_new_sources_pipeline_relevance_prunes(monkeypatch):
    """Each new source is wired through the pipeline's relevance stage."""
    region = {"sources": [
        {"name": "lever", "enabled": True, "require_south_africa": True,
         "companies": [{"site": "acme", "name": "Acme"}]},
        {"name": "recruitee", "enabled": True, "require_south_africa": True,
         "companies": [{"subdomain": "acme", "name": "Acme"}]},
        {"name": "workable", "enabled": True, "require_south_africa": True,
         "companies": [{"account": "acme", "name": "Acme"}]},
    ]}

    def fake_fetch_l(url, params):
        return [lever_entry(title="Nurse Practitioner",
                            description_plain="Patient care in gauteng.")]

    def fake_fetch_r(url, params=None):
        return {"offers": [recruitee_entry(
            title="Junior Accountant",
            description="<p>Pastel and vat bookkeeping in cape town.</p>")]}

    def fake_fetch_w(url, params=None):
        return {"jobs": [workable_entry(
            title="Junior Bookkeeper",
            description="Bookkeeping reconciliations in east london.")]}

    monkeypatch.setattr(lever_module, "fetch", fake_fetch_l)
    monkeypatch.setattr(recruitee_module, "fetch_json", fake_fetch_r)
    monkeypatch.setattr(workable_module, "fetch_json", fake_fetch_w)

    jobs, messages = search_jobs(parse_intent("accounting clerk jobs", {}), region)
    titles = {j.title for j in jobs}
    assert "Nurse Practitioner" not in titles
    assert "Junior Accountant" in titles
    assert "Junior Bookkeeper" in titles


def test_disabled_new_sources_no_http(monkeypatch):
    calls = []
    monkeypatch.setattr(lever_module, "fetch", lambda url, params: calls.append(url))
    monkeypatch.setattr(recruitee_module, "fetch_json",
                        lambda url, params=None: calls.append(url))
    monkeypatch.setattr(workable_module, "fetch_json",
                        lambda url, params=None: calls.append(url))
    region = {"sources": [
        {"name": "lever", "enabled": False, "companies": []},
        {"name": "recruitee", "enabled": False, "companies": []},
        {"name": "workable", "enabled": False, "companies": []},
    ]}
    jobs, messages = search_jobs(parse_intent("accounting jobs", {}), region)
    assert jobs == [] and calls == []
