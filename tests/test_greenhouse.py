"""Regression tests for the second real job source: public Greenhouse boards.

Every fake advert uses a genuine-looking employer URL so records survive
``sources.validation`` exactly like live traffic would.
"""
from __future__ import annotations

import pytest
import requests

import sources.greenhouse as greenhouse_module
from agent.orchestrator import run_pipeline
from agent.parse_intent import JobQuery, parse_intent
from agent.search import SOURCE_REGISTRY, search_jobs
from sources.base import ApplicationPlatformType, JobSourceError
from sources.greenhouse import GreenhouseSource

BOARD_URL = "https://job-boards.greenhouse.io/acme/jobs/123"


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def entry(title="Junior Accountant", jid="123", location="Cape Town, South Africa",
          content="<p>We need an accountant with pastel and vat experience.</p>",
          url=BOARD_URL, updated="2026-08-20T10:00:00Z"):
    e = {"title": title, "id": jid, "location": {"name": location},
         "content": content, "absolute_url": url}
    if updated is not None:
        e["updated_at"] = updated
    return e


def board_payload(*entries):
    return {"jobs": list(entries)}


def install(monkeypatch, responses):
    """responses maps board token -> FakeResp; returns the call log."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        for token, resp in responses.items():
            if f"/boards/{token}/jobs" in url:
                calls.append((token, url))
                return resp
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(greenhouse_module.requests, "get", fake_get)
    return calls


def source_config(**overrides):
    cfg = {"companies": [{"token": "acme", "name": "Acme Ltd"}],
           "require_south_africa": True}
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------- mapping

def test_01_maps_greenhouse_fields_to_job(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(entry()))})
    jobs = GreenhouseSource(source_config()).search(JobQuery())
    assert len(jobs) == 1
    job = jobs[0]
    assert job.title == "Junior Accountant"
    assert job.company == "Acme Ltd"
    assert job.location == "Cape Town, South Africa"
    assert job.url == BOARD_URL
    assert job.source == "greenhouse"
    assert job.posted_date == "2026-08-20T10:00:00Z"
    assert job.id == "acme-123"
    assert job.platform is ApplicationPlatformType.GREENHOUSE


def test_02_strips_html_from_description(monkeypatch):
    html_body = ("<div><p>Bookkeep&nbsp;for <b>us</b>.</p>"
                 "<script>evil()</script><style>x{}</style></div>")
    install(monkeypatch, {"acme": FakeResp(board_payload(entry(content=html_body)))})
    job = GreenhouseSource(source_config()).search(JobQuery())[0]
    assert "Bookkeep for us ." in job.description or "Bookkeep&nbsp;" not in job.description
    assert "<" not in job.description and "evil()" not in job.description


def test_03_caps_description_length(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(content="<p>" + ("word " * 2000) + "</p>")))})
    job = GreenhouseSource(source_config()).search(JobQuery())[0]
    assert len(job.description) <= 4000


def test_04_remote_flag_detected_from_title(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(title="Support Engineer (100% Remote)", location="South Africa")))})
    assert GreenhouseSource(source_config()).search(JobQuery())[0].remote is True


def test_05_missing_title_is_dropped_not_crashed(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(title="", jid="9"), entry(title="Real Role", jid="10")))})
    jobs = GreenhouseSource(source_config()).search(JobQuery())
    assert [j.title for j in jobs] == ["Real Role"]


# ------------------------------------------------------- SA + query filters

def test_06_requires_south_africa_by_default(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(location="London, United Kingdom",
              content="<p>A completely british role in london.</p>")))})
    assert GreenhouseSource(source_config()).search(JobQuery()) == []


def test_07_sa_marker_anywhere_keeps_the_job(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(location="Remote - Global",
              content="<p>You may work remotely from Johannesburg.</p>")))})
    jobs = GreenhouseSource(source_config()).search(JobQuery())
    assert len(jobs) == 1


def test_08_require_south_africa_can_be_disabled(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(location="London, United Kingdom",
              content="<p>A completely british role in london.</p>")))})
    jobs = GreenhouseSource(source_config(require_south_africa=False)).search(JobQuery())
    assert len(jobs) == 1


def test_09_offtopic_adverts_pruned_by_relevance_stage(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(title="Nurse Practitioner", jid="1",
              content="<p>Patient care and ward duties in gauteng.</p>"),
        entry(title="Junior Accountant", jid="2",
              content="<p>Pastel and vat bookkeeping in cape town.</p>")))})
    # The source itself is query-agnostic (like dpsa_circular); the pipeline's
    # relevance stage must drop the nurse advert for an accounting search.
    jobs, messages = search_jobs(parse_intent("accounting clerk jobs", {}),
                                 _region())
    assert [j.title for j in jobs] == ["Junior Accountant"]


def test_10_empty_query_returns_all_sa_jobs(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(jid="1"), entry(jid="2", title="Payroll Clerk")))})
    jobs = GreenhouseSource(source_config()).search(JobQuery())
    assert len(jobs) == 2


# ------------------------------------------------------------- resilience

def test_11_one_failed_board_does_not_kill_the_source(monkeypatch):
    install(monkeypatch, {
        "acme": FakeResp(status=404),
        "beta": FakeResp(board_payload(entry())),
    })
    cfg = source_config()
    cfg["companies"] = [{"token": "acme", "name": "Acme Ltd"},
                        {"token": "beta", "name": "Beta (Pty) Ltd"}]
    jobs = GreenhouseSource(cfg).search(JobQuery())
    assert len(jobs) == 1 and jobs[0].company == "Beta (Pty) Ltd"


def test_12_all_boards_failing_raises_source_error(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(status=500)})
    with pytest.raises(JobSourceError):
        GreenhouseSource(source_config()).search(JobQuery())


def test_13_no_companies_configured_returns_empty_list():
    assert GreenhouseSource({"companies": []}).search(JobQuery()) == []
    assert GreenhouseSource({}).search(JobQuery()) == []


def test_14_plain_string_company_entries_accepted(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(entry()))})
    jobs = GreenhouseSource({"companies": ["acme"]}).search(JobQuery())
    assert jobs[0].company == "acme"


# ------------------------------------------------- pipeline integration

def _region(enabled=True):
    return {"sources": [{"name": "greenhouse", "enabled": enabled,
                         "require_south_africa": True,
                         "companies": [{"token": "acme", "name": "Acme Ltd"}]}]}


def test_15_registry_end_to_end_dedupe_and_stats(monkeypatch):
    assert "greenhouse" in SOURCE_REGISTRY
    # acme repeats one vacancy (identical record); beta contributes its own.
    # Funnel: 3 discovered -> duplicate removed -> 2 kept, per-source stats.
    install(monkeypatch, {
        "acme": FakeResp(board_payload(entry(), entry())),
        "beta": FakeResp(board_payload(entry(
            title="Finance Intern", jid="555",
            content="<p>Rotate through finance operations in durban.</p>",
            url="https://job-boards.greenhouse.io/beta/jobs/555"))),
    })
    region = dict(_region())
    region["sources"][0]["companies"].append({"token": "beta", "name": "Beta"})
    stats: dict = {}
    jobs, messages = search_jobs(parse_intent("accounting jobs", {}), region, stats)
    assert len(jobs) == 2
    assert any("greenhouse: 3 jobs" in m for m in messages)
    assert stats == {"greenhouse": {"discovered": 3, "kept": 2}}


def test_16_disabled_greenhouse_is_skipped_without_http_calls(monkeypatch):
    calls = install(monkeypatch, {"acme": FakeResp(board_payload(entry()))})
    jobs, messages = search_jobs(parse_intent("accounting jobs", {}),
                                 _region(enabled=False))
    assert jobs == [] and calls == []
    assert any("skipped (not enabled)" in m for m in messages)


def test_17_run_pipeline_exposes_source_stats(monkeypatch):
    install(monkeypatch, {"acme": FakeResp(board_payload(
        entry(title="Accounts Clerk", jid="7",
              content="<p>Data entry, filing and pastel reconciliations in durban.</p>")))})
    result = run_pipeline("accounts clerk jobs in durban", _region(), llm=None)
    assert result.query.roles, "sanity: intent parsed"
    assert result.jobs_found and result.ranked
    assert result.source_stats.get("greenhouse", {}).get("discovered") == 1
    assert result.source_stats["greenhouse"]["kept"] >= 1
