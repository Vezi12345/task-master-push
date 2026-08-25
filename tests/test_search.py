import config
from agent.parse_intent import parse_intent
from agent.search import dedupe_jobs, search_jobs
from sources.demo import DEMO_JOBS, DemoSource


def test_demo_source_returns_all_bundled_jobs():
    jobs = DemoSource().search(None)
    assert len(jobs) == len(DEMO_JOBS)
    assert all(job.source == "demo" for job in jobs)


def test_search_uses_enabled_sources_only(monkeypatch):
    region = config.load_region("za")
    enabled = {s["name"] for s in region["sources"] if s.get("enabled")}
    assert enabled == {"demo", "dpsa_circular"}
    from sources.dpsa_circular import DpsaCircularSource

    monkeypatch.setattr(DpsaCircularSource, "search", lambda self, query: [])
    query = parse_intent("entry-level software engineering jobs", region)
    jobs, messages = search_jobs(query, region)
    assert any("demo" in m for m in messages)
    assert any("skipped (not enabled)" in m for m in messages)
    assert len(jobs) == len(DEMO_JOBS)


def test_dedupe_removes_duplicates():
    jobs = DemoSource().search(None) * 2
    unique = dedupe_jobs(jobs)
    assert len(unique) == len(DEMO_JOBS)


def test_job_id_is_stable():
    a = DemoSource().search(None)[0]
    b = DemoSource().search(None)[0]
    assert a.id == b.id
    assert len(a.id) == 16
