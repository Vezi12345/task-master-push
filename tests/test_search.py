import config
from agent.parse_intent import parse_intent
from agent.search import SOURCE_REGISTRY, dedupe_jobs, search_jobs
from evaluation.fixtures import FIXTURE_JOBS, load_fixture_jobs


def test_demo_source_is_not_registered():
    """The live pipeline must never be able to serve demo/mock jobs."""
    assert "demo" not in SOURCE_REGISTRY
    assert all("demo" not in name for name in SOURCE_REGISTRY)


def test_search_uses_enabled_sources_only(monkeypatch):
    region = config.load_region("za")
    enabled = {s["name"] for s in region["sources"] if s.get("enabled")}
    assert "demo" not in enabled
    assert enabled == {"dpsa_circular"}
    from sources.dpsa_circular import DpsaCircularSource

    monkeypatch.setattr(DpsaCircularSource, "search", lambda self, query: [])
    query = parse_intent("entry-level software engineering jobs", region)
    jobs, messages = search_jobs(query, region)
    assert any("skipped (not enabled)" in m for m in messages)
    assert jobs == []


def test_failing_source_does_not_abort_search(monkeypatch):
    from sources.base import JobSourceError
    from sources.dpsa_circular import DpsaCircularSource

    def _boom(self, query):
        raise JobSourceError("could not download circular: boom")

    monkeypatch.setattr(DpsaCircularSource, "search", _boom)
    region = config.load_region("za")
    query = parse_intent("entry-level software engineering jobs", region)
    jobs, messages = search_jobs(query, region)
    assert jobs == []
    assert any(
        "dpsa_circular" in m and "could not download circular" in m for m in messages
    )
    assert any("schemaorg" in m and "skipped (not enabled)" in m for m in messages)


def test_dedupe_removes_duplicates():
    jobs = load_fixture_jobs() * 2
    unique = dedupe_jobs(jobs)
    assert len(unique) == len(FIXTURE_JOBS)


def test_job_id_is_stable():
    a = load_fixture_jobs()[0]
    b = load_fixture_jobs()[0]
    assert a.id == b.id
    assert len(a.id) == 16
