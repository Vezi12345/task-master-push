"""Tests for the bounded career-site crawler source."""
from __future__ import annotations

import pytest
import requests

import sources.crawl as crawl_module
from agent.parse_intent import JobQuery
from agent.search import SOURCE_REGISTRY, search_jobs
from sources.base import JobSourceError
from sources.crawl import CrawlSource


class FakeResp:
    def __init__(self, raw, status=200, is_xml=True):
        self._raw = raw
        self.status_code = status
        self.ok = status < 400
        self.is_xml = is_xml

    @property
    def content(self):
        return self._raw if isinstance(self._raw, bytes) else self._raw.encode("utf-8")

    @property
    def text(self):
        if isinstance(self._raw, bytes):
            return self._raw.decode("utf-8")
        return self._raw

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")


SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://acme.co.za/careers/junior-accountant</loc></url>
  <url><loc>https://acme.co.za/careers/senior-finance</loc></url>
  <url><loc>https://acme.co.za/about</loc></url>
</urlset>
"""


def posting_html(title, location="Cape Town, South Africa", url=""):
    return f"""<html><head>
<script type="application/ld+json">
{{
  "@context": "https://schema.org",
  "@type": "JobPosting",
  "title": "{title}",
  "hiringOrganization": {{"name": "Acme (Pty) Ltd"}},
  "jobLocation": {{
     "@type": "Place",
     "address": {{
        "@type": "PostalAddress",
        "addressLocality": "{location.split(',')[0]}",
        "addressCountry": "ZA"
     }}
  }},
  "description": "<p>Bookkeeping and pastel reconciliations.</p>",
  "url": "{url}"
}}
</script>
</head><body></body></html>"""


def install(monkeypatch, sitemap_resp=FakeResp(SITEMAP), posting_html_by_url=None,
            raise_for=None):
    """Dispatch sitemap/postings by URL. ``raise_for`` maps url -> status."""
    posting_html_by_url = posting_html_by_url or {}

    def fake_get(url, params=None, headers=None, timeout=None):
        for bad, status in (raise_for or {}).items():
            if bad in url:
                return FakeResp("", status=status)
        if ".xml" in url or "sitemap" in url:
            return sitemap_resp
        if url in posting_html_by_url:
            return FakeResp(posting_html_by_url[url], is_xml=False)
        # Unmapped posting URLs: an empty page with no JobPosting markup.
        return FakeResp("<html><body></body></html>", is_xml=False)

    monkeypatch.setattr(crawl_module.requests, "get", fake_get)
    return fake_get


def cfg(seed):
    return {"careers": [seed], "require_south_africa": True}


def seed(url="https://acme.co.za/careers", name="Acme Ltd", **kw):
    d = {"url": url, "name": name, "polite_delay": 0}
    d.update(kw)
    return d


def test_registered():
    assert "crawl" in SOURCE_REGISTRY


def test_maps_sitemap_postings_to_jobs(monkeypatch):
    html = {
        "https://acme.co.za/careers/junior-accountant":
            posting_html("Junior Accountant"),
    }
    install(monkeypatch, posting_html_by_url=html)
    jobs = CrawlSource(cfg(seed())).search(JobQuery())
    assert len(jobs) == 1
    job = jobs[0]
    assert job.title == "Junior Accountant"
    assert job.company == "Acme Ltd", "company name from seed wins"
    assert job.source == "crawl"
    assert not job.remote


def test_remote_and_multiple_postings(monkeypatch):
    html = {
        "https://acme.co.za/careers/junior-accountant":
            posting_html("Junior Accountant (Remote)", location="Durban, South Africa"),
        "https://acme.co.za/careers/senior-finance":
            posting_html("Senior Finance Manager"),
    }
    install(monkeypatch, posting_html_by_url=html)
    jobs = CrawlSource(cfg(seed())).search(JobQuery())
    assert len(jobs) == 2
    by_title = {j.title: j for j in jobs}
    assert by_title["Junior Accountant (Remote)"].remote is True


def test_sa_filter_drops_overseas(monkeypatch):
    install(monkeypatch, posting_html_by_url={
        "https://acme.co.za/careers/junior-accountant":
            posting_html("Junior Accountant", location="London, UK"),
    })
    assert CrawlSource(cfg(seed())).search(JobQuery()) == []


def test_sa_filter_disabled(monkeypatch):
    install(monkeypatch, posting_html_by_url={
        "https://acme.co.za/careers/junior-accountant":
            posting_html("Junior Accountant", location="London, UK"),
    })
    c = cfg(seed())
    c["require_south_africa"] = False
    jobs = CrawlSource(c).search(JobQuery())
    assert len(jobs) == 1


def test_no_seeds_returns_empty():
    assert CrawlSource({"careers": []}).search(JobQuery()) == []
    assert CrawlSource({}).search(JobQuery()) == []


def test_all_sitemaps_failing_raises(monkeypatch):
    install(monkeypatch, sitemap_resp=FakeResp("", status=500))
    with pytest.raises(JobSourceError):
        CrawlSource(cfg(seed())).search(JobQuery())


def test_disabled_crawl_skipped_in_pipeline(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResp("", status=404)

    monkeypatch.setattr(crawl_module.requests, "get", fake_get)
    region = {"sources": [{"name": "crawl", "enabled": False, "careers": [
        seed()]}]}
    jobs, messages = search_jobs(JobQuery(), region)
    assert jobs == [] and calls == []
    assert any("skipped (not enabled)" in m for m in messages)
