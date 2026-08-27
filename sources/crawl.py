"""Bounded career-site crawler source.

Scans the sitemaps of configured employer career sites, discovers job-posting
URLs, then fetches each page and extracts the embedded schema.org
``JobPosting`` JSON-LD (reusing the parser from ``SchemaOrgSource``).

This is deliberately *bounded*: a bounded sitemap hop (root ``sitemap.xml``
plus any sitemap-index children), a hard cap on how many postings we fetch,
anonymised polite delays, and strictly over the configured seed URLs. It
relies on the employer publishing a sitemap that points at its vacancy pages.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

from ._common import (
    HEADERS,
    HTTP_TIMEOUT,
    is_south_african,
)
from .base import Job, JobSource, JobSourceError
from .schemaorg import _iter_postings, _to_job as _schemaorg_to_job

JOB_URL_HINTS = ("job", "career", "vacanc", "position", "opening", "opportunit")
DEFAULT_POLITE_DELAY = 0.4
MAX_SITEMAP_HOPS = 2
MAX_POSTINGS = 50
MAX_SITEMAP_LOCS = 2000


class CrawlSource(JobSource):
    """Fetch vacancies from employer career-site sitemaps."""

    name = "crawl"

    def search(self, query) -> list[Job]:
        seeds = self._companies()
        if not seeds:
            return []
        require_sa = bool((self.config or {}).get("require_south_africa", True))

        jobs: list[Job] = []
        last_err: str | None = None
        for seed in seeds:
            delay = float(seed.get("polite_delay", DEFAULT_POLITE_DELAY))
            try:
                posting_urls = self._discover(seed)
            except requests.RequestException as exc:
                last_err = f"{seed['url']}: {exc}"
                continue
            for url in posting_urls:
                time.sleep(delay)
                try:
                    posting = self._fetch_posting(url)
                except requests.RequestException as exc:
                    last_err = f"{url}: {exc}"
                    continue
                if not posting:
                    continue
                job = _schemaorg_to_job(posting)
                if job is None:
                    continue
                job.company = seed.get("name") or job.company
                job.source = "crawl"
                if require_sa and not is_south_african(job.title, job.location,
                                                       job.description):
                    continue
                jobs.append(job)

        if last_err is not None and not jobs:
            raise JobSourceError(last_err)
        return jobs

    def _companies(self) -> list[dict]:
        raw = (self.config or {}).get("careers") or []
        seeds = []
        for item in raw:
            if isinstance(item, str):
                item = {"url": item}
            if isinstance(item, dict) and item.get("url"):
                seeds.append(item)
        return seeds

    def _discover(self, seed: dict) -> list[str]:
        """Return the job-posting URLs reachable from a career seed."""
        site_url = seed["url"]
        root_url = _origin(site_url)
        sitemaps = []
        explicit = seed.get("sitemap")
        if explicit:
            sitemaps = [urllib.parse.urljoin(site_url, explicit)]
        else:
            sitemaps = [f"{root_url}/sitemap.xml"]

        posting_urls: list[str] = []
        collected_maps: set[str] = set()

        def collect(sitemap_url: str, depth: int = 0) -> None:
            nonlocal collected_maps
            if depth > MAX_SITEMAP_HOPS or sitemap_url in collected_maps:
                return
            if len(posting_urls) > MAX_POSTINGS:
                return
            collected_maps.add(sitemap_url)
            locs = self._fetch_sitemap_locs(sitemap_url)
            child_maps = [u for u in locs if _is_sitemap_url(u)]
            for u in locs:
                if _is_sitemap_url(u):
                    continue
                if _is_posting_url(u):
                    posting_urls.append(u)
                if len(posting_urls) > MAX_POSTINGS:
                    return
            time.sleep(float(seed.get("polite_delay", DEFAULT_POLITE_DELAY)))
            for child in child_maps:
                if len(posting_urls) > MAX_POSTINGS:
                    return
                time.sleep(float(seed.get("polite_delay", DEFAULT_POLITE_DELAY)))
                collect(child, depth + 1)

        for sm in sitemaps:
            collect(sm)

        # Hard cap by posting count.
        return posting_urls[:MAX_POSTINGS]

    def _fetch_sitemap_locs(self, sitemap_url: str) -> list[str]:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return _extract_locs(resp.content, sitemap_url)

    def _fetch_posting(self, url: str) -> dict:
        resp = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
            except json.JSONDecodeError:
                continue
            for posting in _iter_postings(data):
                if not posting.get("url"):
                    posting["url"] = url
                return posting
        return {}


def _origin(site_url: str) -> str:
    parsed = urllib.parse.urlparse(site_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_posting_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(hint in lowered for hint in JOB_URL_HINTS) \
        and not lowered.endswith((".css", ".js", ".png", ".jpg", ".jpeg",
                                  ".gif", ".svg", ".webp", ".ico", ".pdf"))


def _is_sitemap_url(url: str) -> bool:
    lowered = (url or "").lower()
    return ("sitemap" in lowered or lowered.endswith(".xml")) \
        and not _is_posting_url(lowered)


def _extract_locs(content: bytes, base_url: str) -> list[str]:
    text = content.decode("utf-8", errors="replace")
    if "<urlset" not in text and "<sitemapindex" not in text:
        return []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    locs = []
    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text and elem.text.strip():
            locs.append(urllib.parse.urljoin(base_url, elem.text.strip()))
    return locs[:MAX_SITEMAP_LOCS]
