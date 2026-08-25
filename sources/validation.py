from __future__ import annotations

import re
from datetime import date
from typing import Optional
from urllib.parse import urlparse

from .base import Job

MIN_TITLE_LEN = 3
MIN_COMPANY_LEN = 2
MIN_DESCRIPTION_LEN = 30

_FAKE_MARKERS = (
    "lorem ipsum",
    "sample job",
    "demo job",
    "example job",
    "placeholder",
    "test job",
    "dummy job",
    "mock job",
    "acme corp",
    "acme corporation",
    "example company",
    "sample company",
    "test company",
    "fake company",
    "your company",
    "company name here",
)

_FAKE_HOSTS = (
    "example.com",
    "example.org",
    "example.net",
    "test.com",
    "localhost",
    "127.0.0.1",
    "invalid",
)

_GENERIC_URL_FRAGMENTS = (
    "/careers",
    "/career",
    "/jobs",
    "/job-search",
    "/about",
    "/company",
)


def validate_real_job(job: Job) -> tuple[Optional[Job], str]:
    """Validate that a job looks like a genuine online vacancy.

    Returns ``(job, "")`` when the job passes every check, or ``(None,
    rejection_reason)`` when it must be discarded instead of displayed.
    """
    title = (job.title or "").strip()
    company = (job.company or "").strip()

    if len(title) < MIN_TITLE_LEN:
        return None, "missing or too-short title"
    if len(company) < MIN_COMPANY_LEN:
        return None, "missing or too-short company"

    url = (job.url or "").strip()
    parsed = urlparse(url)
    if not url or parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None, f"invalid source URL: {url!r}"

    host = parsed.netloc.lower()
    if any(host == bad or host.endswith("." + bad) for bad in _FAKE_HOSTS):
        return None, f"non-genuine source host: {host}"

    description = (job.description or "").strip()
    if len(description) < MIN_DESCRIPTION_LEN:
        return None, "missing or too-short description"

    haystack = f"{title} {company} {description[:500]}".lower()
    for marker in _FAKE_MARKERS:
        if marker in haystack:
            return None, f"fake/demo marker detected: {marker!r}"

    return job, ""


def filter_real_jobs(jobs: list[Job]) -> tuple[list[Job], list[str]]:
    """Run the full validation pipeline over retrieved jobs.

    Every job must come from a real source with a verifiable URL, title,
    company and description. Invalid records are discarded — never repaired
    with fabricated data.
    """
    valid: list[Job] = []
    rejections: list[str] = []
    today = date.today().isoformat()
    for job in jobs:
        checked, reason = validate_real_job(job)
        if checked is None:
            rejections.append(f"{job.title or '<untitled>'}: {reason}")
            continue
        if not checked.date_found:
            checked.date_found = today
        valid.append(checked)
    return valid, rejections


def is_generic_source_url(url: str) -> bool:
    """True when the URL points at a general careers/jobs page rather than a
    specific vacancy. Such jobs are still real, but the UI must present the
    link as the *source page*, not as the specific vacancy."""
    path = urlparse(url).path.lower().rstrip("/")
    if not path:
        return True
    return any(path.endswith(fragment) for fragment in _GENERIC_URL_FRAGMENTS)


_WORD_RE = re.compile(r"[a-z0-9#.+]+")


def normalise_title_key(title: str) -> str:
    words = _WORD_RE.findall(title.lower())
    stop = {"the", "a", "an", "and", "of", "for", "in", "to"}
    return " ".join(w for w in words if w not in stop)
