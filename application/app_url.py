from __future__ import annotations

"""Real application-page detection.

For a discovered job this module determines whether there is a CONCRETE
application mechanism (ATS apply page or an explicit apply link on the
employer site) and which platform hosts it.

A generic careers homepage is never accepted as proof of an application
mechanism. When no concrete mechanism can be established the caller gets a
result with ``found=False`` and a human-readable reason — never a guessed
URL.
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

from sources.base import Job


# ---------------------------------------------------------------------------
# platform fingerprinting
# ---------------------------------------------------------------------------

_PLATFORM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"boards\.greenhouse\.io|job-boards\.greenhouse\.io|greenhouse\.io", "greenhouse"),
    (r"jobs\.lever\.co|lever\.co", "lever"),
    (r"myworkdayjobs\.com|myworkdaysite\.com|\.myworkday\.com", "workday"),
    (r"(jobs|careers)\.smartrecruiters\.com|smartrecruiters\.com", "smartrecruiters"),
    (r"\.taleo\.net|taleo\.net", "taleo"),
    (r"\.icims\.com|icims\.com", "icims"),
    (r"jobs\.jobvite\.com|jobvite\.com", "jobvite"),
    (r"\.sapsf\.com|successfactors\.com", "successfactors"),
    (r"applytojob\.com", "applytojob"),
    (r"bamboohr\.com", "bamboohr"),
    (r"apply\.workable\.com|workable\.com", "workable"),
    (r"jobs\.personio\.(de|com)", "personio"),
    (r"recruitee\.com", "recruitee"),
    (r"breezy\.hr", "breezy_hr"),
    (r"recruiterbox\.com", "recruiterbox"),
)

_APPLY_PATH_RE = re.compile(
    r"/(apply[^a-z]|apply$|application|careers?/apply|jobs?/apply|e/apply)", re.IGNORECASE
)


def detect_platform(url: str) -> str:
    """Return the ATS/application platform hosting ``url`` ('' if unknown)."""
    lowered = (url or "").lower()
    if not lowered.startswith(("http://", "https://")):
        return ""
    for pattern, name in _PLATFORM_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return ""


def _looks_like_apply_path(url: str) -> bool:
    return bool(_APPLY_PATH_RE.search(url or ""))


# ---------------------------------------------------------------------------
# apply-link extraction from employer pages
# ---------------------------------------------------------------------------

class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []  # (href, text)
        self._current_href: Optional[str] = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_d = dict(attrs)
            self._current_href = attrs_d.get("href")
            self._current_text = []

    def handle_data(self, data):
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._current_href is not None:
            self.links.append((self._current_href, " ".join(self._current_text).strip()))
            self._current_href = None
            self._current_text = []


_APPLY_TEXT_RE = re.compile(r"\bapply\b|\bapply now\b|start your application", re.IGNORECASE)
_GENERIC_CAREERS_RE = re.compile(
    r"^https?://[^/]+/?$|/(careers?|jobs?|work-with-us|join-us)/?$", re.IGNORECASE
)


def _extract_links(page_html: str, base_url: str) -> list[tuple[str, str]]:
    collector = _AnchorCollector()
    try:
        collector.feed(page_html)
    except Exception:
        pass
    resolved: list[tuple[str, str]] = []
    for href, text in collector.links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        resolved.append((absolute, text))
    return resolved


@dataclass
class ApplicationTarget:
    """The concrete outcome of application-mechanism discovery."""

    found: bool = False
    application_url: str = ""
    platform: str = ""
    confidence: float = 0.0
    evidence: str = ""
    requires_user_action: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.found and not self.requires_user_action


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def find_application_url(
    job: Job,
    page_html: Optional[str] = None,
    fetch: Optional[object] = None,
) -> ApplicationTarget:
    """Locate the real application mechanism for ``job``.

    Order of evidence:
      1. the job URL itself is an ATS apply page,
      2. the job page HTML contains an ATS apply link,
      3. the job page HTML contains an explicit apply link to a concrete
         application path (not a generic careers homepage),
      4. DPSA-style circular vacancies advertise offline (Z83) processes,
      5. nothing found → honest failure with a reason.
    """
    # 1. direct ATS link
    platform = detect_platform(job.url)
    if platform:
        return ApplicationTarget(
            found=True,
            application_url=job.url,
            platform=platform,
            confidence=0.95,
            evidence="Job URL is hosted on the platform itself",
        )

    # 2./3. inspect the job page for apply links
    html = page_html
    if html is None and fetch is not None and job.url:
        try:
            html = fetch(job.url)  # type: ignore[misc]
        except Exception:
            html = None
    if html:
        links = _extract_links(html, job.url)
        ats_links = [
            (url, text) for url, text in links if detect_platform(url)
        ]
        if ats_links:
            url, text = ats_links[0]
            return ApplicationTarget(
                found=True,
                application_url=url,
                platform=detect_platform(url),
                confidence=0.9,
                evidence=f"Apply link points to {detect_platform(url)} ({text or 'link'})",
            )
        apply_links = [
            (url, text) for url, text in links
            if _looks_like_apply_path(url) or _APPLY_TEXT_RE.search(text)
        ]
        # reject generic careers homepages masquerading as apply links
        concrete = [
            (url, text) for url, text in apply_links
            if not _GENERIC_CAREERS_RE.match(url)
        ]
        if concrete:
            url, text = concrete[0]
            return ApplicationTarget(
                found=True,
                application_url=url,
                platform="custom" if not detect_platform(url) else detect_platform(url),
                confidence=0.7,
                evidence=f"Explicit apply link on employer page ({text or 'link'})",
            )

    # 4. DPSA circular vacancies use the official Z83 offline process
    if getattr(job, "source", "") == "dpsa_circular":
        text = f"{job.description}".lower()
        if "z83" in text or "public service vacancy circular" in text:
            return ApplicationTarget(
                found=False,
                requires_user_action=True,
                reason=(
                    "This DPSA circular post is applied for offline using the "
                    "official Z83 form (post/email/hand delivery as per the "
                    "circular instructions) — no online application form exists"
                ),
            )

    # 5. honest failure
    return ApplicationTarget(
        found=False,
        reason="No concrete online application mechanism was found for this job",
    )


def summarise_target(target: ApplicationTarget, job: Job) -> dict:
    """Storeable summary of application-mechanism discovery."""
    return {
        "job_url": job.url,
        "application_url": target.application_url,
        "source": job.source,
        "application_platform": target.platform,
        "employer": job.company,
        "job_title": job.title,
        "confidence": target.confidence,
        "evidence": target.evidence,
        "found": target.found,
        "requires_user_action": target.requires_user_action,
        "reason": target.reason,
    }
