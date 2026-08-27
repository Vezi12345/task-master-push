"""Shared helpers for public ATS job-board sources.

Kept deliberately free of network calls so it is trivially unit-testable.
"""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

import requests

MAX_DESCRIPTION_CHARS = 4000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

HTTP_TIMEOUT = 30

from application.pacing import PolitenessPolicy  # noqa: E402

# Shared adaptive politeness policy for all public job-board fetches.
PACER = PolitenessPolicy(
    min_interval=0.25,
    initial_backoff=1.0,
    backoff_factor=2.0,
    max_backoff=30.0,
)

SA_MARKERS = (
    "south africa",
    "johannesburg",
    "cape town",
    "durban",
    "pretoria",
    "gauteng",
    "western cape",
    "kwazulu-natal",
    "east london",
)


def strip_html(raw) -> str:
    """Return the plain-text content of an HTML fragment."""
    if raw is None:
        return ""
    text = str(raw)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def is_south_african(title: str, location: str, description: str) -> bool:
    haystack = f"{title} {location} {description[:1500]}".lower()
    return any(marker in haystack for marker in SA_MARKERS)


def capped(text: str) -> str:
    return text[:MAX_DESCRIPTION_CHARS]


def fetch_json(url: str, *, params: dict | None = None) -> dict | list:
    """GET a JSON response; raises requests.RequestException on failure.

    Routes through the shared :data:`PACER` politeness policy: waits for the
    per-host interval, then records success/failure to drive adaptive backoff
    when a host throttles or errors."""
    netloc = urlparse(url).netloc
    PACER.wait_for(netloc)
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError:
        PACER.reported_failure(netloc)
        raise
    except requests.RequestException:
        PACER.reported_failure(netloc)
        raise
    PACER.reported_success(netloc)
    return resp.json()
