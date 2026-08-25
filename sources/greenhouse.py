from __future__ import annotations

import html
import re
from typing import Any

import requests

from .base import Job, JobSource, JobSourceError

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

HTTP_TIMEOUT = 30
BOARDS_API = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
MAX_DESCRIPTION_CHARS = 4000

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


class GreenhouseSource(JobSource):
    """Public Greenhouse job boards (boards-api.greenhouse.io).

    No API key is required. The board roster lives in the region config as
    ``companies`` (verified tokens only); each advert keeps its original
    employer URL so every application goes to the real vacancy page.
    """

    name = "greenhouse"

    def search(self, query) -> list[Job]:
        """Return every South African vacancy on the configured boards.

        Like the other sources, this ignores ``query`` on purpose: role
        matching is owned centrally by ``agent.relevance``, which understands
        occupation families far better than keyword overlap would.
        """
        companies = self._companies()
        if not companies:
            return []

        require_sa = bool((self.config or {}).get("require_south_africa", True))

        jobs: list[Job] = []
        errors: list[str] = []
        for token, display_name in companies:
            try:
                entries = self._fetch_board(token)
            except requests.RequestException as exc:
                errors.append(f"{token}: {exc}")
                continue
            for entry in entries:
                job = _to_job(entry, token, display_name)
                if job is None:
                    continue
                if require_sa and not _is_south_african(job):
                    continue
                jobs.append(job)

        if errors and not jobs:
            raise JobSourceError("; ".join(errors))
        return jobs

    def _companies(self) -> list[tuple[str, str]]:
        raw = (self.config or {}).get("companies") or []
        pairs: list[tuple[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                token = str(item.get("token") or "").strip()
                name = str(item.get("name") or "").strip() or token
            else:
                token = str(item).strip()
                name = token
            if token:
                pairs.append((token, name))
        return pairs

    def _fetch_board(self, token: str) -> list[dict]:
        resp = requests.get(
            BOARDS_API.format(token=token),
            params={"content": "true"},
            headers=HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        jobs = data.get("jobs") if isinstance(data, dict) else None
        return jobs if isinstance(jobs, list) else []


def _to_job(entry: dict, board_token: str, company_name: str) -> Job | None:
    if not isinstance(entry, dict):
        return None
    title = str(entry.get("title") or "").strip()
    if not title:
        return None
    location_node = entry.get("location") if isinstance(entry.get("location"), dict) else {}
    location = str(location_node.get("name") or "").strip()
    url = str(entry.get("absolute_url") or "").strip()
    description = _strip_html(entry.get("content"))[:MAX_DESCRIPTION_CHARS]
    remote = "remote" in f"{title} {location}".lower()
    job_id = str(entry.get("id") or "").strip()
    return Job(
        title=title,
        company=company_name,
        location=location,
        remote=remote,
        description=description,
        url=url,
        source="greenhouse",
        posted_date=str(entry.get("updated_at") or "").strip() or None,
        id=f"{board_token}-{job_id}" if job_id else "",
    )


def _strip_html(raw: Any) -> str:
    if not raw:
        return ""
    text = str(raw)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _is_south_african(job: Job) -> bool:
    haystack = f"{job.title} {job.location} {job.description[:1500]}".lower()
    return any(marker in haystack for marker in SA_MARKERS)
