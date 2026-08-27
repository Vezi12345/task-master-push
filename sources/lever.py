"""Lever public postings source.

Reads the unauthenticated Lever Postings API
(``https://api.lever.co/v0/postings/{site}?mode=json``). No API key or login
is required. Every vacancy keeps its original ``hostedUrl`` so applications
go to the real employer posting.
"""
from __future__ import annotations

from typing import Any

import requests

from ._common import (
    HEADERS,
    HTTP_TIMEOUT,
    capped,
    is_south_african,
    strip_html,
)
from .base import ApplicationPlatformType, Job, JobSource, JobSourceError

GLOBAL_API = "https://api.lever.co/v0/postings/{site}"
EU_API = "https://api.eu.lever.co/v0/postings/{site}"


class LeverSource(JobSource):
    """Public Lever job boards (api.lever.co postings API, no key)."""

    name = "lever"

    def search(self, query) -> list[Job]:
        companies = self._companies()
        if not companies:
            return []
        require_sa = bool((self.config or {}).get("require_south_africa", True))

        jobs: list[Job] = []
        errors: list[str] = []
        for site, display_name in companies:
            try:
                entries = self._fetch_site(site)
            except requests.RequestException as exc:
                errors.append(f"{site}: {exc}")
                continue
            for entry in entries:
                job = _to_job(entry, display_name)
                if job is None:
                    continue
                if require_sa and not is_south_african(job.title, job.location,
                                                        job.description):
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
                site = str(item.get("site") or item.get("token") or "").strip()
                name = str(item.get("name") or "").strip() or site
            else:
                site = str(item).strip()
                name = site
            if site:
                pairs.append((site, name))
        return pairs

    def _fetch_site(self, site: str) -> list[dict]:
        url = GLOBAL_API.format(site=site)
        try:
            data = fetch(url, {"mode": "json", "limit": 100})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                data = fetch(EU_API.format(site=site), {"mode": "json", "limit": 100})
            else:
                raise
        if isinstance(data, list):
            return data
        return []


def fetch(url: str, params: dict) -> list:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _to_job(entry: dict, company_name: str) -> Job | None:
    if not isinstance(entry, dict):
        return None
    title = str(entry.get("text") or "").strip()
    if not title:
        return None
    categories = entry.get("categories") if isinstance(entry.get("categories"), dict) else {}
    location = _location_string(categories)
    url = str(entry.get("hostedUrl") or entry.get("applyUrl") or "").strip()
    description = _build_description(entry)
    workplace = str(entry.get("workplaceType") or "").lower()
    remote = workplace == "remote" or "remote" in f"{title} {location}".lower()
    salary_min, salary_max, salary_text = _extract_salary(entry.get("salaryRange"))
    job_id = str(entry.get("id") or "").strip()
    return Job(
        title=title,
        company=company_name,
        location=location,
        remote=remote,
        description=description,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_text=salary_text,
        url=url,
        source="lever",
        posted_date=_format_timestamp(entry.get("createdAt")),
        platform=ApplicationPlatformType.LEVER,
        id=f"lever-{job_id}" if job_id else "",
    )


def _location_string(categories: dict) -> str:
    primary = str(categories.get("location") or "").strip()
    extras = categories.get("allLocations") or []
    parts = [primary]
    if isinstance(extras, list):
        for loc in extras:
            loc = str(loc).strip()
            parts.append(loc)
    return _clean_locations(parts)


def _clean_locations(parts: list[str]) -> str:
    seen: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    return " | ".join(seen)


def _build_description(entry: dict) -> str:
    plain = str(entry.get("descriptionPlain")
                 or entry.get("descriptionBodyPlain")
                 or "").strip()
    if not plain:
        plain = strip_html(entry.get("description"))
    lists_text = []
    for item in entry.get("lists") or []:
        if isinstance(item, dict):
            head = str(item.get("text") or "").strip()
            content = strip_html(item.get("content"))
            if content:
                lists_text.append(f"{head}: {content}" if head else content)
    body = "\n\n".join([p for p in [plain, "\n\n".join(lists_text)] if p])
    return capped(body)


def _extract_salary(value: Any) -> tuple[int | None, int | None, str | None]:
    if not isinstance(value, dict):
        return None, None, None
    currency = str(value.get("currency") or "").strip()
    lo = _to_int(value.get("min"))
    hi = _to_int(value.get("max"))
    if lo is None and hi is None:
        return None, None, None
    text = f"{currency} {lo if lo is not None else ''}-{hi if hi is not None else ''}".strip(" -")
    return lo, hi, text


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    # Lever returns a millisecond epoch in the postings API.
    if isinstance(value, (int, float)):
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            value / 1000.0, tz=_dt.timezone.utc
        ).strftime("%Y-%m-%d")
    return str(value)
