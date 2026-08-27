"""Workable public jobs source.

Reads the unauthenticated Workable widget endpoint
(``https://www.workable.com/api/accounts/{account}?details=true``) which
returns published jobs as JSON. No API key or login is required. Each job
keeps its original application URL.
"""
from __future__ import annotations

import requests

from ._common import (
    capped,
    fetch_json,
    is_south_african,
)
from .base import ApplicationPlatformType, Job, JobSource, JobSourceError

ACCOUNTS_API = "https://www.workable.com/api/accounts/{account}"


class WorkableSource(JobSource):
    """Public Workable job boards (no API key required)."""

    name = "workable"

    def search(self, query) -> list[Job]:
        companies = self._companies()
        if not companies:
            return []
        require_sa = bool((self.config or {}).get("require_south_africa", True))

        jobs: list[Job] = []
        errors: list[str] = []
        for account, display_name in companies:
            try:
                entries = self._fetch_board(account)
            except requests.RequestException as exc:
                errors.append(f"{account}: {exc}")
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
                account = str(item.get("account") or item.get("token") or "").strip()
                name = str(item.get("name") or "").strip() or account
            else:
                account = str(item).strip()
                name = account
            if account:
                pairs.append((account, name))
        return pairs

    def _fetch_board(self, account: str) -> list[dict]:
        data = fetch_json(ACCOUNTS_API.format(account=account),
                          params={"details": "true"})
        jobs = data.get("jobs") if isinstance(data, dict) else None
        return jobs if isinstance(jobs, list) else []


def _to_job(entry: dict, company_name: str) -> Job | None:
    if not isinstance(entry, dict):
        return None
    title = str(entry.get("title") or "").strip()
    if not title:
        return None
    location = _extract_location(entry.get("location")) or _fallback_location(entry)
    url = str(entry.get("url") or entry.get("shortlink") or "").strip()
    description = capped(
        " ".join(x for x in (
            entry.get("description") or "",
            entry.get("full_description") or "",
        ) if x).strip()
    )
    remote = _is_remote(title, location, entry)
    job_id = str(entry.get("id") or entry.get("shortcode") or "").strip()
    return Job(
        title=title,
        company=company_name,
        location=location,
        remote=remote,
        description=description,
        url=url,
        source="workable",
        posted_date=str(entry.get("published_on") or entry.get("created_at") or "").strip() or None,
        platform=ApplicationPlatformType.GENERIC_WEB,
        id=f"workable-{job_id}" if job_id else "",
    )


def _extract_location(value) -> str:
    if isinstance(value, dict):
        direct = str(value.get("location_str") or "").strip()
        if direct:
            return direct
        city = str(value.get("city") or "").strip()
        region = str(value.get("state_code") or value.get("region") or "").strip()
        country = str(value.get("country_name") or value.get("country") or "").strip()
        return ", ".join(x for x in (city, region, country) if x)
    return str(value or "").strip()


def _fallback_location(entry: dict) -> str:
    """Compose a location label from Workable's flat fields."""
    parts = [
        str(entry.get("city") or "").strip(),
        str(entry.get("state") or "").strip(),
        str(entry.get("country") or "").strip(),
    ]
    label = ", ".join(x for x in parts if x)
    if label:
        return label
    # Fall back to the structured locations array (e.g. [{country: ...}]).
    locs = entry.get("locations")
    if isinstance(locs, list):
        bits = []
        for loc in locs:
            if isinstance(loc, dict):
                bits.append(_extract_location(loc))
            elif loc:
                bits.append(str(loc))
        label = " | ".join(x for x in bits if x)
    return label


def _is_remote(title: str, location: str, entry: dict) -> bool:
    remote = str(entry.get("remote") or "").strip().lower()
    if remote in ("true", "1", "yes"):
        return True
    telecommuting = entry.get("telecommuting")
    if telecommuting is True or str(telecommuting).strip().lower() == "true":
        return True
    workplace = str(entry.get("workplace_type") or "").strip().lower()
    if workplace == "remote":
        return True
    return "remote" in f"{title} {location}".lower()
