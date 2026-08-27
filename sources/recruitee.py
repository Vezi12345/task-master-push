"""Recruitee public offers source.

Reads the public careers endpoint
(``https://{subdomain}.recruitee.com/api/offers/``), which returns published
offers as JSON without any API key. Each offer keeps its original application
URL.
"""
from __future__ import annotations

import requests

from ._common import (
    capped,
    fetch_json,
    is_south_african,
    strip_html,
)
from .base import ApplicationPlatformType, Job, JobSource, JobSourceError

OFFERS_API = "https://{subdomain}.recruitee.com/api/offers/"


class RecruiteeSource(JobSource):
    """Public Recruitee career sites (no API key required)."""

    name = "recruitee"

    def search(self, query) -> list[Job]:
        companies = self._companies()
        if not companies:
            return []
        require_sa = bool((self.config or {}).get("require_south_africa", True))

        jobs: list[Job] = []
        errors: list[str] = []
        for subdomain, display_name in companies:
            try:
                offers = self._fetch_board(subdomain)
            except requests.RequestException as exc:
                errors.append(f"{subdomain}: {exc}")
                continue
            for offer in offers:
                job = _to_job(offer, display_name)
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
                sub = str(item.get("subdomain") or item.get("token") or "").strip()
                name = str(item.get("name") or "").strip() or sub
            else:
                sub = str(item).strip()
                name = sub
            if sub:
                pairs.append((sub, name))
        return pairs

    def _fetch_board(self, subdomain: str) -> list[dict]:
        data = fetch_json(OFFERS_API.format(subdomain=subdomain))
        offers = data.get("offers") if isinstance(data, dict) else None
        return offers if isinstance(offers, list) else []


def _to_job(offer: dict, company_name: str) -> Job | None:
    if not isinstance(offer, dict):
        return None
    title = str(offer.get("title") or "").strip()
    if not title:
        return None
    location = str(offer.get("location") or "").strip()
    if not location:
        location = ", ".join(x for x in (
            str(offer.get("city") or "").strip(),
            str(offer.get("country") or "").strip(),
        ) if x)
    url = str(offer.get("careers_url") or offer.get("url") or "").strip()
    description = capped(strip_html(
        offer.get("description") or offer.get("requirements") or ""
    ))
    remote = bool(offer.get("remote")) or bool(offer.get("hybrid")) or \
        "remote" in f"{title} {location}".lower()
    job_id = str(offer.get("id") or "").strip()
    return Job(
        title=title,
        company=company_name,
        location=location,
        remote=remote,
        description=description,
        url=url,
        source="recruitee",
        posted_date=str(offer.get("published_at") or "").strip() or None,
        platform=ApplicationPlatformType.GENERIC_WEB,
        id=f"recruitee-{job_id}" if job_id else "",
    )
