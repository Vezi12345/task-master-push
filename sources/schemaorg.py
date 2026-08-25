from __future__ import annotations

import json
from typing import Any, Iterator

import requests
from bs4 import BeautifulSoup

from .base import Job, JobSource, JobSourceError

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}


class SchemaOrgSource(JobSource):
    name = "schemaorg"

    def search(self, query) -> list[Job]:
        url = (self.config or {}).get("search_url")
        if not url:
            return []

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise JobSourceError(f"could not fetch search page: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        jobs: list[Job] = []
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
            except json.JSONDecodeError:
                continue
            for posting in _iter_postings(data):
                job = _to_job(posting)
                if job:
                    jobs.append(job)
        return jobs


def _iter_postings(data: Any) -> Iterator[dict]:
    if isinstance(data, list):
        for item in data:
            yield from _iter_postings(item)
    elif isinstance(data, dict):
        if _is_job_posting(data):
            yield data
        for value in data.values():
            yield from _iter_postings(value)


def _is_job_posting(data: dict) -> bool:
    graph = data.get("@graph")
    if graph:
        return False
    if "@type" not in data:
        return False
    types = data["@type"]
    if isinstance(types, list):
        return "JobPosting" in types
    return types == "JobPosting"


def _to_job(posting: dict) -> Job | None:
    title = _string(posting.get("title") or posting.get("name"))
    if not title:
        return None
    org = posting.get("hiringOrganization") or {}
    company = _string(org.get("name")) or "Unknown"
    location = _extract_location(posting.get("jobLocation"))
    description = _string(posting.get("description")) or ""
    salary_min, salary_max, salary_text = _extract_salary(posting.get("baseSalary"))
    remote = _is_remote(title, location, description)
    return Job(
        title=title,
        company=company,
        location=location,
        remote=remote,
        description=description,
        salary_min=salary_min,
        salary_max=salary_max,
        salary_text=salary_text,
        url=_string(posting.get("url")) or "",
        source="schemaorg",
        posted_date=_string(posting.get("datePosted")),
    )


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_location(value: Any) -> str:
    if not isinstance(value, list):
        value = [value]
    parts: list[str] = []
    for node in value:
        if not isinstance(node, dict):
            continue
        if "@type" in node and node["@type"] != "Place":
            continue
        address = node.get("address") or {}
        if isinstance(address, dict):
            locality = _string(address.get("addressLocality"))
            region = _string(address.get("addressRegion"))
            country = _string(address.get("addressCountry"))
            if isinstance(country, dict):
                country = _string(country.get("name"))
            bit = ", ".join(x for x in (locality, region, country) if x)
        else:
            bit = _string(address)
        if bit:
            parts.append(bit)
    return "; ".join(parts) or _string(posting_location_text(value))


def posting_location_text(nodes: list) -> str:
    return ""


def _extract_salary(value: Any) -> tuple[int | None, int | None, str | None]:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None, None, None
    value_node = value.get("value")
    number: Any = None
    if isinstance(value_node, dict):
        number = value_node.get("value")
    else:
        number = value_node
    currency = _string(value.get("currency"))
    text = f"{currency} {number}".strip() if number is not None else None
    try:
        number = int(number) if number is not None else None
    except (TypeError, ValueError):
        number = None
    return number, number, text


def _is_remote(title: str, location: str, description: str) -> bool:
    text = f"{title} {location} {description}".lower()
    return "remote" in text
