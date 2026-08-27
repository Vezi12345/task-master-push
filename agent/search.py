from __future__ import annotations

from collections import Counter

from sources.base import Job, JobSource, JobSourceError
from sources.crawl import CrawlSource
from sources.dpsa_circular import DpsaCircularSource
from sources.greenhouse import GreenhouseSource
from sources.lever import LeverSource
from sources.recruitee import RecruiteeSource
from sources.schemaorg import SchemaOrgSource
from sources.validation import (
    filter_real_jobs,
    normalise_title_key,
)
from sources.workable import WorkableSource

from .parse_intent import JobQuery
from .relevance import filter_relevant_jobs

SOURCE_REGISTRY: dict[str, type[JobSource]] = {
    "dpsa_circular": DpsaCircularSource,
    "schemaorg": SchemaOrgSource,
    "greenhouse": GreenhouseSource,
    "lever": LeverSource,
    "recruitee": RecruiteeSource,
    "workable": WorkableSource,
    "crawl": CrawlSource,
}


def search_jobs(query: JobQuery, region: dict, stats: dict | None = None) -> tuple[list[Job], list[str]]:
    """Search every enabled source and return surviving jobs plus messages.

    When ``stats`` is a dict it is filled with per-source funnel counts:
    ``{source_name: {"discovered": n, "kept": m}}`` where *discovered* is
    what the source returned and *kept* is how many of those records
    survived validation, relevance filtering and deduplication.
    """
    messages: list[str] = []
    jobs: list[Job] = []
    discovered: Counter[str] = Counter()
    for source_config in region.get("sources", []):
        if not source_config.get("enabled"):
            messages.append(f"{source_config.get('name', '?')}: skipped (not enabled)")
            continue
        source_class = SOURCE_REGISTRY.get(source_config["name"])
        if source_class is None:
            messages.append(f"{source_config['name']}: unknown source type")
            continue
        try:
            source = source_class(source_config)
            found = source.search(query)
            jobs.extend(found)
            discovered[source.name] += len(found)
            messages.append(f"{source.name}: {len(found)} jobs")
        except JobSourceError as exc:
            messages.append(f"{source_config['name']}: {exc}")
        except Exception as exc:
            messages.append(f"{source_config['name']}: unexpected error ({exc})")

    # Stage 1: is each record a legitimate online vacancy?
    jobs, invalid = filter_real_jobs(jobs)
    messages.extend(_summarise_rejections(
        "invalid job record", invalid,
        lambda reason: reason,
    ))

    # Stage 2: does the vacancy's role actually match the search?
    jobs, irrelevant = filter_relevant_jobs(jobs, query)
    messages.extend(_summarise_rejections(
        "not role-relevant", [reason for _, reason in irrelevant],
        lambda reason: reason,
    ))

    final_jobs = dedupe_jobs(jobs)
    if stats is not None:
        kept = Counter(job.source for job in final_jobs)
        stats.update(
            {
                name: {"discovered": count, "kept": kept.get(name, 0)}
                for name, count in discovered.items()
            }
        )
    return final_jobs, messages


def _summarise_rejections(label: str, reasons: list[str], key_of=None) -> list[str]:
    """Compact per-stage rejection summary instead of one line per record."""
    if not reasons:
        return []
    counts = Counter(reasons)
    lines = []
    remaining = sum(counts.values())
    for reason, count in counts.most_common(3):
        lines.append(f"{label}: {count}x {reason}")
        remaining -= count
    if remaining > 0:
        lines.append(f"{label}: {remaining}x other reasons")
    return lines


def dedupe_jobs(jobs: list[Job]) -> list[Job]:
    """Remove duplicate vacancies.

    A URL alone does not identify a vacancy: circular documents (e.g. the
    DPSA PDF) legitimately carry many distinct vacancies under one URL. URLs
    are therefore only used for deduplication when they uniquely identify a
    single role; shared documents are deduplicated by (title, company).
    """
    roles_per_url: dict[str, set[tuple[str, str]]] = {}
    for job in jobs:
        url_key = (job.url or "").strip().lower()
        if url_key:
            roles_per_url.setdefault(url_key, set()).add(
                (normalise_title_key(job.title), job.company.strip().lower())
            )
    shared_urls = {u for u, roles in roles_per_url.items() if len(roles) > 1}

    seen_urls: set[str] = set()
    seen_role_keys: set[tuple[str, str]] = set()
    unique: list[Job] = []
    for job in jobs:
        url_key = (job.url or "").strip().lower()
        role_key = (normalise_title_key(job.title), job.company.strip().lower())
        if role_key in seen_role_keys:
            continue
        if url_key and url_key not in shared_urls and url_key in seen_urls:
            continue
        if url_key:
            seen_urls.add(url_key)
        seen_role_keys.add(role_key)
        unique.append(job)
    return unique
