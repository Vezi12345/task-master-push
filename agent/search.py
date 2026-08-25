from __future__ import annotations

from sources.base import Job, JobSource, JobSourceError
from sources.demo import DemoSource
from sources.dpsa_circular import DpsaCircularSource
from sources.schemaorg import SchemaOrgSource

from .parse_intent import JobQuery

SOURCE_REGISTRY: dict[str, type[JobSource]] = {
    "demo": DemoSource,
    "dpsa_circular": DpsaCircularSource,
    "schemaorg": SchemaOrgSource,
}


def search_jobs(query: JobQuery, region: dict) -> tuple[list[Job], list[str]]:
    messages: list[str] = []
    jobs: list[Job] = []
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
            messages.append(f"{source.name}: {len(found)} jobs")
        except JobSourceError as exc:
            messages.append(f"{source_config['name']}: {exc}")
        except Exception as exc:
            messages.append(f"{source_config['name']}: unexpected error ({exc})")
    return dedupe_jobs(jobs), messages


def dedupe_jobs(jobs: list[Job]) -> list[Job]:
    seen: set[str] = set()
    unique: list[Job] = []
    for job in jobs:
        key = Job.make_id(job.title, job.company, job.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique
