from __future__ import annotations

from dataclasses import dataclass, field

from agent.parse_intent import JobQuery
from agent.rank import RankedJob, rank_jobs
from agent.search import search_jobs
from sources.base import Job


@dataclass
class PipelineResult:
    query: JobQuery
    search_messages: list[str] = field(default_factory=list)
    jobs_found: list[Job] = field(default_factory=list)
    ranked: list[RankedJob] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_pipeline(prompt: str, region: dict, llm=None) -> PipelineResult:
    query = _parse(prompt, region, llm)
    jobs, messages = search_jobs(query, region)
    ranked = rank_jobs(jobs, query, llm)
    notes: list[str] = []
    if llm is not None and not llm.is_available():
        notes.append("Ollama is offline — used built-in rules for intent parsing.")
    return PipelineResult(query=query, search_messages=messages, jobs_found=jobs, ranked=ranked, notes=notes)


def _parse(prompt: str, region: dict, llm=None) -> JobQuery:
    from agent.parse_intent import parse_intent

    return parse_intent(prompt, region, llm)
