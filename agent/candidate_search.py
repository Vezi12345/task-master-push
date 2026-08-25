"""Candidate-driven multi-query job search (search first, filter second).

Generates diverse queries FROM THE CANDIDATE'S PROFILE, runs them across
all configured sources via the existing orchestrator pipeline, pools the
results, de-duplicates, optionally expands with discovered related titles
(bounded), and hands back a ranked pool for strict candidate-side
evaluation.  No profession-specific logic lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.rank import RankedJob
from candidate.search_profile import (
    SearchProfile,
    build_search_profile,
    describe_strategy,
    generate_queries,
)

_TITLE_STRIP_WORDS = {
    "junior", "senior", "entry", "level", "graduate", "intern",
    "internship", "trainee", "assistant", "the", "and", "for", "in", "at",
}


@dataclass
class CandidateSearchResult:
    queries_used: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    strategy: SearchProfile = None
    ranked: list[RankedJob] = field(default_factory=list)
    duplicates_dropped: int = 0

    def summary_line(self) -> str:
        n = len(self.ranked)
        return (f"{len(self.queries_used)} queries -> {n} unique jobs "
                f"({self.duplicates_dropped} duplicates dropped) | "
                f"{describe_strategy(self.strategy)}")


def _job_key(job) -> str:
    return f"{job.title}|{job.company}|{job.url}".lower()


def _merge_ranked(pools: list[list[RankedJob]]) -> tuple[list[RankedJob], int]:
    best: dict[str, RankedJob] = {}
    dropped = 0
    for pool in pools:
        for item in pool:
            key = _job_key(item.job)
            if key in best:
                dropped += 1
                if item.score > best[key].score:
                    best[key] = item
            else:
                best[key] = item
    merged = sorted(best.values(), key=lambda i: i.score, reverse=True)
    return merged, dropped


def _expansion_candidates(ranked: list[RankedJob],
                          used_queries: list[str]) -> list[str]:
    """§10: mine high-scoring result titles for related search cores."""
    used_lower = {" ".join(q.lower().split()) for q in used_queries}
    cores: list[str] = []
    for item in ranked[:25]:
        if item.score < 55:
            break
        words = [w.strip(",:;-()") for w in item.job.title.split()]
        words = [w for w in words
                 if w.lower() not in _TITLE_STRIP_WORDS and len(w) > 1]
        if not 1 <= len(words) <= 4:
            continue
        core = " ".join(words).lower()
        if core in used_lower or core in cores:
            continue
        # only expand toward titles that look like real role names
        if any(ch.isdigit() for ch in core):
            continue
        cores.append(core)
    return cores[:2]


def summarize_related(jobs) -> list[dict]:
    """Bucket raw (pre-filter) listings into occupation labels so the UI can
    say what WAS found when the exact search matched nothing."""
    from candidate.occupations import bucket_for_job_title

    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for job in jobs:
        label = bucket_for_job_title(getattr(job, "title", ""))
        if label is None:
            continue
        counts[label] = counts.get(label, 0) + 1
        seen = samples.setdefault(label, [])
        if len(seen) < 3:
            seen.append(job.title)
    ranked_buckets = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"label": label, "count": n, "samples": samples[label]}
            for label, n in ranked_buckets[:6]]


def search_for_candidate(
    profile,
    region: dict = None,
    llm=None,
    query_text: str = None,
    max_queries: int = 6,
    max_jobs: int = 120,
    pipeline_fn=None,
) -> CandidateSearchResult:
    """Profile-driven broad discovery feeding the strict matching layer."""
    from agent import orchestrator as orch_module
    import config as cfg

    region = region if region is not None else cfg.load_region()
    pipeline_fn = pipeline_fn or (lambda q: orch_module.run_pipeline(
        q, region, llm))

    sp = build_search_profile(profile)
    queries = generate_queries(sp, max_queries=max_queries)

    # an explicit user query REPLACES the generated ones: when someone
    # types what they want, profile-derived guesses only re-crawl the same
    # source and flood the report with duplicate skips
    if query_text and query_text.strip():
        queries = [" ".join(query_text.strip().split())]

    pools: list[list[RankedJob]] = []
    used: list[str] = []
    for q in queries:
        try:
            result = pipeline_fn(q)
        except Exception:
            continue  # one failing source/query never kills the search
        pools.append(list(getattr(result, "ranked", []) or []))
        used.append(q)

    merged, dropped = _merge_ranked(pools)

    # bounded second wave: expand toward related titles actually seen
    expanded: list[str] = []
    if len(used) < max_queries and merged:
        for core in _expansion_candidates(merged, used):
            try:
                result = pipeline_fn(core)
            except Exception:
                continue
            expanded.append(core)
            pools.append(list(getattr(result, "ranked", []) or []))
            used.append(core)
            if len(used) >= max_queries:
                break
        merged, extra_dropped = _merge_ranked(pools)
        dropped += extra_dropped

    return CandidateSearchResult(
        queries_used=used,
        expanded_queries=expanded,
        strategy=sp,
        ranked=merged[:max_jobs],
        duplicates_dropped=dropped,
    )
