from __future__ import annotations

"""Offline search-quality evaluation runner.

Builds a deterministic corpus from the bundled sources (the in-repo demo
jobs plus the three DPSA circular fixtures parsed offline), then runs the
existing ``parse_intent`` + ``rank_jobs`` pipeline against every query in
:mod:`evaluation.dataset` and reports:

  * intent-field accuracy (roles / seniority / locations / remote /
    min_salary / skills / keywords),
  * precision@3 and precision@10,
  * recall@10,
  * mean reciprocal rank (MRR),
  * hard-constraint violation rate (only ``remote: required`` and known
    ``min_salary`` shortfalls count; location is intentionally soft).

    This module never touches the network, never calls ``search_jobs`` with the
live region config, and never invokes an LLM.
"""

from collections import defaultdict

from agent.parse_intent import parse_intent
from agent.rank import RankedJob, rank_jobs
from agent.search import dedupe_jobs
from config import load_region
from sources.demo import DemoSource
from sources.dpsa_circular import parse_circular

from .dataset import QUERIES

FIXTURES = {
    "dpsa_circular": "tests/fixtures/dpsa_circular.txt",
    "dpsa_circular_malformed": "tests/fixtures/dpsa_circular_malformed.txt",
    "dpsa_circular_annexure": "tests/fixtures/dpsa_circular_annexure.txt",
}

FIELDS = ("roles", "seniority", "locations", "remote", "min_salary", "skills", "keywords")


def build_corpus() -> list:
    from pathlib import Path

    region = load_region("za")
    corpus = DemoSource(region).search(None)
    for source, path in FIXTURES.items():
        text = Path(path).read_text(encoding="utf-8")
        corpus.extend(parse_circular(text, source_url=None, default_company=None))
    return dedupe_jobs(corpus)


def build_index(corpus: list) -> dict:
    index: dict[tuple[str, str], object] = {}
    for job in corpus:
        index[(job.source, job.title)] = job
    return index


def _field_correct(kind: str, expected, extracted) -> bool:
    if kind == "roles":
        return _list_ok(expected, extracted)
    if kind == "locations":
        return _list_ok(expected, extracted)
    if kind in ("skills", "keywords"):
        return _list_ok(expected, extracted)
    if kind in ("seniority", "remote", "min_salary"):
        return extracted == expected
    raise ValueError(kind)


def _list_ok(expected: list, extracted: list) -> bool:
    if not expected:
        return not extracted
    return all(item in extracted for item in expected)


def _extract_readable(query) -> dict:
    return {
        "roles": list(query.roles),
        "seniority": query.seniority,
        "locations": [loc.city for loc in query.locations],
        "remote": query.remote,
        "min_salary": query.min_salary,
        "skills": list(query.skills),
        "keywords": list(query.keywords),
    }


def _check_hard_constraints(query, ranked: list[RankedJob], hard: dict) -> tuple[int, list[str]]:
    violations: list[str] = []
    top = ranked[:10]
    if hard.get("remote") == "required":
        for item in top:
            if not item.job.remote:
                violations.append(f"{item.job.company}: {item.job.title} is not remote")
    if hard.get("min_salary") is not None:
        for item in top:
            if item.job.salary_min is not None and item.job.salary_min < hard["min_salary"]:
                violations.append(
                    f"{item.job.company}: {item.job.title} pays R{item.job.salary_min:,} "
                    f"< R{hard['min_salary']:,}"
                )
    return len(violations), violations


def _precision(ranked: list[RankedJob], gold: list, k: int) -> float:
    denom = min(k, len(ranked))
    if denom == 0:
        return 1.0
    top_keys = {(item.job.source, item.job.title) for item in ranked[:k]}
    hits = sum(1 for key in gold if key in top_keys)
    return hits / denom


def _recall(ranked: list[RankedJob], gold: list) -> float:
    if not gold:
        return None
    ranked_keys = {(item.job.source, item.job.title) for item in ranked[:10]}
    hits = sum(1 for key in gold if key in ranked_keys)
    return hits / len(gold)


def _mrr(ranked: list[RankedJob], gold: list) -> float:
    if not gold:
        return None
    gold_keys = set(gold)
    for i, item in enumerate(ranked, start=1):
        if (item.job.source, item.job.title) in gold_keys:
            return 1.0 / i
    return 0.0


def evaluate_query(entry: dict, region: dict, corpus: list, index: dict) -> dict:
    query = parse_intent(entry["query"], region)
    ranked = rank_jobs(corpus, query)
    expected = entry["expected"]
    extracted = _extract_readable(query)

    field_ok = {kind: _field_correct(kind, expected[kind], extracted[kind]) for kind in FIELDS}
    correct_fields = sum(field_ok.values())

    gold = entry["gold"]
    p3 = _precision(ranked, gold, 3)
    p10 = _precision(ranked, gold, 10)
    r10 = _recall(ranked, gold)
    mrr = _mrr(ranked, gold)

    n_violations = 0
    violations: list[str] = []
    if entry.get("hard"):
        n_violations, violations = _check_hard_constraints(query, ranked, entry["hard"])

    top3 = [f"{item.job.company}: {item.job.title}" for item in ranked[:3]]
    return {
        "query": entry["query"],
        "category": entry["category"],
        "gold": list(entry["gold"]),
        "intent_accuracy": correct_fields / len(FIELDS),
        "field_ok": field_ok,
        "expected": expected,
        "extracted": extracted,
        "precision_3": p3,
        "precision_10": p10,
        "recall_10": r10,
        "mrr": mrr,
        "n_violations": n_violations,
        "violations": violations,
        "top3": top3,
        "ranked_count": len(ranked),
    }


def run_evaluation() -> dict:
    region = load_region("za")
    corpus = build_corpus()
    index = build_index(corpus)
    rows = [evaluate_query(entry, region, corpus, index) for entry in QUERIES]

    intent_fields = sum(row["field_ok"][kind] for row in rows for kind in FIELDS)
    total_fields = len(rows) * len(FIELDS)

    def mean(rows_, key, skip_none=True):
        values = [row[key] for row in rows_ if row[key] is not None or not skip_none]
        if not values:
            return None
        return sum(values) / len(values)

    by_category: dict = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    categories = {}
    for cat, cat_rows in by_category.items():
        categories[cat] = {
            "queries": len(cat_rows),
            "intent_accuracy": mean(cat_rows, "intent_accuracy"),
            "precision_3": mean(cat_rows, "precision_3"),
            "precision_10": mean(cat_rows, "precision_10"),
            "recall_10": mean(cat_rows, "recall_10"),
            "mrr": mean(cat_rows, "mrr"),
            "violations": sum(r["n_violations"] for r in cat_rows),
        }

    return {
        "corpus_size": len(corpus),
        "num_queries": len(rows),
        "intent_accuracy": sum(r["intent_accuracy"] for r in rows) / len(rows),
        "intent_field_accuracy": intent_fields / total_fields,
        "precision_3": mean(rows, "precision_3"),
        "precision_10": mean(rows, "precision_10"),
        "recall_10": mean(rows, "recall_10"),
        "mrr": mean(rows, "mrr"),
        "violations": sum(r["n_violations"] for r in rows),
        "categories": categories,
        "rows": rows,
    }


def _fmt(value) -> str:
    if value is None:
        return "   n/a"
    return f"{value:.3f}"


def format_report(metrics: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("SEARCH QUALITY EVALUATION - OFFLINE BASELINE")
    lines.append("=" * 78)
    lines.append(f"Corpus size : {metrics['corpus_size']} jobs (10 demo + 12 DPSA fixtures)")
    lines.append(f"Queries     : {metrics['num_queries']}")
    lines.append("")
    lines.append("OVERALL METRICS")
    lines.append("-" * 78)
    lines.append(f"  Intent-field accuracy : {metrics['intent_field_accuracy']:.3f}")
    lines.append(f"  Intent accuracy/query : {metrics['intent_accuracy']:.3f}")
    lines.append(f"  Precision@3           : {_fmt(metrics['precision_3'])}")
    lines.append(f"  Precision@10          : {_fmt(metrics['precision_10'])}")
    lines.append(f"  Recall@10             : {_fmt(metrics['recall_10'])}")
    lines.append(f"  MRR                   : {_fmt(metrics['mrr'])}")
    lines.append(f"  Hard-constraint viol.  : {metrics['violations']}")
    lines.append("")
    lines.append("BY CATEGORY")
    lines.append("-" * 78)
    header = f"  {'category':<12} {'queries':>7} {'intent':>7} {'p@3':>7} {'p@10':>7} {'r@10':>7} {'mrr':>7} {'viol':>5}"
    lines.append(header)
    for cat, c in sorted(metrics["categories"].items()):
        lines.append(
            f"  {cat:<12} {c['queries']:>7} {c['intent_accuracy']:.3f} "
            f"{_fmt(c['precision_3'])} {_fmt(c['precision_10'])} "
            f"{_fmt(c['recall_10'])} {_fmt(c['mrr'])} {c['violations']:>5}"
        )
    lines.append("")

    failed = [r for r in metrics["rows"] if r["intent_accuracy"] < 1.0 or r["precision_3"] < 0.5 or r["mrr"] == 0 or r["n_violations"]]
    lines.append("FAILED QUERIES (intent<1.0, p@3<0.5, MRR=0, or violations)")
    lines.append("-" * 78)
    if not failed:
        lines.append("  (none)")
    for row in failed:
        lines.append(f"  [{row['category']}] {row['query']}")
        lines.append(f"      intent={row['intent_accuracy']:.3f} p@3={row['precision_3']:.3f} "
                     f"p@10={row['precision_10']:.3f} r@10={_fmt(row['recall_10'])} mrr={_fmt(row['mrr'])} "
                     f"viol={row['n_violations']} ranked={row['ranked_count']}")
        bad_fields = [k for k in FIELDS if not row["field_ok"][k]]
        for kind in bad_fields:
            lines.append(f"      ~ {kind}: expected {row['expected'][kind]!r} "
                         f"got {row['extracted'][kind]!r}")
        if row["n_violations"]:
            for v in row["violations"]:
                lines.append(f"      ! violation: {v}")
        if row["mrr"] == 0 and row["recall_10"] not in (0.0, None):
            lines.append("      ? gold partially retrieved but never first-ranked")
    lines.append("")

    good = sorted(metrics["rows"], key=lambda r: (r["intent_accuracy"] + r["precision_3"] + (r["mrr"] or 0)), reverse=True)
    lines.append("BEST 3 QUERIES")
    lines.append("-" * 78)
    for row in good[:3]:
        lines.append(f"  [{row['category']}] {row['query']}")
        lines.append(f"      intent={row['intent_accuracy']:.3f} p@3={row['precision_3']:.3f} mrr={_fmt(row['mrr'])}")
        lines.append(f"      top3: {', '.join(row['top3']) or '(none)'}")
    lines.append("")
    worst = failed or good[-3:]
    lines.append("WORST 3 QUERIES")
    lines.append("-" * 78)
    for row in worst[-3:]:
        lines.append(f"  [{row['category']}] {row['query']}")
        lines.append(f"      intent={row['intent_accuracy']:.3f} p@3={row['precision_3']:.3f} mrr={_fmt(row['mrr'])}")
        lines.append(f"      top3: {', '.join(row['top3']) or '(none)'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    metrics = run_evaluation()
    print(format_report(metrics))


if __name__ == "__main__":
    main()
