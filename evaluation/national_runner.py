from __future__ import annotations

"""National-scale offline search-quality evaluation runner.

Builds a large, deterministic, offline corpus from:
  1. the DPSA public-service vacancy circulars stored under
     ``evaluation/corpus/dpsa/dpsa_jobs.json`` (raw extraction) -- de-duplicated
     by ``(POST reference, department)`` keeping the most recent advertisement,
  2. the ten in-repo demo jobs (labelled ``source="demo"``, private-sector
     fixtures used only to exercise the tiny non-public slice available).

Then, for every query in :mod:`evaluation.national_dataset`:

  * parses intent with the existing rule-based ``parse_intent``,
  * ranks with the existing ``rank_jobs`` (no LLM, no network),
  * computes gold sets from the deterministic relevance spec (see the dataset
    module docstring) rather than hand-picked rows,
  * reports intent-field accuracy, P@3, P@10, R@10, MRR, NDCG@10,
    hard-constraint violations, zero-result rate, duplicate-result rate, and a
    location-precision diagnostic,
  * compares against a naive keyword baseline so the structured ranker's
    value (or lack of it) is measured, not assumed.

This module never touches the network at evaluation time.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from agent.parse_intent import KEYWORD_STOPWORDS, parse_intent
from agent.rank import RankedJob, rank_jobs
from agent.search import dedupe_jobs
from config import load_region
from sources.base import Job
from sources.demo import DemoSource

from .national_dataset import QUERIES

CORPUS_JSON = Path(__file__).parent / "corpus" / "dpsa" / "dpsa_jobs.json"
REPORTS_DIR = Path(__file__).parent / "reports"

FIELDS = ("roles", "seniority", "locations", "remote", "min_salary", "skills", "keywords")

# gold title markers (independent of the ranker's seniority heuristics)
ENTRY_TITLE = ["junior", "graduate", "intern", "internship", "trainee", "learner",
               "student", "youth", "cadet", "entry", "grade 1"]
SENIOR_TITLE = ["senior", "principal", "chief", "executive", "deputy director",
                "director", "head", "manager"]


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def load_dpsa_jobs() -> list[Job]:
    data = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    return [Job(**d) for d in data]


NOTICE_MARKERS = ("cancelled", "please note", "abridged")


def _is_notice(job: Job) -> bool:
    t = (job.title or "").lower()
    return any(m in t for m in NOTICE_MARKERS)


def build_national_corpus() -> list[Job]:
    """Drop notice/cancellation rows, then dedup by (POST ref, department)
    keeping the most recent occurrence."""
    jobs = [j for j in load_dpsa_jobs() if not _is_notice(j)]
    seen: dict[tuple, Job] = {}
    for job in jobs:
        key = (job.id, job.company) if job.id else (job.title, job.company, job.url)
        seen[key] = job
    region = load_region("za")
    demo = DemoSource(region).search(None)
    return list(seen.values()) + demo


# ---------------------------------------------------------------------------
# relevance gold (documented deterministic rules, independent of the ranker)
# ---------------------------------------------------------------------------

def term_in(term: str, text: str) -> bool:
    if "\\b" in term:
        return re.search(term, text, re.IGNORECASE) is not None
    return term.lower() in (text or "").lower()


def is_relevant(job: Job, rel: dict) -> bool:
    if not rel:
        return True
    title = (job.title or "").lower()
    desc = (job.description or "").lower()
    blob = f"{title} {desc}"

    if rel.get("any") and not any(term_in(t, title) for t in rel["any"]):
        return False
    if rel.get("any_broad") and not any(term_in(t, blob) for t in rel["any_broad"]):
        return False
    if rel.get("must") and not all(term_in(t, blob) for t in rel["must"]):
        return False

    seniority = rel.get("seniority") or ""
    if seniority == "entry-level":
        if not any(term_in(t, title) for t in ENTRY_TITLE):
            return False
        if any(term_in(t, title) for t in SENIOR_TITLE):
            return False
    elif seniority == "senior":
        if not any(term_in(t, title) for t in SENIOR_TITLE):
            return False

    locations = rel.get("locations") or []
    if locations and job.location:
        lowered = job.location.lower()
        if not any(city.lower() in lowered for city in locations):
            return False

    min_salary = rel.get("min_salary")
    if min_salary is not None:
        if job.salary_min is None or job.salary_min < min_salary:
            return False

    remote = rel.get("remote", "any")
    if remote == "required" and not job.remote:
        return False
    if remote == "no" and job.remote:
        return False
    return True


def compute_gold(corpus: list[Job], rel: dict) -> set[int]:
    return {id(job) for job in corpus if is_relevant(job, rel)}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _list_ok(expected: list, extracted: list) -> bool:
    if not expected:
        return not extracted
    return all(item in extracted for item in expected)


def _field_correct(kind: str, expected, extracted) -> bool:
    if kind in ("roles", "locations", "skills", "keywords"):
        return _list_ok(expected, extracted)
    return extracted == expected


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


def _precision(ranked, gold, k):
    denom = min(k, len(ranked))
    if denom == 0:
        return 1.0
    top = {id(it.job) for it in ranked[:k]}
    hits = sum(1 for g in gold if g in top)
    return hits / denom


def _recall(ranked, gold):
    if not gold:
        return None
    top = {id(it.job) for it in ranked[:10]}
    hits = sum(1 for g in gold if g in top)
    return hits / len(gold)


def _mrr(ranked, gold):
    if not gold:
        return None
    for i, item in enumerate(ranked, start=1):
        if id(item.job) in gold:
            return 1.0 / i
    return 0.0


def _ndcg10(ranked, gold):
    if not gold:
        return None
    gains = [1.0 if id(it.job) in gold else 0.0 for it in ranked[:10]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    relevant = min(len(gold), 10)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(relevant))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _check_hard(query, ranked, hard):
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


def _dup_rate(ranked):
    keys = [(it.job.title.strip().lower(), it.job.location.strip().lower()) for it in ranked[:10]]
    seen, dups = set(), 0
    for k in keys:
        if k in seen:
            dups += 1
        seen.add(k)
    return dups / max(len(keys), 1)


def _loc_precision(ranked, cities):
    if not cities:
        return None
    stated = [it for it in ranked[:10] if it.job.location]
    if not stated:
        return None
    hits = sum(1 for it in stated if any(c.lower() in it.job.location.lower() for c in cities))
    return hits / len(stated)


# ---------------------------------------------------------------------------
# naive keyword baseline
# ---------------------------------------------------------------------------

def _baseline_rank(corpus: list[Job], query_text: str):
    tokens = [
        t for t in re.findall(r"[a-z]+", query_text.lower())
        if t not in KEYWORD_STOPWORDS and len(t) >= 3
    ]
    scored = []
    for job in corpus:
        title = (job.title or "").lower()
        blob = f"{title} {(job.description or '').lower()} {(job.location or '').lower()} {(job.company or '').lower()}"
        score = sum(3 if t in title else 1 for t in tokens if t in blob)
        scored.append((score, id(job), job))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [SimpleNamespace(job=job, score=score, reasons=[], summary="") for score, _, job in scored]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_query(entry, region, corpus):
    query = parse_intent(entry["query"], region)
    ranked = rank_jobs(corpus, query)
    expected = entry["expected"]
    extracted = _extract_readable(query)
    rel = entry.get("rel")

    gold = compute_gold(corpus, rel) if rel else set()

    field_ok = {kind: _field_correct(kind, expected[kind], extracted[kind]) for kind in FIELDS}
    correct_fields = sum(field_ok.values())

    p3 = _precision(ranked, gold, 3)
    p10 = _precision(ranked, gold, 10)
    r10 = _recall(ranked, gold)
    mrr = _mrr(ranked, gold)
    ndcg = _ndcg10(ranked, gold)

    n_violations, violations = 0, []
    if entry.get("hard"):
        n_violations, violations = _check_hard(query, ranked, entry["hard"])

    cities = [c for c in (rel or {}).get("locations", [])] if rel else []

    return {
        "query": entry["query"],
        "category": entry["category"],
        "rel": rel,
        "intent_accuracy": correct_fields / len(FIELDS),
        "field_ok": field_ok,
        "expected": expected,
        "extracted": extracted,
        "precision_3": p3,
        "precision_10": p10,
        "recall_10": r10,
        "mrr": mrr,
        "ndcg_10": ndcg,
        "n_violations": n_violations,
        "violations": violations,
        "gold_size": len(gold),
        "ranked_count": len(ranked),
        "zero_results": len(ranked) == 0,
        "dup_rate": _dup_rate(ranked),
        "loc_precision": _loc_precision(ranked, cities),
        "top3": [f"{item.job.company}: {item.job.title}" for item in ranked[:3]],
    }


def _baseline_metrics(rows, corpus):
    stop = set(KEYWORD_STOPWORDS)
    for row in rows:
        ranked = _baseline_rank(corpus, row["query"])
        gold = compute_gold(corpus, row["rel"]) if row["rel"] else set()
        p3 = _precision(ranked, gold, 3)
        p10 = _precision(ranked, gold, 10)
        r10 = _recall(ranked, gold)
        mrr = _mrr(ranked, gold)
        ndcg = _ndcg10(ranked, gold)
        row["base_p3"] = p3
        row["base_p10"] = p10
        row["base_r10"] = r10
        row["base_mrr"] = mrr
        row["base_ndcg"] = ndcg
    return rows


def _mean(rows, key, skip_none=True):
    values = [r[key] for r in rows if r[key] is not None or not skip_none]
    if not values:
        return None
    return sum(values) / len(values)


def corpus_stats() -> dict:
    raw = load_dpsa_jobs()
    notices = [j for j in raw if _is_notice(j)]
    clean = [j for j in raw if not _is_notice(j)]
    canonical = build_national_corpus()
    dpsa_canonical = canonical[:-10] if canonical[-1].source == "demo" else canonical
    re_adv = len(clean) - len(dpsa_canonical)

    with_n = lambda j, f: sum(1 for x in j if x)
    loc = Counter((j.location or "").strip().upper() for j in dpsa_canonical)
    comp = Counter((j.company or "").strip() for j in dpsa_canonical)
    sals = [j.salary_min for j in dpsa_canonical if j.salary_min]

    norm_groups: dict[str, list] = defaultdict(list)
    for j in dpsa_canonical:
        t = re.sub(r"^post\s+[\d/]+\s*:\s*", "", (j.title or "").strip().lower())
        norm_groups[(t, (j.location or "").strip().lower())].append(j)
    near_dup = sum(1 for k, v in norm_groups.items() if len(v) > 1)
    near_dup_extra = sum(len(v) - 1 for v in norm_groups.values() if len(v) > 1)

    return {
        "raw_jobs": len(raw),
        "notice_rows": len(notices),
        "canonical_jobs": len(dpsa_canonical),
        "demo_jobs": 10,
        "re_advertisements_removed": re_adv,
        "circulars": 54,
        "with_title": with_n(dpsa_canonical, lambda j: j.title),
        "with_department": with_n(dpsa_canonical, lambda j: j.company),
        "with_location": with_n(dpsa_canonical, lambda j: j.location),
        "with_salary_min": with_n(dpsa_canonical, lambda j: j.salary_min),
        "with_closing_date": with_n(dpsa_canonical, lambda j: j.posted_date),
        "with_description": with_n(dpsa_canonical, lambda j: j.description),
        "remote_jobs": sum(1 for j in dpsa_canonical if j.remote),
        "median_salary": sorted(sals)[len(sals) // 2] if sals else None,
        "salary_known_pct": round(100 * with_n(dpsa_canonical, lambda j: j.salary_min) / len(dpsa_canonical), 1),
        "top_locations": loc.most_common(10),
        "top_companies": comp.most_common(8),
        "near_duplicate_groups": near_dup,
        "near_duplicate_extra_rows": near_dup_extra,
    }


def run_national_evaluation() -> dict:
    region = load_region("za")
    corpus = build_national_corpus()
    rows = [evaluate_query(entry, region, corpus) for entry in QUERIES]
    rows = _baseline_metrics(rows, corpus)

    def mean(key, skip_none=True):
        return _mean(rows, key, skip_none)

    by_category: dict[str, list] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    categories = {}
    for cat, cat_rows in by_category.items():
        categories[cat] = {
            "queries": len(cat_rows),
            "intent_accuracy": _mean(cat_rows, "intent_accuracy"),
            "precision_3": _mean(cat_rows, "precision_3"),
            "precision_10": _mean(cat_rows, "precision_10"),
            "recall_10": _mean(cat_rows, "recall_10"),
            "mrr": _mean(cat_rows, "mrr"),
            "ndcg_10": _mean(cat_rows, "ndcg_10"),
            "base_p3": _mean(cat_rows, "base_p3"),
            "base_ndcg": _mean(cat_rows, "base_ndcg"),
            "violations": sum(r["n_violations"] for r in cat_rows),
            "zero_results": sum(1 for r in cat_rows if r["zero_results"]),
            "with_gold": sum(1 for r in cat_rows if r["gold_size"] > 0),
        }

    intent_fields = sum(row["field_ok"][k] for row in rows for k in FIELDS)
    field_failures = {k: sum(1 for row in rows if not row["field_ok"][k]) for k in FIELDS}

    with_gold = [r for r in rows if r["gold_size"] > 0]

    return {
        "corpus": corpus_stats(),
        "num_queries": len(rows),
        "queries_with_gold": len(with_gold),
        "intent_accuracy": sum(r["intent_accuracy"] for r in rows) / len(rows),
        "intent_field_accuracy": intent_fields / (len(rows) * len(FIELDS)),
        "field_failures": field_failures,
        "precision_3": mean("precision_3"),
        "precision_10": mean("precision_10"),
        "recall_10": mean("recall_10"),
        "mrr": mean("mrr"),
        "ndcg_10": mean("ndcg_10"),
        "base_p3": mean("base_p3"),
        "base_p10": mean("base_p10"),
        "base_r10": mean("base_r10"),
        "base_mrr": mean("base_mrr"),
        "base_ndcg": mean("base_ndcg"),
        "p3_with_gold": _mean(with_gold, "precision_3"),
        "r10_with_gold": _mean(with_gold, "recall_10"),
        "mrr_with_gold": _mean(with_gold, "mrr"),
        "ndcg_with_gold": _mean(with_gold, "ndcg_10"),
        "violations": sum(r["n_violations"] for r in rows),
        "zero_result_rate": sum(1 for r in rows if r["zero_results"]) / len(rows),
        "dup_result_rate": mean("dup_rate"),
        "loc_precision": mean("loc_precision"),
        "categories": categories,
        "rows": rows,
    }


def _fmt(v) -> str:
    if v is None:
        return "  n/a"
    return f"{v:.3f}"


def format_report(m: dict) -> str:
    lines: list[str] = []
    w = 78
    lines.append("=" * w)
    lines.append("NATIONAL SEARCH QUALITY EVALUATION - SA PUBLIC-SERVICE CORPUS")
    lines.append("=" * w)
    c = m["corpus"]
    lines.append(f"Raw postings parsed        : {c['raw_jobs']} (54 DPSA circulars, 2025-2026; notice rows: {c['notice_rows']})")
    lines.append(f"Canonical jobs (deduped)   : {c['canonical_jobs']} (re-adverts removed: {c['re_advertisements_removed']})")
    lines.append(f"Demo fixture jobs (private): {c['demo_jobs']}")
    lines.append(f"Coverage: salary {c['salary_known_pct']}%, location {c['with_location']}, closing {c['with_closing_date']}, remote {c['remote_jobs']}")
    lines.append(f"Median monthly salary      : R{c['median_salary']:,}")
    lines.append(f"Near-duplicate title+centre groups: {c['near_duplicate_groups']} ({c['near_duplicate_extra_rows']} extra rows)")
    lines.append("")
    lines.append("QUERIES")
    lines.append("-" * w)
    lines.append(f"Queries                    : {m['num_queries']}")
    lines.append("")
    lines.append("OVERALL METRICS (structured ranker)")
    lines.append("-" * w)
    lines.append(f"  Intent-field accuracy : {m['intent_field_accuracy']:.3f}")
    lines.append(f"  Intent accuracy/query : {m['intent_accuracy']:.3f}")
    lines.append(f"  Precision@3           : {_fmt(m['precision_3'])}")
    lines.append(f"  Precision@10          : {_fmt(m['precision_10'])}")
    lines.append(f"  Recall@10             : {_fmt(m['recall_10'])}")
    lines.append(f"  MRR                   : {_fmt(m['mrr'])}")
    lines.append(f"  NDCG@10               : {_fmt(m['ndcg_10'])}")
    lines.append(f"  Hard-constraint viol.  : {m['violations']}")
    lines.append(f"  Zero-result rate      : {m['zero_result_rate']:.3f}")
    lines.append(f"  Duplicate-result rate : {m['dup_result_rate']:.3f}")
    lines.append(f"  Location precision    : {_fmt(m['loc_precision'])} (among stated-location top-10)")
    lines.append(f"  Queries with gold     : {m['queries_with_gold']} / {m['num_queries']}")
    lines.append(f"  P@3 / R@10 (gold>0)   : {_fmt(m['p3_with_gold'])} / {_fmt(m['r10_with_gold'])}")
    lines.append(f"  MRR / NDCG (gold>0)   : {_fmt(m['mrr_with_gold'])} / {_fmt(m['ndcg_with_gold'])}")
    lines.append("")
    lines.append("NAIVE KEYWORD BASELINE")
    lines.append("-" * w)
    lines.append(f"  Base P@3 / P@10       : {_fmt(m['base_p3'])} / {_fmt(m['base_p10'])}")
    lines.append(f"  Base R@10 / MRR       : {_fmt(m['base_r10'])} / {_fmt(m['base_mrr'])}")
    lines.append(f"  Base NDCG@10          : {_fmt(m['base_ndcg'])}")
    lines.append(f"  Ranker - baseline NDCG: {_fmt(None if m['ndcg_10'] is None or m['base_ndcg'] is None else m['ndcg_10'] - m['base_ndcg'])}")
    lines.append("")
    lines.append("INTENT-FIELD FAILURES (parser gaps)")
    lines.append("-" * w)
    for kind in FIELDS:
        lines.append(f"  {kind:<10} : {m['field_failures'][kind]:>3} / {m['num_queries']}")
    lines.append("")
    lines.append("BY CATEGORY")
    lines.append("-" * w)
    header = f"  {'category':<12} {'n':>3} {'gold':>4} {'intent':>6} {'p@3':>6} {'p@10':>6} {'r@10':>6} {'mrr':>6} {'ndcg':>6} {'b-p3':>6} {'b-ndcg':>6} {'zero':>4} {'viol':>4}"
    lines.append(header)
    for cat, x in sorted(m["categories"].items()):
        lines.append(
            f"  {cat:<12} {x['queries']:>3} {x['with_gold']:>4} {x['intent_accuracy']:.3f} "
            f"{_fmt(x['precision_3'])} {_fmt(x['precision_10'])} {_fmt(x['recall_10'])} "
            f"{_fmt(x['mrr'])} {_fmt(x['ndcg_10'])} {_fmt(x['base_p3'])} "
            f"{_fmt(x['base_ndcg'])} {x['zero_results']:>4} {x['violations']:>4}"
        )
    lines.append("")

    failed = [
        r for r in m["rows"]
        if r["precision_3"] < 0.5 or r["mrr"] == 0 or r["n_violations"] or r["zero_results"]
    ]
    lines.append("WORST QUERIES (p@3<0.5, MRR=0, violations, or zero results)")
    lines.append("-" * w)
    if not failed:
        lines.append("  (none)")
    for row in failed:
        lines.append(f"  [{row['category']}] {row['query']}")
        lines.append(
            f"      intent={row['intent_accuracy']:.3f} p@3={row['precision_3']:.3f} "
            f"p@10={row['precision_10']:.3f} r@10={_fmt(row['recall_10'])} mrr={_fmt(row['mrr'])} "
            f"ndcg={_fmt(row['ndcg_10'])} gold={row['gold_size']} ranked={row['ranked_count']} "
            f"viol={row['n_violations']}"
        )
        bad = [k for k in FIELDS if not row["field_ok"][k]]
        if bad:
            details = "; ".join(f"{k}: exp {row['expected'][k]!r} got {row['extracted'][k]!r}" for k in bad)
            lines.append(f"      ~ {details}")
        if row["top3"]:
            lines.append("      top3: " + " | ".join(row["top3"][:3]))
        if row["n_violations"]:
            for v in row["violations"]:
                lines.append(f"      ! {v}")
    lines.append("")
    return "\n".join(lines)


def write_reports(m: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "national_report.txt").write_text(format_report(m), encoding="utf-8")
    summary = {k: v for k, v in m.items() if k not in ("rows",)}
    summary["rows"] = [
        {k: r[k] for k in ("query", "category", "intent_accuracy", "precision_3", "precision_10",
                           "recall_10", "mrr", "ndcg_10", "n_violations", "gold_size",
                           "ranked_count", "zero_results", "dup_rate", "loc_precision",
                           "base_p3", "base_ndcg")}
        for r in m["rows"]
    ]
    (REPORTS_DIR / "national_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def main() -> None:
    metrics = run_national_evaluation()
    write_reports(metrics)
    print(format_report(metrics))


if __name__ == "__main__":
    main()
