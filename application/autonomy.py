from __future__ import annotations

"""Autonomous job-search-and-apply engine.

SEARCH → FILTER → SCORE → DECIDE → ANSWER QUESTIONS → APPLY → VERIFY → TRACK

The user's role is limited to: configure profile once, set preferences,
START an autonomous run (explicit POST /api/autonomous/run — that call IS
the consent gate), and review results. Everything in between is automated.

States reported per job (mapped onto the existing lifecycle — no statuses
were replaced):
    PREPARED            → form analysed + answers ready (READY_FOR_REVIEW)
    READY_TO_SUBMIT     → all mandatory questions answered truthfully
    SUBMITTED           → browser confirmed the submission
    AWAITING_CONFIRMATION / CONFIRMED → Gmail REST verification
    FAILED              → submission attempt failed
    REQUIRES_USER_INPUT → critical unknown question (never guessed)
    SKIPPED             → below threshold, duplicate, or limit reached

Safety/data-integrity (enforced, not advisory):
  * every answer traces to profile data, a verified user answer, or a safe
    derivation — critical unknowns are never invented
  * generated drafts pass humanise+validate before submission
  * duplicates (same job) and limits (per run / per day) enforced BEFORE
    any submission
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Callable, Optional

from config import (
    AUTONOMOUS_RUNS_DIR,
    MAX_APPLICATIONS_PER_DAY,
    MAX_APPLICATIONS_PER_RUN,
    MIN_APPLICATION_SCORE,
)
from sources.base import Job

# questions where a wrong guess would materially damage the application
_CRITICAL_PATTERNS = re.compile(
    r"citizen|citizenship|nationality|right to work|authori[sz]ed to work|"
    r"legally entitled to work|work permit|work visa|disability|"
    r"criminal record|police clearance|background check|"
    r"licen[cs]e\b|\bid number\b|national identity|ethnicity|\brace\b|gender\b",
    re.I,
)
_SALARY_PATTERN = re.compile(r"salary|remuneration|compensation|package|pay", re.I)


def classify_question_criticality(question: str, required: bool = False) -> str:
    """Return ``"critical"`` or ``"noncritical"``.

    Critical: legal/citizenship/disability/criminal/licence/ID/demographic
    questions — plus salary questions when the form marks them mandatory.
    """
    if _CRITICAL_PATTERNS.search(question):
        return "critical"
    if required and _SALARY_PATTERN.search(question):
        return "critical"
    return "noncritical"


@dataclass
class JobDecision:
    job: Job
    suitability: int = 0
    decision: str = "skip"          # apply | skip
    reason: str = ""
    reasons: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    explanation: str = ""
    blockers: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    missing_preferred: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.job.title,
            "company": self.job.company,
            "url": self.job.url,
            "score": self.suitability,
            "decision": self.decision,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "concerns": list(self.concerns),
            "explanation": self.explanation,
            "blockers": list(self.blockers),
            "unknowns": list(self.unknowns),
            "matched": list(self.matched),
            "missing_preferred": list(self.missing_preferred),
        }


def suitability_score(
    profile, job: Job, preference_score: int = 0
) -> tuple[int, list[str], list[str]]:
    """Candidate-vs-job suitability 0–100 from REAL profile evidence.

    Primary component: the explainable suitability engine (weighted,
    configurable dimensions + hard-requirement detection). The search
    query's preference score contributes 20%; candidate fit contributes
    80%. Reasons are the engine's evidence-backed matches.
    """
    from candidate.suitability import evaluate as evaluate_suitability

    result = evaluate_suitability(job, profile)
    blended = round(0.8 * result.score + 0.2 * max(0, min(100, preference_score)))
    reasons = [f"✓ {m}" for m in result.matched]
    concerns = list(result.missing_preferred)
    return int(max(0, min(100, blended))), reasons, concerns


# ---------------------------------------------------------------------------
# selection with limits + dedup
# ---------------------------------------------------------------------------

class AutonomyPolicy:
    def __init__(
        self,
        min_score: int = None,
        max_per_run: int = None,
        max_per_day: int = None,
    ):
        self.min_score = int(min_score if min_score is not None else MIN_APPLICATION_SCORE)
        self.max_per_run = int(max_per_run if max_per_run is not None else MAX_APPLICATIONS_PER_RUN)
        self.max_per_day = int(max_per_day if max_per_day is not None else MAX_APPLICATIONS_PER_DAY)


def submissions_today(tracker) -> int:
    today = date.today().isoformat()
    count = 0
    for app in tracker.all():
        stamp = app.submitted_at or app.submission_time or app.date_submitted
        if stamp and stamp[:10] == today and app.submitted:
            count += 1
    return count


def decide_job(
    item,                      # RankedJob-like: .job .score .reasons
    profile,
    tracker=None,
    policy: AutonomyPolicy = None,
    already_seen_urls: set[str] = None,
) -> JobDecision:
    """Evaluate ONE ranked job → apply/skip with transparent reasons."""
    policy = policy or AutonomyPolicy()
    tracker = tracker if tracker is not None else _default_tracker()
    seen = already_seen_urls or set()
    job = item.job

    # explainable suitability engine: hard requirements override the score,
    # mandatory-but-unknown facts require user input, never guesses.
    # ONE evaluation feeds both the numeric score and the gates.
    from candidate.suitability import evaluate as evaluate_suitability
    engine = evaluate_suitability(job, profile)
    pref = max(0, min(100, getattr(item, "score", 0)))
    score = int(max(0, min(100, round(0.8 * engine.score + 0.2 * pref))))
    reasons = [f"✓ {m}" for m in engine.matched]
    d = JobDecision(
        job=job, suitability=score, reasons=reasons,
        concerns=list(engine.missing_preferred),
        blockers=list(engine.blockers), unknowns=list(engine.unknowns),
        matched=list(engine.matched),
        missing_preferred=list(engine.missing_preferred),
        explanation=engine.explain(),
    )

    # duplicates first — never re-apply
    if tracker is not None and hasattr(tracker, "find_by_job_id"):
        try:
            if tracker.find_by_job_id(job.id):
                d.reason = "duplicate — already tracked for this job"
                return d
        except Exception:
            pass
    key = f"{job.title}|{job.company}|{job.url}".lower()
    if key in seen or (job.url and job.url.lower() in seen):
        d.reason = "duplicate — same job discovered twice in this run"
        return d

    # hard requirement gates from the job text itself
    text = f"{job.title} {job.description}".lower()
    if re.search(r"5\+|10\+ years|senior\b|principal\b|head of\b", text):
        d.reason = "seniority above candidate level"
        return d

    if engine.decision == "reject":
        d.reason = engine.blockers[0] if engine.blockers else \
            "hard requirement not met"
        return d
    if engine.decision == "requires_user_input":
        first = engine.unknowns[0] if engine.unknowns else "unknown mandatory fact"
        d.reason = f"REQUIRES_USER_INPUT - {first}"
        return d

    if score < policy.min_score:
        if 50 <= score < 65:
            d.decision = "review"
            d.reason = (
                f"Possible match - review recommended ({score}% falls in the "
                f"50-64% transferable band; auto-submit needs "
                f">= {policy.min_score}%)")
        else:
            d.reason = f"score below threshold ({score}% < {policy.min_score}%)"
        return d

    d.decision = "apply"
    d.reason = f"selected — score {score}% meets threshold {policy.min_score}%"
    return d


def select_jobs(
    ranked_items: list,
    profile,
    tracker=None,
    policy: AutonomyPolicy = None,
) -> tuple[list[JobDecision], list[JobDecision]]:
    """Rank → select suitable jobs → enforce limits. Returns (selected, skipped)."""
    policy = policy or AutonomyPolicy()
    tracker = tracker if tracker is not None else _default_tracker()

    seen: set[str] = set()   # dedupe jobs that appear twice in one run
    evaluated: list[JobDecision] = []
    for item in ranked_items:
        d = decide_job(item, profile, tracker, policy, already_seen_urls=seen)
        job = item.job
        seen.add(f"{job.title}|{job.company}|{job.url}".lower())
        if job.url:
            seen.add(job.url.lower())
        evaluated.append(d)
    evaluated.sort(key=lambda d: d.suitability, reverse=True)

    selected: list[JobDecision] = []
    skipped: list[JobDecision] = []
    daily_before = submissions_today(tracker) if tracker is not None else 0
    daily_left = max(0, policy.max_per_day - daily_before)

    for d in evaluated:
        if d.decision != "apply":
            skipped.append(d)
            continue
        if len(selected) >= policy.max_per_run:
            d.decision, d.reason = "skip", "run limit reached (MAX_APPLICATIONS_PER_RUN)"
            skipped.append(d)
            continue
        if daily_left <= 0:
            d.decision, d.reason = "skip", f"daily limit reached (MAX_APPLICATIONS_PER_DAY={policy.max_per_day})"
            skipped.append(d)
            continue
        selected.append(d)
        daily_left -= 1

    return selected, skipped


# ---------------------------------------------------------------------------
# question readiness over a prepared fill plan
# ---------------------------------------------------------------------------

def evaluate_plan_readiness(plan_entries: list[dict]) -> dict:
    """Classify every plan entry; never invents anything.

    Returns ready=False when a REQUIRED question cannot be answered
    truthfully. Critical unknowns are listed separately from non-critical.
    """
    critical_missing: list[str] = []
    noncritical_missing: list[str] = []
    answered = 0
    for e in plan_entries or []:
        q = e.get("question", "")
        required = bool(e.get("required"))
        # consent checkboxes are granted at submit time; uploads carry files.
        # Uploads count as handled; consents are neither answered nor missing.
        if e.get("is_consent") or e.get("is_terms"):
            continue
        if e.get("upload_kind"):
            answered += 1
            continue
        has_value = bool((e.get("value") or "").strip())
        if has_value:
            answered += 1
            continue
        kind = classify_question_criticality(q, required)
        if required:
            (critical_missing if kind == "critical" else noncritical_missing).append(q)
    return {
        "ready": not (critical_missing or noncritical_missing),
        "answered": answered,
        "critical_missing": critical_missing,
        "noncritical_missing": noncritical_missing,
    }


# ---------------------------------------------------------------------------
# the autonomous run itself
# ---------------------------------------------------------------------------

def _default_tracker():
    from application.tracker import ApplicationTracker
    return ApplicationTracker()


def run_autonomous_job_search(
    query_text: str = "",
    *,
    search_fn: Optional[Callable] = None,
    driver_factory: Optional[Callable] = None,
    service_factory: Optional[Callable] = None,
    tracker=None,
    connector=None,
    policy: AutonomyPolicy = None,
    dry_run: bool = False,
) -> dict:
    """Full autonomous pipeline. Returns a structured execution report.

    ``search_fn(prompt, profile) -> object with .ranked`` and
    ``driver_factory() -> BrowserDriver`` are injectable so tests can run the
    whole flow without network or browser. ``dry_run`` stops right after
    selection (no applications are opened).
    """
    started = datetime.now().isoformat()
    policy = policy or AutonomyPolicy()
    tracker = tracker if tracker is not None else _default_tracker()

    report: dict = {
        "started_at": started,
        "query": query_text,
        "threshold": policy.min_score,
        "limits": {"per_run": policy.max_per_run, "per_day": policy.max_per_day},
        "jobs_discovered": 0,
        "jobs_evaluated": 0,
        "suitable_jobs": 0,
        "applications": [],
        "skipped": [],
        "error": "",
    }

    def finish(extra_summary: str = "") -> dict:
        report["finished_at"] = datetime.now().isoformat()
        report["summary"] = extra_summary or _render_report_text(report)
        _save_report(report)
        return report

    # 1. profile -----------------------------------------------------------
    from candidate.storage import load_profile
    profile = load_profile()
    if profile is None:
        report["error"] = "No candidate profile found — upload your CV first."
        return finish("AUTONOMOUS RUN ABORTED — profile missing")

    # 2–3. search + dedupe source list -------------------------------------
    if search_fn is None:
        search_fn = _default_search_fn(query_text)
    result = search_fn(query_text, profile)
    ranked_items = list(getattr(result, "ranked", []) or [])
    report["queries_used"] = list(getattr(result, "queries_used", []) or [])
    report["expanded_queries"] = list(
        getattr(result, "expanded_queries", []) or [])
    strategy = getattr(result, "strategy", None)
    if strategy is not None:
        report["search_strategy"] = {
            "career_level": strategy.career_level,
            "occupations": [o.get("label") for o in strategy.occupations],
            "inference": strategy.inference,
            "locations": list(strategy.locations),
        }
    # drop source-level duplicates before scoring anything
    seen_keys: set[str] = set()
    unique_items = []
    for item in ranked_items:
        j = item.job
        key = f"{j.title}|{j.company}|{j.url}".lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(item)
    report["duplicates_dropped"] = len(ranked_items) - len(unique_items)
    ranked_items = unique_items
    report["jobs_discovered"] = len(ranked_items)

    # 4–7. evaluate / rank / select with limits ----------------------------
    selected, skipped = select_jobs(ranked_items, profile, tracker, policy)
    report["jobs_evaluated"] = len(selected) + len(skipped)
    report["skipped"] = [d.to_dict() | {"outcome": "SKIPPED"} for d in skipped]
    report["suitable_jobs"] = len(selected)

    if dry_run:
        report["dry_run"] = True
        report["applications"] = [d.to_dict() | {"outcome": "DRY_RUN"} for d in selected]
        return finish()

    # 8–13. prepare → answer → validate → submit → record ------------------
    driver = None
    if driver_factory is not None:
        driver = driver_factory()
    service = (service_factory() if service_factory is not None
               else _default_service_factory())

    for d in selected:
        entry = {
            "company": d.job.company,
            "title": d.job.title,
            "score": d.suitability,
            "reason": d.reason,
            "reasons": d.reasons,
            "explanation": d.explanation,
            "missing_preferred": d.missing_preferred,
            "application_id": "",
            "outcome": "",
            "detail": "",
        }
        report["applications"].append(entry)
        try:
            app_obj = service.start_application(d.job, profile, tracker, driver=driver)
            if app_obj is None:
                entry["outcome"] = "FAILED"
                entry["detail"] = "could not reach application page"
                continue
            entry["application_id"] = app_obj.id

            readiness = evaluate_plan_readiness(app_obj.fill_plan or [])
            if not readiness["ready"]:
                # critical unknowns must NEVER be guessed — stop safely
                from application.lifecycle import transition
                from application.models import ApplicationStatus
                missing = readiness["critical_missing"] + readiness["noncritical_missing"]
                transition(app_obj, ApplicationStatus.REQUIRES_USER_ACTION,
                           "Autonomous run stopped: unanswered required questions")
                app_obj.error = "Needs your input before submitting: " + "; ".join(missing)
                tracker.update(app_obj)
                entry["outcome"] = "REQUIRES_USER_INPUT"
                entry["detail"] = app_obj.error
                continue

            # humanise AI-drafted answers, validate against profile
            ok_to_submit = _humanise_generated_answers(app_obj, profile)
            if not ok_to_submit:
                entry["outcome"] = "REQUIRES_USER_INPUT"
                entry["detail"] = "generated draft failed validation against your profile"
                continue

            if driver is None:
                driver = driver_factory() if driver_factory is not None else _open_driver()
            # submit EXACTLY the plan we just validated/humanised — never a
            # stale service-internal plan from another job
            from application.form_filler import FillPlan, PlannedAnswer
            fresh_plan = FillPlan(
                entries=[PlannedAnswer(**e) for e in (app_obj.fill_plan or [])])
            result = service.confirm_and_submit(
                app_obj, tracker, driver,
                consent_granted=True,   # consent = you explicitly started this run
                user_answers={},
                plan=fresh_plan,
            )
            entry["outcome"] = str(result.status.value) if hasattr(result.status, "value") else str(result.status)
            entry["detail"] = (result.confirmation_text or result.error or "")[:300]
        except Exception as exc:  # one bad site must not kill the run
            entry["outcome"] = "FAILED"
            entry["detail"] = str(exc)[:300]

    # 14–15. confirmation emails for everything submitted this run ---------
    submitted_ids = [a["application_id"] for a in report["applications"]
                     if a["outcome"] == "submitted"]
    if submitted_ids:
        try:
            report["confirmation_check"] = _check_confirmations(
                submitted_ids, tracker, connector
            )
        except Exception as exc:
            report["confirmation_check"] = {"error": str(exc)[:200]}

    return finish()


# ---------------------------------------------------------------------------

def _humanise_generated_answers(app_obj, profile) -> bool:
    """Humanise every generated_from_evidence answer in place.

    Returns False when a generated draft makes claims the profile cannot
    support — the application then needs user input instead of submission.
    """
    from application.humanise import humanise_and_validate

    changed = []
    for e in app_obj.fill_plan or []:
        if e.get("answer_type") != "generated_from_evidence":
            continue
        final, ok, issues = humanise_and_validate(e.get("value") or "", profile)
        if not ok:
            app_obj.error = ("Draft rejected by validation: " + "; ".join(issues))[:300]
            return False
        if final != e.get("value"):
            e["value"] = final
            e["reason"] = (e.get("reason") or "") + " (humanised)"
            changed.append(e)
    if changed:
        app_obj.fill_plan = list(app_obj.fill_plan)
    return True


def _default_search_fn(query_text: str = None):
    """Profile-driven multi-query search — works for ANY profession.

    Queries are generated from the candidate's own evidence (titles,
    qualifications, skills, inferred occupations) by
    agent.candidate_search; the explicit query_text, when provided, leads.
    """
    def search(_prompt, _profile):
        from agent.candidate_search import search_for_candidate
        return search_for_candidate(_profile, query_text=query_text or None)
    return search


def _default_service_factory():
    from application.submission import ApplicationAutomationService
    return ApplicationAutomationService()


def _open_driver():
    from application.browser import open_driver
    return open_driver(prefer_headless=True)


def _check_confirmations(app_ids: list[str], tracker, connector) -> dict:
    """Single poll of the Gmail REST backend per run — never resubmits."""
    from application.email_confirmation import await_confirmation

    results = {}
    for app_id in app_ids:
        app_obj = tracker.get(app_id)
        if app_obj is None:
            continue
        try:
            updated = await_confirmation(app_obj, tracker, connector=connector)
            results[app_id] = updated.status.value
        except Exception as exc:
            results[app_id] = f"check_failed: {exc}"[:120]
    return results


def _render_report_text(report: dict) -> str:
    lines = ["AUTONOMOUS RUN COMPLETE", ""]
    if report.get("queries_used"):
        lines.append(f"Search queries: {len(report['queries_used'])}"
                     + (f" (+{len(report['expanded_queries'])} expanded)"
                        if report.get("expanded_queries") else ""))
        for q in report["queries_used"][:8]:
            mark = "*" if q in report.get("expanded_queries", []) else "-"
            lines.append(f"  {mark} {q}")
    strategy = report.get("search_strategy") or {}
    if strategy:
        lines.append(
            f"Candidate profile read as: {strategy.get('career_level', '?')}"
            f" — {', '.join(strategy.get('occupations') or ['no occupation inferred'])}")
    lines.append(f"Jobs discovered: {report['jobs_discovered']}")
    lines.append(f"Jobs evaluated: {report['jobs_evaluated']}")
    lines.append(f"Suitable jobs: {report['suitable_jobs']}")
    lines.append(f"Application limit: {report['limits']['per_run']} "
                 f"(daily cap {report['limits']['per_day']})")
    if report.get("error"):
        lines.append(f"ERROR: {report['error']}")
        return "\n".join(lines)
    if report["applications"]:
        lines.append("")
        lines.append("Applications:")
        for a in report["applications"]:
            outcome = a.get("outcome", "")
            mark = "✓" if outcome in ("submitted", "confirmed",
                                       "awaiting_confirmation") \
                else ("!" if outcome in ("failed", "requires_user_action") else "·")
            suffix = f" — {a.get('detail')}" if a.get("detail") else ""
            lines.append(f"{mark} {a['company']} — {a['title']} — "
                         f"{a['score']}% — {outcome or 'selected'}{suffix}")
            for why in (a.get("matched") or [])[:3]:
                lines.append(f"    why: {why}")
            for mp in (a.get("missing_preferred") or [])[:2]:
                lines.append(f"    missing/preferred: {mp}")
    if report["skipped"]:
        lines.append("")
        lines.append("Skipped:")
        for s in report["skipped"]:
            lines.append(f"{s['company']} — {s['title']} — {s['score']}% — "
                         f"{s.get('reason', '')}")
    return "\n".join(lines)


def _save_report(report: dict) -> None:
    try:
        import config as cfg
        runs_dir = cfg.AUTONOMOUS_RUNS_DIR
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = runs_dir / f"run_{stamp}.json"
        path.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")
    except Exception:
        pass  # reporting must never break the pipeline
