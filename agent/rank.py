from __future__ import annotations

import re
from dataclasses import dataclass, field

from sources.base import Job

from .parse_intent import ROLE_PHRASES, JobQuery

W_ROLE = 25
W_SENIORITY = 15
W_LOCATION = 15
W_SALARY = 15
W_SKILLS = 15
W_REMOTE = 15
TOTAL_WEIGHT = W_ROLE + W_SENIORITY + W_LOCATION + W_SALARY + W_SKILLS + W_REMOTE

ENTRY_MARKERS = [
    "junior",
    "entry-level",
    "entry level",
    "graduate",
    "intern",
    "trainee",
    "learner",
    "0-2 years",
    "1-2 years",
    "0 - 2 years",
]
SENIOR_MARKERS = [
    "senior",
    "principal",
    "lead",
    "head of",
    "5+ years",
    "10+ years",
    "5 - 10 years",
]

# Leadership signals judged on the TITLE only: a graduate programme's
# description may mention anything, but a "CHIEF DIRECTOR" title is not an
# entry-level job no matter what the body text says.
LEADERSHIP_TITLE_RE = re.compile(
    r"\b(chief|deputy director|director|regional head|head|senior|principal)\b",
    re.IGNORECASE,
)
# ...unless the title itself marks the post as developmental.
DEVELOPMENTAL_TITLE_RE = re.compile(
    r"\b(intern|internship|trainee|learner|graduate?)\b", re.IGNORECASE
)


@dataclass
class RankedJob:
    job: Job
    score: int
    reasons: list[str] = field(default_factory=list)
    summary: str = ""


def rank_jobs(jobs: list[Job], query: JobQuery, llm=None) -> list[RankedJob]:
    surviving = _filter_jobs(jobs, query)
    ranked = [_rank_job(job, query, llm) for job in surviving]
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _filter_jobs(jobs: list[Job], query: JobQuery) -> list[Job]:
    kept: list[Job] = []
    for job in jobs:
        if not _role_allowed(job, query):
            continue
        if not _seniority_allowed(job, query):
            continue
        if not _location_allowed(job, query):
            continue
        if not _salary_allowed(job, query):
            continue
        if not _remote_allowed(job, query):
            continue
        kept.append(job)
    return kept


def _role_terms(query: JobQuery) -> list[str]:
    """Surface-form terms for the queried role groups, derived from the
    existing ROLE_PHRASES vocabulary (no parallel synonym list).

    For every role group named by ``query.roles`` we reuse that group's own
    synonym phrases, plus the trailing role noun of each canonical label
    (e.g. "software engineer" -> "engineer", "software developer" ->
    "developer"). Falls back to the raw role strings when no known group is
    named so pre-existing behavior is preserved for unknown roles.
    """
    wanted = {role.strip().lower() for role in query.roles}
    terms: list[str] = []
    seen: set[str] = set()
    for canonical, phrases in ROLE_PHRASES:
        if wanted & {c.strip().lower() for c in canonical}:
            for phrase in phrases:
                key = phrase.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    terms.append(key)
    for role in query.roles:
        noun = role.strip().lower().split(" / ")[-1].split()[-1]
        if noun and noun not in seen:
            seen.add(noun)
            terms.append(noun)
    return terms or [role.strip().lower() for role in query.roles]


def _role_match(job: Job, query: JobQuery) -> tuple[str | None, bool]:
    """Return ``(matched_term, matched_in_title)`` using the shared synonym
    set. A term in the title is a stronger signal than one in the description
    only; both share the exact same matching semantics as filtering.
    """
    if not query.roles:
        return None, False
    title = job.title.lower()
    haystack = f"{title} {job.description}".lower()
    terms = _role_terms(query)
    for term in terms:
        if term in title:
            return term, True
    for term in terms:
        if term in haystack:
            return term, False
    return None, False


def _role_allowed(job: Job, query: JobQuery) -> bool:
    if not query.roles:
        return True
    matched, _ = _role_match(job, query)
    return matched is not None


def _seniority_allowed(job: Job, query: JobQuery) -> bool:
    if query.seniority != "entry-level":
        return True
    title = job.title.lower()
    if DEVELOPMENTAL_TITLE_RE.search(title):
        return True
    if LEADERSHIP_TITLE_RE.search(title):
        return False
    text = f"{job.title} {job.description}".lower()
    has_entry = any(marker in text for marker in ENTRY_MARKERS)
    has_senior = any(marker in text for marker in SENIOR_MARKERS)
    return has_entry or not has_senior


def _location_allowed(job: Job, query: JobQuery) -> bool:
    if not query.locations:
        return True
    if job.remote and query.remote != "no":
        return True
    if not job.location:
        return True
    lowered = job.location.lower()
    return any(loc.city.lower() in lowered for loc in query.locations)


def _salary_allowed(job: Job, query: JobQuery) -> bool:
    if query.min_salary is None or job.salary_min is None:
        return True
    return job.salary_min >= query.min_salary


def _remote_allowed(job: Job, query: JobQuery) -> bool:
    if query.remote == "required":
        return job.remote
    if query.remote == "no":
        return not job.remote
    return True


def _rank_job(job: Job, query: JobQuery, llm=None) -> RankedJob:
    points = 0
    reasons: list[str] = []

    points += _role_score(job, query, reasons)
    points += _seniority_score(job, query, reasons)
    points += _location_score(job, query, reasons)
    points += _salary_score(job, query, reasons)
    points += _skills_score(job, query, reasons)
    points += _remote_score(job, query, reasons)

    title = job.title.lower()
    if (LEADERSHIP_TITLE_RE.search(title)
            and not DEVELOPMENTAL_TITLE_RE.search(title)):
        # Softer penalty: leadership roles are penalised less when the user
        # explicitly asked for senior/director roles (query.seniority or
        # keywords contain leadership terms).
        _user_wants_leadership = (
            query.seniority == "senior"
            or any(kw in ("director", "head", "chief", "executive", "manager")
                   for kw in query.keywords)
        )
        penalty = 4 if _user_wants_leadership else 8
        points -= penalty
        reasons.append(f"Seniority: ~ leadership title (penalty -{penalty})")

    score = round(points / TOTAL_WEIGHT * 100)
    score = max(0, min(100, score))
    return RankedJob(job=job, score=score, reasons=reasons, summary=_summary(job, query, llm))


def _role_score(job: Job, query: JobQuery, reasons: list[str]) -> int:
    if not query.roles:
        # When no roles are specified, use free-text keywords as a weak
        # role signal — but only title matches count, since description
        # matches are too broad (e.g. "manager" in any job description).
        title = job.title.lower()
        kw_in_title = [kw for kw in query.keywords
                       if len(kw) >= 3 and kw in title]
        if kw_in_title:
            reasons.append(f"Role:      ~ keyword '{kw_in_title[0]}' in title (no role specified)")
            return round(W_ROLE * 0.7)
        reasons.append("Role:      \u2022 (not specified)")
        return round(W_ROLE * 0.4)
    matched, in_title = _role_match(job, query)
    if in_title:
        reasons.append(f"Role:      \u2713 '{matched}' in title")
        return W_ROLE
    if matched:
        reasons.append(f"Role:      ~ '{matched}' in description only")
        return round(W_ROLE * 0.7)
    reasons.append("Role:      \u2717 no matching role")
    return 0


def _seniority_score(job: Job, query: JobQuery, reasons: list[str]) -> int:
    text = f"{job.title} {job.description}".lower()
    if query.seniority == "entry-level":
        has_entry = any(marker in text for marker in ENTRY_MARKERS)
        has_senior = any(marker in text for marker in SENIOR_MARKERS)
        if has_entry and not has_senior:
            reasons.append("Seniority: ✓ entry-level")
            return W_SENIORITY
        if has_entry:
            reasons.append("Seniority: ~ entry-level role, mentions senior work")
            return round(W_SENIORITY * 0.8)
        if has_senior:
            reasons.append("Seniority: ✗ senior experience demanded")
            return 0
        reasons.append("Seniority: ~ seniority not stated")
        return round(W_SENIORITY * 0.6)
    if query.seniority == "senior":
        if any(marker in text for marker in SENIOR_MARKERS):
            reasons.append("Seniority: ✓ senior role")
            return W_SENIORITY
        reasons.append("Seniority: ~ seniority not confirmed")
        return round(W_SENIORITY * 0.5)
    reasons.append("Seniority: • (not specified)")
    return W_SENIORITY


def _location_score(job: Job, query: JobQuery, reasons: list[str]) -> int:
    if not query.locations:
        if job.remote:
            reasons.append("Location:  ✓ Remote (no city requirement)")
        else:
            reasons.append("Location:  • (no city requirement)")
        return W_LOCATION
    if not job.location:
        reasons.append("Location:  ~ location not stated")
        return round(W_LOCATION * 0.6)
    lowered = job.location.lower()
    matched = [loc for loc in query.locations if loc.city.lower() in lowered]
    if matched:
        reasons.append(f"Location:  ✓ {matched[0].city} (within ~{matched[0].radius_km} km)")
        return W_LOCATION
    if job.remote and query.remote != "no":
        reasons.append("Location:  ✓ Remote (no city requirement)")
        return W_LOCATION
    wanted = ", ".join(loc.city for loc in query.locations)
    reasons.append(f"Location:  ✗ outside {wanted}")
    return 0


def _salary_score(job: Job, query: JobQuery, reasons: list[str]) -> int:
    if job.salary_min is None:
        reasons.append("Salary:    ⚠ not stated")
        return round(W_SALARY * 0.5)
    if query.min_salary is None:
        reasons.append(f"Salary:    • {_fmt_salary(job.salary_min)} (not requested)")
        return W_SALARY
    if job.salary_min >= query.min_salary:
        reasons.append(f"Salary:    ✓ {_fmt_salary(job.salary_min)} ≥ {_fmt_salary(query.min_salary)}")
        return W_SALARY
    reasons.append(f"Salary:    ✗ {_fmt_salary(job.salary_min)} < {_fmt_salary(query.min_salary)}")
    return 0


def _matching_terms(query: JobQuery) -> list[str]:
    """Requirement-grade terms: explicit skills only.

    Free-text keywords are fuzzy search hints, NOT requirements - they must
    never surface as 'missing' items against a job or the candidate.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for term in query.skills:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            terms.append(key)
    return terms


def _keyword_hits(job: Job, query: JobQuery) -> list[str]:
    text = f"{job.title} {job.description}".lower()
    hits: list[str] = []
    for kw in query.keywords:
        key = kw.strip().lower()
        if len(key) >= 3 and key in text:
            hits.append(key)
    return hits


def _skills_score(job: Job, query: JobQuery, reasons: list[str]) -> int:
    terms = _matching_terms(query)
    if not terms:
        # no explicit skills requested: free-text keywords still break ties
        hits = _keyword_hits(job, query)[:1]
        if hits:
            reasons.append(f"Skills:    ✓ keyword '{hits[0]}'")
            return W_SKILLS
        reasons.append("Skills:    • (not specified)")
        return round(W_SKILLS * 0.6)
    text = f"{job.title} {job.description}".lower()
    matched = [t for t in terms if t in text]
    if matched:
        reasons.append(f"Skills:    ✓ {', '.join(matched)}")
        return W_SKILLS
    keyword_hits = _keyword_hits(job, query)[:2]
    if keyword_hits:
        reasons.append(f"Skills:    ~ related keyword '{keyword_hits[0]}' found")
        return round(W_SKILLS * 0.5)
    reasons.append(f"Skills:    ✗ none of {', '.join(terms)}")
    return 0


def _remote_score(job: Job, query: JobQuery, reasons: list[str]) -> int:
    preference = query.remote
    if preference == "required":
        if job.remote:
            reasons.append("Remote:    ✓ remote required and offered")
        else:
            reasons.append("Remote:    ✗ not remote")
        return W_REMOTE if job.remote else 0
    if preference == "no":
        if not job.remote:
            reasons.append("Remote:    ✓ on-site only")
        else:
            reasons.append("Remote:    ✗ remote offered, you wanted on-site")
        return W_REMOTE if not job.remote else 0
    if preference == "preferred":
        if job.remote:
            reasons.append("Remote:    ✓ accepts SA applicants")
            return W_REMOTE
        reasons.append("Remote:    ~ on-site / hybrid only")
        return round(W_REMOTE * 0.4)
    reasons.append("Remote:    • (not specified)")
    return W_REMOTE


def _summary(job: Job, query: JobQuery, llm=None) -> str:
    text = f"{job.title} {job.description}".lower()
    terms = _matching_terms(query)
    matched_skills = [t for t in terms if t in text]
    bits = []
    if matched_skills:
        bits.append(f"asks for {', '.join(matched_skills)}")
    if job.remote and query.remote == "preferred":
        bits.append("offers remote for SA candidates")
    if job.salary_min is None and query.min_salary is not None:
        bits.append("doesn't list salary, so confirm it")
    elif job.salary_min is not None and query.min_salary is not None:
        bits.append(f"pays {_fmt_salary(job.salary_min)}")
    elif job.salary_min is None:
        bits.append("doesn't list salary")
    summary = f"{job.company}: " + ("; ".join(bits) if bits else "review the posting for details") + "."
    if llm and llm.is_available():
        try:
            narrative = llm.chat_json(
                "Write ONE short, honest sentence (max 30 words) summarising why this "
                "job is a fit for the candidate. Only JSON, key 'summary'.",
                f"Candidate is a {query.seniority or 'job'} seeking {', '.join(query.roles) or 'work'} "
                f"wanting {query.remote} remote in {', '.join(l.city for l in query.locations) or 'any location'}. "
                f"Job: {job.title} at {job.company}. Description: {job.description[:800]!r}",
            )
            summary = str(narrative.get("summary", summary)).strip() or summary
        except Exception:
            pass
    return summary


def _fmt_salary(value: int) -> str:
    return f"R{value:,}"
