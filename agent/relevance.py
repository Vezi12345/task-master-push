from __future__ import annotations

"""Job-title relevance filtering.

Separate from ``sources.validation`` (which answers "is this a legitimate
online vacancy?"), this module answers "does this vacancy actually match the
role the user asked for?".

Matching is semantic at the job-title level: titles are normalised (post
numbers, reference numbers and vacancy counts stripped) and compared against
concept vocabularies with word boundaries, so "Software Development Intern"
matches a software-developer query while "Organisational Development Director"
or "Property Developer" does not. Description text is deliberately ignored —
a vacancy that merely mentions "software" in the body is not a software role.
"""

import re
from dataclasses import dataclass, field

from sources.base import Job

from .parse_intent import JobQuery


# ---------------------------------------------------------------------------
# Title normalisation
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    re.compile(r"^\s*post\s+\d+\s*/\s*\d+\s*:?\s*", re.IGNORECASE),
    re.compile(r"\bref\s*(?:no|number)?\s*[:.]?\s*[A-Z0-9/\-()]*\s*$", re.IGNORECASE),
    re.compile(r"\(\s*x\s*\d+\s*(?:posts?)?\s*\)", re.IGNORECASE),
    re.compile(r"\(\s*\d+\s*(?:posts?)?\s*\)", re.IGNORECASE),
    re.compile(r"\bgrade\s+[a-e](?:\s*-\s*[a-e])?\b", re.IGNORECASE),
    re.compile(r"\(\s*\d+\s*months?\s*contract\s*\)", re.IGNORECASE),
    re.compile(r"\bcontract\b", re.IGNORECASE),
]


def normalise_title(title: str) -> str:
    """Strip circular-specific noise so the core role phrase remains."""
    text = (title or "").lower()
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\(\s*\)", " ", text)  # parentheses emptied by stripping
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Role concepts
# ---------------------------------------------------------------------------

@dataclass
class _Concept:
    key: str
    # phrases whose presence (word-bounded) in the title indicates the concept
    phrases: list[str] = field(default_factory=list)
    # domain guards: when present, an otherwise-matching title is rejected
    # (e.g. "Property Developer", "Business Development Manager")
    negative: list[str] = field(default_factory=list)


_CONCEPTS: list[_Concept] = [
    _Concept(
        "software_development",
        phrases=[
            "software developer", "software development", "software engineer",
            "software engineering", "software dev", "programmer", "coding",
            "application developer", "applications developer",
            "application development engineer",
            "systems developer", "system developer",
            "web developer", "web dev", "webdev",
            "full stack developer", "full-stack developer",
            "frontend developer", "front-end developer", "front end developer",
            "backend developer", "back-end developer", "back end developer",
            ".net developer", "dotnet developer", "java developer",
            "c# developer", "c++ developer", "python developer",
            "php developer", "node developer", "ruby developer",
            "graduate developer", "junior developer", "lead developer",
            "senior developer", "principal developer",
            "development intern", "development graduate",
        ],
        negative=[
            "business development", "property", "real estate",
            "organisational development", "organizational development",
            "skills development", "human resource development",
            "staff development", "curriculum development",
            "land development", "rural development", "economic development",
            "community development", "leadership development",
            "personal development", "product development",
            "fundraising", "sales",
        ],
    ),
    _Concept(
        "data",
        phrases=[
            "data analyst", "data analysis", "data scientist",
            "data science", "analytics", "business intelligence",
            "bi developer", "insights analyst",
        ],
    ),
    _Concept(
        "qa",
        phrases=["qa", "quality assurance", "software tester", "test engineer", "quality controller"],
    ),
    _Concept(
        "it_support",
        phrases=[
            "it support", "help desk", "service desk", "desktop support",
            "support technician", "it technician", "ict technician",
            "network administrator", "systems administrator",
            "sysadmin", "infrastructure engineer",
        ],
    ),
    _Concept(
        "finance",
        phrases=[
            "finance", "financial", "accountant", "accounting",
            "bookkeeper", "bcom", "investment", "banking", "auditor",
        ],
    ),
    _Concept(
        "admin",
        phrases=[
            "admin", "administration", "administrative", "clerk",
            "office administrator", "receptionist", "registry",
        ],
    ),
    _Concept(
        "hr",
        phrases=[
            "human resource", "hr officer", "hr advisor", "hr generalist",
            "recruitment", "talent acquisition", "people and culture",
        ],
    ),
]

# Query-role surface forms -> concept keys. Canonical ROLE_PHRASES groups and
# common phrasings both resolve here.
_ROLE_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "software engineer": ("software_development",),
    "software developer": ("software_development",),
    "developer": ("software_development",),
    "programmer": ("software_development",),
    "web developer": ("software_development",),
    "full stack developer": ("software_development",),
    "backend developer": ("software_development",),
    "frontend developer": ("software_development",),
    "application developer": ("software_development",),
    "systems developer": ("software_development",),
    "software development intern": ("software_development",),
    "graduate software developer": ("software_development",),
    "junior developer": ("software_development",),
    "coder": ("software_development",),
    "data analyst": ("data",),
    "data scientist": ("data",),
    "qa / test engineer": ("qa",),
    "qa engineer": ("qa",),
    "test engineer": ("qa",),
    "it support": ("it_support",),
    "it technician": ("it_support",),
    "finance": ("finance", "accountant", "auditor"),
    "accountant": ("finance",),
    "administrator / clerk": ("admin",),
    "admin clerk": ("admin",),
    "hr officer": ("hr",),
}


def _registry_concepts() -> tuple[list[_Concept], dict[str, tuple[str, ...]]]:
    """Concept vocabularies for every registry occupation.

    Generated from the same generic registry the search profile uses, so
    nursing / driving / teaching / artisan searches all filter by title
    semantics without any hand-coded profession list here. Occupations
    already covered by the curated concepts above (software development,
    data) are skipped — those carry domain negative-guards ("property
    developer", "business development") that must not be bypassed.
    """
    from candidate.occupations import OCCUPATIONS

    covered = {"software_developer", "data_analyst"}
    concepts: list[_Concept] = []
    mapping: dict[str, tuple[str, ...]] = {}
    for occ in OCCUPATIONS:
        if occ.key in covered:
            continue
        phrases = [occ.label.lower()]
        phrases.extend(t for t in occ.titles)
        phrases.extend(t for t in occ.adjacent)
        concepts.append(_Concept(occ.key,
                                 phrases=list(dict.fromkeys(phrases))))
        mapping[occ.label.lower()] = (occ.key,)
    return concepts, mapping


_REGISTRY_CONCEPTS, _REGISTRY_ROLE_CONCEPT_MAP = _registry_concepts()

_WORD_RE = re.compile(r"[a-z0-9#+.\-]+")


def _phrase_regex(phrase: str) -> re.Pattern:
    """Word-bounded phrase match, tolerating a simple plural on the final
    word so circular titles like 'IT TECHNICIANS' or 'SOFTWARE DEVELOPERS'
    still match."""
    return re.compile(
        r"(?<![a-z0-9])" + re.escape(phrase) + r"(?:s|es)?(?![a-z0-9])"
    )


def _title_concepts(normalised_title: str) -> set[str]:
    found: set[str] = set()
    for concept in list(_CONCEPTS) + list(_REGISTRY_CONCEPTS):
        for phrase in concept.phrases:
            if _phrase_regex(phrase).search(normalised_title):
                found.add(concept.key)
                break
    return found


def _blocked_by_negative(concept_key: str, normalised_title: str) -> bool:
    concept = next(
        (c for c in list(_CONCEPTS) + list(_REGISTRY_CONCEPTS)
         if c.key == concept_key), None)
    if concept is None:
        return False
    return any(
        _phrase_regex(guard).search(normalised_title)
        for guard in concept.negative
    )


def _significant_tokens(text: str) -> set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "for", "in", "to", "with",
        "senior", "junior", "graduate", "entry", "level", "chief", "deputy",
        "assistant", "director", "manager", "officer", "post", "ref",
    }
    return {
        tok.strip("-") for tok in _WORD_RE.findall(text.lower())
        if len(tok) >= 4 and tok not in stopwords
    }


def is_relevant_job(job: Job, query: JobQuery) -> tuple[bool, str]:
    """Decide whether the vacancy's actual role matches the requested roles.

    Returns ``(True, "")`` or ``(False, reason)``.
    """
    wanted_concepts: set[str] = set()
    for role in query.roles:
        key = role.strip().lower()
        wanted_concepts.update(_ROLE_CONCEPT_MAP.get(key, ()))
        wanted_concepts.update(_REGISTRY_ROLE_CONCEPT_MAP.get(key, ()))

    title = normalise_title(job.title)

    if wanted_concepts:
        title_concepts = _title_concepts(title)
        for concept_key in wanted_concepts:
            if concept_key in title_concepts and not _blocked_by_negative(concept_key, title):
                return True, ""
        return False, (
            f"title '{job.title.strip()}' is not a "
            f"{ ' / '.join(query.roles) } role"
        )

    # Unmapped roles: fall back to significant-token overlap on the title.
    role_tokens: set[str] = set()
    for role in query.roles:
        role_tokens.update(_significant_tokens(role))
    if not role_tokens:
        return True, ""  # nothing to compare against — don't over-reject
    title_tokens = _significant_tokens(title)
    if role_tokens & title_tokens:
        return True, ""
    return False, f"title '{job.title.strip()}' shares no role keywords with {' / '.join(query.roles)}"


def filter_relevant_jobs(
    jobs: list[Job],
    query: JobQuery,
) -> tuple[list[Job], list[tuple[Job, str]]]:
    """Split retrieved jobs into role-relevant and irrelevant records.

    Real-job validation has already run; this pass never looks at whether a
    job is genuine — only whether it matches what the user searched for.
    """
    kept: list[Job] = []
    rejected: list[tuple[Job, str]] = []
    for job in jobs:
        relevant, reason = is_relevant_job(job, query)
        if relevant:
            kept.append(job)
        else:
            rejected.append((job, reason))
    return kept, rejected


__all__ = [
    "filter_relevant_jobs",
    "is_relevant_job",
    "normalise_title",
]
