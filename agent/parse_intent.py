from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from candidate.occupations import OCCUPATIONS


class LocationRef(BaseModel):
    city: str
    radius_km: int = 50


class JobQuery(BaseModel):
    roles: list[str] = Field(default_factory=list)
    seniority: str = ""
    keywords: list[str] = Field(default_factory=list)
    locations: list[LocationRef] = Field(default_factory=list)
    remote: str = "any"
    min_salary: Optional[int] = None
    currency: str = "ZAR"
    skills: list[str] = Field(default_factory=list)


class IntentType(str):
    SEARCH = "search"
    APPLY = "apply"
    SHOW_JOBS = "show_jobs"
    SHOW_APPLICATIONS = "show_applications"
    NEEDS_ATTENTION = "needs_attention"
    APPROVE = "approve"
    CANCEL = "cancel"
    ANSWER = "answer"
    STATUS = "status"
    HELP = "help"
    UNKNOWN = "unknown"


class UserIntent(BaseModel):
    intent_type: str = "search"
    search_query: Optional[JobQuery] = None
    apply_count: Optional[int] = None
    min_match_score: Optional[int] = None
    answers: dict[str, str] = Field(default_factory=dict)
    target_id: Optional[str] = None
    message: str = ""


_APPLY_PATTERNS = [
    re.compile(r"apply\s+to\s+(?:the\s+)?(?:best\s+)?(\d+)", re.IGNORECASE),
    re.compile(r"apply\s+to\s+(?:the\s+)?best\s+(\d+)", re.IGNORECASE),
    re.compile(r"apply\s+to\s+(?:the\s+)?(?:top\s+)?(\d+)", re.IGNORECASE),
    re.compile(r"apply\s+(\d+)", re.IGNORECASE),
    re.compile(r"submit\s+(?:the\s+)?(?:best\s+)?(\d+)", re.IGNORECASE),
]

_MATCH_SCORE_PATTERNS = [
    re.compile(r"(?:where\s+)?(?:i\s+)?(?:have\s+)?(?:at\s+least|above|over|>=?)\s*(\d+)\s*%", re.IGNORECASE),
    re.compile(r"(\d+)\s*%\s*match", re.IGNORECASE),
]

_SHOW_APPLICATIONS_PATTERNS = [
    re.compile(r"show\s+(?:my\s+)?applications?", re.IGNORECASE),
    re.compile(r"application\s+(?:status|history)", re.IGNORECASE),
    re.compile(r"list\s+(?:my\s+)?applications?", re.IGNORECASE),
]

_NEEDS_ATTENTION_PATTERNS = [
    re.compile(r"(?:show\s+)?applications?\s+(?:that\s+)?(?:need|needs)\s+(?:my\s+)?attention", re.IGNORECASE),
    re.compile(r"(?:show\s+)?pending\s+questions?", re.IGNORECASE),
]

_APPROVE_PATTERNS = [
    re.compile(r"^approve$", re.IGNORECASE),
    re.compile(r"approve\s+(?:application\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"approve\s+(?:all|the|these|those)\s*(?:applications?)?", re.IGNORECASE),
    re.compile(r"submit\s+(?:application\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"submit\s+(?:all|the|these|those)\s*(?:applications?)?", re.IGNORECASE),
    re.compile(r"^yes$", re.IGNORECASE),
    re.compile(r"yes\s*,?\s*submit", re.IGNORECASE),
]

_CANCEL_PATTERNS = [
    re.compile(r"cancel\s+(?:application\s+)?(\w+)", re.IGNORECASE),
    re.compile(r"cancel\s+(?:all|the|these|those)\s*(?:applications?)?", re.IGNORECASE),
    re.compile(r"no\s*,?\s*(?:don'?t|do\s+not)\s+submit", re.IGNORECASE),
    re.compile(r"stop", re.IGNORECASE),
]


def parse_user_intent(prompt: str, region: dict, llm=None) -> UserIntent:
    lowered = prompt.strip().lower()

    for pattern in _NEEDS_ATTENTION_PATTERNS:
        if pattern.search(lowered):
            return UserIntent(intent_type="needs_attention", message=prompt)

    for pattern in _SHOW_APPLICATIONS_PATTERNS:
        if pattern.search(lowered):
            return UserIntent(intent_type="show_applications", message=prompt)

    for pattern in _APPROVE_PATTERNS:
        m = pattern.search(lowered)
        if m:
            target_id = None
            try:
                target_id = m.group(1)
            except IndexError:
                pass
            if target_id and target_id in ("all", "the", "these", "those", "applications", "application"):
                target_id = None
            return UserIntent(intent_type="approve", target_id=target_id, message=prompt)

    for pattern in _CANCEL_PATTERNS:
        m = pattern.search(lowered)
        if m:
            target_id = None
            try:
                target_id = m.group(1)
            except IndexError:
                pass
            if target_id and target_id in ("all", "the", "these", "those", "applications", "application"):
                target_id = None
            return UserIntent(intent_type="cancel", target_id=target_id, message=prompt)

    apply_count = None
    for pattern in _APPLY_PATTERNS:
        m = pattern.search(lowered)
        if m:
            apply_count = int(m.group(1))
            break

    min_match = None
    for pattern in _MATCH_SCORE_PATTERNS:
        m = pattern.search(lowered)
        if m:
            min_match = int(m.group(1))
            break

    has_apply_intent = any(
        word in lowered
        for word in ["apply", "submit", "send application"]
    )

    query = parse_intent(prompt, region, llm)

    if apply_count is not None or has_apply_intent:
        return UserIntent(
            intent_type="apply",
            search_query=query,
            apply_count=apply_count,
            min_match_score=min_match,
            message=prompt,
        )

    return UserIntent(
        intent_type="search",
        search_query=query,
        message=prompt,
    )


ROLE_PHRASES: list[tuple[list[str], list[str]]] = [
    (
        ["software engineer", "software developer"],
        [
            "software engineering",
            "software engineer",
            "software developer",
            "software dev",
            "programmer",
            "developer",
            "coding",
            "backend",
            "frontend",
            "full stack",
            "full-stack",
            "web developer",
            "web dev",
        ],
    ),
    (
        ["data analyst", "data scientist"],
        ["data analyst", "data science", "data scientist", "analytics"],
    ),
    (
        ["qa / test engineer"],
        ["qa", "quality assurance", "software tester", "test engineer"],
    ),
    (
        ["finance"],
        ["finance", "financial", "accounting", "bcom", "investment", "banking"],
    ),
    (
        ["administrator / clerk"],
        ["admin", "clerk", "administration", "office administrator"],
    ),
    (
        ["it support"],
        ["it support", "help desk", "desktop support", "support technician"],
    ),
]

def _registry_role_phrases() -> list[tuple[list[str], list[str]]]:
    """Derive role triggers from the generic occupation registry.

    Every occupation contributes its label, direct titles, adjacent titles
    and sector qualification nouns ("nursing", "accounting", ...) as
    trigger phrases mapping to the occupation label — so any profession in
    the registry is recognised without hand-coded branches and no
    profession is privileged by default.
    """
    groups: list[tuple[list[str], list[str]]] = []
    for occ in OCCUPATIONS:
        triggers: list[str] = [occ.label.lower()]
        triggers.extend(t for t in occ.titles)
        triggers.extend(t for t in occ.adjacent)
        # sector nouns from qualifications ("nursing" → Nurse) but skip
        # tiny acronyms that would false-match as substrings
        triggers.extend(
            q for q in occ.qualifications
            if len(q) >= 5 or " " in q
        )
        unique = list(dict.fromkeys(triggers))
        unique.sort(key=len, reverse=True)
        groups.append(([occ.label], unique))
    return groups


_REGISTRY_ROLE_PHRASES = _registry_role_phrases()

MONEY_RE = re.compile(r"(?<!\d)(?:R|r|ZAR)?\s?(\d{1,3}(?:[, ]\d{3})*|\d+)\s?([kK])?")

DISTANCE_RE = re.compile(r"within\s+(\d+)\s*kms?|radius\s+of\s+(\d+)\s*kms?")

SENIORITY_ENTRY = [
    "entry-level",
    "entry level",
    "junior",
    "graduate",
    "recent graduate",
    "recently graduated",
    "intern",
    "internship",
    "internships",
    "learner",
    "trainee",
    "no experience",
    "0-2 years",
    "0 - 2 years",
]
SENIORITY_MID = ["mid-level", "mid level", "experienced", "2+ years", "3+ years"]
SENIORITY_SENIOR = ["senior", "principal", "lead", "head of", "5+ years", "10+ years"]

KEYWORD_STOPWORDS = {
    "find", "looking", "for", "the", "a", "an", "of", "to", "in", "on", "at",
    "and", "or", "with", "me", "my", "i", "am", "job", "jobs", "position",
    "positions", "role", "roles", "best", "matches", "show", "please", "can",
    "you", "help", "want", "need", "list", "search", "about", "any", "some",
    "also", "then", "really", "great", "good", "like", "would", "from", "up",
    "around", "new", "based", "across", "all", "within", "our", "their",
    "preferably", "preferred", "preference", "prefer", "preferable", "remote",
    "onsite", "on-site", "site", "office", "hybrid", "fully", "must", "only",
    "100", "paid", "paying", "pays", "pay", "salary", "salaries", "minimum",
    "least", "monthly", "month", "per", "annum", "annual", "week", "weeks",
    "year", "years", "experience", "recently", "recent", "engineering",
    "using",
}


def _matched_role_words(text: str) -> set[str]:
    """Words belonging to role phrases that actually matched *text*.

    Only consumed when the phrase is a substring of the text, so unmatched
    phrases don't silently eat keywords like "officer" or "enrolled".
    """
    words: set[str] = set()
    for _, phrases in ROLE_PHRASES:
        for phrase in phrases:
            if phrase in text:
                words.update(phrase.split())
    for _, phrases in _REGISTRY_ROLE_PHRASES:
        for phrase in phrases:
            if phrase in text:
                words.update(phrase.split())
    return words


def _consumed_words(region: dict, text: str = "") -> set[str]:
    words: set[str] = set(KEYWORD_STOPWORDS)
    # Role-phrase words are NOT consumed globally — only when the phrase
    # actually matched the query text.  This lets unmatched words like
    # "officer", "manager", "deputy" survive as keywords instead of
    # silently vanishing, which was the root cause of empty keyword lists
    # for queries like "HR officer" or "senior manager".
    if text:
        words.update(_matched_role_words(text))
    for markers in (SENIORITY_ENTRY, SENIORITY_MID, SENIORITY_SENIOR):
        for marker in markers:
            words.update(marker.split())
    words.update(str(region.get("name", "")).lower().split())
    for entry in region.get("locations", {}).values():
        words.add(str(entry.get("city", "")).lower())
        words.update(str(alias).lower() for alias in entry.get("aliases", []))
    for keywords in region.get("skills_dictionary", {}).values():
        for keyword in keywords:
            words.update(str(keyword).lower().split())
    return words


def parse_intent(prompt: str, region: dict, llm=None) -> JobQuery:
    query = _parse_with_llm(prompt, region, llm)
    if query is None:
        query = _parse_with_rules(prompt, region)
    return _normalize(query, region)


def _parse_with_llm(prompt: str, region: dict, llm) -> Optional[JobQuery]:
    if llm is None or not llm.is_available():
        return None
    schema = {
        "roles": ["list of job role strings"],
        "seniority": '"entry-level", "mid-level", "senior", or ""',
        "keywords": ["list of free-text keywords from the prompt"],
        "locations": [{"city": "city name as written", "radius_km": 50}],
        "remote": '"required", "preferred", "no", or "any"',
        "min_salary": "monthly minimum in prompt as a number, or null",
        "currency": "3-letter currency code from the region config",
        "skills": ["list of skills mentioned"],
    }
    system = (
        "You extract a structured job-search query from a user's natural-language "
        "request. Respond with ONLY valid JSON matching exactly this shape: "
        f"{json_dumps(schema)}. Use an empty string for unknown seniority, null for "
        "unknown salary, empty lists when nothing matches, and 'any' for remote when "
        "the user is indifferent."
    )
    user_prompt = (
        f"The user is a job seeker in this region: {region['name']} "
        f"(currency {region['currency']}). Request: {prompt!r}"
    )
    for attempt in range(2):
        try:
            raw = llm.chat_json(system, user_prompt)
            query = JobQuery(**raw)
            return query
        except (ValidationError, ValueError):
            continue
    return None


def _parse_with_rules(prompt: str, region: dict) -> JobQuery:
    lowered = prompt.lower()
    query = JobQuery(
        roles=_extract_roles(lowered),
        seniority=_extract_seniority(lowered),
        keywords=_extract_keywords(lowered, region),
        locations=_extract_locations(lowered, region),
        remote=_extract_remote(lowered),
        min_salary=_extract_salary(lowered),
        currency=region.get("currency", "ZAR"),
        skills=_extract_skills(lowered, region),
    )
    return query


def _extract_roles(text: str) -> list[str]:
    for roles, phrases in ROLE_PHRASES:
        if any(phrase in text for phrase in phrases):
            return roles
    for roles, phrases in _REGISTRY_ROLE_PHRASES:
        if any(phrase in text for phrase in phrases):
            return roles
    return []


def _matched_role_vocab(text: str) -> set[str]:
    """Words belonging to whichever role group matched the text.

    These words describe the SEARCH TARGET, never the candidate's skills,
    so skill extraction must not treat them as evidence of ability.
    """
    vocab: set[str] = set()
    groups = list(ROLE_PHRASES) + list(_REGISTRY_ROLE_PHRASES)
    for _, phrases in groups:
        if any(phrase in text for phrase in phrases):
            for phrase in phrases:
                vocab.update(phrase.split())
            break
    return vocab


def _extract_seniority(text: str) -> str:
    if any(word in text for word in SENIORITY_ENTRY):
        return "entry-level"
    if any(word in text for word in SENIORITY_SENIOR):
        return "senior"
    if any(word in text for word in SENIORITY_MID):
        return "mid-level"
    return ""


_KEYWORD_STOPWORDS = {
    "about", "again", "all", "also", "any", "area", "areas", "are", "best",
    "can", "career", "careers", "could", "find", "for", "from", "good",
    "great", "has", "have", "hello", "here", "hire", "hiring", "hope",
    "into", "job", "jobs", "just", "kind", "look", "looking", "lots",
    "may", "me", "might", "more", "most", "much", "must", "my", "near",
    "need", "okay", "openings", "opportunit", "over", "please", "plenty",
    "position", "positions", "really", "role", "roles", "search", "shall",
    "show", "some", "sort", "thanks", "that", "the", "their", "then",
    "there", "these", "they", "this", "those", "type", "under", "very",
    "vacanc", "want", "was", "well", "were", "what", "when", "where",
    "which", "will", "with", "work", "working", "would", "you", "your",
}


def _is_filler(token: str) -> bool:
    if token in _KEYWORD_STOPWORDS:
        return True
    # partial matches catch prefixed forms like "vacancies" / "opportunities"
    return any(token.startswith(sw) for sw in ("vacanc", "opportunit"))


def _extract_keywords(text: str, region: dict) -> list[str]:
    residue = text.lower()
    for word in _consumed_words(region, text=residue):
        residue = re.sub(rf"\b{re.escape(word)}\b", " ", residue)
    tokens = re.findall(r"[a-z][a-z-]+", residue)
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip("-")
        if len(token) < 3 or token in seen or _is_filler(token):
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= 10:
            break
    return keywords


def _extract_locations(text: str, region: dict) -> list[LocationRef]:
    distance = _extract_distance(text)
    found: list[LocationRef] = []
    for name, entry in region.get("locations", {}).items():
        aliases = [name] + list(entry.get("aliases", []))
        if any(alias in text for alias in aliases):
            found.append(LocationRef(city=entry["city"], radius_km=distance))
    return found


def _extract_distance(text: str) -> int:
    match = DISTANCE_RE.search(text)
    if match:
        for group in match.groups():
            if group:
                return int(group)
    return 50


def _extract_remote(text: str) -> str:
    if any(word in text for word in ["fully remote", "remote only", "must be remote", "100% remote"]):
        return "required"
    if "remote" in text:
        if any(word in text for word in ["not remote", "no remote", "onsite", "on-site", "in office"]):
            return "no"
        return "preferred"
    if any(word in text for word in ["onsite", "on-site", "in office"]):
        return "no"
    return "any"


def _extract_salary(text: str) -> Optional[int]:
    candidate: Optional[int] = None
    for match in MONEY_RE.finditer(text):
        amount = _parse_amount(match.group(1), match.group(2))
        if amount is None:
            continue
        if 500 <= amount <= 2_000_000:
            candidate = amount
            break
    return candidate


def _parse_amount(digits: Optional[str], suffix: Optional[str]) -> Optional[int]:
    if not digits:
        return None
    clean = re.sub(r"[^\d]", "", digits)
    if not clean:
        return None
    try:
        value = int(clean)
    except ValueError:
        return None
    if suffix and suffix.lower() == "k":
        value *= 1000
    return value


def _extract_skills(text: str, region: dict) -> list[str]:
    """Skills the user CLAIMS, never what they are searching for.

    "Find me finance jobs" is a search target ("finance"), not a claimed
    skill. The matched role group's vocabulary is stripped before
    dictionary matching so sector nouns in the request cannot masquerade
    as ability — while genuinely mentioned skills ("using Python")
    survive untouched.
    """
    residue = text.lower()
    role_vocab = _matched_role_vocab(residue)
    for word in role_vocab:
        residue = re.sub(rf"\b{re.escape(word)}\b", " ", residue)

    skills: list[str] = []
    for skill, keywords in region.get("skills_dictionary", {}).items():
        if any(keyword in residue for keyword in keywords):
            skills.append(skill)
    for word in ["computer science", "bcom"]:
        if word in residue:
            skills.append(word)

    # final guard: drop anything whose words are all search-target vocab
    cleaned: list[str] = []
    for skill in skills:
        tokens = {t for t in re.findall(r"[a-z0-9+#]+", skill)}
        if tokens and tokens <= role_vocab:
            continue
        cleaned.append(skill)
    return cleaned


def _normalize(query: JobQuery, region: dict) -> JobQuery:
    query.currency = region.get("currency", query.currency)
    seen: set[str] = set()
    roles: list[str] = []
    for role in query.roles:
        key = role.strip().lower()
        if key and key not in seen:
            seen.add(key)
            roles.append(role.strip())
    query.roles = roles
    query.skills = list(dict.fromkeys(s.strip().lower() for s in query.skills if s.strip()))
    query.keywords = list(dict.fromkeys(k.strip().lower() for k in query.keywords if k.strip()))
    seen_cities: set[str] = set()
    locations: list[LocationRef] = []
    for loc in query.locations:
        key = loc.city.strip().lower()
        if key and key not in seen_cities:
            seen_cities.add(key)
            locations.append(LocationRef(city=loc.city.strip(), radius_km=max(loc.radius_km, 0)))
    query.locations = locations
    return query


def json_dumps(value) -> str:
    import json

    return json.dumps(value)
