from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError


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
    "learner",
    "trainee",
    "no experience",
    "0-2 years",
    "0 - 2 years",
]
SENIORITY_MID = ["mid-level", "mid level", "experienced", "2+ years", "3+ years"]
SENIORITY_SENIOR = ["senior", "principal", "lead", "head of", "5+ years", "10+ years"]


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
        keywords=_extract_keywords(lowered),
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
    return []


def _extract_seniority(text: str) -> str:
    if any(word in text for word in SENIORITY_ENTRY):
        return "entry-level"
    if any(word in text for word in SENIORITY_SENIOR):
        return "senior"
    if any(word in text for word in SENIORITY_MID):
        return "mid-level"
    return ""


def _extract_keywords(text: str) -> list[str]:
    words = ["computer science", "bcom", "degree", "matric", "graduat"]
    return [word for word in words if word in text]


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
    skills: list[str] = []
    for skill, keywords in region.get("skills_dictionary", {}).items():
        if any(keyword in text for keyword in keywords):
            skills.append(skill)
    for word in ["computer science", "bcom"]:
        if word in text:
            skills.append(word)
    return skills


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
