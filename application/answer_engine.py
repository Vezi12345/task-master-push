from __future__ import annotations

"""Application answer engine.

Answers arbitrary application questions from every verified source the
system holds: stored user answers, the candidate knowledge profile, the CV,
and derivations computed from verified facts. Questions are classified
semantically — an unseen phrasing still maps to the right candidate
attribute. The engine never fabricates: when information is genuinely
unknown it returns ``UNKNOWN`` and asks the user.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from candidate.profile import CandidateProfile, SENSITIVE_KEYS
from application.skill_format import format_skill_list, format_skill_name


class AnswerType(str, Enum):
    VERIFIED = "verified"
    USER_PROVIDED = "user_provided"
    DERIVED = "derived"
    GENERATED_FROM_EVIDENCE = "generated_from_evidence"
    UNKNOWN = "unknown"
    NEEDS_USER = "needs_user"
    SENSITIVE = "sensitive"
    CONSENT_REQUIRED = "consent_required"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class AnswerResult:
    question: str
    answer: Optional[str]
    answer_type: AnswerType
    confidence: Confidence
    auto_fill: bool
    needs_user: bool
    field_key: str = ""
    category: str = ""
    explanation: str = ""
    source: str = ""

    @property
    def is_answered(self) -> bool:
        return self.answer is not None


# ---------------------------------------------------------------------------
# question classification
# ---------------------------------------------------------------------------

_QUESTION_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    # (field_key, category, keyword patterns) — order matters, most specific first
    ("recent_graduate", "eligibility", (
        "recent graduate", "recently graduated", "graduate within",
        "new graduate", "completed your studies within",
    )),
    ("hs_completion_year", "education", (
        "high school completion", "matric year", "matriculated",
        "high school year", "when did you complete high school",
        "when did you matric", "year of matric",
    )),
    ("preferred_name", "identity", (
        "preferred name", "what should we call you", "nickname",
        "what name do you prefer", "preferred first name",
    )),
    ("south_african_citizen", "eligibility", (
        "south african citizen", "sa citizen", "sa id", "south african id",
        "citizen of south africa",
    )),
    ("citizenship", "eligibility", (
        "citizenship", "nationality", "citizen of", "which country",
        "legal right to work in", "permitted to work in",
    )),
    ("nationality", "eligibility", (
        "nationality", "what is your nationality",
    )),
    ("work_authorisation", "eligibility", (
        "authorised to work", "authorized to work", "right to work",
        "legally entitled to work", "eligible to work", "work permit",
        "work visa", "work authorisation", "work authorization",
    )),
    ("id_number", "eligibility", (
        "id number", "id nr", "national id", "passport number",
        "identity number", "sa id number", "south african id",
    )),
    ("gender_pronouns", "demographic", (
        "pronouns", "preferred pronouns", "gender pronouns",
    )),
    ("discipline", "education", (
        "discipline", "field of study", "field of discipline",
        "area of study", "specialisation", "specialization",
        "study field", "academic discipline",
    )),
    ("country_of_residence", "identity", (
        "country of residence", "country you live in", "which country do you live",
        "country do you reside", "residential country", "current country",
        "^country$",
    )),
    ("heard_of_company", "other", (
        "heard of", "familiar with", "know about", "know of",
        "prior to applying", "aware of",
    )),
    ("drivers_licence", "requirements", (
        "driver's licence", "drivers licence", "driver's license",
        "drivers license", "driver license", "driving licence",
        "driving license", "code 8", "code 10",
        "code b", "code eb", "code c1", "in possession of a valid",
        "have a licence", "have a license", "licence do you hold",
        "license do you hold", "legally drive", "legal to drive",
        "can you drive", "able to drive", "do you drive",
        "valid licence", "valid license",
    )),
    ("vehicle", "requirements", (
        "own vehicle", "own car", "own transport", "reliable transport",
    )),
    ("expected_salary", "compensation", (
        "salary expectation", "expected salary", "salary requirement",
        "remuneration", "compensation", "pay expectation",
        "what salary", "salary range", "package you expect",
        "expecting to earn", "minimum pay",
    )),
    ("notice_period", "logistics", (
        "notice period", "currently employed notice", "how much notice",
        "notice do you have to give",
    )),
    ("availability", "availability", (
        "availability", "start date", "how soon could you start",
        "how soon can you start", "when could you start",
        "when can you start", "available to start", "earliest start",
        "when would you be able to start", "commence work", "commence employment",
    )),
    ("relocation", "preferences", (
        "relocate", "relocation", "willing to move", "moving to another city",
        "open to moving", "relocating",
    )),
    ("travel_preference", "preferences", (
        "willing to travel", "comfortable travelling", "comfortable traveling",
        "travel to client", "travel for work", "travelling", "traveling",
        "travel as part of", "travel requirement",
    )),
    ("teamwork_experience", "experience", (
        "distributed team", "remote team", "virtual team",
        "cross-functional", "cross functional", "collaborat",
        "team player", "teamwork", "interpersonal",
        "work in a team", "working in a team", "work well in a team",
        "team environment", "worked with distributed",
    )),
    ("remote_work_experience", "experience", (
        "worked remotely", "working remotely", "remote work experience",
        "experience working from home", "remote experience",
    )),
    ("work_preference", "preferences", (
        "remote work", "work remotely", "hybrid", "on-site", "onsite",
        "work from home", "work from the office", "office based",
    )),
    ("preferred_locations", "preferences", (
        "preferred location", "location preference", "which city would you",
        "where would you prefer to work", "preferred area",
    )),
    ("date_of_birth", "demographic", (
        "date of birth", "birth date", "how old are you", "your age",
        "age bracket", "how old",
    )),
    ("race", "demographic", (
        "race", "equity group", "population group", "bbbee",
        "broad-based black", "designated group", "african", "coloured",
        "indian", "white",  # equity-group vocabulary; only ever answered from stored data
    )),
    ("gender", "demographic", (
        "gender", "male or female", "female", "male",
    )),
    ("disability", "demographic", (
        "disability", "disabled", "handicap", "impairment",
    )),
    # academic RESULT must match BEFORE the qualification rule below
    # ('degree' alone would otherwise swallow 'degree result')
    ("education_result", "education", (
        "degree result", "university result", "final result", "academic result",
        "what was your result", "your average", "final mark", "final grade",
        "gpa", "percentage", "grading system", "grading scale",
        "expected result if you have not yet graduated",
    )),
    ("highest_qualification", "education", (
        "highest qualification", "qualification", "degree", "diploma",
        "certificate you obtained", "educational background",
        "highest education", "what did you study", "field of study",
        "studied", "academic background", "tertiary",
    )),
    ("graduation_year", "education", (
        "graduate", "graduated", "graduation", "when did you complete",
        "year of completion", "completed your studies",
    )),
    ("years_experience", "experience", (
        "years of experience", "years' experience", "years experience",
        "how long have you worked", "work experience do you have",
        "professional experience", "experience do you have",
        "how many years",
    )),
    ("motivation", "motivation", (
        "why do you want", "why are you interested", "why this position",
        "why this role", "why this company", "why our company",
        "why should we hire", "what attracts you", "motivation",
        "motivates you", "what drives you",
        "what interests you", "why would you like", "why do you wish",
        "suitable for this position", "good fit for this",
    )),
    ("email", "personal", ("email address", "email", "e mail", "contact email")),
    ("phone", "personal", ("phone number", "phone", "contact number",
                           "mobile number", "cellphone")),
    # first/last must be checked BEFORE the generic full-name rule
    ("first_name", "personal", ("first name", "given name")),
    ("last_name", "personal", ("surname", "last name", "family name")),
    ("preferred_name", "personal", ("preferred name", "nickname", "name you prefer",
                                    "what should we call you")),
    ("name", "personal", ("full name", "your name")),
    # high school — only ever answered from an explicit high-school record
    ("hs_mathematics_result", "education", (
        "mathematics result", "maths result", "math mark", "mathematics mark",
        "how did you perform in mathematics", "mathematics performance",
    )),
    ("hs_native_language", "education", (
        "native language", "home language",
    )),
    ("hs_overall_result", "education", (
        "matric result", "high school result", "high school overall",
        "secondary school result",
    )),
    # online presence — verified URLs ONLY, never names
    ("online_website", "web_link", ("website", "personal site", "web url")),
    ("online_linkedin", "web_link", ("linkedin profile", "linkedin url", "linkedin")),
    ("online_github", "web_link", ("github profile", "github url", "github")),
    ("online_portfolio", "web_link", ("portfolio site", "portfolio url", "portfolio link")),
    # projects — first-class evidence source
    ("personal_project", "projects", (
        "personal project", "personal software project", "side project",
        "own project", "project outside of curriculum",
        "outside of curriculum or work", "software project, outside",
        "project you are most proud",
    )),
    ("languages_spoken", "languages", (
        "languages do you speak", "spoken languages", "language proficiency",
        "which languages", "what languages",
    )),
    ("location", "personal", ("where are you based", "current location", "which city do you live")),
]

_STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does did for from had has have "
    "he her his how i if in into is it its may me my no not of on or our shall she "
    "should so some such than that the their them then there these they this those "
    "to us was we were what when where which who whom why will with would you your "
    "yours do does did".split()
)

_WORD_BOUNDARY_SKILLS = {"r", "go", "c", "c#", "c++"}


def classify_question(question: str) -> tuple[str, str]:
    """Semantically classify a question into ``(field_key, category)``.

    Never requires an exact string match — unseen phrasings map to the
    canonical attribute they ask about.
    """
    text = _normalise(question)
    for field_key, category, patterns in _QUESTION_RULES:
        for pattern in patterns:
            if pattern in text:
                # "Start date month" / "Start date year" are date-picker
                # sub-fields, not availability text.  They need numeric
                # values or select-dropdown picks, not "AVAILABLE IMMEDIATELY".
                if (
                    field_key == "availability"
                    and re.search(r"year|month|day", text)
                ):
                    return "", "other"
                return field_key, category
    if "?" in question or _looks_like_question(question):
        return "", "other"
    return "", "other"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).strip()


def _looks_like_question(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith((
        "are ", "is ", "do ", "does ", "did ", "can ", "could ", "would ",
        "will ", "have ", "has ", "what ", "when ", "where ", "who ", "why ",
        "how ", "which ", "please ", "tell ",
    ))


# ---------------------------------------------------------------------------
# city → country, discipline mapping, date parsing helpers
# ---------------------------------------------------------------------------

_CITY_COUNTRY: dict[str, str] = {
    "durban": "South Africa",
    "johannesburg": "South Africa",
    "cape town": "South Africa",
    "pretoria": "South Africa",
    "centurion": "South Africa",
    "sandton": "South Africa",
    "port elizabeth": "South Africa",
    "bloemfontein": "South Africa",
    "stellenbosch": "South Africa",
    "midrand": "South Africa",
    "rosebank": "South Africa",
    "fourways": "South Africa",
    "randburg": "South Africa",
    "century city": "South Africa",
    "london": "United Kingdom",
    "manchester": "United Kingdom",
    "birmingham": "United Kingdom",
    "edinburgh": "United Kingdom",
    "glasgow": "United Kingdom",
    "new york": "United States",
    "san francisco": "United States",
    "los angeles": "United States",
    "chicago": "United States",
    "seattle": "United States",
    "austin": "United States",
    "boston": "United States",
    "washington": "United States",
    "toronto": "Canada",
    "vancouver": "Canada",
    "montreal": "Canada",
    "ottawa": "Canada",
    "sydney": "Australia",
    "melbourne": "Australia",
    "brisbane": "Australia",
    "perth": "Australia",
    "auckland": "New Zealand",
    "wellington": "New Zealand",
    "berlin": "Germany",
    "munich": "Germany",
    "amsterdam": "Netherlands",
    "dublin": "Ireland",
    "paris": "France",
    "singapore": "Singapore",
    "dubai": "United Arab Emirates",
    "abu dhabi": "United Arab Emirates",
    "nairobi": "Kenya",
    "lagos": "Nigeria",
    "accra": "Ghana",
}


def _city_to_country(location: str) -> Optional[str]:
    """Infer country from a city or 'City, Country' string."""
    loc = (location or "").strip().lower()
    if not loc:
        return None
    # "Durban, South Africa" → split and try last part first
    parts = [p.strip() for p in re.split(r"[,;]", loc) if p.strip()]
    if len(parts) >= 2:
        country = _CITY_COUNTRY.get(parts[-1])
        if country:
            return country
    # Try the full location
    country = _CITY_COUNTRY.get(loc)
    if country:
        return country
    # Try each part
    for part in parts:
        country = _CITY_COUNTRY.get(part)
        if country:
            return country
    return None


_DISCIPLINE_MAP: dict[str, str] = {
    "application development": "Information Technology",
    "application development": "Information Technology",
    "software development": "Computer Science",
    "computer science": "Computer Science",
    "information technology": "Information Technology",
    "information technology": "Information Technology",
    "computer engineering": "Computer Engineering",
    "electrical engineering": "Electrical Engineering",
    "mechanical engineering": "Mechanical Engineering",
    "data science": "Data Science",
    "data analytics": "Data Science",
    "mathematics": "Mathematics",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "business administration": "Business Administration",
    "business management": "Business Administration",
    "finance": "Finance",
    "accounting": "Accounting",
    "marketing": "Marketing",
    "communications": "Communications",
    "design": "Design",
    "graphic design": "Design",
    "web design": "Design",
    "networking": "Information Technology",
    "cybersecurity": "Information Technology",
    "information systems": "Information Technology",
    "software engineering": "Software Engineering",
    "systems engineering": "Systems Engineering",
}


def _map_discipline(field_of_study: str) -> str:
    """Map a free-text education field to a standard discipline value."""
    lowered = (field_of_study or "").strip().lower()
    if not lowered:
        return ""
    # Exact match
    if lowered in _DISCIPLINE_MAP:
        return _DISCIPLINE_MAP[lowered]
    # Partial match
    for key, value in _DISCIPLINE_MAP.items():
        if key in lowered or lowered in key:
            return value
    # Capitalize and return as-is
    return field_of_study.strip().title()


def _extract_discipline_from_qualification(qualification: str) -> Optional[str]:
    """Try to extract a discipline from a free-text qualification string.
    
    For example, 'Diploma in Information and Communication Technology – Application Development'
    should yield 'Information Technology'."""
    if not qualification:
        return None
    # Try the full qualification string first
    mapped = _map_discipline(qualification)
    if mapped and mapped.lower() != qualification.strip().lower():
        return mapped
    # Try each segment split by common delimiters
    for sep in ("–", "-", "–", "—", ",", "|", "/"):
        if sep in qualification:
            for part in qualification.split(sep):
                part = part.strip()
                if part:
                    mapped = _map_discipline(part)
                    if mapped and mapped.lower() != part.lower():
                        return mapped
    return None


def _parse_date(text: str) -> Optional[date]:
    """Best-effort parse of a date string into a date object."""
    text = (text or "").strip()
    if not text:
        return None
    # Try common formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m", "%B %Y",
                "%b %Y", "%Y", "%d %B %Y", "%d %b %Y"):
        try:
            return date.strptime(text, fmt)
        except ValueError:
            continue
    # Year only
    m = re.match(r"^(19|20)\d{2}$", text)
    if m:
        return date(int(text), 6, 1)  # mid-year default
    return None


# ---------------------------------------------------------------------------
# semantic equivalence of questions (memory reuse)
# ---------------------------------------------------------------------------

def content_tokens(text: str) -> set[str]:
    return {
        tok for tok in _normalise(text).split()
        if tok not in _STOPWORDS and len(tok) > 1
    }


def token_similarity(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def questions_equivalent(q1: str, q2: str, threshold: float = 0.55) -> bool:
    k1, _ = classify_question(q1)
    k2, _ = classify_question(q2)
    if k1 and k1 == k2:
        return True
    return token_similarity(q1, q2) >= threshold


# ---------------------------------------------------------------------------
# derived facts
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _extract_year(value: str) -> Optional[int]:
    match = _YEAR_RE.search(value or "")
    return int(match.group(0)) if match else None


def latest_graduation_year(profile: CandidateProfile) -> Optional[int]:
    years = [
        y for edu in profile.education
        for y in [_extract_year(edu.end_date)]
        if y is not None
    ]
    return max(years) if years else None


def compute_experience_months(profile: CandidateProfile) -> Optional[int]:
    """Total professional experience in whole months, derived only from
    explicit employment date ranges. Returns None when dates are unknown."""
    total_months = 0
    today = date.today()
    found_dates = False
    for exp in profile.experience:
        etype = (exp.experience_type or "employment").lower()
        if etype == "project":
            continue
        start = _parse_date(exp.start_date)
        end = _parse_date(exp.end_date)
        if start is None:
            continue
        found_dates = True
        if end is None:
            end_text = (exp.end_date or "").lower()
            if end_text and not any(w in end_text for w in ("present", "current", "now")):
                continue
            end = today
        if end < start:
            continue
        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_months += max(0, months)
    if not found_dates:
        return None
    return total_months


def compute_years_experience(profile: CandidateProfile) -> Optional[float]:
    """Total professional experience in years, derived only from explicit
    employment date ranges. Returns None when dates are unknown."""
    months = compute_experience_months(profile)
    if months is None:
        return None
    return round(months / 12, 1)


def compute_age(profile: CandidateProfile) -> Optional[int]:
    dob = _parse_date(getattr(profile, "date_of_birth", "") or "")
    if dob is None:
        year = _extract_year(getattr(profile, "date_of_birth", "") or "")
        if year is None:
            return None
        today = date.today()
        return today.year - year
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _parse_date(value: str):
    value = (value or "").strip()
    if not value:
        return None
    iso = re.match(r"^(\d{4})-(\d{1,2})(?:-(\d{1,2}))?$", value)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3) or 1))
        except ValueError:
            return None
    month_year = re.match(
        r"^([a-z]{3,9})\s+(\d{4})$", value.lower(), re.IGNORECASE
    )
    if month_year:
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        key = month_year.group(1)[:3]
        if key in months:
            try:
                return date(int(month_year.group(2)), months[key], 1)
            except ValueError:
                return None
    year_only = re.match(r"^(19|20)\d{2}$", value)
    if year_only:
        return date(int(value), 1, 1)
    return None


# ---------------------------------------------------------------------------
# evidence search
# ---------------------------------------------------------------------------

def _word_boundary_in(term: str, text: str) -> bool:
    pattern = r"\b" + re.escape(term.strip()) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def find_skill_evidence(topic: str, profile: CandidateProfile) -> Optional[str]:
    """Locate verified evidence that the candidate knows ``topic``.

    Returns a truthful human-readable justification, or None."""
    topic_clean = topic.strip().lower()
    if not topic_clean:
        return None
    short = len(topic_clean) <= 3

    def matches(text: str) -> bool:
        if short and topic_clean.rstrip("+#. ") in _WORD_BOUNDARY_SKILLS:
            return _word_boundary_in(topic_clean, text)
        return topic_clean in text.lower()

    def contains_term(term: str, text: str) -> bool:
        """Containment check that never lets a short token (e.g. 'r')
        match inside an ordinary word like 'travelling'."""
        t = term.strip().lower()
        if not t:
            return False
        if len(t) <= 3 or t.rstrip("+#. ") in _WORD_BOUNDARY_SKILLS:
            return _word_boundary_in(t, text)
        return t in text.lower()

    for skill in profile.skills:
        s = skill.strip().lower()
        if not s:
            continue
        if contains_term(topic_clean, skill) or contains_term(s, topic_clean):
            return f"{format_skill_name(skill)} is listed among my skills"
    for cert in profile.certifications:
        if contains_term(topic_clean, cert.name):
            return f"I hold the certification \"{cert.name}\""
    for proj in profile.projects:
        hay = f"{proj.name} {proj.description} {' '.join(proj.technologies)}"
        if matches(hay):
            used = f" using {', '.join(proj.technologies[:3])}" if proj.technologies else ""
            return f"it was applied in my project \"{proj.name}\"{used}"
    for exp in profile.experience:
        hay = f"{exp.title} {exp.description} {' '.join(exp.skills)}"
        if matches(hay):
            at = f" at {exp.company}" if exp.company else ""
            return f"it formed part of my work as {exp.title or 'a team member'}{at}"
    if matches(f"{profile.professional_summary} {profile.education[0].field if profile.education else ''}"):
        return "it aligns with my field of study"
    return None


def relevant_skills_for_job(profile: CandidateProfile, job_context: dict) -> list[str]:
    """Candidate skills that the real job description also mentions."""
    job_text = " ".join(str(v) for v in [
        job_context.get("title", ""),
        job_context.get("description", ""),
        job_context.get("requirements", ""),
    ]).lower()
    relevant = []
    for skill in profile.skills:
        s = skill.strip().lower()
        if not s or len(s) < 2:
            continue
        if s in _WORD_BOUNDARY_SKILLS:
            if _word_boundary_in(s, job_text):
                relevant.append(skill)
        elif s in job_text:
            relevant.append(skill)
    return relevant


# ---------------------------------------------------------------------------
# generation from evidence
# ---------------------------------------------------------------------------

def generate_motivation_answer(
    profile: CandidateProfile,
    job_context: dict,
) -> Optional[str]:
    """Build a truthful motivation answer from verified candidate facts plus
    the real job context. Mentions only skills/evidence that exist."""
    company = job_context.get("company") or "your organisation"
    title = job_context.get("title") or "this position"
    relevant = relevant_skills_for_job(profile, job_context)

    parts: list[str] = []
    qual = ""
    if profile.education:
        edu = profile.education[0]
        qual = " ".join(p for p in (edu.qualification, edu.field) if p).strip()
    if qual:
        parts.append(f"As a {qual} graduate")
    elif profile.experience:
        parts.append("With my professional background")

    if relevant:
        listed = format_skill_list(relevant[:5])
        parts.append(
            f"I am excited about the {title} role at {company}. "
            f"My experience with {listed} matches the skills the role requires"
        )
    elif profile.skills:
        listed = format_skill_list(profile.skills[:5])
        parts.append(
            f"I am excited about the {title} role at {company}. "
            f"My skills in {listed} give me a foundation to contribute quickly"
        )
    else:
        parts.append(f"The {title} role at {company} aligns closely with my studies and career goals")

    if profile.projects and relevant:
        proj = profile.projects[0]
        parts.append(f"For example, I built \"{proj.name}\", which gave me hands-on practice I would bring to this role")
    elif profile.experience and relevant:
        exp = profile.experience[0]
        at = f" at {exp.company}" if exp.company else ""
        parts.append(f"In my role as {exp.title or 'a developer'}{at}, I applied these skills in practice")
    elif profile.certifications:
        cert_names = ", ".join(c.name for c in profile.certifications[:2] if c.name)
        if cert_names:
            parts.append(f"I also hold certifications such as {cert_names}, demonstrating my commitment to continuous learning")

    sentence = ". ".join(p for p in parts if p)
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def generate_skill_answer(topic: str, profile: CandidateProfile) -> Optional[str]:
    evidence = find_skill_evidence(topic, profile)
    if evidence is None:
        return None
    return f"Yes. {evidence[0].upper() + evidence[1:]}."


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

_SENSITIVE_FIELD_KEYS = frozenset(SENSITIVE_KEYS)


def answer_question(
    question: str,
    candidate_profile: Optional[CandidateProfile],
    job_context: Optional[dict] = None,
    answer_store=None,
) -> AnswerResult:
    """Answer one application question truthfully.

    Resolution order: remembered answers → answer store → structured profile
    knowledge → logical derivation → generation from verified evidence →
    unknown (ask the user).
    """
    job_context = job_context or {}
    field_key, category = classify_question(question)

    # 1. previously answered (semantic memory on the profile)
    if candidate_profile is not None:
        mem = _lookup_memory(candidate_profile, question, field_key)
        if mem is not None:
            return AnswerResult(
                question=question, answer=mem.answer,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=mem.field_key or field_key, category=category,
                explanation="Previously supplied by you",
                source="memory",
            )

    # 2. persisted answer store keyed by canonical field
    if answer_store is not None and field_key and answer_store.has(field_key):
        return AnswerResult(
            question=question, answer=answer_store.get(field_key),
            answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
            auto_fill=True, needs_user=False,
            field_key=field_key, category=category,
            explanation="Saved answer", source="answer_store",
        )

    if candidate_profile is None:
        return _unknown(question, field_key, category, "No candidate information available")

    # 3. sensitive attributes: only ever echoed from explicitly stored values
    if field_key in _SENSITIVE_FIELD_KEYS:
        stored = candidate_profile.get_known_value(field_key)
        if stored:
            return AnswerResult(
                question=question, answer=str(stored),
                answer_type=AnswerType.SENSITIVE, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="Explicitly provided by you", source="profile",
            )
        return _unknown(
            question, field_key, category,
            "Sensitive information is never inferred — please provide it",
        )

    # 3b. structured evidence sections
    if field_key == "education_result":
        edu = next((e for e in candidate_profile.education if e.result), None)
        if edu:
            return AnswerResult(
                question=question, answer=edu.result,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation=(
                    f"Result recorded on your {(edu.institution or 'education')} record"
                ),
                source="profile.education",
            )
        return _unknown(
            question, field_key, category,
            "Your degree result is not in your profile — please provide it "
            "(a qualification name is NOT a result)",
        )

    if field_key == "hs_mathematics_result":
        hs = candidate_profile.high_school
        if hs.mathematics_result:
            return AnswerResult(
                question=question, answer=hs.mathematics_result,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="From your high-school profile", source="profile.high_school",
            )
        return _unknown(
            question, field_key, category,
            "Not in your profile — high-school results are never inferred",
        )

    if field_key == "hs_native_language":
        hs = candidate_profile.high_school
        if hs.native_language:
            value = hs.native_language + (
                f" ({hs.native_language_result})" if hs.native_language_result else ""
            )
            return AnswerResult(
                question=question, answer=value,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="From your high-school profile", source="profile.high_school",
            )
        return _unknown(
            question, field_key, category,
            "Not in your profile — please provide it",
        )

    if field_key == "hs_overall_result":
        hs = candidate_profile.high_school
        if hs.overall_result:
            return AnswerResult(
                question=question, answer=hs.overall_result,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="From your high-school profile", source="profile.high_school",
            )
        return _unknown(
            question, field_key, category,
            "Not in your profile — please provide it",
        )

    if field_key.startswith("online_"):
        link_key = field_key[len("online_"):]
        url = candidate_profile.online_profiles.get(link_key)
        if url:
            return AnswerResult(
                question=question, answer=url,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="Verified URL saved in your profile",
                source="profile.online_profiles",
            )
        return _unknown(
            question, field_key, category,
            "No verified URL in your profile — links are never guessed",
        )

    if field_key == "personal_project":
        projects = candidate_profile.projects
        chosen = next((p for p in projects if p.is_personal), None) \
            or (projects[0] if projects else None)
        if chosen:
            bits = []
            if chosen.description:
                bits.append(chosen.description.strip())
            elif chosen.name:
                bits.append(f"Project: {chosen.name}")
            tech = ", ".join(chosen.technologies)
            if tech:
                bits.append(f"Technologies used: {tech}")
            role_bits = []
            if chosen.role:
                role_bits.append(f"My role: {chosen.role}")
            role_bits.extend(a.strip() for a in chosen.achievements if a.strip())
            if role_bits:
                bits.append(". ".join(role_bits))
            draft = ". ".join(b.rstrip('.') for b in bits if b) + "."
            return AnswerResult(
                question=question, answer=draft,
                answer_type=AnswerType.GENERATED_FROM_EVIDENCE,
                confidence=Confidence.MEDIUM,
                auto_fill=False, needs_user=False,
                field_key=field_key, category=category,
                explanation=(
                    f"AI-generated draft based only on your '{chosen.name}' "
                    "project record — review before submitting"
                ),
                source=f"profile.projects:{chosen.name}",
            )
        return _unknown(
            question, field_key, category,
            "No project details in your profile — please describe one yourself",
        )

    if field_key == "languages_spoken":
        if candidate_profile.languages:
            names = ", ".join(l.name for l in candidate_profile.languages)
            return AnswerResult(
                question=question, answer=names,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="Languages listed in your profile", source="profile.languages",
            )
        return _unknown(question, field_key, category,
                        "Languages not listed in your profile")

    # 4. directly verified structured knowledge
    stored = candidate_profile.get_known_value(field_key) if field_key else None
    if stored:
        return AnswerResult(
            question=question, answer=str(stored),
            answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
            auto_fill=True, needs_user=False,
            field_key=field_key, category=category,
            explanation="From your CV/profile", source="profile",
        )

    # 4a. names split from the full profile name
    if field_key in ("first_name", "last_name") and candidate_profile.name:
        parts = candidate_profile.name.split()
        if field_key == "first_name" and parts:
            return AnswerResult(
                question=question, answer=parts[0],
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="From your CV/profile", source="profile",
            )
        if field_key == "last_name":
            if len(parts) > 1:
                return AnswerResult(
                    question=question, answer=parts[-1],
                    answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                    auto_fill=True, needs_user=False,
                    field_key=field_key, category=category,
                    explanation="From your CV/profile", source="profile",
                )
            return _unknown(
                question, field_key, category,
                "Profile only has one name — please provide your surname",
            )

    if field_key == "highest_qualification" and candidate_profile.education:
        edu = candidate_profile.education[0]
        qual = " ".join(p for p in (edu.qualification, edu.field) if p).strip()
        if qual:
            return AnswerResult(
                question=question, answer=qual,
                answer_type=AnswerType.VERIFIED, confidence=Confidence.HIGH,
                auto_fill=True, needs_user=False,
                field_key=field_key, category=category,
                explanation="From your education history", source="profile",
            )

    # 5. derivation from verified facts
    derived = _try_derive(field_key, candidate_profile)
    if derived is not None:
        answer, explanation = derived
        return AnswerResult(
            question=question, answer=answer,
            answer_type=AnswerType.DERIVED, confidence=Confidence.HIGH,
            auto_fill=True, needs_user=False,
            field_key=field_key, category=category,
            explanation=explanation, source="derived",
        )

    # 6. generation strictly from verified evidence
    generated = _try_generate(question, field_key, candidate_profile, job_context)
    if generated is not None:
        answer, explanation = generated
        return AnswerResult(
            question=question, answer=answer,
            answer_type=AnswerType.GENERATED_FROM_EVIDENCE,
            confidence=Confidence.MEDIUM,
            auto_fill=True, needs_user=False,
            field_key=field_key, category=category,
            explanation=explanation, source="generated",
        )

    return _unknown(question, field_key, category, "Not enough reliable information")


def _unknown(question: str, field_key: str, category: str, reason: str) -> AnswerResult:
    return AnswerResult(
        question=question, answer=None,
        answer_type=AnswerType.UNKNOWN, confidence=Confidence.UNKNOWN,
        auto_fill=False, needs_user=True,
        field_key=field_key, category=category,
        explanation=reason, source="",
    )


def _lookup_memory(
    profile: CandidateProfile,
    question: str,
    field_key: str,
) -> Optional["object"]:
    best = None
    best_score = 0.0
    for mem in profile.question_memory:
        if mem.question.lower() == question.lower():
            return mem
        if field_key and mem.field_key and mem.field_key == field_key:
            score = 1.0
        else:
            score = token_similarity(mem.question, question)
            if score < 0.55:
                continue
        if score > best_score:
            best, best_score = mem, score
    return best


def _try_derive(field_key: str, profile: CandidateProfile) -> Optional[tuple[str, str]]:
    if field_key == "recent_graduate":
        year = latest_graduation_year(profile)
        if year is None:
            return None
        delta = date.today().year - year
        if delta <= 2:
            return "Yes", f"Graduated {year}, within two years"
        return "No", f"Graduated {year}, more than two years ago"

    if field_key == "years_experience":
        months = compute_experience_months(profile)
        if months is None:
            return None
        if months >= 12:
            years = months / 12
            display = str(int(years)) if years == int(years) else f"{years:.1f}"
            return f"{display} year{'s' if years != 1 else ''}", \
                f"Calculated from your employment history ({months} months total)"
        unit = "month" if months == 1 else "months"
        return f"Less than 1 year ({months} {unit})", \
            "Calculated from your employment history"

    if field_key == "age":
        age = compute_age(profile)
        if age is not None and getattr(profile, "date_of_birth", ""):
            return str(age), "Calculated from your date of birth"
        return None

    if field_key == "south_african_citizen":
        citizenship = (profile.citizenship or "").lower()
        if "south africa" in citizenship or "sa citizen" in citizenship:
            return "Yes", f"Derived from your stated citizenship ({profile.citizenship})"
        if citizenship:
            return "No", f"Derived from your stated citizenship ({profile.citizenship})"
        return None

    if field_key == "work_authorisation":
        citizenship = (profile.citizenship or "").lower()
        if "south africa" in citizenship or "sa citizen" in citizenship:
            return "Yes — South African citizen with the legal right to work in South Africa", \
                "Derived from your stated citizenship"
        auth = (profile.work_authorisation or "").strip()
        if auth:
            return auth, "From your saved work-authorisation status"
        return None

    if field_key == "graduation_year":
        year = latest_graduation_year(profile)
        if year is not None:
            return str(year), "From your education history"
        return None

    # Preferred name → first name
    if field_key == "preferred_name":
        stored = profile.preferred_name or profile.get_known_value("preferred_name")
        if stored:
            return str(stored), "From your profile"
        if profile.name:
            parts = profile.name.split()
            if parts:
                return parts[0], "Derived from your full name"
        return None

    # High-school completion year
    if field_key == "hs_completion_year":
        hs = profile.high_school
        if hs.completion_year:
            return str(hs.completion_year), "From your high-school record"
        return None

    # -- new inference rules ------------------------------------------------

    # City → Country mapping
    if field_key in ("country_of_residence", "country"):
        stored = profile.country_of_residence or profile.get_known_value("country_of_residence")
        if stored:
            return str(stored), "From your profile"
        location = (profile.location or "").strip()
        country = _city_to_country(location)
        if country:
            return country, f"Inferred from your location ({location})"
        return None

    # Education field → Discipline
    if field_key == "discipline":
        stored = profile.get_known_value("discipline")
        if stored:
            return str(stored), "From your education history"
        if profile.education:
            for edu in profile.education:
                if edu.field:
                    mapped = _map_discipline(edu.field)
                    return mapped, f"Mapped from your field of study ({edu.field})"
                # Try extracting from qualification string
                if edu.qualification:
                    extracted = _extract_discipline_from_qualification(edu.qualification)
                    if extracted:
                        return extracted, f"Extracted from qualification ({edu.qualification})"
        return None

    # Start date decomposition — "available immediately" → current month/year
    if field_key == "start_date_month":
        availability = (profile.availability or "").lower()
        if "immediate" in availability or "asap" in availability or "available now" in availability:
            month = date.today().strftime("%B")
            return month, "Current month (available immediately)"
        return None

    if field_key == "start_date_year":
        availability = (profile.availability or "").lower()
        if "immediate" in availability or "asap" in availability or "available now" in availability:
            return str(date.today().year), "Current year (available immediately)"
        return None

    # End date decomposition — pull from most recent education
    if field_key in ("end_date_month", "end_date_year"):
        if profile.education:
            edu = profile.education[0]
            end = (edu.end_date or "").strip()
            if end:
                parsed = _parse_date(end)
                if parsed:
                    if field_key == "end_date_month":
                        return parsed.strftime("%B"), f"From your education record ({edu.institution})"
                    return str(parsed.year), f"From your education record ({edu.institution})"
        return None

    # Start date from education — when question asks for start date of studies
    if field_key in ("start_date_month_of_study", "start_date_year_of_study"):
        if profile.education:
            edu = profile.education[0]
            start = (edu.start_date or "").strip()
            if start:
                parsed = _parse_date(start)
                if parsed:
                    if field_key == "start_date_month_of_study":
                        return parsed.strftime("%B"), f"From your education record ({edu.institution})"
                    return str(parsed.year), f"From your education record ({edu.institution})"
        return None

    # "Heard of company" — safe default is "No"
    if field_key == "heard_of_company":
        return "No", "Default answer — cannot verify company familiarity"

    # Nationality — alias for citizenship
    if field_key == "nationality":
        stored = profile.nationality or profile.citizenship
        if stored:
            return stored, "From your profile"
        return None

    # ID number — only from explicit storage, never inferred
    if field_key == "id_number":
        stored = profile.id_number or profile.get_known_value("id_number")
        if stored:
            return str(stored), "From your profile"
        return None

    # Gender pronouns — only from explicit storage
    if field_key == "gender_pronouns":
        stored = profile.gender_pronouns or profile.get_known_value("gender_pronouns")
        if stored:
            return str(stored), "From your profile"
        return None

    return None


_TEAMWORK_MARKERS = (
    "team", "collaborat", "distributed", "remote", "agile", "scrum",
    "cross-functional", "cross functional", "stakeholder",
)

# Personal facts (preferences, logistics, compensation) are known only to
# the candidate. The engine must never generate them from unrelated profile
# data such as skills — missing facts stay UNKNOWN and get asked.
_PERSONAL_FACT_FIELDS = frozenset({
    "travel_preference",
    "relocation",
    "work_preference",
    "preferred_locations",
    "availability",
    "notice_period",
    "drivers_licence",
    "vehicle",
    "expected_salary",
})


def _sentence_containing(text: str, marker: str) -> str:
    for part in re.split(r"(?<=[.!?])\s+", text.strip()):
        if marker in part.lower():
            return part.strip()
    return text.strip()


def _lower_first(sentence: str) -> str:
    words = sentence.split()
    if not words:
        return sentence
    first = words[0]
    if first.isupper() or (len(first) > 1 and first[1:].isupper()):
        return sentence  # acronym / proper noun — leave as-is
    return first[0].lower() + sentence[1:]


def generate_teamwork_answer(profile: CandidateProfile) -> Optional[str]:
    """Answer distributed-team / remote-work / collaboration questions from
    actual employment or project records only.

    A programming language is NOT evidence of teamwork: ``profile.skills``
    is deliberately never consulted here."""
    for exp in profile.experience:
        text = " ".join(p for p in (exp.title, exp.description) if p).strip()
        if not text:
            continue
        marker = next((m for m in _TEAMWORK_MARKERS if m in text.lower()), None)
        if marker is None:
            continue
        source_text = exp.description or exp.title
        sentence = _sentence_containing(source_text, marker)
        at = f" at {exp.company}" if exp.company else ""
        role = exp.title or "a team member"
        answer = f"Yes. As {role}{at}, {_lower_first(sentence)}"
        return answer if answer.endswith(".") else answer + "."
    for proj in profile.projects:
        text = " ".join(p for p in (proj.name, proj.description) if p).strip()
        if not text:
            continue
        marker = next((m for m in _TEAMWORK_MARKERS if m in text.lower()), None)
        if marker is None:
            continue
        source_text = proj.description or proj.name
        sentence = _sentence_containing(source_text, marker)
        answer = f"Yes. In my project \"{proj.name}\", {_lower_first(sentence)}"
        return answer if answer.endswith(".") else answer + "."
    # Fallback: if no specific markers found but we have education
    # (not just skills — a programming language is NOT teamwork evidence),
    # generate a truthful generic answer based on the collaborative nature
    # of software development
    if profile.education:
        edu = profile.education[0]
        qual = " ".join(p for p in (edu.qualification, edu.field) if p).strip()
        if qual:
            answer = (
                f"Yes. As a {qual} graduate, I have experience working in "
                "collaborative environments through academic group projects "
                "and team-based development work"
            )
            return answer + "."
    return None


def _try_generate(
    question: str,
    field_key: str,
    profile: CandidateProfile,
    job_context: dict,
) -> Optional[tuple[str, str]]:
    lowered = _normalise(question)

    # Evidence-sufficiency gate: these intents have exactly one admissible
    # evidence source (a stored personal fact). If it is absent → UNKNOWN.
    if field_key in _PERSONAL_FACT_FIELDS:
        return None

    if field_key in ("teamwork_experience", "remote_work_experience"):
        answer = generate_teamwork_answer(profile)
        if answer:
            return answer, "Based on your recorded work/project experience"
        return None

    if field_key == "motivation":
        answer = generate_motivation_answer(profile, job_context)
        if answer:
            return answer, "Generated from your verified background and the actual job description"
        return None

    # comfort/experience/familiarity questions map onto skill evidence
    topic_match = re.search(
        r"(?:comfortable|familiar|experienced?|confident|proficient|knowledgeable)"
        r"(?:\s+working)?(?:\s+with|\s+in|\s+using)?\s+(.+?)[?.]*$",
        lowered,
    )
    if topic_match:
        raw_topic = topic_match.group(1)
        raw_topic = re.sub(r"^(the|a|an)\s+", "", raw_topic)
        raw_topic = re.sub(r"\s+(environment|setting|methodolog\w*|practices?)$", "", raw_topic)
        answer = generate_skill_answer(raw_topic, profile)
        if answer:
            return answer, "Based on verified evidence in your CV/profile"
        return None

    # "Do you know/have experience with X?"
    know_match = re.search(
        r"(?:know|used|worked with|experience (?:with|in)|familiarity with)\s+([a-z0-9#+.\- ]+?)[?.]*$",
        lowered,
    )
    if know_match:
        answer = generate_skill_answer(know_match.group(1).strip(), profile)
        if answer:
            return answer, "Based on verified evidence in your CV/profile"
        return None

    return None
