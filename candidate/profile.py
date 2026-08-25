from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    INFERRED = "inferred"
    FROM_USER = "from_user"


class KnownField(BaseModel):
    key: str
    value: str
    status: KnowledgeStatus = KnowledgeStatus.KNOWN
    source: str = ""


class Education(BaseModel):
    institution: str = ""
    qualification: str = ""
    field: str = ""
    start_date: str = ""
    end_date: str = ""
    # the ACADEMIC RESULT is distinct from the qualification itself:
    # "Diploma in ICT" != "65%"
    result: str = ""
    grading_system: str = ""          # Percentage | GPA 4.0 | ... (user-supplied)
    qualification_level: str = ""     # Certificate | Diploma | Bachelor | ...
    is_highest: bool = False


class Experience(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_type: str = "employment"
    employment_type: str = ""         # full-time | internship | contract | ...
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    location: str = ""


class Certification(BaseModel):
    name: str = ""
    issuer: str = ""
    date: str = ""
    credential_id: str = ""
    url: str = ""


class Project(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    url: str = ""
    github_url: str = ""
    role: str = ""
    achievements: list[str] = Field(default_factory=list)
    is_personal: bool = False
    is_academic: bool = False
    is_work_related: bool = False


class OnlineProfiles(BaseModel):
    """Verified URLs only — never names, never guesses."""
    website: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""

    def get(self, key: str) -> str:
        return (getattr(self, key, "") or "").strip()


class HighSchoolRecord(BaseModel):
    """Optional high-school profile. Nothing here is ever inferred —
    every value comes from the user or their CV text."""
    school: str = ""
    province: str = ""
    country: str = ""
    completion_year: str = ""
    mathematics_result: str = ""
    mathematics_grade: str = ""
    native_language: str = ""
    native_language_result: str = ""
    overall_result: str = ""
    scoring_system: str = ""
    awards: list[str] = Field(default_factory=list)
    rankings: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.school or self.completion_year or self.mathematics_result
            or self.overall_result or self.native_language
        )


class SkillDetail(BaseModel):
    """A skill WITH provenance. A bare skill name never implies
    professional experience."""
    name: str
    category: str = ""
    proficiency: str = ""             # e.g. beginner/intermediate/advanced
    source: str = ""                  # cv | user | project:<name> | job:<company>
    evidence: str = ""                # free text pointing at real usage


class LanguageRecord(BaseModel):
    name: str
    proficiency: str = ""


class DocumentRef(BaseModel):
    kind: str = ""                    # cv | cover_letter | transcript | other
    path: str = ""
    uploaded_at: str = ""


class QuestionMemory(BaseModel):
    """A previously answered application question, kept so the same or a
    semantically equivalent question never has to be asked twice."""
    question: str
    answer: str
    field_key: str = ""
    source: str = "user"
    updated_at: str = ""
    confidence: str = "high"
    evidence: str = ""


# Canonical knowledge keys grouped by category. Demographic keys are special:
# they are NEVER inferred — only stored when explicitly supplied by the user.
IDENTITY_KEYS = (
    "name", "email", "phone", "location", "date_of_birth",
    "preferred_name", "country_of_residence", "city", "address",
)
EDUCATION_KEYS = ("highest_qualification", "education_result")
ELIGIBILITY_KEYS = ("citizenship", "south_african_citizen", "work_authorisation")
PREFERENCE_KEYS = (
    "expected_salary", "minimum_salary", "relocation", "work_preference",
    "preferred_locations", "travel_preference", "international_travel",
)
LOGISTICS_KEYS = ("notice_period", "availability")
REQUIREMENT_KEYS = ("drivers_licence", "vehicle")
DEMOGRAPHIC_KEYS = ("race", "gender", "disability")

SENSITIVE_KEYS = frozenset(DEMOGRAPHIC_KEYS) | {
    "citizenship", "south_african_citizen", "date_of_birth",
}


class CandidateProfile(BaseModel):
    # -- identity ------------------------------------------------------------
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    professional_summary: str = ""
    preferred_name: str = ""
    date_of_birth: str = ""
    country_of_residence: str = ""
    city: str = ""
    address: str = ""

    # -- skills / history ----------------------------------------------------
    skills: list[str] = Field(default_factory=list)
    skill_details: list[SkillDetail] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    high_school: HighSchoolRecord = Field(default_factory=HighSchoolRecord)
    experience: list[Experience] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    languages: list[LanguageRecord] = Field(default_factory=list)

    # -- online presence (verified URLs only) ---------------------------------
    online_profiles: OnlineProfiles = Field(default_factory=OnlineProfiles)

    # -- documents -------------------------------------------------------------
    documents: list[DocumentRef] = Field(default_factory=list)

    # -- skills / history ----------------------------------------------------
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    # -- eligibility ---------------------------------------------------------
    citizenship: str = ""
    work_authorisation: str = ""

    # -- preferences ---------------------------------------------------------
    expected_salary: str = ""
    minimum_salary: str = ""
    relocation: str = ""
    work_preference: str = ""
    preferred_locations: list[str] = Field(default_factory=list)
    travel_preference: str = ""
    international_travel: str = ""
    preferred_roles: list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)

    # -- logistics -----------------------------------------------------------
    notice_period: str = ""
    availability: str = ""

    # -- requirements --------------------------------------------------------
    drivers_licence: str = ""
    vehicle: str = ""

    # -- demographics (only ever set from explicit user input) ---------------
    race: str = ""
    gender: str = ""
    disability: str = ""

    # -- knowledge bookkeeping -----------------------------------------------
    known_fields: list[KnownField] = Field(default_factory=list)
    question_memory: list[QuestionMemory] = Field(default_factory=list)

    def get_known_value(self, key: str) -> Optional[str]:
        structured = self._structured_field(key)
        if structured not in (None, ""):
            return str(structured)
        for kf in self.known_fields:
            if kf.key == key and kf.status != KnowledgeStatus.UNKNOWN:
                return kf.value
        return None

    # -- name splitting --------------------------------------------------------
    @property
    def first_name(self) -> str:
        """First token of the full name. 'Lucky Vezi' -> 'Lucky'.
        Never the full name."""
        parts = (self.name or "").split()
        return parts[0] if parts else ""

    @property
    def last_name(self) -> Optional[str]:
        """Last token of a multi-part name. A single-word name has NO
        surname — callers must treat that as UNKNOWN and ask the user."""
        parts = (self.name or "").split()
        return parts[-1] if len(parts) > 1 else None

    def is_known(self, key: str) -> bool:
        return self.get_known_value(key) is not None

    def set_known(self, key: str, value: str, source: str = "profile") -> None:
        value = (value or "").strip()
        if not value:
            return
        if key in _SCALAR_STRUCTURED_KEYS and hasattr(self, key):
            setattr(self, key, value)
        for i, kf in enumerate(self.known_fields):
            if kf.key == key:
                self.known_fields[i] = KnownField(
                    key=key, value=value,
                    status=KnowledgeStatus.KNOWN, source=source,
                )
                return
        self.known_fields.append(KnownField(
            key=key, value=value,
            status=KnowledgeStatus.KNOWN, source=source,
        ))

    def set_unknown(self, key: str) -> None:
        for i, kf in enumerate(self.known_fields):
            if kf.key == key:
                self.known_fields[i] = KnownField(
                    key=key, value="",
                    status=KnowledgeStatus.UNKNOWN, source="",
                )
                return
        self.known_fields.append(KnownField(
            key=key, value="",
            status=KnowledgeStatus.UNKNOWN, source="",
        ))

    def remember_answer(
        self,
        question: str,
        answer: str,
        field_key: str = "",
        source: str = "user",
        confidence: str = "high",
        evidence: str = "",
    ) -> None:
        """Store a user-supplied answer so equivalent future questions can be
        answered without asking again."""
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return
        now = datetime.now().isoformat(timespec="seconds")
        for mem in self.question_memory:
            if mem.question.lower() == question.lower():
                mem.answer = answer
                mem.field_key = field_key or mem.field_key
                mem.source = source
                mem.confidence = confidence or mem.confidence
                mem.evidence = evidence or mem.evidence
                mem.updated_at = now
                return
        self.question_memory.append(QuestionMemory(
            question=question,
            answer=answer,
            field_key=field_key,
            source=source,
            updated_at=now,
            confidence=confidence,
            evidence=evidence,
        ))

    def populate_known_fields(self) -> None:
        if self.name:
            self.set_known("name", self.name, "cv")
        if self.email:
            self.set_known("email", self.email, "cv")
        if self.phone:
            self.set_known("phone", self.phone, "cv")
        if self.location:
            self.set_known("location", self.location, "cv")
        if self.country_of_residence:
            self.set_known("country_of_residence", self.country_of_residence, "cv")
        edu = next((e for e in self.education if e.is_highest), None) \
            or (self.education[0] if self.education else None)
        if edu:
            qual = " ".join(p for p in (edu.qualification, edu.field) if p)
            if qual:
                self.set_known("highest_qualification", qual, "cv")
            if edu.result and edu.result not in qual:
                self.set_known("education_result", edu.result, "cv")
        if self.skills:
            self.set_known("skills", ", ".join(self.skills), "cv")
        for link in ("website", "linkedin", "github", "portfolio"):
            url = self.online_profiles.get(link)
            if url:
                self.set_known(f"online_{link}", url, "cv")

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.name:
            parts.append(self.name)
        if self.email:
            parts.append(self.email)
        if self.location:
            parts.append(f"Location: {self.location}")
        if self.education:
            edu = self.education[0]
            edu_str = " - ".join(
                p for p in (edu.qualification, edu.field) if p
            )
            if edu_str:
                parts.append(f"Education: {edu_str}")
        if self.experience:
            parts.append(f"Experience: {len(self.experience)} role(s)")
        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills[:8])}")
        return " | ".join(parts)

    def _structured_field(self, key: str):
        # online_* keys resolve against the verified-URL sub-model
        if key.startswith("online_") and hasattr(self, "online_profiles"):
            return self.online_profiles.get(key[len("online_"):]) or None
        if key in _STRUCTURED_KEYS and hasattr(self, key):
            value = getattr(self, key)
            if isinstance(value, list):
                return ", ".join(str(v) for v in value) if value else None
            return value
        return None


_STRUCTURED_KEYS = frozenset(IDENTITY_KEYS) | frozenset(EDUCATION_KEYS) \
    | frozenset(ELIGIBILITY_KEYS) | frozenset(PREFERENCE_KEYS) \
    | frozenset(LOGISTICS_KEYS) | frozenset(REQUIREMENT_KEYS) \
    | frozenset(DEMOGRAPHIC_KEYS) | {"skills"}

_SCALAR_STRUCTURED_KEYS = _STRUCTURED_KEYS - {
    "skills", "preferred_locations", "preferred_roles", "employment_types",
}
