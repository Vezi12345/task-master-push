from __future__ import annotations

import uuid
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


class Experience(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""
    skills: list[str] = Field(default_factory=list)


class Certification(BaseModel):
    name: str = ""
    issuer: str = ""
    date: str = ""


class Project(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    professional_summary: str = ""
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    known_fields: list[KnownField] = Field(default_factory=list)

    def get_known_value(self, key: str) -> Optional[str]:
        for kf in self.known_fields:
            if kf.key == key and kf.status != KnowledgeStatus.UNKNOWN:
                return kf.value
        return None

    def is_known(self, key: str) -> bool:
        return self.get_known_value(key) is not None

    def set_known(self, key: str, value: str, source: str = "profile") -> None:
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

    def populate_known_fields(self) -> None:
        if self.name:
            self.set_known("name", self.name, "cv")
        if self.email:
            self.set_known("email", self.email, "cv")
        if self.phone:
            self.set_known("phone", self.phone, "cv")
        if self.location:
            self.set_known("location", self.location, "cv")
        if self.education:
            edu = self.education[0]
            qual = " ".join(
                p for p in (edu.qualification, edu.field) if p
            )
            if qual:
                self.set_known("highest_qualification", qual, "cv")
        if self.experience:
            self.set_known(
                "years_experience", str(len(self.experience)), "cv"
            )
        if self.skills:
            self.set_known("skills", ", ".join(self.skills), "cv")

        always_check = [
            "expected_salary", "relocation", "drivers_licence",
            "notice_period", "work_authorisation", "availability",
            "citizenship",
        ]
        for key in always_check:
            if not self.is_known(key):
                self.set_unknown(key)

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
