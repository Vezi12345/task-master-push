from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ApplicationPlatformType(str, Enum):
    GENERIC_WEB = "generic_web"
    WORKDAY = "workday"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    SMARTRECRUITERS = "smartrecruiters"
    EMAIL = "email"
    UNKNOWN = "unknown"


def detect_platform_type(url: str) -> ApplicationPlatformType:
    if not url:
        return ApplicationPlatformType.UNKNOWN
    lowered = url.lower()
    if "workday" in lowered:
        return ApplicationPlatformType.WORKDAY
    if "greenhouse.io" in lowered or "greenhouse" in lowered:
        return ApplicationPlatformType.GREENHOUSE
    if "lever.co" in lowered or "lever" in lowered:
        return ApplicationPlatformType.LEVER
    if "smartrecruiters" in lowered:
        return ApplicationPlatformType.SMARTRECRUITERS
    if lowered.startswith("mailto:") or "mailto" in lowered:
        return ApplicationPlatformType.EMAIL
    return ApplicationPlatformType.GENERIC_WEB


@dataclass
class Job:
    title: str
    company: str
    location: str = ""
    remote: bool = False
    description: str = ""
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary_text: Optional[str] = None
    url: str = ""
    source: str = ""
    posted_date: Optional[str] = None
    platform: ApplicationPlatformType = ApplicationPlatformType.UNKNOWN
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.make_id(self.title, self.company, self.url)
        if self.platform == ApplicationPlatformType.UNKNOWN and self.url:
            self.platform = detect_platform_type(self.url)

    @staticmethod
    def make_id(title: str, company: str, url: str) -> str:
        import hashlib

        raw = f"{title}|{company}|{url}".strip().lower()
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class JobSourceError(RuntimeError):
    pass


class JobSource(ABC):
    name = "base"

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def search(self, query) -> list[Job]:
        raise NotImplementedError
