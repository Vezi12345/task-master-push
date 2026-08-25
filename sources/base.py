from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


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
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.make_id(self.title, self.company, self.url)

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
