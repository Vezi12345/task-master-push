from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class BrowserResult(BaseModel):
    success: bool = False
    application_url: str = ""
    confirmation_id: str = ""
    error: str = ""
    requires_login: bool = False
    requires_human_input: bool = False
    partial: bool = False
    screenshot_path: str = ""


class BrowserAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def inspect_form(self, url: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def fill_and_submit(
        self,
        url: str,
        fields: dict[str, str],
        files: Optional[dict[str, str]] = None,
        headless: bool = True,
    ) -> BrowserResult:
        raise NotImplementedError

    @abstractmethod
    async def check_status(self, url: str) -> str:
        raise NotImplementedError
