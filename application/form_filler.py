from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from sources.base import ApplicationPlatformType


class FieldType(str, Enum):
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    NUMBER = "number"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXTAREA = "textarea"
    FILE = "file"
    DATE = "date"
    URL = "url"
    DROPDOWN = "dropdown"
    UNKNOWN = "unknown"


class FormField(BaseModel):
    name: str = ""
    label: str = ""
    field_type: FieldType = FieldType.UNKNOWN
    required: bool = False
    options: list[str] = Field(default_factory=list)
    placeholder: str = ""
    value: str = ""
    mapped_value: Optional[str] = None
    confidence: float = 0.0
    needs_human_input: bool = False


class SubmissionResult(BaseModel):
    success: bool = False
    application_url: str = ""
    confirmation_id: str = ""
    error: str = ""
    requires_login: bool = False
    requires_human_input: bool = False
    partial: bool = False


class ApplicationPlatform(ABC):
    name: str = "generic"

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def inspect_form(self, url: str) -> list[FormField]:
        raise NotImplementedError

    @abstractmethod
    def fill_and_submit(
        self,
        url: str,
        fields: dict[str, str],
        files: Optional[dict[str, Any]] = None,
    ) -> SubmissionResult:
        raise NotImplementedError

    @abstractmethod
    def check_status(self, url: str) -> str:
        raise NotImplementedError


class GenericApplicationAdapter(ApplicationPlatform):
    name = "generic"

    def can_handle(self, url: str) -> bool:
        return True

    def inspect_form(self, url: str) -> list[FormField]:
        return []

    def fill_and_submit(
        self,
        url: str,
        fields: dict[str, str],
        files: Optional[dict[str, Any]] = None,
    ) -> SubmissionResult:
        return SubmissionResult(
            success=False,
            error="Browser automation not yet implemented. "
            "This job's application requires manual submission.",
            requires_human_input=True,
        )

    def check_status(self, url: str) -> str:
        return "unknown"


class PlaywrightApplicationAdapter(ApplicationPlatform):
    name = "playwright"

    def __init__(self, browser_adapter=None) -> None:
        self._browser_adapter = browser_adapter

    def can_handle(self, url: str) -> bool:
        return True

    def inspect_form(self, url: str) -> list[FormField]:
        return []

    def fill_and_submit(
        self,
        url: str,
        fields: dict[str, str],
        files: Optional[dict[str, Any]] = None,
    ) -> SubmissionResult:
        import asyncio
        adapter = self._get_browser_adapter(url)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        adapter.fill_and_submit(url, fields, files),
                    )
                    result = future.result(timeout=60)
            else:
                result = loop.run_until_complete(
                    adapter.fill_and_submit(url, fields, files)
                )
        except Exception as exc:
            return SubmissionResult(
                success=False,
                error=f"Browser automation error: {exc}",
                requires_human_input=True,
            )
        return SubmissionResult(
            success=result.success,
            application_url=result.application_url,
            confirmation_id=result.confirmation_id,
            error=result.error,
            requires_login=result.requires_login,
            requires_human_input=result.requires_human_input,
            partial=result.partial,
        )

    def check_status(self, url: str) -> str:
        return "unknown"

    def _get_browser_adapter(self, url: str):
        from browser import (
            GenericBrowserAdapter,
            WorkdayBrowserAdapter,
            GreenhouseBrowserAdapter,
            LeverBrowserAdapter,
            SmartRecruitersBrowserAdapter,
        )
        adapters = [
            WorkdayBrowserAdapter(),
            GreenhouseBrowserAdapter(),
            LeverBrowserAdapter(),
            SmartRecruitersBrowserAdapter(),
        ]
        for adapter in adapters:
            if adapter.can_handle(url):
                return adapter
        return GenericBrowserAdapter()


class PlatformRegistry:
    def __init__(self) -> None:
        self._platforms: list[ApplicationPlatform] = []

    def register(self, platform: ApplicationPlatform) -> None:
        self._platforms.append(platform)

    def get_platform(self, url: str) -> ApplicationPlatform:
        for platform in self._platforms:
            if platform.can_handle(url):
                return platform
        return GenericApplicationAdapter()

    def register_default(self) -> None:
        self.register(PlaywrightApplicationAdapter())


_platform_registry = PlatformRegistry()
_platform_registry.register_default()


def get_platform(url: str) -> ApplicationPlatform:
    return _platform_registry.get_platform(url)


def get_playwright_adapter(url: str) -> PlaywrightApplicationAdapter:
    return PlaywrightApplicationAdapter()


def register_platform(platform: ApplicationPlatform) -> None:
    _platform_registry.register(platform)
