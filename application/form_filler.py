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


# ---------------------------------------------------------------------------
# Evidence-grounded form filling (real application pipeline)
#
# Every discovered form question is routed through the semantic answer
# engine. Only VERIFIED / DERIVED / GENERATED_FROM_EVIDENCE answers are
# auto-filled; UNKNOWN questions become explicit user actions. Consent and
# terms checkboxes are NEVER auto-checked, and demographic questions can
# only ever be answered from explicitly stored candidate data (enforced by
# the engine's sensitive-field guard).
# ---------------------------------------------------------------------------

import re as _re

from application.answer_engine import (
    AnswerType,
    answer_question,
    classify_question,
    questions_equivalent,
)
from application.form_analyzer import AnalyzedField, FormAnalysis
from candidate.profile import CandidateProfile


class PlannedAnswer(BaseModel):
    selector: str = ""
    name: str = ""
    question: str = ""
    field_type: str = ""
    category: str = "other"
    required: bool = False
    value: Optional[str] = None
    answer_type: str = ""          # verified|derived|generated_from_evidence|unknown
    source: str = ""               # memory|answer_store|profile|derived|generated|document
    needs_user: bool = False
    reason: str = ""
    options: list[str] = Field(default_factory=list)
    upload_kind: str = ""          # "", "cv", "cover_letter"
    is_consent: bool = False
    is_terms: bool = False
    is_demographic: bool = False
    # set when the profile's current value and a remembered application answer
    # disagree — the user must pick which one to use, never silently
    conflict: dict = Field(default_factory=dict)


class FillPlan(BaseModel):
    entries: list[PlannedAnswer] = Field(default_factory=list)
    job_title: str = ""
    company: str = ""

    @property
    def needs_user_questions(self) -> list[str]:
        return [e.question for e in self.entries if e.needs_user and not e.is_consent and not e.is_terms]

    @property
    def consent_entries(self) -> list[PlannedAnswer]:
        return [e for e in self.entries if e.is_consent or e.is_terms]

    @property
    def unanswered_required(self) -> list[str]:
        return [
            e.question for e in self.entries
            if e.required and (e.needs_user or not e.value)
        ]

    @property
    def ready_to_fill(self) -> bool:
        return len(self.unanswered_required) == 0


_CANONICAL_QUESTIONS: dict[tuple[str, str], str] = {
    ("contact", "email"): "What is your email address?",
    ("contact", "tel"): "What is your phone number?",
    ("identity", "text"): "What is your full name?",
    ("location", "text"): "Where are you based?",
}


def _canonical_question(field: AnalyzedField) -> str:
    """Terse labels ('Email', 'Phone') are rephrased as canonical questions
    so the semantic engine classifies them; real questions pass through
    verbatim."""
    q = (field.question or "").strip()
    if len(q) > 15 or q.endswith("?"):
        return q
    lowered = q.lower()
    # explicit first/last-name labels are passed through so the engine can
    # split the candidate's name instead of defaulting to the full name
    if any(
        hint in lowered
        for hint in ("first name", "last name", "surname", "given name", "family name")
    ):
        return q
    mapped = _CANONICAL_QUESTIONS.get((field.category, field.field_type))
    return mapped or q or f"Provide {field.name or 'this field'}"


def _match_option(answer: str, options: list[str]) -> Optional[str]:
    lowered = (answer or "").strip().lower()
    if not lowered:
        return None
    for option in options:
        if option.strip().lower() == lowered:
            return option
    for option in options:
        if lowered in option.strip().lower() or option.strip().lower() in lowered:
            return option
    return None


class FormFiller:
    """Builds a fill plan for an analysed REAL application form."""

    def __init__(self, cv_path=None, cover_letter_path=None) -> None:
        self.cv_path = str(cv_path) if cv_path else ""
        self.cover_letter_path = str(cover_letter_path) if cover_letter_path else ""

    def build_plan(
        self,
        analysis: FormAnalysis,
        profile: CandidateProfile,
        job_context: dict,
        remembered_answers: Optional[dict[str, str]] = None,
    ) -> FillPlan:
        plan = FillPlan(
            job_title=job_context.get("title", ""),
            company=job_context.get("company", ""),
        )
        remembered = remembered_answers or {}

        for field_obj in analysis.fields:
            entry = self._plan_field(field_obj, profile, job_context, remembered)
            plan.entries.append(entry)
        return plan

    # -- per-field planning --------------------------------------------------
    def _plan_field(
        self,
        field_obj: AnalyzedField,
        profile: CandidateProfile,
        job_context: dict,
        remembered: dict[str, str],
    ) -> PlannedAnswer:
        entry = PlannedAnswer(
            selector=field_obj.selector,
            name=field_obj.name,
            question=field_obj.display_question,
            field_type=field_obj.field_type,
            category=field_obj.category,
            required=field_obj.required,
            options=list(field_obj.options),
            is_consent=field_obj.is_consent,
            is_terms=field_obj.is_terms,
            is_demographic=field_obj.is_demographic,
        )

        # 1. documents: upload the real files
        if field_obj.field_type == "file":
            kind = (
                "cover_letter"
                if _re.search(r"cover\s*letter", field_obj.display_question, _re.I)
                else "cv"
            )
            path = self.cover_letter_path if kind == "cover_letter" else self.cv_path
            entry.upload_kind = kind
            if path:
                entry.value = path
                entry.answer_type = "verified"
                entry.source = "document"
                entry.reason = f"Upload the actual candidate {kind.replace('_', ' ')}"
            else:
                entry.needs_user = True
                entry.reason = f"No {kind.replace('_', ' ')} file available to upload"
            return entry

        # 2. consent / terms: never auto-checked
        if field_obj.is_consent or field_obj.is_terms:
            entry.needs_user = True
            entry.reason = (
                "Requires your explicit agreement — review it before submitting"
            )
            return entry

        # 3. every genuine question goes through the evidence-grounded engine
        question = _canonical_question(field_obj)

        # remembered user answers for this exact intent take precedence —
        # but if the profile's CURRENT value disagrees, surface the conflict
        # instead of silently picking a side
        for mem_q, mem_a in remembered.items():
            if questions_equivalent(mem_q, question):
                # compare against the profile's STRUCTURED value only —
                # question_memory already holds the remembered answer itself
                fk, _cat = classify_question(question)
                structured = profile.get_known_value(fk) if fk else None
                current_val = str(structured).strip() \
                    if structured not in (None, "") else ""
                remembered_val = str(mem_a).strip()
                if (
                    current_val
                    and field_obj.field_type not in ("textarea",)
                    and current_val.lower() != remembered_val.lower()
                ):
                    entry.conflict = {
                        "profile_value": current_val,
                        "remembered_value": remembered_val,
                    }
                    entry.needs_user = True
                    entry.answer_type = "unknown"
                    entry.reason = (
                        "CONFLICTING INFORMATION — your profile says "
                        f'"{current_val}" but an earlier application said '
                        f'"{remembered_val}". Choose which to use.'
                    )
                else:
                    entry.value = remembered_val
                    entry.answer_type = "verified"
                    entry.source = "memory"
                    entry.reason = "Previously supplied by you"
                break

        if entry.value is None and not entry.conflict:
            result = answer_question(question, profile, job_context=job_context)
            if result.answer is not None:
                entry.value = str(result.answer)
                entry.answer_type = result.answer_type.value
                entry.source = result.source
                entry.reason = result.explanation
            else:
                entry.needs_user = True
                entry.answer_type = "unknown"
                entry.reason = result.explanation or "No reliable evidence — please answer"

        # 4. choice fields must match a real option — never invent one
        if field_obj.options and entry.value is not None and not entry.needs_user:
            matched = _match_option(entry.value, field_obj.options)
            if matched is None:
                entry.needs_user = True
                entry.reason = (
                    f'Answer "{entry.value}" does not match any offered option'
                )
                entry.value = None
            else:
                entry.value = matched

        return entry
