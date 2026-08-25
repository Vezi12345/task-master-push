from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

import config
from candidate.profile import CandidateProfile

from .answer_engine import (
    AnswerResult,
    AnswerType,
    Confidence,
    answer_question,
    classify_question,
)
from .models import MissingInfo


_COMMON_QUESTIONS: list[dict] = [
    {
        "question": "What is your highest qualification?",
        "field_key": "highest_qualification",
        "category": "education",
    },
    {
        "question": "What is your expected salary?",
        "field_key": "expected_salary",
        "category": "compensation",
    },
    {
        "question": "How many years of experience do you have?",
        "field_key": "years_experience",
        "category": "experience",
    },
    {
        "question": "Are you willing to relocate?",
        "field_key": "relocation",
        "category": "preferences",
    },
    {
        "question": "Do you have a valid driver's licence?",
        "field_key": "drivers_licence",
        "category": "requirements",
    },
    {
        "question": "What is your notice period?",
        "field_key": "notice_period",
        "category": "logistics",
    },
    {
        "question": "Are you authorised to work in South Africa?",
        "field_key": "work_authorisation",
        "category": "requirements",
    },
    {
        "question": "What is your availability / start date?",
        "field_key": "availability",
        "category": "logistics",
    },
    {
        "question": "What is your citizenship status?",
        "field_key": "citizenship",
        "category": "eligibility",
    },
    {
        "question": "What is your race / equity group?",
        "field_key": "race",
        "category": "demographic",
    },
    {
        "question": "What is your gender?",
        "field_key": "gender",
        "category": "demographic",
    },
    {
        "question": "Do you have a disability?",
        "field_key": "disability",
        "category": "demographic",
    },
    {
        "question": "What is your age / date of birth?",
        "field_key": "date_of_birth",
        "category": "demographic",
    },
    {
        "question": "Are you a South African citizen?",
        "field_key": "south_african_citizen",
        "category": "eligibility",
    },
    {
        "question": "Are you a recent graduate (within 2 years)?",
        "field_key": "recent_graduate",
        "category": "eligibility",
    },
]

_ELIGIBILITY_KEYWORDS: dict[str, list[str]] = {
    "work_authorisation": [
        "work authorisation", "work authorization", "legally entitled",
        "right to work", "authorised to work", "authorized to work",
        "entitled to work",
    ],
    "disability": ["disability", "disabled", "broad-based black economic empowerment", "bbbee", "employment equity"],
    "race": ["race", "equity group", "diversity", "bbbee", "employment equity", "affirmative"],
    "gender": ["gender", "female", "male", "women in", "diversity"],
    "recent_graduate": ["recent graduate", "recently graduated", "new graduate", "graduate programme", "graduate program"],
    "south_african_citizen": ["south african citizen", "sa citizen", "citizen"],
    "drivers_licence": ["driver's licence", "drivers licence", "driver license", "own transport", "valid licence"],
    "relocation": ["relocate", "relocation", "willing to move"],
    "notice_period": ["notice period", "available immediately", "start date"],
    "expected_salary": ["salary", "remuneration", "compensation", "package"],
}


def stable_question_key(question_text: str) -> str:
    """Stable storage key for questions that have no canonical field —
    must survive process restarts (unlike ``hash()``)."""
    digest = hashlib.sha1(question_text.strip().lower().encode("utf-8")).hexdigest()
    return f"q_{digest[:10]}"


class AnswerRecord(BaseModel):
    """One persisted answer with full provenance.

    ``verified`` is True only when the user explicitly supplied/confirmed
    the value. Evidence-derived or generated values are stored with
    status "needs_confirmation" / "draft" and verified=False."""
    field_key: str
    answer: str
    source: str = "user"            # user | cv | derived | generated
    status: str = "verified"        # verified | needs_confirmation | draft
    verified: bool = True
    evidence: str = ""
    question: str = ""
    updated_at: str = ""


def _record_from_legacy(field_key: str, value: str) -> AnswerRecord:
    """Legacy answers.json entries were plain strings typed by the user."""
    return AnswerRecord(
        field_key=field_key, answer=value,
        source="user", status="verified", verified=True,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )


class AnswerStore:
    """Persistent store of answers keyed by canonical field.

    Storage format: ``{field_key: {answer, source, status, verified, ...}}``.
    Legacy files holding plain ``{field_key: "value"}`` strings load
    transparently (treated as user-verified)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path(config.ANSWERS_FILE)
        self._answers: dict[str, AnswerRecord] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        if self._path and self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            self._answers = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    try:
                        self._answers[key] = AnswerRecord(**{**value, "field_key": key})
                    except Exception:
                        self._answers[key] = _record_from_legacy(
                            key, str(value.get("answer", "")))
                else:
                    self._answers[key] = _record_from_legacy(key, str(value))

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: r.model_dump() for k, r in self._answers.items()}
            self._path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # -- record-level API ---------------------------------------------------

    def record(self, field_key: str) -> Optional[AnswerRecord]:
        rec = self._answers.get(field_key)
        return rec.model_copy() if rec else None

    def set_record(self, record: AnswerRecord) -> None:
        if not str(record.answer).strip():
            return
        record.updated_at = datetime.now().isoformat(timespec="seconds")
        self._answers[record.field_key] = record
        self._save()

    def all_records(self) -> dict[str, AnswerRecord]:
        return {k: r.model_copy() for k, r in self._answers.items()}

    def remove(self, field_key: str) -> None:
        if field_key in self._answers:
            del self._answers[field_key]
            self._save()

    # -- backward-compatible string API --------------------------------------

    def get(self, field_key: str) -> Optional[str]:
        rec = self._answers.get(field_key)
        return rec.answer if rec else None

    def set(self, field_key: str, answer: str) -> None:
        self.set_record(AnswerRecord(
            field_key=field_key, answer=str(answer),
            source="user", status="verified", verified=True,
        ))

    def has(self, field_key: str) -> bool:
        rec = self._answers.get(field_key)
        return bool(rec and rec.answer.strip())

    def all_answers(self) -> dict[str, str]:
        return {k: r.answer for k, r in self._answers.items()}

    def bulk_set(self, answers: dict[str, str]) -> None:
        for key, value in answers.items():
            if str(value).strip():
                self.set(key, str(value))


class QuestionEngine:
    """Resolves application questions through the semantic answer engine.

    The CV is only one input: stored user answers, structured profile
    knowledge, derivations and evidence-based generation all contribute.
    Only questions that genuinely cannot be answered become MissingInfo.
    """

    def __init__(self, answer_store: Optional[AnswerStore] = None) -> None:
        self.answer_store = answer_store or AnswerStore()

    def try_answer_from_profile(
        self,
        question_entry: dict,
        profile: Optional[CandidateProfile],
        job_context: Optional[dict] = None,
    ) -> tuple[Optional[str], bool]:
        """Backward-compatible helper: returns ``(answer, needs_input)``."""
        result = self.answer(
            question_entry.get("question", ""),
            profile,
            job_context=job_context,
        )
        if result.is_answered:
            return result.answer, False
        return None, True

    def answer(
        self,
        question: str,
        profile: Optional[CandidateProfile],
        job_context: Optional[dict] = None,
    ) -> AnswerResult:
        return answer_question(
            question,
            profile,
            job_context=job_context,
            answer_store=self.answer_store,
        )

    def resolve_common_questions(
        self,
        profile: Optional[CandidateProfile],
        job_description: str = "",
        job_context: Optional[dict] = None,
    ) -> tuple[dict[str, str], list[MissingInfo]]:
        context = job_context or {"description": job_description}
        answered: dict[str, str] = {}
        missing: list[MissingInfo] = []

        for entry in _COMMON_QUESTIONS:
            result = self.answer(entry["question"], profile, job_context=context)
            if result.is_answered:
                answered[entry["field_key"]] = result.answer
            else:
                missing.append(MissingInfo(
                    question=entry["question"],
                    field_key=entry["field_key"],
                    category=entry.get("category", ""),
                ))

        return answered, missing

    def resolve_job_specific_questions(
        self,
        screening_questions: list[dict],
        profile: Optional[CandidateProfile],
        job_context: Optional[dict] = None,
    ) -> tuple[dict[str, str], list[MissingInfo]]:
        """Handle arbitrary employer questions — including ones never seen
        before — through semantic classification."""
        answered: dict[str, str] = {}
        missing: list[MissingInfo] = []

        for q in screening_questions:
            question_text = q.get("question", "")
            if not question_text.strip():
                continue
            field_key, category = classify_question(question_text)
            result = self.answer(question_text, profile, job_context=job_context)
            key = field_key or stable_question_key(question_text)
            if result.is_answered:
                answered[key] = result.answer
            else:
                missing.append(MissingInfo(
                    question=question_text,
                    field_key=key,
                    category=q.get("category") or category or "screening",
                ))

        return answered, missing

    def detect_job_requirements(
        self,
        job_description: str,
        profile: Optional[CandidateProfile],
        job_context: Optional[dict] = None,
    ) -> tuple[dict[str, str], list[MissingInfo]]:
        context = job_context or {"description": job_description}
        answered: dict[str, str] = {}
        missing: list[MissingInfo] = []
        desc_lower = job_description.lower()

        for field_key, keywords in _ELIGIBILITY_KEYWORDS.items():
            if not any(kw in desc_lower for kw in keywords):
                continue
            entry = next(
                (q for q in _COMMON_QUESTIONS if q["field_key"] == field_key),
                {"question": field_key.replace("_", " ").capitalize() + "?"},
            )
            result = self.answer(entry["question"], profile, job_context=context)
            if result.is_answered:
                answered[field_key] = result.answer
            else:
                missing.append(MissingInfo(
                    question=entry["question"],
                    field_key=field_key,
                    category=result.category or entry.get("category", ""),
                    priority=(
                        "required"
                        if field_key in ("work_authorisation", "citizenship")
                        else "preferred"
                    ),
                ))

        return answered, missing


__all__ = [
    "AnswerStore",
    "QuestionEngine",
    "AnswerResult",
    "AnswerType",
    "Confidence",
]
