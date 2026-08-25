from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from candidate.profile import CandidateProfile

from .models import MissingInfo


_COMMON_QUESTIONS: list[dict] = [
    {
        "question": "What is your highest qualification?",
        "field_key": "highest_qualification",
        "category": "education",
        "profile_lookup": "qualification",
    },
    {
        "question": "What is your expected salary?",
        "field_key": "expected_salary",
        "category": "compensation",
        "profile_lookup": None,
    },
    {
        "question": "How many years of experience do you have?",
        "field_key": "years_experience",
        "category": "experience",
        "profile_lookup": "years_experience",
    },
    {
        "question": "Are you willing to relocate?",
        "field_key": "relocation",
        "category": "preferences",
        "profile_lookup": None,
    },
    {
        "question": "Do you have a valid driver's licence?",
        "field_key": "drivers_licence",
        "category": "requirements",
        "profile_lookup": None,
    },
    {
        "question": "What is your notice period?",
        "field_key": "notice_period",
        "category": "logistics",
        "profile_lookup": None,
    },
    {
        "question": "Are you authorised to work in South Africa?",
        "field_key": "work_authorisation",
        "category": "requirements",
        "profile_lookup": None,
    },
    {
        "question": "What is your availability / start date?",
        "field_key": "availability",
        "category": "logistics",
        "profile_lookup": None,
    },
    {
        "question": "What is your citizenship status?",
        "field_key": "citizenship",
        "category": "requirements",
        "profile_lookup": None,
    },
    {
        "question": "What is your race / equity group?",
        "field_key": "race",
        "category": "demographic",
        "profile_lookup": None,
    },
    {
        "question": "What is your gender?",
        "field_key": "gender",
        "category": "demographic",
        "profile_lookup": None,
    },
    {
        "question": "Do you have a disability?",
        "field_key": "disability",
        "category": "demographic",
        "profile_lookup": None,
    },
    {
        "question": "What is your age / date of birth?",
        "field_key": "date_of_birth",
        "category": "demographic",
        "profile_lookup": None,
    },
    {
        "question": "Are you a South African citizen?",
        "field_key": "south_african_citizen",
        "category": "eligibility",
        "profile_lookup": None,
    },
    {
        "question": "Are you a recent graduate (within 2 years)?",
        "field_key": "recent_graduate",
        "category": "eligibility",
        "profile_lookup": None,
    },
]

_ELIGIBILITY_KEYWORDS: dict[str, list[str]] = {
    "work_authorisation": ["work authorisation", "work authorization", "legally entitled", "right to work", "south africa"],
    "disability": ["disability", "disabled", "broad-based black economic empowerment", "bbbee", "employment equity"],
    "race": ["race", "equity", "diversity", "bbbee", "employment equity", "affirmative"],
    "gender": ["gender", "female", "male", "women in", "diversity"],
    "recent_graduate": ["recent graduate", "recently graduated", "new graduate", "graduate programme", "graduate program"],
    "south_african_citizen": ["south african citizen", "sa citizen", "citizen"],
    "drivers_licence": ["driver's licence", "drivers licence", "driver license", "own transport", "valid licence"],
    "relocation": ["relocate", "relocation", "willing to move"],
    "notice_period": ["notice period", "available immediately", "start date"],
    "expected_salary": ["salary", "remuneration", "compensation", "package"],
}


class AnswerStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._answers: dict[str, str] = {}
        if path and path.exists():
            self._load()

    def _load(self) -> None:
        if self._path and self._path.exists():
            try:
                self._answers = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._answers = {}

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._answers, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def get(self, field_key: str) -> Optional[str]:
        return self._answers.get(field_key)

    def set(self, field_key: str, answer: str) -> None:
        self._answers[field_key] = answer
        self._save()

    def has(self, field_key: str) -> bool:
        return field_key in self._answers and self._answers[field_key].strip() != ""

    def all_answers(self) -> dict[str, str]:
        return dict(self._answers)

    def bulk_set(self, answers: dict[str, str]) -> None:
        self._answers.update(answers)
        self._save()


class QuestionEngine:
    def __init__(self, answer_store: Optional[AnswerStore] = None) -> None:
        self.answer_store = answer_store or AnswerStore()

    def try_answer_from_profile(
        self, question_entry: dict, profile: Optional[CandidateProfile]
    ) -> Optional[str]:
        if profile is None:
            return None

        lookup = question_entry.get("profile_lookup")
        if not lookup:
            return None

        if lookup == "qualification":
            if profile.education:
                edu = profile.education[0]
                parts = [p for p in (edu.qualification, edu.field) if p]
                return " ".join(parts) if parts else None
            return None

        if lookup == "years_experience":
            if profile.experience:
                return str(len(profile.experience))
            return None

        return None

    def answer_question(
        self,
        question_entry: dict,
        profile: Optional[CandidateProfile],
    ) -> tuple[Optional[str], bool]:
        field_key = question_entry.get("field_key", "")

        if self.answer_store.has(field_key):
            return self.answer_store.get(field_key), False

        answer = self.try_answer_from_profile(question_entry, profile)
        if answer is not None:
            return answer, False

        return None, True

    def resolve_common_questions(
        self,
        profile: Optional[CandidateProfile],
        job_description: str = "",
    ) -> tuple[dict[str, str], list[MissingInfo]]:
        answered: dict[str, str] = {}
        missing: list[MissingInfo] = []

        for entry in _COMMON_QUESTIONS:
            answer, needs_input = self.answer_question(entry, profile)
            if needs_input:
                missing.append(MissingInfo(
                    question=entry["question"],
                    field_key=entry["field_key"],
                    category=entry.get("category", ""),
                ))
            elif answer is not None:
                answered[entry["field_key"]] = answer

        return answered, missing

    def resolve_job_specific_questions(
        self,
        screening_questions: list[dict],
        profile: Optional[CandidateProfile],
    ) -> tuple[dict[str, str], list[MissingInfo]]:
        answered: dict[str, str] = {}
        missing: list[MissingInfo] = []

        for q in screening_questions:
            field_key = q.get("field_key", q.get("question", "")[:50])
            if self.answer_store.has(field_key):
                answered[field_key] = self.answer_store.get(field_key)
                continue

            answer = self.try_answer_from_profile(q, profile)
            if answer is not None:
                answered[field_key] = answer
            else:
                missing.append(MissingInfo(
                    question=q.get("question", ""),
                    field_key=field_key,
                    category=q.get("category", "screening"),
                ))

        return answered, missing

    def detect_job_requirements(
        self,
        job_description: str,
        profile: Optional[CandidateProfile],
    ) -> tuple[dict[str, str], list[MissingInfo]]:
        answered: dict[str, str] = {}
        missing: list[MissingInfo] = []
        desc_lower = job_description.lower()

        for field_key, keywords in _ELIGIBILITY_KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                if self.answer_store.has(field_key):
                    answered[field_key] = self.answer_store.get(field_key)
                    continue
                if profile is not None:
                    answer = self.try_answer_from_profile(
                        {"field_key": field_key, "profile_lookup": field_key},
                        profile,
                    )
                    if answer is not None:
                        answered[field_key] = answer
                        continue
                question_entry = next(
                    (q for q in _COMMON_QUESTIONS if q["field_key"] == field_key),
                    None,
                )
                if question_entry:
                    missing.append(MissingInfo(
                        question=question_entry["question"],
                        field_key=field_key,
                        category=question_entry.get("category", ""),
                        priority="required" if field_key in ("work_authorisation", "citizenship") else "preferred",
                    ))

        return answered, missing
