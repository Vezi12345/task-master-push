from __future__ import annotations

"""Persistent candidate profile / answer-memory system.

One canonical registry of every recurring application question. Each
answer carries provenance:

    source    user | cv | derived | generated
    status    verified | needs_confirmation | draft | unknown | user_required
    verified  True only when the USER explicitly supplied/confirmed it

Resolution priority (highest wins for display):
  1. explicit user answer            → verified
  2. existing verified profile data  → verified
  3. reliable CV/profile evidence    → evidence-based (needs_confirmation)
  4. AI-generated                    → draft only, never presented as verified
  5. nothing reliable                → unknown / user_required (sensitive)

CONFLICTS: a new answer that differs from an existing VERIFIED answer is
never written automatically — it becomes a pending conflict the user must
resolve ("which is correct?").

Sensitive attributes (demographics, citizenship, DOB) are NEVER inferred:
they are answered only from explicitly stored values, otherwise they are
marked ``user_required`` and asked.
"""

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from candidate.profile import CandidateProfile
from candidate.storage import load_profile, save_profile

from .question_engine import AnswerRecord, AnswerStore, QuestionEngine, stable_question_key
from .answer_engine import AnswerType


# ---------------------------------------------------------------------------
# canonical field registry
# ---------------------------------------------------------------------------

CATEGORY_ORDER = (
    "Personal",
    "Contact",
    "Education",
    "Work & Experience",
    "Technical Skills",
    "Preferences",
    "Application Questions",
)


@dataclass(frozen=True)
class RegistryField:
    key: str                # canonical engine/profile field key
    label: str              # dashboard label
    category: str           # dashboard group
    question: str           # canonical ask-question (memory lookup key)
    sensitive: bool = False
    input_kind: str = "text"   # text | bool | year | salary | url | list
    ask: bool = True           # shown on the "Complete your profile" screen


REGISTRY: tuple[RegistryField, ...] = (
    # -- Personal ----------------------------------------------------------
    RegistryField("first_name", "First name", "Personal", "What is your first name?"),
    RegistryField("last_name", "Last name", "Personal", "What is your last name?"),
    RegistryField("preferred_name", "Preferred name", "Personal", "What is your preferred name?"),
    RegistryField("date_of_birth", "Date of birth", "Personal",
                  "What is your date of birth?", sensitive=True),
    RegistryField("country_of_residence", "Country", "Personal",
                  "In which country do you currently live?"),
    RegistryField("citizenship", "Citizenship", "Personal",
                  "What is your citizenship?", sensitive=True),
    RegistryField("south_african_citizen", "South African citizen", "Personal",
                  "Are you a South African citizen?", sensitive=True, input_kind="bool"),
    RegistryField("work_authorisation", "Work authorisation", "Personal",
                  "Are you legally authorised to work in South Africa?"),
    RegistryField("location", "Location / city", "Personal",
                  "Where are you currently based?"),
    # demographics: asked ONLY here, never inferred (SA equity forms need
    # them; the engine refuses to answer these from any other source)
    RegistryField("race", "Race / equity group", "Personal",
                  "What is your race / equity group?", sensitive=True),
    RegistryField("gender", "Gender", "Personal",
                  "What is your gender?", sensitive=True),
    RegistryField("disability", "Disability status", "Personal",
                  "Do you have a disability?", sensitive=True),
    # -- Contact -------------------------------------------------------------
    RegistryField("email", "Email", "Contact", "What is your email address?",
                  input_kind="text"),
    RegistryField("phone", "Phone", "Contact", "What is your phone number?"),
    RegistryField("online_linkedin", "LinkedIn", "Contact",
                  "What is your LinkedIn profile URL?", input_kind="url"),
    RegistryField("online_github", "GitHub", "Contact",
                  "What is your GitHub profile URL?", input_kind="url"),
    RegistryField("online_portfolio", "Portfolio", "Contact",
                  "What is your portfolio URL?", input_kind="url"),
    RegistryField("online_website", "Website", "Contact",
                  "What is your personal website URL?", input_kind="url"),
    # -- Education -----------------------------------------------------------
    RegistryField("university", "University", "Education",
                  "Which university did you attend?"),
    RegistryField("highest_qualification", "Highest qualification", "Education",
                  "What is your highest qualification?"),
    RegistryField("education_result", "Academic result", "Education",
                  "What was your final academic result?"),
    RegistryField("graduation_year", "Graduation year", "Education",
                  "In which year did you graduate?", input_kind="year"),
    RegistryField("recent_graduate", "Recent graduate", "Education",
                  "Are you a recent graduate (within 2 years)?", input_kind="bool"),
    # -- Work & Experience -----------------------------------------------------
    RegistryField("years_experience", "Years of experience", "Work & Experience",
                  "How many years of work experience do you have?"),
    RegistryField("notice_period", "Notice period", "Work & Experience",
                  "What is your notice period?"),
    RegistryField("availability", "Availability / start date", "Work & Experience",
                  "When can you start?"),
    RegistryField("drivers_licence", "Driver's licence", "Work & Experience",
                  "Do you have a valid driver's licence?", input_kind="bool"),
    RegistryField("vehicle", "Own vehicle", "Work & Experience",
                  "Do you have your own vehicle or transport?", input_kind="bool"),
    # -- Technical Skills --------------------------------------------------------
    RegistryField("skills", "Technical skills", "Technical Skills",
                  "List your technical skills (comma separated)", input_kind="list",
                  ask=False),  # managed by the dedicated skills editor
    # -- Preferences ---------------------------------------------------------
    RegistryField("expected_salary", "Expected salary", "Preferences",
                  "What is your expected salary?", input_kind="salary"),
    RegistryField("minimum_salary", "Minimum salary", "Preferences",
                  "What is your minimum acceptable salary?", input_kind="salary"),
    RegistryField("relocation", "Willing to relocate", "Preferences",
                  "Are you willing to relocate?", input_kind="bool"),
    RegistryField("travel_preference", "Willing to travel", "Preferences",
                  "Are you willing to travel for work?", input_kind="bool"),
    RegistryField("work_preference", "Remote / hybrid / on-site", "Preferences",
                  "Do you prefer remote, hybrid or on-site work?"),
    RegistryField("preferred_locations", "Preferred locations", "Preferences",
                  "Which locations would you prefer to work in?", input_kind="list"),
)

REGISTRY_BY_KEY = {f.key: f for f in REGISTRY}
SENSITIVE_REGISTRY_KEYS = frozenset(f.key for f in REGISTRY if f.sensitive)
_LIST_KEYS = frozenset(f.key for f in REGISTRY if f.input_kind == "list")

# keys that never live on CandidateProfile as scalars — set_known would fail
_NON_PROFILE_KEYS = _LIST_KEYS | {"first_name", "last_name", "university"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalize_answer(value: str) -> str:
    """Loose equality for answers: 'Yes' vs 'yes!' are the same fact."""
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def answers_equivalent(a: str, b: str) -> bool:
    return normalize_answer(a) == normalize_answer(b) and bool(normalize_answer(a))


def field_for_question(question: str) -> Optional[RegistryField]:
    """Map an employer's phrasing to a canonical registry field.

    Uses the semantic classifier; returns None when the meaning is
    uncertain — callers must ASK rather than blindly match."""
    from .answer_engine import classify_question

    key, _category = classify_question(question)
    if key and key in REGISTRY_BY_KEY:
        return REGISTRY_BY_KEY[key]
    return None


_STATUS_ORDER = {
    "verified": 0, "needs_confirmation": 1, "draft": 2,
    "user_required": 3, "unknown": 4,
}


# ---------------------------------------------------------------------------
# pending conflicts store
# ---------------------------------------------------------------------------

def _conflicts_path(path: Optional[Path] = None) -> Path:
    return path or config.ANSWER_CONFLICTS_FILE


def _load_conflicts(path: Optional[Path] = None) -> list[dict]:
    p = _conflicts_path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_conflicts(items: list[dict], path: Optional[Path] = None) -> None:
    p = _conflicts_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------

class ProfileMemoryService:
    """Reads/writes the candidate's persistent answer memory with full
    provenance, status derivation and conflict gating."""

    def __init__(
        self,
        profile: Optional[CandidateProfile] = None,
        store: Optional[AnswerStore] = None,
        conflicts_path: Optional[Path] = None,
        persist_profile: bool = True,
    ) -> None:
        self._profile = profile
        self.store = store or AnswerStore()
        self.engine = QuestionEngine(self.store)
        self.conflicts_path = conflicts_path
        self.persist_profile = persist_profile

    # -- profile access ------------------------------------------------------

    @property
    def profile(self) -> Optional[CandidateProfile]:
        if self._profile is None:
            self._profile = load_profile()
        return self._profile

    def _save_profile(self) -> None:
        if self.persist_profile and self._profile is not None:
            save_profile(self._profile)

    # -- snapshot -------------------------------------------------------------

    def snapshot(self) -> dict:
        fields = [self._entry(rf) for rf in REGISTRY]
        fields.extend(self._custom_question_entries())
        categories: dict[str, list[dict]] = {name: [] for name in CATEGORY_ORDER}
        for entry in fields:
            categories.setdefault(entry["category"], []).append(entry)
        counts = {"verified": 0, "needs_confirmation": 0, "draft": 0,
                  "unknown": 0, "user_required": 0}
        for entry in fields:
            if entry["status"] in counts:
                counts[entry["status"]] += 1
        return {
            "categories": [
                {"name": name, "fields": entries}
                for name, entries in categories.items() if entries
            ],
            "counts": counts,
            "pending_conflicts": self.pending_conflicts(),
        }

    def _entry(self, rf: RegistryField) -> dict:
        base = {
            "key": rf.key, "label": rf.label, "category": rf.category,
            "question": rf.question, "sensitive": rf.sensitive,
            "input_kind": rf.input_kind, "custom": False,
        }
        rec = self.store.record(rf.key)
        if rec is not None:
            base.update({
                "value": rec.answer, "source": rec.source,
                "status": rec.status, "verified": rec.verified,
                "updated_at": rec.updated_at,
                "explanation": "Saved by you" if rec.verified else (rec.evidence or ""),
            })
            return base

        result = self.engine.answer(rf.question, self.profile)
        value, status, verified = result.answer, "unknown", False
        source, explanation = result.source, result.explanation

        if result.is_answered:
            if result.answer_type in (AnswerType.VERIFIED, AnswerType.USER_PROVIDED):
                status, verified = "verified", True
                source = "cv" if result.source.startswith("profile") else result.source
                explanation = result.explanation
            elif result.answer_type == AnswerType.SENSITIVE:
                status, verified = "verified", True
                source = "user"
                explanation = result.explanation
            elif result.answer_type == AnswerType.DERIVED:
                status = "needs_confirmation"
                source = "derived"
            else:  # GENERATED_FROM_EVIDENCE — draft only
                status, source = "draft", "generated"
        elif rf.sensitive:
            status = "user_required"
            explanation = "Sensitive information is never inferred — please provide it"

        return {**base, "value": value, "source": source, "status": status,
                "verified": verified, "updated_at": "",
                "explanation": explanation}

    def _custom_question_entries(self) -> list[dict]:
        """Free-form questions previously answered without a canonical field."""
        entries = []
        profile = self.profile
        seen_keys = set()
        mems = profile.question_memory if profile else []
        for mem in mems:
            if mem.field_key in REGISTRY_BY_KEY or not mem.field_key.startswith("q_"):
                continue
            if mem.field_key in seen_keys:
                continue
            seen_keys.add(mem.field_key)
            verified = mem.source == "user"
            entries.append({
                "key": mem.field_key, "label": mem.question,
                "category": "Application Questions", "question": mem.question,
                "sensitive": False, "input_kind": "text", "custom": True,
                "value": mem.answer,
                "source": mem.source,
                "status": "verified" if verified else "needs_confirmation",
                "verified": verified,
                "updated_at": mem.updated_at,
                "explanation": mem.evidence or "Saved application question",
            })
        return entries

    # -- asking flow ------------------------------------------------------------

    def missing_questions(self) -> list[dict]:
        """The 'Complete your profile' queue: everything unknown or
        user-required that we would otherwise have to ask."""
        out = []
        for entry in self.snapshot()["categories"]:
            for f in entry["fields"]:
                if f["status"] in ("unknown", "user_required") \
                        and not f["custom"] and f.get("ask", True):
                    out.append({
                        "key": f["key"], "label": f["label"],
                        "question": f["question"], "category": f["category"],
                        "priority": "high" if f["status"] == "user_required" else "normal",
                        "input_kind": f["input_kind"],
                    })
        return out

    # -- saving ------------------------------------------------------------------

    def save_user_answer(
        self, key: str, answer: str, question: str = "",
    ) -> dict:
        """Save a user-supplied answer. If a different VERIFIED answer
        already exists, create a conflict instead of overwriting."""
        key = (key or "").strip()
        answer = (answer or "").strip()
        if not key or not answer:
            return {"ok": False, "error": "key and answer are required"}

        rf = REGISTRY_BY_KEY.get(key)
        current = self._entry(rf) if rf else self._current_custom(key, question)

        if current and current.get("verified") and current.get("value"):
            if answers_equivalent(current["value"], answer):
                return {"ok": True, "saved": True, "unchanged": True}
            conflict_id = secrets.token_urlsafe(8)
            ask_q = (current.get("question") if current else "") \
                or question or (rf.question if rf else "")
            items = _load_conflicts(self.conflicts_path)
            items.append({
                "id": conflict_id,
                "key": key,
                "question": ask_q,
                "existing_value": current["value"],
                "proposed_value": answer,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            })
            _save_conflicts(items, self.conflicts_path)
            return {
                "ok": True, "conflict": True, "conflict_id": conflict_id,
                "message": "A verified answer already exists — confirm which "
                           "one is correct before anything is changed.",
                "existing_value": current["value"],
                "proposed_value": answer,
            }
        return self.commit_answer(key, answer, question=question)

    def commit_answer(self, key: str, answer: str, question: str = "") -> dict:
        """Write the answer as USER-VERIFIED across all memory layers."""
        key = (key or "").strip()
        answer = (answer or "").strip()
        if not key or not answer:
            return {"ok": False, "error": "key and answer are required"}

        rf = REGISTRY_BY_KEY.get(key)
        ask_question = question or (rf.question if rf else "") or key

        self.store.set_record(AnswerRecord(
            field_key=key, answer=answer,
            source="user", status="verified", verified=True,
            question=ask_question,
        ))

        profile = self.profile
        if profile is None:
            from candidate.storage import save_profile as _sp
            profile = CandidateProfile()
            self._profile = profile
            if self.persist_profile:
                _sp(profile)
        profile.remember_answer(
            ask_question, answer,
            field_key=key if rf else (key if key.startswith("q_") else ""),
            source="user", confidence="high",
        )
        if rf and key not in _NON_PROFILE_KEYS:
            profile.set_known(key, answer, "user")
        self._save_profile()
        return {"ok": True, "saved": True, "key": key}

    # -- conflicts ------------------------------------------------------------------

    def pending_conflicts(self) -> list[dict]:
        return _load_conflicts(self.conflicts_path)

    def resolve_conflict(self, conflict_id: str, choice: str) -> dict:
        """choice: 'existing' keeps the verified value (and re-confirms it);
        'new' replaces it with the proposed value. Both close the conflict."""
        items = _load_conflicts(self.conflicts_path)
        found = next((c for c in items if c.get("id") == conflict_id), None)
        if found is None:
            return {"ok": False, "error": "Conflict not found"}
        if choice not in ("existing", "new"):
            return {"ok": False, "error": "choice must be 'existing' or 'new'"}

        if choice == "new":
            outcome = self.commit_answer(found["key"], found["proposed_value"],
                                         question=found.get("question", ""))
            if not outcome.get("ok"):
                return outcome
        else:
            # keep existing — record explicit re-confirmation so the audit
            # trail shows the user actively chose it
            rec = self.store.record(found["key"])
            if rec is not None:
                rec.verified = True
                rec.status = "verified"
                rec.source = "user"
                self.store.set_record(rec)

        _save_conflicts(
            [c for c in items if c.get("id") != conflict_id],
            self.conflicts_path,
        )
        return {"ok": True, "resolved": conflict_id, "choice": choice}

    # -- custom questions -----------------------------------------------------------

    def _current_custom(self, key: str, question: str) -> Optional[dict]:
        profile = self.profile
        mems = profile.question_memory if profile else []
        for mem in mems:
            if mem.field_key == key or (
                not mem.field_key and mem.question.lower() == (question or "").lower()
            ):
                return {
                    "key": key, "question": mem.question, "value": mem.answer,
                    "verified": mem.source == "user", "status":
                        "verified" if mem.source == "user" else "needs_confirmation",
                    "custom": True, "label": mem.question,
                }
        rec = self.store.record(key)
        if rec is not None:
            return {"key": key, "question": rec.question or question,
                    "value": rec.answer, "verified": rec.verified,
                    "status": rec.status, "custom": True, "label": question}
        return None


__all__ = [
    "ProfileMemoryService",
    "RegistryField",
    "REGISTRY",
    "REGISTRY_BY_KEY",
    "CATEGORY_ORDER",
    "field_for_question",
    "answers_equivalent",
    "normalize_answer",
]
