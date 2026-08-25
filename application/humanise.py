from __future__ import annotations

"""Humanise AI-generated application answers and validate them against the
candidate profile.

Pipeline required before any generated answer is submitted:

    QUESTION → GENERATE DRAFT → HUMANISE DRAFT
             → VALIDATE AGAINST CANDIDATE PROFILE → SUBMIT

Rules:
  * no corporate filler ("highly motivated", "leverage my skillset", ...)
  * natural first-person sentences a real candidate would write
  * never claim experience, qualifications or certifications the profile
    does not support — validation strips or rejects such claims
"""

import re
from typing import Optional

from candidate.profile import CandidateProfile

# corporate / robotic phrases mapped to plain-language replacements
_CLICHE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bas a highly motivated and passionate individual,?\s*", re.I), ""),
    (re.compile(r"\bhighly motivated\b[,\s]*", re.I), ""),
    (re.compile(r"\bpassionate about leveraging\b", re.I), "interested in using"),
    (re.compile(r"\bexcited to leverage\b", re.I), "keen to use"),
    (re.compile(r"\bleverage my comprehensive technical skillset\b", re.I), "use my skills"),
    (re.compile(r"\bleverage\b", re.I), "use"),
    (re.compile(r"\bmy comprehensive (technical )?skill-?set\b", re.I), "my skills"),
    (re.compile(r"\ba results-driven\b", re.I), "a"),
    (re.compile(r"\bdynamic professional\b", re.I), "worker"),
    (re.compile(r"\bsynergy\b", re.I), "teamwork"),
    (re.compile(r"\bthink outside the box\b", re.I), "find practical solutions"),
    (re.compile(r"\bproven track record of\b", re.I), "history of"),
    (re.compile(r"\bin today's fast-paced (world|environment)\b,?\s*", re.I), ""),
    (re.compile(r"\bat the end of the day\b,?\s*", re.I), ""),
    (re.compile(r"\bI am writing to express my keen interest\b", re.I), "I am interested in this role because"),
]

# sentence fragments left over after removing clichés
_JUNK_START = re.compile(r"^(?:and|but|so|also|,\s*|\s+)+", re.I)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_SENTENCE_GAP = re.compile(r"\.\s*\.")

_YEARS_CLAIM = re.compile(
    r"\b(?P<years>\d+(?:\.\d+)?|\w+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:work\s+)?experience\b",
    re.I,
)
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_CERT_CLAIM = re.compile(r"\bI am certified in ([^.,;]+)|\bmy ([\w\s]+) certification\b", re.I)
_QUALIFICATION_CLAIM = re.compile(
    r"\b(?:degree|diploma|certificate|bachelors?|bachelor's?|masters?|master's?|national diploma)\s+(?:in|of)\s+([^.,;]+)",
    re.I,
)


def humanise(text: str, profile: Optional[CandidateProfile] = None) -> str:
    """Rewrite an AI draft into plain, natural first-person language."""
    if not text:
        return ""
    out = text.strip()
    for pattern, replacement in _CLICHE_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    out = _MULTI_SENTENCE_GAP.sub(". ", out)
    out = "\n".join(line.strip() for line in out.splitlines())
    out = _MULTI_SPACE.sub(" ", out)
    # drop empty leading fragments like ", and"
    lines = []
    for line in out.splitlines():
        line = _JUNK_START.sub("", line).strip()
        if line:
            lines.append(line)
    out = "\n".join(lines)
    if out and not out.endswith((".", "!", "?")):
        out += "."
    return out


def _claimed_years(text: str) -> list[tuple[str, float]]:
    claimed = []
    for m in _YEARS_CLAIM.finditer(text):
        raw = m.group("years").lower()
        if raw in _WORD_NUMBERS:
            claimed.append((m.group(0), float(_WORD_NUMBERS[raw])))
        else:
            try:
                claimed.append((m.group(0), float(raw)))
            except ValueError:
                continue
    return claimed


def validate_against_profile(
    text: str, profile: CandidateProfile
) -> tuple[bool, list[str]]:
    """Return ``(ok, issues)``. ``ok`` False means the draft claims facts the
    candidate cannot back up — the draft must NOT be submitted."""
    from application.answer_engine import compute_years_experience  # local: avoids import cycle at module load

    issues: list[str] = []

    actual_years = compute_years_experience(profile)
    for claim_text, years in _claimed_years(text):
        if years <= 1.5:
            continue  # "looking for my first year of experience" style is fine
        if actual_years is None or years > max(actual_years + 0.5, 1.5):
            issues.append(f"Claims '{claim_text.strip()}' but profile supports "
                          f"{actual_years if actual_years is not None else 'no'} year(s)")

    edu_blob = " ".join(
        " ".join(filter(None, [e.qualification, e.field, e.institution]))
        for e in (profile.education or [])
    ).lower()
    for m in _QUALIFICATION_CLAIM.finditer(text):
        subject = m.group(1).strip().lower()
        # cut trailing clauses: "… Diploma in X and completed an internship"
        subject = re.split(
            r"\b(?:and|also|which|that|where|while|i)\b", subject)[0].strip()
        if len(subject) < 4:
            continue
        key_terms = [t for t in re.split(r"\W+", subject) if len(t) >= 3]
        # tolerate abbreviations/wording drift: at least half the key terms
        # must literally appear in the recorded education text
        matched = sum(1 for t in key_terms if t in edu_blob)
        supported = bool(edu_blob) and bool(key_terms) \
            and matched / len(key_terms) >= 0.5
        if not supported:
            issues.append(f"Claims qualification '{m.group(0).strip()}' "
                          "not supported by CV education")

    certs = getattr(profile, "certifications", None) or []
    cert_blob = " ".join(
        c.name if hasattr(c, "name") else str(c) for c in certs
    ).lower()
    for m in _CERT_CLAIM.finditer(text):
        subject = (m.group(1) or m.group(2) or "").strip().lower()
        if len(subject) < 4:
            continue
        words = [w for w in re.split(r"\W+", subject) if len(w) >= 3]
        if not cert_blob or not any(w in cert_blob for w in words):
            issues.append(f"Claims certification '{subject}' not present in profile")

    return (not issues), issues


def humanise_and_validate(
    text: str, profile: CandidateProfile
) -> tuple[str, bool, list[str]]:
    """Convenience wrapper: humanise then validate. Returns
    ``(final_text, ok, issues)`` — only submit when ``ok`` is True."""
    final = humanise(text, profile)
    ok, issues = validate_against_profile(final, profile)
    return final, ok, issues
