from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from .profile import (
    CandidateProfile,
    Certification,
    Education,
    Experience,
    Project,
)


class CvExtractionError(RuntimeError):
    pass


def extract_pdf_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise CvExtractionError(f"File not found: {path}")
    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise CvExtractionError(f"Could not read PDF: {exc}") from exc
    text = "\n".join(pages).strip()
    if not text:
        raise CvExtractionError("PDF contains no extractable text")
    return text


def parse_cv(text: str, llm=None) -> CandidateProfile:
    profile = _parse_with_llm(text, llm)
    if profile is None:
        profile = _parse_with_rules(text)
    return profile


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------

_CV_SCHEMA = {
    "name": "full name string",
    "email": "email address or empty string",
    "phone": "phone number or empty string",
    "location": "city / region or empty string",
    "professional_summary": "one-paragraph career summary or empty string",
    "skills": ["list of skills"],
    "education": [
        {
            "institution": "string",
            "qualification": "e.g. BSc, Diploma",
            "field": "e.g. Computer Science",
            "start_date": "string or empty",
            "end_date": "string or empty",
        }
    ],
    "experience": [
        {
            "company": "string",
            "title": "job title",
            "start_date": "string or empty",
            "end_date": "string or empty",
            "description": "brief description of role",
            "skills": ["skills used in this role"],
        }
    ],
    "certifications": [
        {"name": "string", "issuer": "string", "date": "string or empty"}
    ],
    "projects": [
        {"name": "string", "description": "string", "technologies": ["list"]}
    ],
    "achievements": ["list of notable achievements"],
}


def _parse_with_llm(text: str, llm) -> CandidateProfile | None:
    if llm is None or not llm.is_available():
        return None
    import json

    system = (
        "You extract a structured candidate profile from the text of a CV / resume. "
        "Respond with ONLY valid JSON matching exactly this shape: "
        f"{json.dumps(_CV_SCHEMA)}. "
        "Use empty strings for missing fields, empty lists when nothing matches. "
        "Be faithful to the CV content — do not invent information."
    )
    user_prompt = f"Candidate CV text:\n\n{text[:6000]}"
    for _attempt in range(2):
        try:
            raw = llm.chat_json(system, user_prompt)
            return CandidateProfile(**raw)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Deterministic fallback parser
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?27|0)\s?[\d][\d\s\-]{7,}")

_SKILL_SECTIONS = re.compile(
    r"(?:skills|technical skills|technologies|competencies|core competencies)\s*:?\s*\n",
    re.IGNORECASE,
)

_SECTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:work\s+)?experience|employment\s+history|"
    r"education|qualifications?|academic\s+background|"
    r"certifications?|licenses?|"
    r"projects?|key\s+projects|"
    r"achievements?|awards?|accolades?|"
    r"(?:professional\s+)?summary|profile|about|objective"
    r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_EDU_KW = re.compile(
    r"\b(bachelor|master|b\.?sc|m\.?sc|b\.?a|m\.?a|b\.?tech|b\.?eng|diploma|degree|"
    r"national\s+diploma|advanced\s+diploma|h\.?d|ph\.?d|mba|associate|b\.?com|m\.?com)\b",
    re.IGNORECASE,
)

_CERT_KW = re.compile(
    r"\b(certified|certification|certificate|aws|azure|google cloud|gcp|pmp|"
    r"comptia|cisco|ccna|ccnp|oracle|itil|scrum|agile)\b",
    re.IGNORECASE,
)

_DATE_RANGE_RE = re.compile(
    r"(\w+\s+\d{4}|\d{4})\s*[-–— to]+\s*(\w+\s+\d{4}|\d{4}|present|current)",
    re.IGNORECASE,
)


def _parse_with_rules(text: str) -> CandidateProfile:
    lines = text.split("\n")

    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)

    sections = _split_sections(lines)
    summary_text = ""
    for header in ("summary", "profile", "about", "objective", "professional summary"):
        if header in sections:
            summary_text = "\n".join(sections[header]).strip()
            break

    skills = _extract_skills_from_text(text, sections)
    education = _extract_education(sections)
    experience = _extract_experience(sections)
    certifications = _extract_certifications(sections)

    name = _extract_name(lines)

    return CandidateProfile(
        name=name,
        email=email_match.group(0) if email_match else "",
        phone=phone_match.group(0).strip() if phone_match else "",
        location="",
        professional_summary=summary_text,
        skills=skills,
        education=education,
        experience=experience,
        certifications=certifications,
    )


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_header = "_preamble"
    sections[current_header] = []
    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            current_header = _SECTION_RE.match(line).group(0).strip().lower().rstrip(":")
            sections[current_header] = []
        else:
            sections[current_header].append(line)
    return sections


_CV_HEADER_PHRASES = {
    "curriculum vitae", "cv", "resume", "résumé",
    "personal details", "personal information", "contact details",
    "contact information", "profile", "summary", "about me",
    "career objective", "career summary", "professional summary",
    "professional profile", "work experience", "education",
    "skills", "certifications", "projects", "references",
    "key skills", "technical skills", "core competencies",
    "employment history", "academic background", "qualifications",
}


def _extract_name(lines: list[str]) -> str:
    for line in lines[:15]:
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue
        if _EMAIL_RE.search(stripped) or _PHONE_RE.search(stripped):
            continue
        if any(ch.isdigit() for ch in stripped):
            continue
        lower = stripped.lower().strip()
        if lower in _CV_HEADER_PHRASES:
            continue
        if any(lower.startswith(phrase) for phrase in _CV_HEADER_PHRASES):
            continue
        words = stripped.split()
        if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w):
            return stripped
    return ""


def _extract_skills_from_text(text: str, sections: dict) -> list[str]:
    blob = text.lower()
    skills: list[str] = []
    seen: set[str] = set()

    known_skills = [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
        "rust", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
        "html", "css", "react", "angular", "vue", "node.js", "node", "django",
        "flask", "fastapi", "spring", "express", "rails", "laravel",
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
        "linux", "bash", "powershell", "terraform", "ansible",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "pandas", "numpy", "scikit-learn", "spark", "hadoop",
        "agile", "scrum", "jira", "confluence",
        "excel", "power bi", "tableau", "sap", "salesforce",
        "photoshop", "illustrator", "figma", "sketch",
        "communication", "leadership", "teamwork", "problem solving",
        "project management", "time management",
    ]

    for skill in known_skills:
        if skill.lower() in blob:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                skills.append(skill)

    for header in sections:
        if "skill" in header or "technolog" in header or "competenc" in header:
            for line in sections[header]:
                for item in re.split(r"[,;|•\-\n]", line):
                    item = item.strip().strip("*").strip()
                    if item and len(item) < 50 and item.lower() not in seen:
                        seen.add(item.lower())
                        skills.append(item)

    return skills


def _extract_education(sections: dict) -> list[Education]:
    results: list[Education] = []
    for header in sections:
        if any(kw in header for kw in ("education", "qualification", "academic")):
            current: dict[str, str] = {}
            for line in sections[header]:
                line = line.strip()
                if not line:
                    if current:
                        results.append(Education(**current))
                        current = {}
                    continue
                date_match = _DATE_RANGE_RE.search(line)
                if _EDU_KW.search(line):
                    if current:
                        results.append(Education(**current))
                    current = {
                        "institution": "",
                        "qualification": "",
                        "field": "",
                        "start_date": date_match.group(1) if date_match else "",
                        "end_date": date_match.group(2) if date_match else "",
                    }
                    cleaned = _DATE_RANGE_RE.sub("", line).strip().rstrip("-–—")
                    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                    if len(parts) >= 2:
                        current["qualification"] = parts[0]
                        current["field"] = parts[1]
                        if len(parts) >= 3:
                            current["institution"] = parts[2]
                    elif parts:
                        current["qualification"] = parts[0]
                elif current:
                    if not current["institution"]:
                        current["institution"] = line
            if current:
                results.append(Education(**current))
    return results


def _extract_experience(sections: dict) -> list[Experience]:
    results: list[Experience] = []
    for header in sections:
        if any(kw in header for kw in ("experience", "employment", "work history")):
            current: dict = {}
            for line in sections[header]:
                line = line.strip()
                if not line:
                    if current:
                        results.append(Experience(**current))
                        current = {}
                    continue
                date_match = _DATE_RANGE_RE.search(line)
                if date_match and (any(c.isalpha() for c in line.split("-")[0]) or re.search(r"\b\d{4}\b", line)):
                    if current:
                        results.append(Experience(**current))
                    cleaned = _DATE_RANGE_RE.sub("", line).strip().rstrip("-–—")
                    parts = [p.strip() for p in cleaned.split(" at ") if p.strip()]
                    if len(parts) >= 2:
                        company = parts[1].strip(" ,;-–—")
                        current = {
                            "title": parts[0],
                            "company": company,
                            "start_date": date_match.group(1),
                            "end_date": date_match.group(2),
                            "description": "",
                            "skills": [],
                        }
                    else:
                        current = {
                            "title": cleaned or line,
                            "company": "",
                            "start_date": date_match.group(1),
                            "end_date": date_match.group(2),
                            "description": "",
                            "skills": [],
                        }
                elif current:
                    current["description"] = (
                        f"{current['description']} {line}".strip()
                        if current["description"]
                        else line
                    )
            if current:
                results.append(Experience(**current))
    return results


def _extract_certifications(sections: dict) -> list[Certification]:
    results: list[Certification] = []
    for header in sections:
        if any(kw in header for kw in ("certification", "certificate", "license")):
            for line in sections[header]:
                line = line.strip()
                if not line or not _CERT_KW.search(line):
                    continue
                date_match = _DATE_RANGE_RE.search(line)
                cleaned = _DATE_RANGE_RE.sub("", line).strip().rstrip("-–—")
                parts = [p.strip() for p in cleaned.split(",") if p.strip()]
                results.append(
                    Certification(
                        name=parts[0] if parts else cleaned,
                        issuer=parts[1] if len(parts) > 1 else "",
                        date=date_match.group(2) if date_match else "",
                    )
                )
    return results
