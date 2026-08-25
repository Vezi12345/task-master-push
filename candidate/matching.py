from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from sources.base import Job

from .profile import CandidateProfile


# ---------------------------------------------------------------------------
# Skill normalisation (small, maintainable)
# ---------------------------------------------------------------------------

_SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "react.js": "react",
    "reactjs": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "postgres": "postgresql",
    "psql": "postgresql",
    "mssql": "sql server",
    "ms sql": "sql server",
    ".net": "c#/.net",
    "dotnet": "c#/.net",
    "c sharp": "c#/.net",
    "csharp": "c#/.net",
    "gcp": "google cloud",
    "k8s": "kubernetes",
    "tf": "terraform",
    "py": "python",
    "cplusplus": "c++",
    "cplusplus": "c++",
    "deep learning": "machine learning",
    "ml": "machine learning",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
}

_KNOWN_SKILLS: list[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    "html", "css", "react", "angular", "vue", "node.js", "django",
    "flask", "fastapi", "spring", "express", "rails", "laravel",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "jenkins", "git", "linux", "bash", "terraform", "ansible",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "pandas", "numpy", "scikit-learn", "spark", "hadoop",
    "agile", "scrum", "jira", "confluence",
    "excel", "power bi", "tableau", "sap", "salesforce",
    "photoshop", "illustrator", "figma",
    "c#/.net", ".net",
]

# Education field keywords
_EDU_FIELDS = {
    "computer science", "information technology", "ict", "software",
    "engineering", "data science", "mathematics", "statistics",
    "commerce", "bcom", "finance", "accounting", "business",
    "design", "graphics", "multimedia",
}

_EXPERIENCE_LEVELS = {
    "intern": 0, "junior": 1, "entry": 1, "graduate": 1, "trainee": 1,
    "mid": 2, "intermediate": 2, "experienced": 3,
    "senior": 4, "lead": 5, "principal": 5, "head": 5, "director": 5,
}


def _normalise_skill(skill: str) -> str:
    key = skill.strip().lower()
    return _SKILL_ALIASES.get(key, key)


_SHORT_SKILLS_ALLOW = {"go", "c#", "c++"}
_WORD_BOUNDARYSkills = {"r"}


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract known skill terms from free-text (job description)."""
    lowered = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for skill in _KNOWN_SKILLS:
        if len(skill) < 3 and skill not in _SHORT_SKILLS_ALLOW and skill not in _WORD_BOUNDARYSkills:
            continue
        if skill in _WORD_BOUNDARYSkills:
            if re.search(r'\b' + re.escape(skill) + r'\b', lowered) and skill not in seen:
                seen.add(skill)
                found.append(skill)
        elif skill in lowered and skill not in seen:
            seen.add(skill)
            found.append(skill)
    return found


def _candidate_all_skills(profile: CandidateProfile) -> set[str]:
    """Gather all skills from profile: declared skills + experience skills + project tech."""
    skills: set[str] = set()
    for s in profile.skills:
        skills.add(_normalise_skill(s))
    for exp in profile.experience:
        for s in exp.skills:
            skills.add(_normalise_skill(s))
    for proj in profile.projects:
        for t in proj.technologies:
            skills.add(_normalise_skill(t))
    return skills


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CandidateMatch(BaseModel):
    score: int = 0
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_match: str = ""
    education_match: str = ""
    location_match: bool = False
    certification_match: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class ApplicationReadiness(BaseModel):
    ready: bool = False
    score: int = 0
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MatchDimension(BaseModel):
    name: str = ""
    score: int = 0
    weight: float = 0.0
    reason: str = ""


class DetailedMatch(BaseModel):
    overall_score: int = 0
    dimensions: list[MatchDimension] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_match: str = ""
    education_match: str = ""
    location_match: bool = False
    remote_match: str = ""
    salary_match: str = ""
    certification_match: list[str] = Field(default_factory=list)
    employment_type_match: str = ""
    graduate_eligible: bool = False


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def match_candidate_to_job(
    profile: CandidateProfile,
    job: Job,
) -> CandidateMatch:
    candidate_skills = _candidate_all_skills(profile)
    job_skills = _extract_skills_from_text(f"{job.title} {job.description}")

    matched = sorted(candidate_skills & set(job_skills))
    missing = sorted(set(job_skills) - candidate_skills)

    skill_score = 0
    if job_skills:
        skill_score = round(100 * len(matched) / len(job_skills))
    else:
        skill_score = 50

    strengths: list[str] = []
    concerns: list[str] = []

    if matched:
        strengths.append(f"Matches {len(matched)} skill(s): {', '.join(matched[:6])}")
    if missing:
        concerns.append(f"Missing {len(missing)} skill(s): {', '.join(missing[:6])}")

    experience_match = _match_experience(profile, job)
    education_match = _match_education(profile, job)
    location_match = _match_location(profile, job)
    cert_match = _match_certifications(profile, job)

    if experience_match == "strong":
        strengths.append("Strong experience match")
    elif experience_match == "partial":
        strengths.append("Some relevant experience")
    elif experience_match == "weak":
        concerns.append("Limited relevant experience")

    if education_match == "relevant":
        strengths.append("Relevant qualification")
    elif education_match == "present":
        strengths.append("Has a qualification")
    elif education_match == "missing":
        concerns.append("No qualification found on CV")

    if location_match:
        strengths.append("Location matches")
    else:
        concerns.append("Location may not match")

    if cert_match:
        strengths.append(f"Relevant certification(s): {', '.join(cert_match[:3])}")

    raw = (
        skill_score * 0.50
        + _experience_score(experience_match) * 0.20
        + (80 if education_match in ("relevant", "present") else 30) * 0.10
        + (90 if location_match else 40) * 0.10
        + (80 if cert_match else 40) * 0.10
    )
    score = max(0, min(100, round(raw)))

    return CandidateMatch(
        score=score,
        matched_skills=matched,
        missing_skills=missing,
        experience_match=experience_match,
        education_match=education_match,
        location_match=location_match,
        certification_match=cert_match,
        strengths=strengths,
        concerns=concerns,
    )


# ---------------------------------------------------------------------------
# Dimension matchers
# ---------------------------------------------------------------------------

def _match_experience(profile: CandidateProfile, job: Job) -> str:
    text = f"{job.title} {job.description}".lower()
    job_level = _infer_job_level(text)
    candidate_level = _infer_candidate_level(profile)
    if job_level is None or candidate_level is None:
        return "unknown"
    if candidate_level >= job_level:
        return "strong"
    if job_level - candidate_level <= 1:
        return "partial"
    return "weak"


def _infer_job_level(text: str) -> int | None:
    for marker, level in _EXPERIENCE_LEVELS.items():
        if marker in text:
            return level
    if re.search(r"\b\d+\+?\s*years?\b", text):
        m = re.search(r"(\d+)\+?\s*years?", text)
        if m:
            y = int(m.group(1))
            if y <= 2:
                return 1
            if y <= 4:
                return 2
            if y <= 7:
                return 3
            return 4
    return None


def _infer_candidate_level(profile: CandidateProfile) -> int | None:
    levels: list[int] = []
    text = " ".join(
        [e.title.lower() for e in profile.experience]
        + [e.description.lower() for e in profile.experience]
    )
    for marker, level in _EXPERIENCE_LEVELS.items():
        if marker in text:
            levels.append(level)
    if profile.education:
        for edu in profile.education:
            qual = edu.qualification.lower()
            if any(k in qual for k in ("master", "mba", "m.sc", "meng")):
                levels.append(3)
            elif any(k in qual for k in ("bachelor", "b.sc", "beng", "b.com", "degree")):
                levels.append(1)
            elif any(k in qual for k in ("diploma", "certificate", "nd")):
                levels.append(1)
    if profile.projects:
        levels.append(1)
    if not levels:
        return None
    return max(levels)


def _experience_score(match: str) -> int:
    return {"strong": 90, "partial": 60, "weak": 30, "unknown": 50}.get(match, 50)


def _match_education(profile: CandidateProfile, job: Job) -> str:
    if not profile.education:
        return "unknown"
    text = f"{job.title} {job.description}".lower()
    relevant_fields = [f for f in _EDU_FIELDS if f in text]
    if not relevant_fields:
        return "present"
    for edu in profile.education:
        edu_blob = f"{edu.qualification} {edu.field}".lower()
        if any(f in edu_blob for f in relevant_fields):
            return "relevant"
        if any(f in text for f in relevant_fields):
            return "present"
    return "missing" if relevant_fields else "present"


def _match_location(profile: CandidateProfile, job: Job) -> bool:
    if job.remote:
        return True
    if not job.location or not profile.location:
        return True
    job_loc = job.location.lower()
    cand_loc = profile.location.lower()
    return cand_loc in job_loc or job_loc in cand_loc


def _match_certifications(profile: CandidateProfile, job: Job) -> list[str]:
    if not profile.certifications:
        return []
    text = f"{job.title} {job.description}".lower()
    matched: list[str] = []
    for cert in profile.certifications:
        name_lower = cert.name.lower()
        if any(word in text for word in name_lower.split() if len(word) > 2):
            matched.append(cert.name)
    return matched


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------

def match_jobs_to_candidate(
    profile: CandidateProfile,
    ranked: list,
) -> list[dict]:
    """Attach CandidateMatch to each RankedJob-like item.

    Each item must have a .job attribute (a Job).
    Returns a list of dicts with 'job', 'rank', and 'candidate_match' keys.
    """
    results: list[dict] = []
    for item in ranked:
        match = match_candidate_to_job(profile, item.job)
        results.append({
            "job": item.job,
            "rank": item,
            "candidate_match": match,
        })
    return results


# ---------------------------------------------------------------------------
# Application readiness
# ---------------------------------------------------------------------------

def assess_readiness(
    profile: CandidateProfile,
    job: Job,
    match: CandidateMatch,
) -> ApplicationReadiness:
    reasons: list[str] = list(match.strengths)
    blockers: list[str] = []
    warnings: list[str] = list(match.concerns)

    if not profile.email and not profile.phone:
        blockers.append("No contact details on CV — cannot apply")

    ready = len(blockers) == 0 and match.score >= 40
    score = match.score

    return ApplicationReadiness(
        ready=ready,
        score=score,
        reasons=reasons,
        blockers=blockers,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Multi-dimensional detailed matching
# ---------------------------------------------------------------------------

def _match_remote_preference(profile: CandidateProfile, job: Job) -> tuple[str, str]:
    if job.remote:
        return "match", "Job is remote — compatible with any location"
    return "unknown", ""


def _match_salary(profile: CandidateProfile, job: Job) -> tuple[str, str]:
    known = profile.get_known_value("expected_salary")
    if not known or not job.salary_min:
        return "unknown", ""
    try:
        expected = int(re.sub(r"[^\d]", "", known))
    except (ValueError, TypeError):
        return "unknown", ""
    if expected <= job.salary_min:
        return "match", f"Expected salary (R{expected:,}) is within range"
    if expected <= job.salary_min * 1.2:
        return "partial", f"Expected salary (R{expected:,}) slightly above minimum (R{job.salary_min:,})"
    return "concern", f"Expected salary (R{expected:,}) may exceed budget (R{job.salary_min:,})"


def _match_employment_type(profile: CandidateProfile, job: Job) -> tuple[str, str]:
    text = f"{job.title} {job.description}".lower()
    if any(kw in text for kw in ("intern", "internship", "learner", "trainee")):
        if profile.education:
            return "match", "Graduate/intern role — candidate has education"
        return "partial", "Intern/trainee role — education status unknown"
    if any(kw in text for kw in ("contract", "freelance", "temporary")):
        return "match", "Contract/temporary role"
    return "match", "Permanent/full-time role"


def _match_graduate_eligibility(profile: CandidateProfile, job: Job) -> tuple[bool, str]:
    text = f"{job.title} {job.description}".lower()
    is_graduate_role = any(kw in text for kw in ("graduate", "recent graduate", "entry level", "entry-level", "no experience"))
    if not is_graduate_role:
        return False, ""
    if profile.education:
        return True, "Candidate has qualifications suitable for graduate role"
    return False, "Candidate may lack formal qualifications for graduate role"


def match_candidate_to_job_detailed(
    profile: CandidateProfile,
    job: Job,
) -> DetailedMatch:
    candidate_skills = _candidate_all_skills(profile)
    job_skills = _extract_skills_from_text(f"{job.title} {job.description}")

    matched = sorted(candidate_skills & set(job_skills))
    missing = sorted(set(job_skills) - candidate_skills)

    skill_score = 0
    if job_skills:
        skill_score = round(100 * len(matched) / len(job_skills))
    else:
        skill_score = 50

    experience_match = _match_experience(profile, job)
    education_match = _match_education(profile, job)
    location_match = _match_location(profile, job)
    cert_match = _match_certifications(profile, job)
    remote_match, remote_reason = _match_remote_preference(profile, job)
    salary_match, salary_reason = _match_salary(profile, job)
    emp_type_match, emp_type_reason = _match_employment_type(profile, job)
    grad_eligible, grad_reason = _match_graduate_eligibility(profile, job)

    dimensions: list[MatchDimension] = [
        MatchDimension(name="skills", score=skill_score, weight=0.35,
                       reason=f"Matched {len(matched)}/{len(job_skills)} required skills" if job_skills else "No specific skills required"),
        MatchDimension(name="experience", score=_experience_score(experience_match), weight=0.20,
                       reason=f"Experience level: {experience_match}"),
        MatchDimension(name="education", score=80 if education_match in ("relevant", "present") else 30, weight=0.10,
                       reason=f"Education: {education_match}"),
        MatchDimension(name="location", score=90 if location_match else 40, weight=0.10,
                       reason="Location matches" if location_match else "Location may not match"),
        MatchDimension(name="certifications", score=80 if cert_match else 40, weight=0.05,
                       reason=f"Relevant certs: {', '.join(cert_match[:3])}" if cert_match else "No relevant certifications"),
        MatchDimension(name="remote", score=90 if remote_match == "match" else 50, weight=0.05,
                       reason=remote_reason or "Remote preference unknown"),
        MatchDimension(name="salary", score=70 if salary_match == "unknown" else (80 if salary_match == "match" else (60 if salary_match == "partial" else 40)), weight=0.05,
                       reason=salary_reason or "Salary expectations not specified"),
        MatchDimension(name="employment_type", score=80 if emp_type_match == "match" else 60, weight=0.05,
                       reason=emp_type_reason),
    ]

    strengths: list[str] = []
    concerns: list[str] = []
    if matched:
        strengths.append(f"Matches {len(matched)} skill(s): {', '.join(matched[:6])}")
    if missing:
        concerns.append(f"Missing {len(missing)} skill(s): {', '.join(missing[:6])}")
    if experience_match == "strong":
        strengths.append("Strong experience match")
    elif experience_match == "partial":
        strengths.append("Some relevant experience")
    elif experience_match == "weak":
        concerns.append("Limited relevant experience")
    if education_match == "relevant":
        strengths.append("Relevant qualification")
    elif education_match == "missing":
        concerns.append("No qualification found on CV")
    if location_match:
        strengths.append("Location matches")
    if cert_match:
        strengths.append(f"Relevant certification(s): {', '.join(cert_match[:3])}")
    if salary_match == "match" and salary_reason:
        strengths.append(salary_reason)
    elif salary_match == "concern":
        concerns.append(salary_reason)
    if grad_eligible and grad_reason:
        strengths.append(grad_reason)

    overall = max(0, min(100, round(
        sum(d.score * d.weight for d in dimensions) / sum(d.weight for d in dimensions)
        if dimensions else 0
    )))

    return DetailedMatch(
        overall_score=overall,
        dimensions=dimensions,
        strengths=strengths,
        concerns=concerns,
        matched_skills=matched,
        missing_skills=missing,
        experience_match=experience_match,
        education_match=education_match,
        location_match=location_match,
        remote_match=remote_match,
        salary_match=salary_match,
        certification_match=cert_match,
        employment_type_match=emp_type_match,
        graduate_eligible=grad_eligible,
    )
