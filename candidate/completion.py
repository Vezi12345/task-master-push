"""Profile-completion meter.

The overall percentage is weighted: contact details, education and skills are
high-value for screening; high-school extras are low-value. A section only
counts when it holds real information — empty lists/records contribute zero.
"""
from __future__ import annotations

from candidate.profile import CandidateProfile


def _pct(filled: int, total: int) -> float:
    return round(100.0 * filled / total) if total else 100.0


def _section_personal(profile: CandidateProfile) -> tuple[float, list[str]]:
    keys = [
        ("name", bool((profile.name or "").strip())),
        ("first_name", bool(profile.first_name)),
        ("last_name", profile.last_name is not None),
        ("email", bool((profile.email or "").strip())),
        ("phone", bool((profile.phone or "").strip())),
        ("location", bool((profile.location or "").strip())),
        ("country", bool((profile.country_of_residence or "").strip())),
        ("date_of_birth", bool((profile.date_of_birth or "").strip())),
    ]
    missing = [k for k, ok in keys if not ok]
    return _pct(len(keys) - len(missing), len(keys)), missing


def _section_education(profile: CandidateProfile) -> tuple[float, list[str]]:
    if not profile.education:
        return 0.0, ["Add at least one qualification"]
    edu = next(
        (e for e in profile.education if e.is_highest), None
    ) or profile.education[0]
    checks = [
        ("institution", bool(edu.institution.strip())),
        ("qualification", bool(edu.qualification.strip())),
        ("dates", bool((edu.start_date or edu.end_date).strip())),
        # a result is NOT the qualification name — "Diploma in ICT" != "65%"
        ("result", bool(edu.result.strip())),
    ]
    missing = [f"education.{k}" for k, ok in checks if not ok]
    return _pct(len(checks) - len(missing), len(checks)), missing


def _section_experience(profile: CandidateProfile) -> tuple[float, list[str]]:
    if not profile.experience:
        return 0.0, ["Add work experience (or mark yourself as a student/fresh graduate)"]
    exp = profile.experience[0]
    checks = [
        ("company", bool(exp.company.strip())),
        ("title", bool(exp.title.strip())),
        ("dates", bool((exp.start_date or exp.end_date).strip())),
        ("description", bool(exp.description.strip() or exp.responsibilities)),
    ]
    missing = [f"experience.{k}" for k, ok in checks if not ok]
    return _pct(len(checks) - len(missing), len(checks)), missing


def _section_projects(profile: CandidateProfile) -> tuple[float, list[str]]:
    if not profile.projects:
        return 0.0, ["Add at least one project (personal, academic or work)"]
    proj = profile.projects[0]
    checks = [
        ("name_or_description",
         bool(proj.name.strip() or proj.description.strip())),
        ("technologies", bool(proj.technologies)),
        ("link", bool((proj.url or proj.github_url).strip())),
    ]
    missing = [f"projects.{k}" for k, ok in checks if not ok]
    return _pct(len(checks) - len(missing), len(checks)), missing


def _section_skills(profile: CandidateProfile) -> tuple[float, list[str]]:
    if not profile.skills and not profile.skill_details:
        return 0.0, ["Add your key skills"]
    with_evidence = sum(
        1 for s in profile.skill_details
        if s.evidence.strip() or s.source == "work"
    )
    total = max(len(profile.skill_details), len(profile.skills))
    # evidence-backed skills raise the score above the bare-name baseline
    bonus = min(with_evidence / max(total, 1), 1.0)
    return min(60 + int(40 * bonus), 100), []


def _section_certifications(profile: CandidateProfile) -> tuple[float, list[str]]:
    if profile.certifications:
        return 100.0, []
    return 50.0, ["List certifications (or achievements) — optional but valuable"]


def _section_preferences(profile: CandidateProfile) -> tuple[float, list[str]]:
    checks = [
        ("expected_salary", bool((profile.expected_salary or "").strip())),
        ("relocation", bool((profile.relocation or "").strip())),
        ("notice_period", bool((profile.notice_period or "").strip())),
    ]
    missing = [f"preferences.{k}" for k, ok in checks if not ok]
    return _pct(len(checks) - len(missing), len(checks)), missing


def _section_online_profiles(profile: CandidateProfile) -> tuple[float, list[str]]:
    op = profile.online_profiles
    filled = sum(bool(op.get(k)) for k in ("linkedin", "website", "github", "portfolio"))
    return _pct(filled, 2), []          # linkedin + one other link is enough


def _section_high_school(profile: CandidateProfile) -> tuple[float, list[str]]:
    hs = profile.high_school
    if hs.is_empty:
        return 0.0, []
    checks = [
        ("school", bool(hs.school.strip())),
        ("mathematics_result", bool(hs.mathematics_result.strip())),
        ("overall_result", bool(hs.overall_result.strip())),
    ]
    missing = [f"high_school.{k}" for k, ok in checks if not ok]
    return _pct(len(checks) - len(missing), len(checks)), []


# weights favour information employers actually screen on first
_WEIGHTS = {
    "personal": 20,
    "education": 18,
    "skills": 14,
    "projects": 12,
    "experience": 12,
    "preferences": 8,
    "online_profiles": 6,
    "certifications": 4,
    "high_school": 3,
    "application_answers": 3,
}


def compute_completion(profile: CandidateProfile) -> dict:
    """Return {overall, sections:{name:{percent,weight,missing}}}."""
    personal_pct, personal_missing = _section_personal(profile)
    edu_pct, edu_missing = _section_education(profile)
    exp_pct, exp_missing = _section_experience(profile)
    proj_pct, proj_missing = _section_projects(profile)
    skills_pct, skills_missing = _section_skills(profile)
    cert_pct, cert_missing = _section_certifications(profile)
    pref_pct, pref_missing = _section_preferences(profile)
    online_pct, online_missing = _section_online_profiles(profile)
    hs_pct, hs_missing = _section_high_school(profile)

    answered = len(profile.question_memory)
    answers_pct = min(100, answered * 25)

    sections = {
        "personal": {"percent": personal_pct,
                     "missing": personal_missing},
        "education": {"percent": edu_pct, "missing": edu_missing},
        "experience": {"percent": exp_pct, "missing": exp_missing},
        "projects": {"percent": proj_pct, "missing": proj_missing},
        "skills": {"percent": skills_pct, "missing": skills_missing},
        "certifications": {"percent": cert_pct, "missing": cert_missing},
        "preferences": {"percent": pref_pct, "missing": pref_missing},
        "online_profiles": {"percent": online_pct, "missing": online_missing},
        "high_school": {"percent": hs_pct, "missing": hs_missing},
        "application_answers": {"percent": answers_pct, "missing": []},
    }

    total_weight = sum(_WEIGHTS.values())
    overall = sum(
        data["percent"] * _WEIGHTS[name] for name, data in sections.items()
    ) / total_weight

    result = {
        "overall": round(overall),
        "sections": {
            name: {**data, "weight": _WEIGHTS[name]}
            for name, data in sections.items()
        },
    }
    return result


def high_value_missing(profile: CandidateProfile) -> list[dict]:
    """Ordered prompts for progressive onboarding — highest value first."""
    completion = compute_completion(profile)
    prompts: list[tuple[int, str, str]] = []

    def add(section: str, label: str, route: str, weight_bonus: int = 0) -> None:
        prompts.append((-(_WEIGHTS[section]) - weight_bonus, label, route))

    p = completion["sections"]
    if p["personal"]["percent"] < 100:
        add("personal", "Confirm your contact details (name, email, phone)",
            "#personal")
    if p["education"]["percent"] < 100 and profile.education:
        add("education", "Add your degree RESULT (marks/GPA) to your "
                         "highest qualification", "#education")
    if p["skills"]["percent"] < 60:
        add("skills", "List your key technical skills", "#skills")
    if p["projects"]["percent"] == 0:
        add("projects", "Describe one project you can talk about in "
                        "interviews", "#projects")
    if p["experience"]["percent"] == 0:
        add("experience", "Add any work experience or internships",
            "#experience")
    if p["online_profiles"]["percent"] < 100:
        add("online_profiles", "Save your LinkedIn URL (and portfolio/GitHub)",
            "#online")
    if p["preferences"]["percent"] < 100:
        add("preferences", "Set salary expectations and notice period",
            "#preferences")
    if p["high_school"]["percent"] < 100 and not profile.high_school.is_empty:
        add("high_school", "Complete your high-school results record "
                           "(mathematics etc.)", "#high-school")

    prompts.sort(key=lambda t: t[0])
    return [{"label": label, "target": route} for _, label, route in prompts[:6]]
