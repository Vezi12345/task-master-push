"""Derive a structured, candidate-specific job-search profile.

The candidate's profile/CV is the source of truth.  Nothing in this module
is profession-specific: occupations are inferred by matching the
candidate's own evidence (titles, qualifications, skills, certifications)
against the generic occupation registry, and queries are generated from
whatever evidence exists.  When nothing matches, the candidate's literal
titles and field-of-study words are used directly instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from candidate.occupations import (
    OCCUPATIONS,
    Occupation,
    contains_phrase,
    normalize,
)


@dataclass
class SearchProfile:
    career_level: str = "entry-level"
    locations: list[str] = field(default_factory=list)
    remote_preference: str = "any"
    employment_types: list[str] = field(default_factory=list)
    occupations: list[dict] = field(default_factory=list)  # [{key,label,score}]
    direct_titles: list[str] = field(default_factory=list)
    adjacent_titles: list[str] = field(default_factory=list)
    qualification_terms: list[str] = field(default_factory=list)
    skill_terms: list[str] = field(default_factory=list)
    own_titles: list[str] = field(default_factory=list)
    inference: str = "evidence"  # "evidence" | "literal" | "minimal"
    # title -> why this role is searched (explainable expansion)
    expansion_reasons: dict[str, str] = field(default_factory=dict)


def _candidate_evidence(profile) -> dict[str, str]:
    exp_titles = [normalize(e.title) for e in profile.experience if e.title]
    proj_roles = [normalize(p.role) for p in getattr(profile, "projects", [])
                  if getattr(p, "role", "")]
    edu_fields = [normalize(e.field) for e in profile.education if e.field]
    quals = [normalize(e.qualification) for e in profile.education
             if e.qualification]
    certs = [normalize(c.name) for c in profile.certifications if c.name]
    skills = [normalize(s) for s in profile.skills if s]
    return {
        "titles": " | ".join(t for t in exp_titles + proj_roles if t),
        "education": " | ".join(x for x in quals + edu_fields if x),
        "certs": " | ".join(certs),
        "skills": ", ".join(skills),
        "summary": normalize(getattr(profile, "professional_summary", "") or ""),
    }


def infer_occupations(profile) -> list[dict]:
    """Score every registry occupation against the candidate's evidence.

    Uniform treatment for all professions — no per-profession branches.
    """
    ev = _candidate_evidence(profile)
    title_hay = f"{ev['titles']} | {ev['summary']}"
    qual_hay = ev["education"]
    cert_hay = f"{ev['certs']} | {ev['summary']}"
    skill_hay = ev["skills"]

    scored: list[dict] = []
    for occ in OCCUPATIONS:
        score = 0
        for phrase in occ.titles:
            if contains_phrase(title_hay, phrase):
                score += 3
        for phrase in occ.qualifications:
            if contains_phrase(qual_hay, phrase):
                score += 2
        for phrase in occ.qualifications:
            if contains_phrase(cert_hay, phrase):
                score += 1
        for kw in occ.keywords:
            if contains_phrase(skill_hay, kw) or contains_phrase(title_hay, kw):
                score += 1
        if score > 0:
            scored.append({"key": occ.key, "label": occ.label,
                           "score": score})
    scored.sort(key=lambda item: (-item["score"], item["label"]))
    return scored


def derive_career_level(profile) -> str:
    """Infer career level strictly from evidence; never invent seniority."""
    from application.answer_engine import (
        compute_experience_months,
        latest_graduation_year,
    )

    summary = normalize(getattr(profile, "professional_summary", "") or "")
    if any(m in summary for m in ("student", "studying", "undergraduate")):
        return "student"

    months = compute_experience_months(profile)
    grad_year = latest_graduation_year(profile)
    now = date.today()

    if months is None or months <= 0:
        if grad_year and 0 <= now.year - grad_year <= 2:
            return "graduate"
        return "entry-level"
    years = months / 12.0
    if years < 1:
        return "entry-level"
    if years < 3:
        return "junior"
    if years < 7:
        return "mid-level"
    return "senior"


_LEVEL_PREFIXES = {
    "student": ("internship", "graduate internship"),
    "graduate": ("graduate",),
    "entry-level": ("junior", "entry level"),
    "junior": ("junior",),
}


def _location_for_queries(profile) -> str:
    prefs = [str(p).strip() for p in (profile.preferred_locations or [])]
    if prefs:
        return prefs[0]
    city = (getattr(profile, "city", "") or "").strip()
    if city:
        return city.split(",")[0].strip()
    location = (getattr(profile, "location", "") or "").strip()
    if location:
        return location.split(",")[0].strip()
    return ""


def build_search_profile(profile) -> SearchProfile:
    sp = SearchProfile()

    sp.career_level = derive_career_level(profile)
    sp.locations = ([str(p) for p in profile.preferred_locations]
                    if profile.preferred_locations else
                    ([profile.city] if getattr(profile, "city", "") else []))
    wp = normalize(getattr(profile, "work_preference", "") or "")
    if "remote" in wp:
        sp.remote_preference = "preferred"
    elif any(w in wp for w in ("hybrid",)):
        sp.remote_preference = "preferred"
    for etype in ("full-time", "part-time", "contract", "internship",
                  "temporary"):
        if etype in wp:
            sp.employment_types.append(etype)

    inferred = infer_occupations(profile)
    strong = [o for o in inferred if o["score"] >= 3]
    chosen = (strong or inferred[:1]) if inferred else []
    sp.occupations = chosen

    if chosen:
        by_key = {occ.key: occ for occ in OCCUPATIONS}
        seen_direct: set[str] = set()
        seen_adjacent: set[str] = set()
        for entry in chosen[:3]:
            occ: Occupation = by_key[entry["key"]]
            label = occ.label
            for t in occ.titles:
                if t not in seen_direct:
                    seen_direct.add(t)
                    sp.direct_titles.append(t)
                    sp.expansion_reasons[t] = (
                        f"Direct role of {label} (matched from your "
                        "experience/qualification evidence)")
            for t in occ.adjacent:
                if t not in seen_adjacent:
                    seen_adjacent.add(t)
                    sp.adjacent_titles.append(t)
                    sp.expansion_reasons[t] = (
                        f"Adjacent to {label} - employers commonly accept "
                        "the same skills")
            for q in occ.qualifications:
                if q not in sp.qualification_terms:
                    sp.qualification_terms.append(q)
                    if q not in sp.expansion_reasons:
                        sp.expansion_reasons[q] = (
                            f"Qualification field associated with {label}")
        sp.skill_terms = [
            s for s in (normalize(x) for x in profile.skills)
            if s][:8]
        own = [t for t in (normalize(e.title) for e in profile.experience)
               if t and t not in sp.direct_titles]
        sp.own_titles = own[:4]
        sp.inference = "evidence"
    else:
        # Generic path: no registry match — use the candidate's own words.
        own = [normalize(e.title) for e in profile.experience if e.title]
        fields = [normalize(e.field) for e in profile.education if e.field]
        quals = [normalize(e.qualification) for e in profile.education
                 if e.qualification]
        sp.direct_titles = list(dict.fromkeys(own))[:6]
        sp.qualification_terms = list(dict.fromkeys(fields + quals))[:4]
        sp.skill_terms = [normalize(s) for s in profile.skills if s][:6]
        for t in sp.direct_titles:
            sp.expansion_reasons[t] = "Your own previous job title"
        for q in sp.qualification_terms:
            sp.expansion_reasons[q] = "Taken from your stated qualification"
        sp.inference = "literal" if (sp.direct_titles or
                                     sp.qualification_terms) else "minimal"
    return sp


def generate_queries(sp: SearchProfile, max_queries: int = 8) -> list[str]:
    """Diverse, level-aware, location-anchored query strings."""
    place = ""
    if sp.locations:
        place = sp.locations[0]

    prefixes: tuple[str, ...] = _LEVEL_PREFIXES.get(sp.career_level, ())
    if sp.career_level == "student" and not prefixes:
        prefixes = ("internship",)

    direct: list[str] = list(sp.direct_titles)[:5]
    adjacent: list[str] = list(sp.adjacent_titles)[:3]
    quals: list[str] = list(sp.qualification_terms)[:2]

    queries: list[str] = []

    def emit(core: str, prefix: str = "") -> None:
        core_l = core.lower()
        # never double the level marker ("junior junior accountant")
        if prefix and any(core_l.startswith(p) for p in prefixes):
            prefix = ""
        text = " ".join(p for p in (prefix, core, place) if p)
        key = " ".join(text.lower().split())
        if key and key not in {q.lower() for q in queries}:
            queries.append(text)

    # 1. direct matches with level prefix on the first two only
    for i, title in enumerate(direct):
        prefix = prefixes[i % len(prefixes)] if prefixes and i < 2 else ""
        emit(title, prefix)
    # 2. adjacent / transferable roles (no prefix — they are their own niche)
    for title in adjacent:
        emit(title)
    # 3. qualification-based opportunities
    for q in quals:
        if sp.career_level in ("graduate", "student", "entry-level"):
            emit(f"{q} graduate")
        else:
            emit(q)
    # 4. own previous titles as-is (people often continue what they did)
    for t in sp.own_titles[:2]:
        emit(t)

    # 5. honest last resort when the CV gives us almost nothing
    if not queries:
        fallback_core = "entry level" if sp.career_level in (
            "entry-level", "student", "graduate") else ""
        emit(fallback_core or "jobs")

    return queries[:max_queries]


def describe_strategy(sp: SearchProfile) -> str:
    occ_labels = ", ".join(o["label"] for o in sp.occupations) or "none inferred"
    return (f"career level: {sp.career_level}; occupations: {occ_labels}; "
            f"inference: {sp.inference}; locations: "
            f"{', '.join(sp.locations) or 'any'}")
