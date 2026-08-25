"""Explainable candidate-vs-job suitability engine.

Evaluates the *ability to perform a job*, not title similarity:

* weighted, configurable scoring dimensions (no profession-specific code)
* hard requirements (licences, registrations, statutory conditions,
  mandatory qualifications/experience) OVERRIDE the numeric score
* REQUIRED vs PREFERRED distinction — missing preferred items only
  reduce the score
* unknown-but-mandatory facts become ``requires_user_input``, never guesses
* every result carries an explanation for the UI / run reports
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from application.answer_engine import compute_years_experience
from candidate.occupations import OCCUPATIONS as _ALL_OCCS
from candidate.occupations import contains_phrase, normalize
from candidate.search_profile import build_search_profile

DEFAULT_WEIGHTS = {
    "qualification": 20,
    "skills": 25,
    "experience": 15,
    "occupation_fit": 10,
    "transferability": 10,
    "location": 5,
    "employment_type": 5,
    "preferences": 10,
}

REQUIRED_CUE = re.compile(
    r"\b(must|required|requireds|essential|compulsory|minimum of|at least|"
    r"only)\b|\brequirements?\s*:?",
    re.IGNORECASE,
)
PREFERRED_CUE = re.compile(
    r"(advantageous|preferred|desirable|bonus|nice to have|added advantage)",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"[.;\n]\s*")

# kind -> regex over a single sentence
_REQUIREMENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("registration", re.compile(
        r"\b(sanc|sapc|hpcsa|ecsa|saica|psira)\b"
        r"|\bprofessional registration\b|\bregistration with\b"
        r"|\bregistered with\b", re.IGNORECASE)),
    ("licence_code", re.compile(
        r"\bcode\s*(8|10|14|b|eb|c1|c)\b[^.;]{0,40}licen[cs]e"
        r"|licen[cs]e[^.;]{0,40}\bcode\s*(8|10|14|b|eb|c1|c)\b"
        r"|^\s*(code\s*(8|10|14))\b", re.IGNORECASE)),
    ("licence_generic", re.compile(
        r"\bvalid\b[^.;]{0,25}\blicen[cs]e\b|\b(driver'?s?|driving)\s+licen[cs]e\b",
        re.IGNORECASE)),
    ("criminal_record", re.compile(r"\bcriminal record\b", re.IGNORECASE)),
    ("security_clearance", re.compile(
        r"\bsecurity clearance\b", re.IGNORECASE)),
    ("citizenship", re.compile(
        r"\bsouth africans? only\b|\bsa citizens? only\b|\bcitizens? only\b"
        r"|\bvalid work (permit|visa)\b", re.IGNORECASE)),
    ("qualification", re.compile(
        r"\b((?:b ?)?(?:tech|com|sc|eng)?\s?degree|national diploma|"
        r"diploma|matric|grade 12|n6|n4 certificate|certificate)\b",
        re.IGNORECASE)),
    ("years_experience", re.compile(
        r"(\d+)\s*\+?\s*(?:-|to)?\s*(?:\d+)?\s*years?"
        r"[^.;]{0,30}experience", re.IGNORECASE)),
]

_LICENCE_CODES = re.compile(r"\bcode\s*(8|10|14|b|eb|c1|c)\b", re.IGNORECASE)

_SENIOR_TITLE_MARKERS = (
    "senior", "principal", "head of", "manager", "director", "chief",
    "specialist", "lead",
)


@dataclass
class Requirement:
    kind: str
    sentence: str
    required: bool


@dataclass
class SuitabilityResult:
    score: int
    verdict: str
    decision: str  # "apply" | "reject" | "requires_user_input"
    matched: list[str] = field(default_factory=list)
    missing_preferred: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    dimension_scores: dict[str, int] = field(default_factory=dict)

    def explain(self) -> str:
        lines: list[str] = []
        if self.blockers:
            lines.append("Reason:")
            lines += [f"- {b}" for b in self.blockers]
        if self.unknowns:
            lines.append("REQUIRES_USER_INPUT reason:" if self.decision ==
                         "requires_user_input" else "Unknown:")
            lines += [f"- {u}" for u in self.unknowns]
        if self.matched:
            lines.append("Why:")
            lines += [f"- {m}" for m in self.matched[:6]]
        if self.missing_preferred:
            lines.append("Missing/preferred:")
            lines += [f"- {m}" for m in self.missing_preferred[:4]]
        return "\n".join(lines).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def detect_requirements(job_text: str) -> list[Requirement]:
    found: list[Requirement] = []
    for sentence in _sentences(job_text):
        matched_kind = None
        for kind, pattern in _REQUIREMENT_PATTERNS:
            if not pattern.search(sentence):
                continue
            preferred = bool(PREFERRED_CUE.search(sentence))
            explicit_required = bool(REQUIRED_CUE.search(sentence))
            # statutory/operational musts are required even when stated bare;
            # qualifications and experience need an explicit cue or they are
            # treated as merely preferred.
            bare_is_hard = kind in (
                "registration", "licence_code", "licence_generic",
                "criminal_record", "security_clearance", "citizenship",
            )
            required = (not preferred) and (
                explicit_required or bare_is_hard)
            found.append(Requirement(kind=kind, sentence=sentence.strip(),
                                     required=required))
            matched_kind = kind
        if matched_kind is None and PREFERRED_CUE.search(sentence):
            # a nice-to-have the specific patterns don't cover
            # (e.g. "SAP experience advantageous") — never blocking
            found.append(Requirement(kind="preferred_extra",
                                     sentence=sentence.strip(),
                                     required=False))
    return found


def _profile_evidence_text(profile) -> str:
    bits = [profile.skills and ", ".join(profile.skills) or "",
            getattr(profile, "professional_summary", "") or ""]
    bits += [f"{e.qualification} {e.field}" for e in profile.education]
    bits += [c.name for c in profile.certifications]
    return normalize(" | ".join(b for b in bits if b))


def _token_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#]+", text.lower()) if len(w) > 1}


def _phrase_in_profile(profile_text: str, phrase: str) -> bool:
    """Fuzzy containment of the requirement phrase in profile evidence."""
    req_tokens = _token_set(phrase)
    if not req_tokens:
        return False
    prof_tokens = _token_set(profile_text)
    overlap = len(req_tokens & prof_tokens) / max(3, min(len(req_tokens), 6))
    return overlap >= 0.45


# A school-leaving (matric-level) demand is satisfied by any tertiary
# qualification: a diploma or degree proves Grade 12 was completed.
_MATRIC_LEVEL_RE = re.compile(
    r"grade\s*1[0-2]|\bmatric\b|senior certificate|nqf\s*(?:level\s*)?[1-4]\b",
    re.I)
_HIGHER_QUAL_RE = re.compile(
    r"diploma|degree|btech|b\.?\s?tech|bachelor|honours|hons|master"
    r"|nqf\s*(?:level\s*)?[5-9]", re.I)
_TERTIARY_QUAL_RE = re.compile(
    r"diploma|degree|btech|b\.?\s?tech|bachelor|honours|hons|master"
    r"|advanced certificate|nqf\s*(?:level\s*)?[5-9]|n\d\b", re.I)
_SCHOOL_SUBJECT_RE = re.compile(
    r"grade\s*1[0-2][^.;]*?\b(?:with|including|plus)\s+([A-Za-z][A-Za-z ,&/]{2,40})",
    re.I)


def _highest_tertiary_qualification(profile) -> str:
    for edu in profile.education or []:
        qual = normalize(getattr(edu, "qualification", "") or "")
        if qual and _TERTIARY_QUAL_RE.search(qual):
            return qual
    return ""


def _is_matric_only_requirement(sentence: str) -> bool:
    return bool(_MATRIC_LEVEL_RE.search(sentence)) and not _HIGHER_QUAL_RE.search(
        sentence)


def evaluate(job, profile, weights: Optional[dict] = None) -> SuitabilityResult:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    job_text = normalize(f"{job.title} {job.description}")
    sp = build_search_profile(profile)
    profile_text = _profile_evidence_text(profile)
    requirements = detect_requirements(f"{job.title}. {job.description}")

    blockers: list[str] = []
    unknowns: list[str] = []
    matched: list[str] = []
    missing_preferred: list[str] = []

    # ---------- hard requirements (one evaluation per kind, strongest first) --
    by_kind: dict[str, Requirement] = {}
    for req in requirements:
        if req.kind == "licence_generic" and any(
                r.kind == "licence_code" for r in requirements):
            continue  # the code-specific rule is strictly stronger
        by_kind.setdefault(req.kind, req)

    matric_exceeded = False
    for kind, req in by_kind.items():
        required = req.required

        if req.kind in ("registration",):
            body = _REGISTRATION_BODY_RE.search(req.sentence)
            token = body.group(0).lower() if body else "professional registration"
            if token in profile_text:
                matched.append(f"Registration evidence present ({token})")
            elif not req.required:
                missing_preferred.append(req.sentence)
            else:
                unknowns.append(
                    f"Job requires {token} - profile does not state whether "
                    "you are registered")
        elif req.kind == "licence_code":
            codes = {c.lower() for c in _LICENCE_CODES.findall(req.sentence)}
            licence_text = normalize(getattr(profile, "drivers_licence", "") or "")
            held_codes = {c.lower() for c in _LICENCE_CODES.findall(licence_text)}
            wanted = "/".join(f"code {c}" for c in sorted(codes)) or "a driver's licence"
            if not licence_text:
                unknowns.append(
                    f"Job requires {wanted} - profile does not say whether "
                    "you have one")
            elif codes and not (codes & held_codes):
                blockers.append(
                    f"Job requires {wanted}; your profile shows: "
                    f"{licence_text}")
            else:
                matched.append("Driver's licence requirement satisfied")
        elif req.kind == "licence_generic":
            licence_text = normalize(getattr(profile, "drivers_licence", "") or "")
            if not licence_text:
                unknowns.append(
                    "Job requires a driver's licence - profile does not say "
                    "whether you have one")
            else:
                matched.append("Driver's licence requirement satisfied")
        elif req.kind == "citizenship":
            cit = normalize(getattr(profile, "citizenship", "") or "")
            if not cit:
                unknowns.append(
                    "Role is restricted to South African citizens / valid "
                    "work permits - profile does not state citizenship")
            elif any(word in cit for word in ("south african", "sa citizen")):
                matched.append("Citizenship requirement satisfied")
            else:
                blockers.append(
                    f"Role restricted to SA citizens; profile states '{cit}'")
        elif req.kind == "criminal_record":
            unknowns.append(
                "Advert mentions criminal-record checks - confirm your "
                "status before applying")
        elif req.kind == "security_clearance":
            unknowns.append(
                "Role requires security clearance - this cannot be verified "
                "from a CV")
        elif req.kind == "qualification":
            if not req.required:
                missing_preferred.append(req.sentence)
            elif not profile.education:
                unknowns.append(
                    f"Job requires qualification ('{req.sentence}') - no "
                    "education captured in profile")
            elif _phrase_in_profile(profile_text, req.sentence):
                matched.append(f"Qualification matches: {req.sentence}")
            elif _is_matric_only_requirement(req.sentence) and (
                    top_qual := _highest_tertiary_qualification(profile)):
                matric_exceeded = True
                matched.append(
                    "Matric-level requirement satisfied - you hold "
                    f"{top_qual}")
                subject = _SCHOOL_SUBJECT_RE.search(req.sentence)
                if subject and not normalize(subject.group(1)) in profile_text:
                    unknowns.append(
                        f"Advert also asks for '{subject.group(1).strip()}' "
                        "at school level - double-check you meet that detail")
            else:
                blockers.append(
                    f"Job requires qualification ('{req.sentence}') which "
                    "your profile does not show")
        elif req.kind == "preferred_extra":
            if _phrase_in_profile(profile_text, req.sentence):
                matched.append(f"You already have: {req.sentence}")
            else:
                missing_preferred.append(req.sentence)
        elif req.kind == "years_experience":
            m = re.search(r"(\d+)", req.sentence)
            need = int(m.group(1)) if m else 0
            years = compute_years_experience(profile)
            if not req.required:
                if years is not None and years < need:
                    missing_preferred.append(req.sentence)
                continue
            if years is None:
                unknowns.append(
                    f"Job requires ~{need} years experience - profile has no "
                    "dated employment to verify this")
            elif years < need:
                blockers.append(
                    f"Job requires ~{need} years experience; dated profile "
                    f"evidence shows {years:g} years")
            else:
                matched.append(f"Experience requirement met ({years:g} yrs)")

    # ---------- scored dimensions ----------
    dims: dict[str, int] = {}
    occ_fit, transfer = _occupation_fit(job, sp)

    # pollution control: a missing role-specific CREDENTIAL (professional
    # registration) on an obviously misaligned role is a plain reject;
    # identity facts (licence, citizenship, criminal record, clearance)
    # always stay answerable questions regardless of alignment
    _registration_unknown = re.compile(r"you are registered", re.I)
    if occ_fit < 0.5 and unknowns and not blockers and all(
            _registration_unknown.search(u) for u in unknowns):
        blockers.append(
            "Role does not align with your evidenced occupations - " +
            unknowns[0].rstrip())
        unknowns = []

    dims["skills"] = round(_skill_score(profile, job_text) * w["skills"])
    skill_ratio, skill_hits, skill_total = _skill_match(profile, job_text)
    if skill_hits:
        matched.append(f"{skill_hits}/{skill_total} of your skill keywords "
                       "appear in the advert")
    dims["qualification"] = round(_qualification_score(
        requirements, profile_text, matric_exceeded=matric_exceeded) *
        w["qualification"])
    dims["experience"], exp_note = _experience_score(sp, job, job_text)
    dims["experience"] = round(dims["experience"] * w["experience"])
    dims["occupation_fit"] = round(occ_fit * w["occupation_fit"])
    dims["transferability"] = round(transfer * w["transferability"])
    dims["location"] = round(_location_score(job, sp) * w["location"])
    dims["employment_type"] = round(_employment_score(job, sp) *
                                    w["employment_type"])
    dims["preferences"] = round(_preference_score(job, profile, sp) *
                                w["preferences"])

    score = max(0, min(100, sum(dims.values())))
    verdict = ("excellent" if score >= 80 else "strong" if score >= 65
               else "possible" if score >= 50 else "unsuitable")

    if blockers:
        decision = "reject"
    elif unknowns:
        decision = "requires_user_input"
    else:
        decision = "apply"

    if exp_note:
        matched.append(exp_note)
    if occ_fit >= 0.99:
        matched.append("Occupation aligns with your background")
    elif occ_fit >= 0.8:
        matched.append("Adjacent role - skills appear transferable")
    elif sp.occupations:
        # cross-domain possibility: allowed, but clearly labelled so the
        # candidate sees this is NOT their home occupation
        strong_labels = ", ".join(
            o["label"] for o in sp.occupations[:2]) or "no clear occupation"
        matched.append(
            "Transferable match: advert is outside your strongest "
            f"occupation ({strong_labels}); scored only on your actual "
            "evidence, review before applying")

    return SuitabilityResult(
        score=score, verdict=verdict, decision=decision, matched=matched,
        missing_preferred=missing_preferred, blockers=blockers,
        unknowns=unknowns, dimension_scores=dims)


_REGISTRATION_BODY_RE = re.compile(
    r"\b(sanc|sapc|hpcsa|ecsa|saica|psira)\b", re.IGNORECASE)


def _skill_score(profile, job_text: str) -> float:
    ratio, _, _ = _skill_match(profile, job_text)
    return ratio


def _skill_match(profile, job_text: str) -> tuple[float, int, int]:
    """(ratio, hits, total) of the candidate's skill words found in the ad."""
    tokens = []
    seen: set[str] = set()
    for s in list(profile.skills)[:8]:
        for word in s.split():
            wl = word.lower().strip(",;")
            if len(wl) > 1 and wl not in seen:
                seen.add(wl)
                tokens.append(wl)
    if not tokens:
        return 0.35, 0, 0     # no skill evidence at all — below demonstrated
    hits = sum(1 for t in tokens if contains_phrase(job_text, t))
    if hits == 0:
        return 0.15, 0, len(tokens)
    ratio = min(1.0, hits / max(3, min(len(tokens), 5)))
    return ratio, hits, len(tokens)


def _qualification_score(requirements: list[Requirement],
                         profile_text: str,
                         matric_exceeded: bool = False) -> float:
    quals = [r for r in requirements if r.kind == "qualification"]
    if not quals:
        return 0.7  # nothing stated either way
    required = [q for q in quals if q.required]
    if not required:
        return 0.9
    for q in required:
        if _phrase_in_profile(profile_text, q.sentence):
            return 1.0
    if matric_exceeded:
        # school-level demand proven by a tertiary qualification: strong,
        # though subject-specific details may still need checking
        return 0.85
    return 0.2  # blocked anyway via blockers path


def _experience_score(sp, job, job_text: str) -> tuple[float, str]:
    text = f"{job.title} {job.description}".lower()
    senior_job = any(m in text for m in _SENIOR_TITLE_MARKERS) or \
        bool(re.search(r"(?:minimum of|at least)\s+\d+\s*years?", text))
    entry_job = any(m in text for m in (
        "entry level", "entry-level", "graduate", "intern", "trainee",
        "junior", "no experience"))
    level = sp.career_level
    if senior_job:
        if level in ("mid-level", "senior"):
            return 1.0, ""
        return 0.35, ""
    if entry_job:
        if level in ("student", "graduate", "entry-level", "junior"):
            return 1.0, "Entry-level requirement fits your career stage"
        return 0.7, ""
    return 0.75, ""


def _occupation_fit(job, sp) -> tuple[float, float]:
    """(fit, transferability) from taxonomy vs the JOB's own words."""
    title = normalize(job.title)
    text = normalize(f"{job.title} {job.description}")[:600]
    if not sp.occupations:
        return 0.5, 0.5
    best_fit, best_transfer = 0.3, 0.3
    for occ_entry in sp.occupations[:3]:
        occ_key = occ_entry["key"]
        occ = next((o for o in _ALL_OCCS if o.key == occ_key), None)
        if occ is None:
            continue
        fit = 0.3
        for phrase in occ.titles:
            if contains_phrase(title, phrase):
                fit = 1.0
                break
        if fit < 0.85:
            for kw in occ.keywords:
                if contains_phrase(text, kw):
                    fit = max(fit, 0.55)
                    break
        transfer = fit
        if fit < 1.0:
            for phrase in occ.adjacent:
                if contains_phrase(title, phrase):
                    transfer = max(transfer, 0.9)
                    break
        best_fit = max(best_fit, fit)
        best_transfer = max(best_transfer, transfer)
    return best_fit, best_transfer


def _location_score(job, sp) -> float:
    loc = normalize(getattr(job, "location", "") or "")
    if getattr(job, "remote", False) and sp.remote_preference != "no":
        return 1.0
    if not sp.locations:
        return 1.0
    if not loc:
        return 0.6
    wanted = [normalize(p) for p in sp.locations]
    if any(p and (p in loc or loc in p) for p in wanted):
        return 1.0
    # same-province style partial match on first comma segment
    city = wanted[0].split()[0]
    if city and city in loc:
        return 0.8
    return 0.15


def _employment_score(job, sp) -> float:
    if not sp.employment_types:
        return 1.0
    text = f"{getattr(job, 'title', '')} {getattr(job, 'description', '')}".lower()
    for etype in sp.employment_types:
        if etype.replace("-", " ") in text or etype in text:
            return 1.0
    if "internship" in sp.employment_types and any(
            m in text for m in ("intern", "graduate programme",
                                "graduate program")):
        return 1.0
    return 0.4


def _preference_score(job, profile, sp) -> float:
    score = 1.0
    salary_min = getattr(job, "salary_min", None)
    prefs = normalize(getattr(profile, "expected_salary", "") or
                      getattr(profile, "minimum_salary", "") or "")
    m = re.search(r"(\d[\d,\s]*)", prefs)
    if salary_min and m:
        try:
            want = int(float(m.group(1).replace(",", "").replace(" ", "")))
            if salary_min < want:
                score -= 0.6
        except ValueError:
            pass
    if sp.remote_preference == "preferred" and not getattr(
            job, "remote", False):
        score -= 0.3
    return max(0.0, score)
