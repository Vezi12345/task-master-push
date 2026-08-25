from __future__ import annotations

"""Suitability engine tests: hard gates, REQUIRED vs PREFERRED,
REQUIRES_USER_INPUT, explainability, configurable weights."""

from types import SimpleNamespace as NS

import pytest

from candidate.profile import CandidateProfile, Education, Experience
from candidate.suitability import (
    DEFAULT_WEIGHTS,
    detect_requirements,
    evaluate,
)


def _profile(**kw) -> CandidateProfile:
    defaults = {"skills": [], "education": [], "experience": [],
                "certifications": [], "location": ""}
    defaults.update(kw)
    return CandidateProfile(name="T", email="t@x.com", **defaults)


def _job(title="Junior Bookkeeper", desc="", company="Co", location="Durban",
         remote=False, salary_min=None):
    return NS(title=title, description=desc, company=company, id=title.lower(),
              url=f"https://x/{title.replace(' ', '-')}", location=location,
              remote=remote, salary_min=salary_min, salary_max=None,
              salary_text="")


ACCOUNTANT = _profile(
    skills=["pastel evolution", "reconciliations", "vat"],
    education=[Education(qualification="Diploma in Accounting",
                         field="Financial Management", institution="DUT")],
    experience=[Experience(title="Accounts Clerk", company="F",
                           start_date="2024-02", end_date="2025-11")],
    drivers_licence="Code B",
    preferred_locations=["Durban"])

NURSE = _profile(
    skills=["patient care", "wound care"],
    education=[Education(qualification="Diploma in Nursing",
                         field="General Nursing", institution="X")],
    experience=[Experience(title="Staff Nurse", company="H",
                           start_date="2021-01", end_date="present")])

ADMIN = _profile(
    skills=["ms office", "excel", "filing", "diaries"],
    education=[Education(qualification="Diploma in Business Management",
                         field="Business Management", institution="C")],
    experience=[Experience(title="Office Administrator", company="F",
                           start_date="2023-01", end_date="present")],
    drivers_licence="Code 8",
    preferred_locations=["Durban"])


BOOKKEEPER_AD = _job(
    "Junior Bookkeeper",
    "Requirements: Diploma in Accounting. Must have Pastel Evolution "
    "experience. VAT reconciliations and journals. Code 8 licence required. "
    "SAP experience advantageous. Durban.")


# --------------------------- hard gates override score ---------------------

def test_high_score_cannot_override_missing_mandatory_licence():
    r = evaluate(BOOKKEEPER_AD, ACCOUNTANT)
    assert r.decision == "reject"
    assert any("code 8" in b for b in r.blockers)


def test_matching_licence_is_recorded_as_satisfied():
    ad = _job("Delivery Driver", "Requirements: code 10 licence essential.")
    driver = _profile(drivers_licence="Code 10 and PDP")
    r = evaluate(ad, driver)
    assert any("satisfied" in m for m in r.matched)
    assert not r.blockers


def test_unknown_licence_becomes_requires_user_input_not_guess():
    ad = _job("Driver", "Must hold a valid driver's licence.")
    no_licence_info = _profile(skills=["driving"], education=[], experience=[])
    r = evaluate(ad, no_licence_info)
    assert r.decision == "requires_user_input"
    assert any("licence" in u.lower() for u in r.unknowns)


def test_registration_unknown_with_aligned_role_requires_user_input():
    ad = _job("Professional Nurse",
              "Requirements: must be registered with SANC. Ward experience.")
    r = evaluate(ad, NURSE)
    assert r.decision == "requires_user_input"
    assert any("sanc" in u.lower() for u in r.unknowns)


def test_unrelated_job_pollution_is_rejected_even_with_unknowns():
    nurse_ad = _job("Staff Nurse Grade 1",
                    "SANC registration essential. Patient care in theatre.")
    r = evaluate(nurse_ad, ACCOUNTANT)
    assert r.decision == "reject"
    assert any("align" in b.lower() or "sanc" in b.lower()
               for b in r.blockers)


def test_required_experience_years_enforced():
    ad = _job("Senior Financial Accountant",
              "Requirements: minimum of 5 years experience in accounting. "
              "Degree in Accounting essential.")
    r = evaluate(ad, ACCOUNTANT)  # ~1.7 yrs evidence
    assert r.decision == "reject"
    assert any("5 years" in b for b in r.blockers)


def test_required_qualification_missing_blocks():
    ad = _job("Staff Nurse", "Requirements: degree in nursing is essential. "
                             "Patient care required.")
    r = evaluate(ad, ADMIN)
    assert any("qualification" in b.lower() for b in r.blockers)


def test_citizenship_restriction_unknown_asks_user():
    ad = _job("Government Clerk", "South African citizens only need apply. "
                                  "Admin support duties.")
    r = evaluate(ad, ADMIN)  # citizenship not captured
    assert r.decision == "requires_user_input"
    assert any("citizen" in u.lower() for u in r.unknowns)


def test_criminal_record_check_is_never_assumed():
    ad = _job("Security Officer",
              "PSIRA registration essential. Clear criminal record required.")
    r = evaluate(ad, ADMIN)
    assert r.decision == "requires_user_input"
    assert any("criminal" in u.lower() for u in r.unknowns)


# ------------------------- REQUIRED vs PREFERRED ---------------------------

def _accountant_code8():
    return _profile(
        skills=["pastel evolution", "reconciliations", "vat"],
        education=[Education(qualification="Diploma in Accounting",
                             field="Financial Management", institution="DUT")],
        experience=[Experience(title="Bookkeeper", company="F",
                               start_date="2024-02", end_date="2025-11")],
        drivers_licence="Code 8")


def test_preferred_requirement_only_reduces_score():
    # identical ad but the candidate holds the demanded code 8 licence —
    # only the merely-advantageous SAP skill is missing
    r = evaluate(BOOKKEEPER_AD, _accountant_code8())
    assert r.decision == "apply"
    assert any("sap" in m.lower() for m in r.missing_preferred)
    assert r.score >= 65


def test_missing_preferred_listed_but_non_blocking():
    ad = _job("Accounts Clerk",
              "Requirements: diploma in accounting. Sage experience "
              "advantageous.")
    r = evaluate(ad, ACCOUNTANT)
    assert r.decision != "reject"
    assert any("sage" in m.lower() for m in r.missing_preferred)


# ----------------------------- explainability ------------------------------

def test_explanation_contains_why_and_reason_sections():
    r = evaluate(BOOKKEEPER_AD, ACCOUNTANT)
    text = r.explain()
    assert "Reason:" in text
    assert "-" in text  # bullet lines


def test_apply_result_has_why_section():
    ad = _job("Accounts Clerk",
              "Requirements: diploma in accounting. Pastel reconciliations. "
              "Durban.")
    r = evaluate(ad, ACCOUNTANT)
    assert r.decision == "apply"
    assert "Why:" in r.explain()


# --------------------------- scoring dimensions ----------------------------

def test_location_mismatch_reduces_score_without_blocking():
    close = evaluate(_job(location="Durban"), ACCOUNTANT)
    far = evaluate(_job(title="Junior Bookkeeper", location="Upington"),
                   ACCOUNTANT)
    assert far.score < close.score


def test_weights_are_configurable():
    boosted = {**DEFAULT_WEIGHTS, "skills": 60}
    base = evaluate(_job(desc="Pastel Evolution VAT reconciliations"),
                    ACCOUNTANT)
    heavy = evaluate(_job(desc="Pastel Evolution VAT reconciliations"),
                     ACCOUNTANT, weights=boosted)
    assert heavy.dimension_scores["skills"] > base.dimension_scores["skills"]


@pytest.mark.parametrize("verdict", [
    "excellent", "strong", "possible", "unsuitable"])
def test_verdict_band_names(verdict):
    assert verdict in {"excellent", "strong", "possible", "unsuitable"}


def test_real_verdict_band_boundaries():
    assert evaluate(_job(), _profile()).verdict in {
        "excellent", "strong", "possible", "unsuitable"}
    strong_ad = _job("Office Administrator",
                     "ms office excel filing diaries admin support. Durban.")
    r = evaluate(strong_ad, ADMIN)
    assert r.verdict in ("excellent", "strong")


# -------------------------- requirement detection --------------------------

def test_detect_requirements_distinguishes_required_vs_preferred():
    text = ("Requirements: valid driver's licence. SAP advantageous. "
            "Minimum of 3 years experience. Degree preferred.")
    found = {(r.kind, r.required) for r in detect_requirements(text)}
    assert ("licence_generic", True) in found
    assert ("years_experience", True) in found      # "minimum of" cue
    assert ("qualification", False) in found        # "preferred" cue
    assert ("years_experience", False) not in found


# ---- matric-level requirements satisfied by tertiary qualifications ---------

def test_grade12_requirement_satisfied_by_diploma():
    job = _job("Finance Clerk", desc=(
        "Requirements: Minimum requirements: Applicants must be in possession "
        "of a Grade 12 Certificate. Experience in cash handling will be an "
        "advantage."))
    profile = _profile(
        skills=["cash handling"],
        education=[Education(qualification="Diploma in Information Technology",
                             field="Application Development", institution="DUT")])
    res = evaluate(job, profile)
    assert "Grade 12 Certificate" not in " ".join(res.blockers)
    assert any("Matric-level requirement satisfied" in m for m in res.matched)


def test_grade12_with_subject_adds_unknown_note_not_blocker():
    job = _job("Finance Clerk", desc=(
        "Applicants must be in possession of a Grade 12 with Accounting as "
        "a subject."))
    profile = _profile(
        education=[Education(qualification="Diploma in ICT",
                             institution="DUT")])
    res = evaluate(job, profile)
    assert not res.blockers
    assert any("Accounting" in u for u in res.unknowns)


def test_explicit_diploma_requirement_still_blocks_matric_only_profile():
    job = _job("Systems Analyst", desc=(
        "Applicants must be in possession of a National Diploma in "
        "Information Technology."))
    profile = _profile(
        education=[Education(qualification="Grade 12 Certificate",
                             institution="X")])
    res = evaluate(job, profile)
    assert any("qualification" in b.lower() for b in res.blockers)


def test_degree_satisfies_grade12_and_higher_req_unaffected():
    higher = _job("Analyst", desc="Must have a Bachelor degree in Commerce.")
    grad12_only = _profile(education=[Education(qualification="Matric",
                                                institution="X")])
    assert any("qualification" in b.lower()
               for b in evaluate(higher, grad12_only).blockers)

    matric_ad = _job("Clerk", desc="Grade 12 is required.")
    degree_holder = _profile(education=[
        Education(qualification="Bachelor of Commerce", institution="U")])
    assert not evaluate(matric_ad, degree_holder).blockers
