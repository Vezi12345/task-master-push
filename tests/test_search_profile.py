from __future__ import annotations

"""Adaptive search-profile derivation tests — one candidate shape per
profession proves queries are generated from THE CANDIDATE, not from any
hard-coded occupation."""

import pytest

from candidate.profile import CandidateProfile, Education, Experience
from candidate.search_profile import (
    build_search_profile,
    derive_career_level,
    generate_queries,
)


def _profile(**kw) -> CandidateProfile:
    defaults = {"skills": [], "education": [], "experience": [],
                "certifications": [], "location": ""}
    defaults.update(kw)
    return CandidateProfile(name="T", email="t@x.com", **defaults)


def _edu(qual, field, inst="DUT", end="2024-11"):
    return Education(qualification=qual, field=field, institution=inst,
                     end_date=end)


def _exp(title, company="Co", start="2023-01", end="present"):
    return Experience(title=title, company=company, start_date=start,
                      end_date=end, description="")


NURSE = _profile(
    skills=["patient care", "sanc guidelines"],
    education=[_edu("Diploma in Nursing", "General Nursing")],
    experience=[_exp("Staff Nurse", "Netcare")],
    city="Durban")

ACCOUNTANT = _profile(
    skills=["pastel evolution", "vat reconciliations"],
    education=[_edu("Diploma in Accounting", "Financial Management")],
    experience=[_exp("Accounts Clerk", "Firm")],
    preferred_locations=["Durban"])

TEACHER = _profile(
    skills=["lesson planning", "caps curriculum"],
    education=[_edu("B Ed", "Education", end="2019-11")],
    experience=[_exp("Educator", "School", start="2020-01")],
    city="Pietermaritzburg")

ELECTRICIAN = _profile(
    skills=["coc testing", "single phase wiring"],
    education=[_edu("N6 Electrical", "Electrical Engineering")],
    experience=[_exp("Maintenance Electrician", "Works", start="2021-03")])

ADMIN = _profile(
    skills=["ms office", "diary management", "filing"],
    education=[_edu("Diploma in Business Management", "Business Management")],
    experience=[_exp("Office Administrator", "Company")])

SALES = _profile(
    skills=["crm", "cold calling"],
    education=[_edu("Diploma in Marketing Management", "Marketing")],
    experience=[_exp("Sales Representative", "Distributor")])

DEV = _profile(
    skills=["python", "sql", "git"],
    education=[_edu("National Diploma", "ICT Application Development")],
    experience=[])

GRADUATE = _profile(
    education=[_edu("Bachelor of Commerce", "Accounting", end="2025-11")])


@pytest.mark.parametrize("profile,expected,forbidden", [
    (NURSE, ["nurse"], ["software", "developer"]),
    (ACCOUNTANT, ["accountant", "bookkeeper"], ["software", "developer"]),
    (TEACHER, ["teacher", "educator"], ["software"]),
    (ELECTRICIAN, ["electrician"], ["software", "nurse"]),
    (ADMIN, ["administrator", "admin clerk", "receptionist"], ["software"]),
    (SALES, ["sales"], ["software", "nurse"]),
    (DEV, ["developer", "software engineer"], []),
])
def test_queries_are_candidate_specific(profile, expected, forbidden):
    sp = build_search_profile(profile)
    assert sp.inference == "evidence"
    queries = [q.lower() for q in generate_queries(sp)]
    joined = " | ".join(queries)
    for word in expected:
        assert word in joined, f"{word} missing from {queries}"
    for word in forbidden:
        assert word not in joined


def test_non_developer_never_receives_software_queries():
    sp = build_search_profile(NURSE)
    queries = [q.lower() for q in generate_queries(sp, max_queries=8)]
    assert not any("software" in q or "developer" in q or "python" in q
                   for q in queries)


def test_changing_profile_changes_queries_without_code_changes():
    dev_qs = generate_queries(build_search_profile(DEV))
    nurse_qs = generate_queries(build_search_profile(NURSE))
    assert dev_qs != nurse_qs
    assert any("developer" in q.lower() for q in dev_qs)
    assert any("nurse" in q.lower() for q in nurse_qs)


def test_location_is_appended_from_preferences_or_city():
    qs = generate_queries(build_search_profile(NURSE))       # city Durban
    assert any(q.lower().endswith("durban") for q in qs)
    qs2 = generate_queries(build_search_profile(ACCOUNTANT))  # pref Durban
    assert any(q.lower().endswith("durban") for q in qs2)


@pytest.mark.parametrize("profile,level", [
    (_profile(professional_summary="Student studying part time",
              education=[_edu("BCom", "Accounting", end="2027-11")]), "student"),
    (GRADUATE, "graduate"),
    (_profile(experience=[_exp("Clerk", start="2026-03")]), "entry-level"),
    (_profile(experience=[_exp("Senior Clerk", start="2024-01")]), "junior"),
    (_profile(experience=[_exp("Manager", start="2020-01")]), "mid-level"),
    (_profile(experience=[_exp("Director", start="2015-01")]), "senior"),
])
def test_career_level_derived_from_evidence_only(profile, level):
    assert derive_career_level(profile) == level


def test_level_prefixes_match_career_stage():
    grad_qs = generate_queries(build_search_profile(GRADUATE))
    assert any(q.lower().startswith("graduate ") for q in grad_qs)
    mid = _profile(skills=["excel"], experience=[_exp("Analyst",
                                                     start="2019-01")],
                   education=[_edu("BSc", "Statistics")])
    mid_qs = generate_queries(build_search_profile(mid))
    assert not any(q.lower().startswith(("junior ", "graduate "))
                   for q in mid_qs)


def test_weak_cv_falls_back_to_honest_generic_query():
    weak = _profile()
    sp = build_search_profile(weak)
    assert sp.inference == "minimal"
    queries = generate_queries(sp)
    assert queries == ["entry level"]


def test_qualification_only_candidate_gets_qualification_queries():
    only_qual = _profile(education=[_edu("Diploma in Human Resource "
                                         "Management", "Human Resources")])
    sp = build_search_profile(only_qual)
    # registry match or literal fallback — both must yield HR-flavoured queries
    assert sp.inference in ("evidence", "literal")
    joined = " ".join(generate_queries(sp)).lower()
    assert "human resource" in joined or " hr " in f" {joined} "


def test_adjacent_roles_included_for_transferability():
    sp = build_search_profile(ACCOUNTANT)
    adjacent = [t.lower() for t in sp.adjacent_titles]
    assert any("clerk" in a or "bookkeeper" in a or "procurement" in a
               for a in adjacent)
    qs = " | ".join(generate_queries(sp)).lower()
    # at least one non-direct alternative is searched
    assert "audit clerk" in qs or "cashier supervisor" in qs \
        or "stock controller" in qs


def test_query_cap_respected_and_deduplicated():
    sp = build_search_profile(DEV)
    qs = generate_queries(sp, max_queries=3)
    assert len(qs) <= 3
    lowered = [q.lower() for q in qs]
    assert len(lowered) == len(set(lowered))


def test_strategy_description_mentions_inferred_occupation():
    line = build_search_profile(NURSE)
    desc = __import__("candidate.search_profile",
                      fromlist=["describe_strategy"]).describe_strategy(line)
    assert "Nurse" in desc and "evidence" in desc
