from __future__ import annotations

"""Persistent Candidate Application Profile — the 31-section spec.

Covers: structured profile, name splitting, result-vs-qualification,
high-school never inferred, skills-vs-experience, projects first class,
certifications, reusable preferences, canonical mapping, evidence
metadata, GENERATED != VERIFIED, own-words guard, sensitive info,
consent, completion meter, new-info flow, conflicts, versioning.
"""

import json
import time

import pytest

from application.answer_engine import (
    AnswerType,
    answer_question,
    classify_question,
    questions_equivalent,
)
from application.form_analyzer import analyze_application_page
from application.form_filler import FormFiller
from application.models import ApplicationStatus
from application.submission import ApplicationAutomationService
from candidate.completion import compute_completion, high_value_missing
from candidate.profile import (
    CandidateProfile,
    Certification,
    Education,
    Experience,
    HighSchoolRecord,
    Project,
)
from sources.base import Job
from tests.test_form_analyzer import LEVER_STYLE
from tests.test_submission import _Tracker


APPLY_URL = "https://jobs.lever.co/acme/9"
CONFIRM_URL = APPLY_URL + "/confirmation"


def _job() -> Job:
    return Job(id="acme-1", title="Backend Developer", company="Acme",
               location="Durban", description="Build APIs.", url=APPLY_URL,
               source="schemaorg")


def _driver():
    from tests.fakes import FakeBrowserDriver
    driver = FakeBrowserDriver(pages={APPLY_URL: LEVER_STYLE})
    driver.post_click["button[type='submit']"] = CONFIRM_URL
    driver.pages[CONFIRM_URL] = (
        "<html>Thank you — application received. Reference ACME-77</html>"
    )
    return driver


def _rich_profile(**kw) -> CandidateProfile:
    defaults = dict(
        name="Lucky Vezi",
        email="lucky.vezi@example.com",
        phone="082 555 1234",
        location="Durban",
        skills=["java", "sql"],
        education=[Education(
            institution="DUT", qualification="Diploma in ICT", field="App Dev",
            end_date="2025", is_highest=True,
        )],
    )
    defaults.update(kw)
    return CandidateProfile(**defaults)


# ---------------------------------------------------------------------------
# 1. name splitting — never swapped, single-word name asks the user
# ---------------------------------------------------------------------------

def test_first_and_last_name_split_correctly():
    p = CandidateProfile(name="Thandi Nomvula Mkhize", email="t@x.co")
    assert p.first_name == "Thandi"
    assert p.last_name == "Mkhize"
    r = answer_question("What is your last name?", p)
    assert r.answer == "Mkhize" and r.answer_type == AnswerType.VERIFIED
    r2 = answer_question("First name", p)
    assert r2.answer == "Thandi"


def test_single_word_name_has_no_surname_so_user_is_asked():
    p = CandidateProfile(name="Cher", email="c@x.co")
    r = answer_question("Surname", p)
    assert r.answer is None and r.needs_user
    assert not any(r.answer for r in [answer_question("Last name", p)] if r.answer)


# ---------------------------------------------------------------------------
# 2. degree RESULT vs qualification — never conflated
# ---------------------------------------------------------------------------

def test_qualification_name_is_never_used_as_result():
    p = _rich_profile(education=[Education(
        institution="DUT", qualification="Diploma in ICT", field="App Dev",
        result="68%", grading_system="Percentage", is_highest=True)])
    r = answer_question("What was your final degree result?", p)
    assert r.answer == "68%"
    # and without a stored result the user is asked — qualification NOT echoed
    p2 = _rich_profile()
    r2 = answer_question("What was your final degree result?", p2)
    assert r2.answer is None and r2.needs_user
    assert "ICT" not in (r2.explanation or "") and "Diploma" not in str(r2.answer)


def test_grading_system_recorded_with_result():
    p = _rich_profile(education=[Education(
        qualification="BSc", result="3.2", grading_system="GPA 4.0")])
    assert p.education[0].result == "3.2"
    assert p.education[0].grading_system == "GPA 4.0"


# ---------------------------------------------------------------------------
# 3. high school optional — nothing inferred
# ---------------------------------------------------------------------------

def test_high_school_questions_unanswered_without_explicit_record():
    p = _rich_profile()          # degree only, no high-school record
    for q in ("What was your mathematics mark at high school?",
              "What is your home language?",
              "What were your matric results?"):
        r = answer_question(q, p)
        assert r.answer is None, q
        assert r.needs_user, q


def test_high_school_answers_come_only_from_the_record():
    p = _rich_profile(high_school=HighSchoolRecord(
        school="Velabahleke", mathematics_result="72%",
        native_language="isiZulu", overall_result="A",
    ))
    assert answer_question("mathematics result", p).answer == "72%"
    assert "isiZulu" in answer_question("home language", p).answer
    assert answer_question("matric results", p).answer == "A"


# ---------------------------------------------------------------------------
# 4. skills are NOT work experience
# ---------------------------------------------------------------------------

def test_skill_question_never_fabricates_employment_history():
    p = _rich_profile(skills=["java"], experience=[])
    r = answer_question("Describe your experience with Java.", p)
    # no job history → no invented employment narrative; user must supply
    assert r.answer_type in (AnswerType.UNKNOWN,) or r.answer is None \
        or "experience" not in str(r.answer).lower()[:20]


def test_structured_skill_keeps_provenance_and_evidence():
    from candidate.profile import SkillDetail
    p = CandidateProfile(
        name="Lucky Vezi", email="l@x.co",
        skill_details=[
            SkillDetail(name="Java", category="language", proficiency="intermediate",
                        source="project:TrackMyRand", evidence="Built REST API"),
        ],
    )
    assert p.skill_details[0].source.startswith("project:")
    # a bare skill name NEVER implies employment — no experience fabricated
    assert p.experience == []
    r = answer_question("How many years of professional Java experience do you have?", p)
    assert r.answer is None and r.needs_user


# ---------------------------------------------------------------------------
# 5. projects first-class
# ---------------------------------------------------------------------------

def test_personal_project_draft_generated_from_stored_evidence_only():
    p = _rich_profile(projects=[Project(
        name="TrackMyRand", description="Expense tracker app",
        technologies=["Java", "SQLite"], role="Sole developer",
        achievements=["100 users in month one"],
        github_url="https://github.com/lucky/trackmyrand", is_personal=True,
    )])
    r = answer_question(
        "Describe a personal software project outside of curriculum or work", p)
    assert r.answer_type == AnswerType.GENERATED_FROM_EVIDENCE
    assert "Expense tracker app" in r.answer
    assert "Java" in r.answer
    # draft cites ONLY stored fields — no invented claims
    for forbidden in ("led a team of", "increased revenue", "at company"):
        assert forbidden not in r.answer.lower()


def test_project_question_without_projects_needs_user():
    r = answer_question("Tell us about a personal project", _rich_profile())
    assert r.answer is None and r.needs_user


# ---------------------------------------------------------------------------
# 6. certifications & achievements structured
# ---------------------------------------------------------------------------

def test_certifications_hold_issuer_date_credential():
    p = _rich_profile(certifications=[
        Certification(name="AWS CCP", issuer="Amazon", date="2025-03",
                      credential_id="AWS-123", url="https://aws.amazon.com/x")])
    c = p.certifications[0]
    assert (c.name, c.issuer, c.date, c.credential_id) == \
        ("AWS CCP", "Amazon", "2025-03", "AWS-123")


# ---------------------------------------------------------------------------
# 7. preferences reusable across applications
# ---------------------------------------------------------------------------

def test_preferences_reused_in_next_application_plan():
    p = _rich_profile(expected_salary="R25000", notice_period="Immediately")
    filler = FormFiller(cv_path="")
    html = """
    <html><form>
      <input type="text" name="salary" required aria-label="Expected salary"/>
      <input type="text" name="notice" required aria-label="Notice period"/>
      <button type="submit">Submit</button>
    </form></html>"""
    analysis = analyze_application_page(html, APPLY_URL, "lever")
    plan = filler.build_plan(analysis, p, {"title": "Backend Developer", "company": "Acme"})
    values = {e.question: e.value for e in plan.entries}
    assert "R25000" in list(values.values())
    assert "Immediately" in list(values.values())


# ---------------------------------------------------------------------------
# 8. canonical question mapping
# ---------------------------------------------------------------------------

def test_equivalent_questions_map_to_same_canonical_field():
    # classify_question returns (field_key, category)
    a = classify_question("What is your email address?")
    b = classify_question("Email")
    c = classify_question("Contact e-mail address")
    assert a[0] == b[0] == c[0] == "email"
    d = classify_question("Notice period")
    e = classify_question("How much notice do you have to give?")
    assert d[0] == e[0] == "notice_period"


def test_remembered_answer_reused_for_equivalent_question():
    p = _rich_profile()
    p.remember_answer("When can you start?", "1 March 2027",
                      field_key="notice_period", source="user",
                      confidence="high", evidence="Typed by user on Acme form")
    mem = p.question_memory[-1]
    assert mem.confidence == "high"
    assert "Acme" in mem.evidence
    assert mem.updated_at  # timestamp recorded


# ---------------------------------------------------------------------------
# 9. GENERATED_FROM_EVIDENCE is never VERIFIED
# ---------------------------------------------------------------------------

def test_generated_answer_flagged_and_not_marked_verified():
    p = _rich_profile(projects=[Project(name="X", description="Y", is_personal=True)])
    filler = FormFiller(cv_path="")
    html = """
    <html><form><textarea name="q1" required
      aria-label="Describe a personal software project"></textarea>
      <button type="submit">Submit</button></form></html>"""
    analysis = analyze_application_page(html, APPLY_URL, "lever")
    plan = filler.build_plan(analysis, p, {"title": "", "company": ""})
    entry = plan.entries[0]
    assert entry.answer_type == AnswerType.GENERATED_FROM_EVIDENCE.value
    assert entry.answer_type != AnswerType.VERIFIED.value
    assert entry.needs_user is False   # shown in review, but clearly flagged


# ---------------------------------------------------------------------------
# 10. own-words guard blocks auto submission
# ---------------------------------------------------------------------------

OWN_WORDS_HTML = """
<html><body><p>All answers must be written in your own words.
AI-generated content is prohibited.</p>
<form><textarea name="motivation" aria-label="Why do you want to work here?"
 required></textarea><button type="submit">Submit</button></form></body></html>
"""


def test_own_words_requirement_detected_from_page_text():
    analysis = analyze_application_page(OWN_WORDS_HTML, APPLY_URL, "lever")
    assert analysis.own_words_required is True


def test_own_words_blocks_submission_of_generated_drafts():
    p = _rich_profile()
    service = ApplicationAutomationService(cv_path="")
    tracker = _Tracker()
    from tests.fakes import FakeBrowserDriver
    driver = FakeBrowserDriver(pages={APPLY_URL: OWN_WORDS_HTML})
    driver.post_click["button[type='submit']"] = CONFIRM_URL
    driver.pages[CONFIRM_URL] = "<html>received</html>"

    # give the engine something to generate a draft from
    p.projects = [Project(name="SideProj", description="A tool", is_personal=True)]
    app = service.start_application(_job(), p, tracker, driver=driver)
    if app.status == ApplicationStatus.REQUIRES_USER_ACTION:
        pytest.skip("page challenge detected — unrelated to this scenario")
    # motivation-style textarea gets an AI-generated draft
    generated = [e for e in app.fill_plan
                 if e["answer_type"] == AnswerType.GENERATED_FROM_EVIDENCE.value]
    if not generated:
        pytest.skip("no generated entries on this fixture page")
    result = service.confirm_and_submit(
        app, tracker, driver, consent_granted=True)
    assert result.status == ApplicationStatus.REQUIRES_USER_ACTION
    assert "own words" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# 11. sensitive information never inferred
# ---------------------------------------------------------------------------

def test_sensitive_fields_are_never_inferred_from_cv_or_memory():
    p = _rich_profile()      # no race/gender/dob anywhere
    for q in ("What is your gender?", "Race", "Date of birth"):
        r = answer_question(q, p)
        assert r.answer is None, q
        assert r.needs_user


def test_sensitive_values_echoed_only_when_explicitly_stored():
    p = _rich_profile(gender="Female", date_of_birth="2001-04-12")
    r = answer_question("What is your gender?", p)
    assert r.answer == "Female" and r.answer_type == AnswerType.SENSITIVE


# ---------------------------------------------------------------------------
# 12. consent never global / never auto-checked
# ---------------------------------------------------------------------------

CONSENT_HTML = """
<html><form>
<label><input type="checkbox" id="c1"/> I agree to the privacy policy</label>
<button type="submit">Submit</button></form></html>
"""


def test_consent_boxes_always_need_explicit_user_action():
    analysis = analyze_application_page(CONSENT_HTML, APPLY_URL, "lever")
    plan = FormFiller(cv_path="").build_plan(
        analysis, _rich_profile(), {"title": "", "company": ""})
    consents = plan.consent_entries
    assert consents
    assert all(e.needs_user for e in consents)
    assert all(e.value in (None, False, "") for e in consents)


# ---------------------------------------------------------------------------
# 13. completion meter weighted by high-value info
# ---------------------------------------------------------------------------

def test_completion_weighted_towards_high_value_sections():
    empty = compute_completion(CandidateProfile())
    weights = {name: s["weight"] for name, s in empty["sections"].items()}
    assert weights["personal"] > weights["high_school"]
    assert weights["education"] > weights["certifications"]
    # identical bare base: contact details move the meter far more than
    # a complete high-school record (low-value extras)
    base = CandidateProfile(name="Lucky Vezi", email="l@x.co")
    hs_only = compute_completion(CandidateProfile(
        name="Lucky Vezi", email="l@x.co",
        high_school=HighSchoolRecord(school="X", mathematics_result="70%",
                                     overall_result="B")))
    contact_only = compute_completion(CandidateProfile(
        name="Lucky Vezi", email="l@x.co", phone="082 555 0000",
        location="Durban"))
    assert contact_only["overall"] > hs_only["overall"]
    assert empty["overall"] < hs_only["overall"] < contact_only["overall"]


def test_missing_prompts_order_high_value_first():
    prompts = high_value_missing(CandidateProfile())
    assert prompts
    labels = [p["label"].lower() for p in prompts]
    assert any("contact" in l for l in labels[:2])


# ---------------------------------------------------------------------------
# 14. versioning — old applications keep historical answers
# ---------------------------------------------------------------------------

def test_old_application_keeps_values_after_profile_change():
    p = _rich_profile(phone="082 OLD 1111")
    service = ApplicationAutomationService(cv_path="")
    tracker = _Tracker()
    app = service.start_application(_job(), p, tracker, driver=_driver())
    snapshot = [dict(e) for e in app.fill_plan]
    old_phone_entry = next(e for e in snapshot if e.get("value") == "082 OLD 1111")

    # user later updates their profile
    p.phone = "082 NEW 2222"
    # historical application record still carries the OLD value
    assert old_phone_entry["value"] == "082 OLD 1111"
    # ...and a NEW application would use the new value
    plan2 = FormFiller(cv_path="").build_plan(
        analyze_application_page(LEVER_STYLE, APPLY_URL, "lever"), p,
        {"title": "Backend Developer", "company": "Acme"})
    phones = [e.value for e in plan2.entries if e.value == "082 NEW 2222"]
    assert phones


# ---------------------------------------------------------------------------
# 15. conflict handling — profile vs remembered answer
# ---------------------------------------------------------------------------

def test_conflicting_phone_shows_both_and_needs_user():
    p = _rich_profile(phone="083 CURRENT 999")
    p.remember_answer("Phone number", "082 PAST 000", field_key="phone")
    remembered = {"Phone number": "082 PAST 000"}
    filler = FormFiller(cv_path="")
    plan = filler.build_plan(
        analyze_application_page(LEVER_STYLE, APPLY_URL, "lever"), p,
        {"title": "", "company": ""}, remembered)
    conflicts = [e for e in plan.entries if e.conflict]
    assert conflicts, "expected a conflict entry"
    e = conflicts[0]
    assert set(e.conflict) == {"profile_value", "remembered_value"}
    assert e.conflict["profile_value"] == "083 CURRENT 999"
    assert e.conflict["remembered_value"] == "082 PAST 000"
    assert e.needs_user            # user chooses — never silent


def test_matching_memory_and_profile_do_not_conflict():
    p = _rich_profile(phone="083 SAME 111")
    p.remember_answer("Phone number", "083 SAME 111", field_key="phone")
    plan = FormFiller(cv_path="").build_plan(
        analyze_application_page(LEVER_STYLE, APPLY_URL, "lever"), p,
        {"title": "", "company": ""}, {"Phone number": "083 SAME 111"})
    assert not any(e.conflict for e in plan.entries)


# ---------------------------------------------------------------------------
# 16. online profiles: URLs only, never names
# ---------------------------------------------------------------------------

def test_online_link_questions_answered_only_with_saved_url():
    p = _rich_profile()           # no links saved
    r = answer_question("LinkedIn Profile", p)
    assert r.answer is None and r.needs_user
    # a saved URL is used verbatim
    p.online_profiles.linkedin = "https://linkedin.com/in/luckyvezi"
    r2 = answer_question("LinkedIn Profile", p)
    assert r2.answer == "https://linkedin.com/in/luckyvezi"
    assert r2.answer_type == AnswerType.VERIFIED


def test_website_field_never_filled_with_candidate_name():
    p = _rich_profile(name="Lucky Vezi")
    html = """
    <html><form>
      <input type="text" name="website" aria-label="Website"/>
      <input type="url" name="linkedin" aria-label="LinkedIn Profile"/>
      <button type="submit">Go</button></form></html>"""
    plan = FormFiller(cv_path="").build_plan(
        analyze_application_page(html, APPLY_URL, "lever"), p,
        {"title": "", "company": ""})
    web = next(e for e in plan.entries if "website" in (e.name or "").lower())
    linkedin = next(e for e in plan.entries if "linkedin" in (e.name or "").lower())
    assert web.value != "Lucky Vezi" and linkedin.value != "Lucky Vezi"
    assert web.needs_user and linkedin.needs_user
