from __future__ import annotations

"""Phase 5 — form filling through the evidence-grounded answer engine.

Uses the Phase-4 analyzer output plus a real candidate profile. UNKNOWN
answers must become user actions, consent must never be auto-checked, and
demographic questions must never be inferred.
"""

from application.form_analyzer import analyze_application_page
from application.form_filler import FormFiller
from candidate.profile import CandidateProfile, Education
from tests.test_form_analyzer import LEVER_STYLE


def _profile(**kw) -> CandidateProfile:
    defaults = dict(
        name="Lucky Vezi",
        email="lucky.vezi@example.com",
        phone="082 555 1234",
        location="Durban",
        skills=["java", "c#", "sql"],
        education=[Education(
            qualification="Diploma in ICT",
            field="Application Development",
            end_date="2025-11-01",
        )],
    )
    defaults.update(kw)
    return CandidateProfile(**defaults)


def _analysis():
    return analyze_application_page(
        LEVER_STYLE, page_url="https://jobs.lever.co/dvt/123", platform="lever"
    )


def _job_context() -> dict:
    return {
        "title": "Graduate Software Developer",
        "company": "DVT",
        "description": "Graduate development programme.",
        "requirements": "",
    }


def _plan(profile=None):
    filler = FormFiller(cv_path="/tmp/candidate_cv.pdf")
    return filler.build_plan(_analysis(), profile or _profile(), _job_context())


def _entry(plan, name):
    for e in plan.entries:
        if e.name == name or e.selector == f"#{name}":
            return e
    raise AssertionError(f"no entry for {name}")


# ---------------------------------------------------------------------------
# verified auto-filling
# ---------------------------------------------------------------------------

def test_identity_and_contact_fields_auto_filled_from_profile():
    plan = _plan()
    assert _entry(plan, "name").value == "Lucky Vezi"
    assert _entry(plan, "email").value == "lucky.vezi@example.com"
    assert _entry(plan, "phone").value == "082 555 1234"
    assert _entry(plan, "city").value == "Durban"
    for name in ("name", "email", "phone"):
        e = _entry(plan, name)
        assert not e.needs_user, name


def test_every_answer_carries_provenance():
    plan = _plan()
    for entry in plan.entries:
        if entry.value is not None:
            assert entry.answer_type in (
                "verified", "derived", "generated_from_evidence"
            ), f"{entry.question}: {entry.answer_type}"


def test_unknown_question_becomes_user_action_not_invention():
    """Driver's licence is asked on the form but not stored → the plan must
    demand user input rather than invent an answer."""
    plan = _plan()
    licence = _entry(plan, "licence")
    assert licence.needs_user is True
    assert licence.value is None
    assert "driver" in licence.question.lower()


def test_salary_question_not_invented_without_stored_data():
    profile = _profile()  # no expected_salary stored
    plan = _plan(profile)
    salary = _entry(plan, "salary_expectation")
    assert salary.needs_user is True
    assert salary.value is None


def test_demographic_question_never_auto_answered():
    plan = _plan()
    race = _entry(plan, "race")
    assert race.needs_user is True
    assert race.value is None
    assert race.is_demographic is True


def test_demographic_question_uses_only_explicitly_stored_data():
    profile = _profile(race="Black African")
    plan = _plan(profile)
    race = _entry(plan, "race")
    assert race.value == "Black African"
    # demographic echoes are marked SENSITIVE, never presented as verified
    assert race.answer_type == "sensitive"


# ---------------------------------------------------------------------------
# choice fields
# ---------------------------------------------------------------------------

def test_select_answer_must_match_offered_option():
    profile = _profile(race="Black African")
    plan = _plan(profile)
    race = _entry(plan, "race")
    assert race.value in ("Black African",)


def test_unmatchable_choice_answer_requires_user():
    profile = _profile(relocation="Only Cape Town")  # stored, but not an offered option
    filler = FormFiller(cv_path="/tmp/cv.pdf")
    analysis = analyze_application_page("""
    <form>
      <label for="reloc">Willing to relocate?</label>
      <select id="reloc" name="relocation" required>
        <option value="">Choose…</option>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    </form>
    """)
    plan = filler.build_plan(analysis, profile, _job_context())
    reloc = plan.entries[0]
    assert reloc.needs_user is True
    assert reloc.value is None
    assert "does not match any offered option" in reloc.reason


def test_matching_choice_option_selected():
    profile = _profile(relocation="Yes")
    filler = FormFiller(cv_path="/tmp/cv.pdf")
    analysis = analyze_application_page("""
    <form>
      <label for="reloc">Willing to relocate?</label>
      <select id="reloc" name="relocation" required>
        <option value="">Choose…</option>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    </form>
    """)
    plan = filler.build_plan(analysis, profile, _job_context())
    reloc = plan.entries[0]
    assert reloc.needs_user is False
    assert reloc.value == "Yes"


# ---------------------------------------------------------------------------
# documents + consent
# ---------------------------------------------------------------------------

def test_cv_upload_planned_with_real_file():
    plan = _plan()
    cv = _entry(plan, "cv")
    assert cv.upload_kind == "cv"
    assert cv.value == "/tmp/candidate_cv.pdf"
    assert cv.required is True
    assert not cv.needs_user


def test_cover_letter_upload_preferred_when_field_names_cover_letter():
    filler = FormFiller(cv_path="/tmp/cv.pdf", cover_letter_path="/tmp/cover.txt")
    plan = filler.build_plan(_analysis(), _profile(), _job_context())
    cover = _entry(plan, "cover_letter")
    assert cover.upload_kind == "cover_letter"
    assert cover.value == "/tmp/cover.txt"


def test_missing_cv_file_is_a_user_action():
    filler = FormFiller()  # no files at all
    plan = filler.build_plan(_analysis(), _profile(), _job_context())
    cv = _entry(plan, "cv")
    assert cv.needs_user is True
    assert cv.value is None


def test_consent_checkboxes_never_auto_checked():
    plan = _plan()
    consents = plan.consent_entries
    assert len(consents) == 1
    consent = consents[0]
    assert consent.is_consent is True
    assert consent.needs_user is True
    assert consent.value is None
    assert "explicit" in consent.reason.lower()


def test_plan_reports_unanswered_required_fields():
    plan = _plan()
    unanswered = plan.unanswered_required
    # required: name (filled), email (filled), cv (file ok), consent (user), licence? optional
    assert any("consent" in q.lower() for q in unanswered)
    assert not any("Full name" == q for q in unanswered)


def test_ready_to_fill_false_until_user_answers():
    plan = _plan()
    assert plan.ready_to_fill is False

    from application.form_filler import PlannedAnswer
    for entry in plan.entries:
        if entry.needs_user and not entry.is_consent:
            entry.value = entry.value or "User provided"
            entry.needs_user = False
    # consent still outstanding
    assert plan.ready_to_fill is False


def test_remembered_answers_reused_for_equivalent_questions(tmp_path):
    profile = _profile(drivers_licence="Yes — Code 8")
    remembered = {"Do you have a valid driver's licence?": "Yes — Code 8"}
    filler = FormFiller(cv_path="/tmp/cv.pdf")
    plan = filler.build_plan(_analysis(), profile, _job_context(), remembered)
    licence = _entry(plan, "licence")
    # the stored answer is matched to the actual offered option
    assert licence.value == "yes"
    assert licence.answer_type == "verified"


_GREENHOUSE_LINKS = """
<html><body><form>
  <label for="first_name">First Name *</label>
  <input id="first_name" name="first_name" type="text" required/>
  <label for="last_name">Last Name *</label>
  <input id="last_name" name="last_name" type="text" required/>
  <label for="website">Website</label>
  <input id="website" name="website" type="text"/>
  <label for="linkedin">LinkedIn Profile</label>
  <input id="linkedin" name="linkedin" type="text"/>
</form></body></html>
"""


def test_website_and_linkedin_fields_never_filled_with_name():
    """Real-world misfill on Greenhouse: Website/LinkedIn were tagged
    identity and received the candidate's full name."""
    analysis = analyze_application_page(_GREENHOUSE_LINKS, platform="greenhouse")
    filler = FormFiller(cv_path="")
    plan = filler.build_plan(analysis, _profile(), _job_context())
    by_name = {e.name: e for e in plan.entries}
    assert by_name["first_name"].value == "Lucky"
    assert by_name["last_name"].value == "Vezi"
    for link_field in ("website", "linkedin"):
        entry = by_name[link_field]
        assert entry.needs_user is True, link_field
        assert entry.value is None, link_field


_ANON_INPUTS = """
<html><body><form>
  <input type="hidden" name="csrf" value="x"/>
  <label for="first_name">First Name *</label>
  <input id="first_name" type="text" required/>
  <label>School</label>
  <input type="text"/>
  <label>Degree</label>
  <input type="text"/>
</form></body></html>
"""


def test_anonymous_inputs_get_unique_positional_selectors():
    """Greenhouse live bug: inputs without id/name all resolved to the
    first text input on the page and overwrote First Name."""
    analysis = analyze_application_page(_ANON_INPUTS, platform="greenhouse")
    # hidden csrf is skipped; first_name has an id (no name attr), the
    # School/Degree inputs have neither
    anon = [f for f in analysis.fields if f.name in ("", "first_name")]
    assert len(anon) == 3
    s1, s2 = anon[1].selector, anon[2].selector
    assert s1 != s2, "selectors must be unique"
    assert "nth=" in s1 and "nth=" in s2
    # filling in plan order targets two DIFFERENT controls
    filler = FormFiller()
    profile = _profile()
    plan = filler.build_plan(analysis, profile, _job_context())
    school = next(e for e in plan.entries if e.question == "School")
    degree = next(e for e in plan.entries if e.question == "Degree")
    assert school.selector != degree.selector
