from __future__ import annotations

"""Phase 4 — application form analysis on real-style markup."""

from application.form_analyzer import analyze_application_page, classify_question_label


LEVER_STYLE = """
<html><head><title>Apply for this job</title></head><body>
<h2>Apply for Graduate Software Developer at DVT</h2>
<form id="application-form" action="/apply" method="post">
  <div class="field">
    <label for="name">Full name <span class="req">*</span></label>
    <input id="name" name="name" type="text" required/>
  </div>
  <div class="field">
    <label for="email">Email *</label>
    <input id="email" name="email" type="text" class="required"/>
  </div>
  <div class="field">
    <label for="phone">Phone</label>
    <input id="phone" name="phone" type="tel"/>
  </div>
  <div class="field">
    <label for="city">City</label>
    <input id="city" name="city" type="text"/>
  </div>
  <div class="field">
    <label for="grad-year">Graduation year</label>
    <input id="grad-year" name="grad_year" type="date"/>
  </div>
  <div class="field">
    <label for="years-exp">How many years of software development experience do you have?</label>
    <textarea id="years-exp" name="years_experience"></textarea>
  </div>
  <div class="field">
    <label for="salary">What are your salary expectations?</label>
    <input id="salary" name="salary_expectation" type="text"/>
  </div>
  <fieldset>
    <legend>Equity information (optional)</legend>
    <label for="race">Race / population group</label>
    <select id="race" name="race">
      <option value="">Select…</option>
      <option value="black_african">Black African</option>
      <option value="coloured">Coloured</option>
      <option value="indian">Indian</option>
      <option value="white">White</option>
      <option value="none">Prefer not to say</option>
    </select>
  </fieldset>
  <div class="field">
    <label>Do you have a valid driver's licence?</label>
    <input type="radio" name="licence" value="yes"/> Yes
    <input type="radio" name="licence" value="no"/> No
  </div>
  <div class="field">
    <label for="cv">Upload your CV</label>
    <input id="cv" name="cv" type="file" required/>
  </div>
  <div class="field">
    <label for="cover">Cover letter (optional)</label>
    <input id="cover" name="cover_letter" type="file"/>
  </div>
  <div class="field">
    <label for="consent">I consent to my data being processed for recruitment purposes</label>
    <input id="consent" name="consent" type="checkbox" required/>
  </div>
  <div class="field">
    <label for="karaoke">What is your favourite karaoke song?</label>
    <input id="karaoke" name="karaoke_song" type="text"/>
  </div>
  <button type="submit" class="apply-button">Submit application</button>
</form>
</body></html>
"""


def _fields_by_name(analysis):
    return {f.name: f for f in analysis.fields}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_form_discovered_with_all_field_types():
    analysis = analyze_application_page(LEVER_STYLE, page_url="https://jobs.lever.co/dvt/123", platform="lever")
    assert analysis.has_form
    types = {f.name: f.field_type for f in analysis.fields}
    assert types["name"] == "text"
    assert types["email"] == "email"
    assert types["phone"] == "tel"
    assert types["grad_year"] == "date"
    assert types["years_experience"] == "textarea"
    assert types["race"] == "select"
    assert types["licence"] == "radio"
    assert types["cv"] == "file"
    assert types["cover_letter"] == "file"
    assert types["consent"] == "checkbox"


def test_dropdown_options_extracted():
    analysis = analyze_application_page(LEVER_STYLE)
    race = _fields_by_name(analysis)["race"]
    assert "Black African" in race.options
    assert "Prefer not to say" in race.options


def test_radio_group_collapsed_with_options():
    analysis = analyze_application_page(LEVER_STYLE)
    licence = _fields_by_name(analysis)["licence"]
    assert licence.field_type == "radio"
    assert set(licence.options) == {"yes", "no"}
    # only one logical field for the group
    radios = [f for f in analysis.fields if f.name == "licence"]
    assert len(radios) == 1


def test_submit_button_found():
    analysis = analyze_application_page(LEVER_STYLE)
    assert analysis.submit_button is not None
    assert analysis.submit_button.text == "Submit application"
    assert analysis.submit_button.selector


# ---------------------------------------------------------------------------
# required vs optional
# ---------------------------------------------------------------------------

def test_required_fields_detected():
    analysis = analyze_application_page(LEVER_STYLE)
    by_name = _fields_by_name(analysis)
    assert by_name["name"].required is True          # html5 required attr
    assert by_name["email"].required is True         # label asterisk + class
    assert by_name["cv"].required is True            # file upload required
    assert by_name["consent"].required is True       # consent checkbox


def test_optional_fields_detected():
    analysis = analyze_application_page(LEVER_STYLE)
    by_name = _fields_by_name(analysis)
    for optional in ("phone", "city", "grad_year", "years_experience",
                     "salary_expectation", "race", "licence", "cover_letter",
                     "karaoke_song"):
        assert by_name[optional].required is False, optional
    assert len(analysis.optional_fields) >= 9


# ---------------------------------------------------------------------------
# semantic categories route questions to the right evidence source
# ---------------------------------------------------------------------------

def test_categories_classified_for_evidence_routing():
    analysis = analyze_application_page(LEVER_STYLE)
    by_name = _fields_by_name(analysis)
    assert by_name["email"].category == "contact"
    assert by_name["phone"].category == "contact"
    assert by_name["name"].category == "identity"
    assert by_name["salary_expectation"].category == "salary"
    assert by_name["years_experience"].category == "experience"
    assert by_name["race"].category == "demographic"
    assert by_name["cv"].category == "documents"
    assert by_name["cover_letter"].category == "documents"
    assert by_name["consent"].is_consent is True


def test_unseen_question_discovered_not_hard_coded():
    """A question the system has never seen must still be discovered and
    preserved verbatim as 'other' — ready for the answer engine."""
    analysis = analyze_application_page(LEVER_STYLE)
    karaoke = _fields_by_name(analysis)["karaoke_song"]
    assert karaoke.question == "What is your favourite karaoke song?"
    assert karaoke.category == "other"
    assert karaoke.display_question.startswith("What is your favourite")


def test_demographic_flag_set_for_equity_questions():
    analysis = analyze_application_page(LEVER_STYLE)
    race = _fields_by_name(analysis)["race"]
    assert race.is_demographic is True


def test_consent_and_terms_distinguished():
    assert classify_question_label("I agree to the terms and conditions") == "terms"
    assert classify_question_label("I consent to processing of my data") == "consent"


# ---------------------------------------------------------------------------
# gates and empty pages
# ---------------------------------------------------------------------------

def test_cloudflare_gated_page_stops_analysis():
    html = "<html><body>Checking your browser before accessing jobs.example.com.</body></html>"
    analysis = analyze_application_page(html, page_url="https://jobs.example.com/apply")
    assert analysis.challenge is not None
    assert analysis.challenge.kind == "cloudflare"
    assert analysis.fields == []


def test_page_without_form_reports_honestly():
    html = "<html><body><h1>Careers at DVT</h1><p>Email us at jobs@dvt.co.za</p></body></html>"
    analysis = analyze_application_page(html, page_url="https://dvt.co.za/careers")
    assert analysis.has_form is False
    assert analysis.fields == []
    assert any("No application form" in n for n in analysis.notes)


def test_summary_counts():
    analysis = analyze_application_page(LEVER_STYLE, platform="lever")
    summary = analysis.summary()
    assert summary["platform"] == "lever"
    assert summary["field_count"] == len(analysis.fields)
    assert summary["required_count"] == 4
    assert summary["challenge"] is None
    assert "demographic" in summary["categories"]
