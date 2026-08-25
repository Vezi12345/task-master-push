"""LIVE browser automation tests — real Chromium, LOCAL-ONLY pages.

A throwaway HTTP server bound to 127.0.0.1 serves a static test form.
No employer site is contacted and NO submission is ever performed:
the submit button is never clicked on a page that could submit anywhere
(the validation test clicks it only while required fields are empty, so
the browser blocks submission locally).

Skipped automatically if Playwright/Chromium are unavailable.
"""
from __future__ import annotations

import http.server
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("playwright", reason="Playwright not installed")

from application.browser import PlaywrightDriver  # noqa: E402
from application.form_analyzer import analyze_application_page  # noqa: E402
from application.submission import detect_submission_confirmation  # noqa: E402

FORM_HTML = """<!DOCTYPE html>
<html><head><title>Local Test Application Form</title></head>
<body>
<h1>Graduate Developer Application</h1>
<form id="apply" action="/never-submitted" method="post">
  <label for="first_name">First Name</label>
  <input id="first_name" name="first_name" type="text" required/>

  <label for="email">Email</label>
  <input id="email" name="email" type="email" required/>

  <label for="country">Country</label>
  <select id="country" name="country" required>
    <option value="">Please select</option>
    <option value="ZA">South Africa</option>
    <option value="NA">Namibia</option>
  </select>

  <fieldset>
    <legend>Work authorisation</legend>
    <input type="radio" id="auth_yes" name="authorised" value="yes"/>
    <label for="auth_yes">Yes</label>
    <input type="radio" id="auth_no" name="authorised" value="no"/>
    <label for="auth_no">No</label>
  </fieldset>

  <input type="checkbox" id="consent" name="consent"/>
  <label for="consent">I agree to the terms and consent to processing</label>

  <input type="checkbox" id="newsletter" name="newsletter"/>
  <label for="newsletter">Send me job alerts</label>

  <label for="cv">Upload CV</label>
  <input type="file" id="cv" name="cv"/>

  <button type="submit" id="submit-btn">Submit Application</button>
</form>
</body></html>"""

CHALLENGE_HTML = """<!DOCTYPE html>
<html><head><title>Just a moment...</title></head>
<body>Checking your browser before accessing the site.</body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = CHALLENGE_HTML.encode() if self.path == "/challenge" else FORM_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 — must NEVER succeed
        self.send_response(403)
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


@pytest.fixture(scope="module")
def base_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture()
def driver():
    drv = PlaywrightDriver(headless=True)
    drv.start()
    yield drv
    drv.close()


# ----------------------------------------------------------------- tests ---

def test_browser_launches_and_navigates(driver, base_url):
    snapshot = driver.goto(f"{base_url}/form")
    assert snapshot.status_ok is True
    assert "Local Test Application Form" in snapshot.title
    assert driver.current_url().endswith("/form")


def test_form_fields_discovered_live(driver, base_url):
    driver.goto(f"{base_url}/form")
    analysis = analyze_application_page(
        driver.page_html(), page_url=driver.current_url(), platform=""
    )
    assert analysis.has_form is True
    by_name = {f.name: f for f in analysis.fields}
    assert "first_name" in by_name and by_name["first_name"].required
    assert "email" in by_name and by_name["email"].required
    assert by_name["country"].field_type == "select"
    assert "South Africa" in by_name["country"].options
    assert analysis.submit_button is not None


def test_text_field_can_be_filled(driver, base_url):
    driver.goto(f"{base_url}/form")
    driver.fill("#first_name", "Thabo")
    assert driver.evaluate("document.querySelector('#first_name').value") == "Thabo"


def test_select_radio_and_checkboxes_operable(driver, base_url):
    driver.goto(f"{base_url}/form")
    analysis = analyze_application_page(driver.page_html())
    by_name = {f.name: f for f in analysis.fields}
    assert by_name["country"].field_type == "select"
    assert by_name["authorised"].field_type == "radio"
    consent = [f for f in analysis.fields if f.name == "consent"]
    assert consent and consent[0].is_consent

    driver.select_option("#country", "ZA")
    driver.click("#auth_yes")
    # newsletter checkbox is NOT a consent box — safe to toggle in a test;
    # the consent checkbox is deliberately left untouched, as in production.
    driver.set_checkbox("#newsletter", True)

    assert driver.evaluate("document.querySelector('#country').value") == "ZA"
    assert driver.evaluate("document.querySelector('#auth_yes').checked") is True
    assert driver.evaluate("document.querySelector('#newsletter').checked") is True
    assert driver.evaluate("document.querySelector('#consent').checked") is False


def test_validation_errors_detected_then_cleared(driver, base_url):
    driver.goto(f"{base_url}/form")
    errors_before = driver.validation_errors()
    assert errors_before, "empty required fields should produce validation messages"

    driver.fill("#first_name", "Thabo")
    driver.fill("#email", "thabo@example.com")
    driver.select_option("#country", "ZA")
    assert driver.validation_errors() == []


def test_file_upload_prepared(driver, base_url, tmp_path):
    cv_file = tmp_path / "candidate_cv.txt"
    cv_file.write_text("Test CV content", encoding="utf-8")
    driver.goto(f"{base_url}/form")
    driver.upload_file("#cv", str(cv_file))
    count = driver.evaluate("document.querySelector('#cv').files.length")
    assert count == 1


def test_cloudflare_style_challenge_detected_live(driver, base_url):
    driver.goto(f"{base_url}/challenge")
    challenge = driver.check_for_challenge()
    assert challenge is not None
    assert challenge.kind == "cloudflare"
    assert driver.check_for_challenge() is not None  # stable detection


def test_submission_confirmation_detection_on_mock_data():
    # MOCK data only — no browser involved, no real submission exists.
    confirmed, text, reference = detect_submission_confirmation(
        "Thank you for applying! Your application has been received. "
        "Reference: DVT-12345",
        "https://careers.example.com/application-confirmation",
    )
    assert confirmed is True
    assert reference == "DVT-12345"
    not_confirmed, _, _ = detect_submission_confirmation(
        "Your application was saved as a draft.", "https://careers.example.com/form"
    )
    assert not_confirmed is False
