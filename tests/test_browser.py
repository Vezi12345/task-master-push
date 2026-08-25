from __future__ import annotations

"""Phase 3 — browser abstraction layer.

The real driver is Playwright; tests here use the clearly-marked
``FakeBrowserDriver`` test double (is_real=False) plus pure challenge
detection. No test pretends a fake browser is a real one.
"""

import pytest

from application.browser import (
    BrowserError,
    BrowserUnavailable,
    PlaywrightDriver,
    detect_challenge,
)
from tests.fakes import FakeBrowserDriver


# ---------------------------------------------------------------------------
# challenge detection (never bypassed)
# ---------------------------------------------------------------------------

def test_cloudflare_page_detected():
    challenge = detect_challenge(
        "Checking your browser before accessing careers.example.com.",
        url="https://careers.example.com/apply",
    )
    assert challenge is not None
    assert challenge.kind == "cloudflare"
    assert "Cloudflare" in challenge.user_message()


def test_recaptcha_detected():
    challenge = detect_challenge('<div class="g-recaptcha" data-sitekey="x"></div>')
    assert challenge is not None
    assert challenge.kind == "captcha"
    assert "CAPTCHA" in challenge.user_message()


def test_mfa_screen_detected():
    challenge = detect_challenge("Enter the verification code sent to your phone")
    assert challenge is not None
    assert challenge.kind == "mfa"


def test_login_wall_detected():
    challenge = detect_challenge("Please sign in to apply for this position")
    assert challenge is not None
    assert challenge.kind == "login_wall"
    assert "sign in" in challenge.user_message().lower()


def test_normal_application_page_has_no_challenge():
    html = """
    <html><body><h1>Apply — Graduate Developer</h1>
    <form><label>First name</label><input name="first_name"/></form>
    </body></html>
    """
    assert detect_challenge(html, url="https://jobs.lever.co/dvt/123") is None


# ---------------------------------------------------------------------------
# driver contract via the scripted test double
# ---------------------------------------------------------------------------

APPLY_PAGE = """
<html><body>
  <h1>Apply for this job</h1>
  <form id="application-form">
    <input name="name" id="id_name"/>
    <input name="email" id="id_email"/>
    <select name="country" id="id_country"><option value="ZA">South Africa</option></select>
    <input type="checkbox" name="consent" id="id_consent"/>
    <input type="file" name="cv" id="id_cv"/>
  </form>
  <button type="submit" id="submit-btn">Submit application</button>
</body></html>
"""


def _fake_driver() -> FakeBrowserDriver:
    driver = FakeBrowserDriver(pages={"https://jobs.lever.co/dvt/123": APPLY_PAGE})
    driver.start()
    driver.goto("https://jobs.lever.co/dvt/123")
    return driver


def test_fake_driver_is_explicitly_not_real():
    assert FakeBrowserDriver.is_real is False
    assert PlaywrightDriver.is_real is True


def test_navigation_and_interaction_contract():
    driver = _fake_driver()
    driver.fill("#id_name", "Lucky Vezi")
    driver.select_option("#id_country", "ZA")
    driver.set_checkbox("#id_consent", True)
    driver.upload_file("#id_cv", "cv.pdf")
    driver.click("#submit-btn")

    assert ("fill", "#id_name", "Lucky Vezi") in driver.actions
    assert ("select", "#id_country", "ZA") in driver.actions
    assert ("checkbox", "#id_consent", True) in driver.actions
    assert ("upload", "#id_cv", "cv.pdf") in driver.actions
    assert ("click", "#submit-btn") in driver.actions
    assert driver.history[0] == "https://jobs.lever.co/dvt/123"
    driver.close()
    assert driver.closed


def test_validation_errors_surfaced_by_driver():
    driver = _fake_driver()
    driver.validation_error_text = ["Email is required", "Please complete all required fields"]
    assert driver.validation_errors() == [
        "Email is required",
        "Please complete all required fields",
    ]


def test_driver_failure_raises_instead_of_faking_success():
    driver = _fake_driver()
    driver.fail_on["click"] = BrowserError("submit button disabled")
    with pytest.raises(BrowserError):
        driver.click("#submit-btn")


def test_challenge_check_uses_current_page():
    driver = FakeBrowserDriver(pages={
        "https://x.com/guard": "<html>Just a moment...</html>",
    })
    driver.start()
    driver.goto("https://x.com/guard")
    challenge = driver.check_for_challenge()
    assert challenge is not None
    assert challenge.kind == "cloudflare"


# ---------------------------------------------------------------------------
# real driver guards
# ---------------------------------------------------------------------------

def test_real_driver_requires_start_before_use():
    driver = PlaywrightDriver()
    with pytest.raises(BrowserError):
        driver.page_html()


def test_real_driver_reports_missing_playwright_honestly(monkeypatch):
    driver = PlaywrightDriver()

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("No module named 'playwright'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(BrowserUnavailable):
        driver.start()
