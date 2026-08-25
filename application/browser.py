from __future__ import annotations

"""Real-browser automation layer.

``PlaywrightDriver`` drives an actual Chromium instance against REAL
employer application pages. There is deliberately no simulated success:
if the browser cannot perform an action, the exception/failure propagates
and the caller must surface ``REQUIRES_USER_ACTION``.

Test doubles MUST subclass ``FakeBrowserDriver`` (``is_real = False``);
the submission layer refuses to record ``submission_mode="real"`` unless
``driver.is_real`` is true, so mocked runs can never be reported as real
submissions.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# challenges we must never bypass
# ---------------------------------------------------------------------------

_CHALLENGE_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("checking your browser before accessing", "cloudflare"),
    ("attention required! | cloudflare", "cloudflare"),
    ("cf-challenge", "cloudflare"),
    ("challenge-platform", "cloudflare"),
    ("just a moment...", "cloudflare"),
    ("g-recaptcha", "captcha"),
    ("recaptcha/api.js", "captcha"),
    ("hcaptcha.com", "captcha"),
    ("h-captcha", "captcha"),
    ("are you a robot", "captcha"),
    ("prove you are human", "captcha"),
    ("two-factor authentication", "mfa"),
    ("two-factor verification", "mfa"),
    ("enter the verification code", "mfa"),
    ("multi-factor authentication", "mfa"),
    ("verify your email address to continue", "email_verification"),
)

_LOGIN_WALL_SIGNATURES: tuple[str, ...] = (
    "sign in to continue",
    "log in to continue",
    "please sign in to apply",
)


@dataclass
class UserActionRequired:
    """A security/human gate the agent must never bypass."""

    kind: str  # cloudflare | captcha | mfa | email_verification | login_wall
    message: str

    def user_message(self) -> str:
        labels = {
            "cloudflare": "Cloudflare verification required. "
            "Please complete the verification in the browser.",
            "captcha": "CAPTCHA detected. Please complete the CAPTCHA in the browser.",
            "mfa": "Multi-factor authentication required. "
            "Please complete the sign-in in the browser.",
            "email_verification": "Email verification required. "
            "Please verify the email address in the browser.",
            "login_wall": "Employer login required. Please sign in manually.",
        }
        return labels.get(self.kind, self.message)


def detect_challenge(page_text: str, url: str = "") -> Optional[UserActionRequired]:
    hay = f"{url}\n{page_text}".lower()
    for signature, kind in _CHALLENGE_SIGNATURES:
        if signature in hay:
            return UserActionRequired(kind=kind, message=f"Matched signature: {signature!r}")
    for signature in _LOGIN_WALL_SIGNATURES:
        if signature in hay:
            return UserActionRequired(
                kind="login_wall", message=f"Matched signature: {signature!r}"
            )
    return None


# ---------------------------------------------------------------------------
# driver interface
# ---------------------------------------------------------------------------

@dataclass
class PageSnapshot:
    url: str
    title: str
    status_ok: bool = True
    error: str = ""


class BrowserError(Exception):
    """The real browser could not perform the requested action."""


class BrowserUnavailable(BrowserError):
    """Browser automation prerequisites are missing."""


class BrowserDriver(ABC):
    """Interface used by the form analyzer / filler / submitter."""

    is_real: bool = False

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def goto(self, url: str) -> PageSnapshot: ...

    @abstractmethod
    def current_url(self) -> str: ...

    @abstractmethod
    def page_html(self) -> str: ...

    @abstractmethod
    def page_text(self) -> str: ...

    @abstractmethod
    def fill(self, css_selector: str, value: str) -> None: ...

    @abstractmethod
    def select_option(self, css_selector: str, value: str) -> None: ...

    @abstractmethod
    def set_checkbox(self, css_selector: str, checked: bool) -> None: ...

    @abstractmethod
    def upload_file(self, css_selector: str, file_path: str) -> None: ...

    @abstractmethod
    def click(self, css_selector: str) -> None: ...

    @abstractmethod
    def scroll_to(self, css_selector: str) -> None: ...

    @abstractmethod
    def validation_errors(self) -> list[str]: ...

    def evaluate(self, script: str):
        """Read-only JS evaluation hook (diagnostics/verification).
        Real drivers override this; test doubles may ignore it."""
        raise BrowserError("This driver does not support JS evaluation")

    def check_for_challenge(self) -> Optional[UserActionRequired]:
        return detect_challenge(self.page_text(), self.current_url())

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# real driver
# ---------------------------------------------------------------------------

class PlaywrightDriver(BrowserDriver):
    """Drives a real Chromium browser via Playwright (sync API)."""

    is_real = True

    def __init__(self, headless: bool = True, timeout_ms: int = 30000) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._pw = None
        self._browser = None
        self._page = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._page is not None:
            return  # already started
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserUnavailable(
                "Playwright is not installed. Run: pip install playwright "
                "and then: playwright install chromium"
            ) from exc
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page()
        self._page.set_default_timeout(self._timeout_ms)

    def _require_page(self):
        if self._page is None:
            raise BrowserError("Browser not started — call start() first")
        return self._page

    def close(self) -> None:
        try:
            if self._page is not None:
                self._page.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        if self._pw is not None:
            # sync_playwright().start() returns a Playwright object whose
            # teardown method is stop() — close() does not exist and leaving
            # it running poisons the event loop for future drivers.
            try:
                self._pw.stop()
            except Exception:
                pass
        self._page = None
        self._browser = None
        self._pw = None

    # -- navigation --------------------------------------------------------
    def goto(self, url: str) -> PageSnapshot:
        page = self._require_page()
        response = page.goto(url, wait_until="domcontentloaded")
        ok = response is not None and response.ok
        return PageSnapshot(
            url=page.url,
            title=page.title(),
            status_ok=ok,
            error="" if ok else f"HTTP {response.status if response else 'no response'}",
        )

    def current_url(self) -> str:
        return self._require_page().url

    def page_html(self) -> str:
        return self._require_page().content()

    def page_text(self) -> str:
        return self._require_page().inner_text("body")

    # -- interaction -------------------------------------------------------
    def fill(self, css_selector: str, value: str) -> None:
        page = self._require_page()
        locator = page.locator(css_selector).first
        if locator.count() == 0:
            raise BrowserError(f"Element not found: {css_selector}")
        locator.fill(value)

    def select_option(self, css_selector: str, value: str) -> None:
        page = self._require_page()
        locator = page.locator(css_selector).first
        if locator.count() == 0:
            raise BrowserError(f"Element not found: {css_selector}")
        locator.select_option(value)

    def set_checkbox(self, css_selector: str, checked: bool) -> None:
        page = self._require_page()
        locator = page.locator(css_selector).first
        if locator.count() == 0:
            raise BrowserError(f"Element not found: {css_selector}")
        locator.set_checked(checked)

    def upload_file(self, css_selector: str, file_path: str) -> None:
        page = self._require_page()
        locator = page.locator(css_selector).first
        if locator.count() == 0:
            raise BrowserError(f"Element not found: {css_selector}")
        locator.set_input_files(file_path)

    def click(self, css_selector: str) -> None:
        page = self._require_page()
        locator = page.locator(css_selector).first
        if locator.count() == 0:
            raise BrowserError(f"Element not found: {css_selector}")
        locator.click()

    def scroll_to(self, css_selector: str) -> None:
        page = self._require_page()
        locator = page.locator(css_selector).first
        if locator.count() == 0:
            raise BrowserError(f"Element not found: {css_selector}")
        locator.scroll_into_view_if_needed()

    def validation_errors(self) -> list[str]:
        page = self._require_page()
        js = """
        () => {
          const out = [];
          document.querySelectorAll(
            '.error, .field-error, .invalid-feedback, [aria-invalid="true"], .alert-danger'
          ).forEach(el => { const t = el.innerText && el.innerText.trim();
                            if (t) out.push(t); });
          document.querySelectorAll(':invalid').forEach(el => {
            if (el.validationMessage) out.push(el.validationMessage);
          });
          return out;
        }
        """
        try:
            return [str(t) for t in page.evaluate(js) or []]
        except Exception:
            return []

    def evaluate(self, script: str):
        return self._require_page().evaluate(script)


def open_driver(prefer_headless: bool = True) -> BrowserDriver:
    """Factory used by the submission service."""
    return PlaywrightDriver(headless=prefer_headless)
