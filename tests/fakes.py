from __future__ import annotations

"""Shared TEST DOUBLE browser driver.

``FakeBrowserDriver`` exists ONLY for unit tests. It reports
``is_real = False``, and the submission layer uses that flag to tag any
application record produced through it as ``submission_mode="mocked_test"``.
Mocked runs must never be reported to users as real submissions.
"""

from application.browser import BrowserDriver, PageSnapshot


class FakeBrowserDriver(BrowserDriver):
    """Scriptable in-memory browser used exclusively by tests."""

    is_real = False

    def __init__(self, pages: dict[str, str] | None = None) -> None:
        self.pages = pages or {}
        self.started = False
        self.closed = False
        self.url = ""
        self.history: list[str] = []
        self.actions: list[tuple] = []
        self.validation_error_text: list[str] = []
        self.fail_on: dict[str, Exception] = {}
        # selector -> url to navigate to after clicking (simulates submit)
        self.post_click: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    # -- navigation --------------------------------------------------------
    def goto(self, url: str) -> PageSnapshot:
        if "goto" in self.fail_on:
            raise self.fail_on["goto"]
        self.url = url
        self.history.append(url)
        html = self.pages.get(url)
        return PageSnapshot(
            url=url,
            title="Fake page",
            status_ok=html is not None,
            error="" if html is not None else f"No scripted page for {url}",
        )

    def current_url(self) -> str:
        return self.url

    def _page_for(self, url: str) -> str:
        html = self.pages.get(url)
        if html is None:
            raise AssertionError(f"FakeBrowserDriver has no scripted page for {url}")
        return html

    def page_html(self) -> str:
        return self._page_for(self.url)

    def page_text(self) -> str:
        import re

        return re.sub(r"<[^>]+>", " ", self.page_html())

    # -- interaction -------------------------------------------------------
    def fill(self, css_selector: str, value: str) -> None:
        if "fill" in self.fail_on:
            raise self.fail_on["fill"]
        self.actions.append(("fill", css_selector, value))

    def select_option(self, css_selector: str, value: str) -> None:
        self.actions.append(("select", css_selector, value))

    def set_checkbox(self, css_selector: str, checked: bool) -> None:
        self.actions.append(("checkbox", css_selector, checked))

    def upload_file(self, css_selector: str, file_path: str) -> None:
        if "upload" in self.fail_on:
            raise self.fail_on["upload"]
        self.actions.append(("upload", css_selector, file_path))

    def click(self, css_selector: str) -> None:
        if "click" in self.fail_on:
            raise self.fail_on["click"]
        self.actions.append(("click", css_selector))
        self.history.append(f"click:{css_selector}")
        target_url = self.post_click.get(css_selector)
        if target_url:
            self.url = target_url
            self.history.append(target_url)

    def scroll_to(self, css_selector: str) -> None:
        self.actions.append(("scroll", css_selector))

    def validation_errors(self) -> list[str]:
        return list(self.validation_error_text)
