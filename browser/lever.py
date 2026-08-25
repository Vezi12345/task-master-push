from __future__ import annotations

from typing import Any, Optional

from .base import BrowserAdapter, BrowserResult


class LeverBrowserAdapter(BrowserAdapter):
    name = "lever"

    def can_handle(self, url: str) -> bool:
        lowered = url.lower()
        return "lever.co" in lowered or "jobs.lever" in lowered

    async def inspect_form(self, url: str) -> list[dict[str, Any]]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return []

        fields: list[dict[str, Any]] = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                inputs = await page.query_selector_all(
                    ".application-form input, .application-form select, "
                    ".application-form textarea, form input, form select, form textarea"
                )
                for inp in inputs:
                    name = await inp.get_attribute("name") or ""
                    label_text = ""
                    label_el = await page.query_selector(f'label[for="{await inp.get_attribute("id") or ""}"]')
                    if label_el:
                        label_text = (await label_el.inner_text()).strip()
                    tag = await inp.evaluate("el => el.tagName.toLowerCase()")
                    field_type = await inp.get_attribute("type") or tag
                    required = await inp.get_attribute("required") is not None
                    placeholder = await inp.get_attribute("placeholder") or ""
                    options: list[str] = []
                    if tag == "select":
                        option_els = await inp.query_selector_all("option")
                        for opt in option_els:
                            val = (await opt.inner_text()).strip()
                            if val:
                                options.append(val)
                    fields.append({
                        "name": name,
                        "label": label_text,
                        "field_type": field_type,
                        "required": required,
                        "placeholder": placeholder,
                        "options": options,
                    })
            except Exception:
                pass
            finally:
                await browser.close()
        return fields

    async def fill_and_submit(
        self,
        url: str,
        fields: dict[str, str],
        files: Optional[dict[str, str]] = None,
        headless: bool = True,
    ) -> BrowserResult:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return BrowserResult(
                success=False,
                error="Playwright not installed",
                requires_human_input=True,
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                for field_name, value in fields.items():
                    if not value:
                        continue
                    selectors = [
                        f'#{field_name}',
                        f'input[name="{field_name}"]',
                        f'select[name="{field_name}"]',
                        f'textarea[name="{field_name}"]',
                    ]
                    filled = False
                    for sel in selectors:
                        el = await page.query_selector(sel)
                        if el:
                            tag = await el.evaluate("el => el.tagName.toLowerCase()")
                            if tag == "select":
                                await el.select_option(label=value)
                            else:
                                await el.fill(value)
                            filled = True
                            break
                if files:
                    for field_name, file_path in files.items():
                        file_input = await page.query_selector(
                            f'#{field_name}[type="file"], input[name="{field_name}"][type="file"]'
                        )
                        if file_input:
                            await file_input.set_input_files(file_path)
                submit_btn = await page.query_selector(
                    'button[data-qa="btn-submit"], button[type="submit"]'
                )
                if submit_btn:
                    await submit_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    return BrowserResult(
                        success=True,
                        application_url=page.url,
                    )
                return BrowserResult(
                    success=False,
                    error="No submit button found on Lever page",
                    requires_human_input=True,
                )
            except Exception as exc:
                return BrowserResult(
                    success=False,
                    error=f"Lever browser error: {exc}",
                    requires_human_input=True,
                )
            finally:
                await browser.close()

    async def check_status(self, url: str) -> str:
        return "unknown"
