from __future__ import annotations

import re
from typing import Any, Optional

from .base import BrowserAdapter, BrowserResult


class WorkdayBrowserAdapter(BrowserAdapter):
    name = "workday"

    def can_handle(self, url: str) -> bool:
        lowered = url.lower()
        return "workday" in lowered or "myworkdayjobs" in lowered

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
                await page.goto(url, wait_until="networkidle", timeout=45000)
                inputs = await page.query_selector_all(
                    '[data-automation-id], input, select, textarea'
                )
                for inp in inputs:
                    automation_id = await inp.get_attribute("data-automation-id") or ""
                    name = await inp.get_attribute("name") or automation_id
                    label_text = ""
                    aria_label = await inp.get_attribute("aria-label") or ""
                    if aria_label:
                        label_text = aria_label
                    else:
                        label_for = await inp.get_attribute("id")
                        if label_for:
                            label_el = await page.query_selector(f'label[for="{label_for}"]')
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
                            val = await opt.get_attribute("value") or ""
                            text = (await opt.inner_text()).strip()
                            if val or text:
                                options.append(text or val)
                    fields.append({
                        "name": name,
                        "label": label_text,
                        "field_type": field_type,
                        "required": required,
                        "placeholder": placeholder,
                        "options": options,
                        "automation_id": automation_id,
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
                await page.goto(url, wait_until="networkidle", timeout=45000)
                for field_name, value in fields.items():
                    if not value:
                        continue
                    selectors = [
                        f'[data-automation-id="{field_name}"]',
                        f'input[name="{field_name}"]',
                        f'#{field_name}',
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
                    if not filled:
                        label_el = await page.query_selector(f'label:has-text("{field_name}")')
                        if label_el:
                            inp = await label_el.query_selector("input, select, textarea")
                            if inp:
                                await inp.fill(value)
                if files:
                    for field_name, file_path in files.items():
                        file_input = await page.query_selector(
                            f'input[name="{field_name}"][type="file"], '
                            f'[data-automation-id="{field_name}"][type="file"]'
                        )
                        if file_input:
                            await file_input.set_input_files(file_path)
                submit_btn = await page.query_selector(
                    '[data-automation-id="bottomNavigationNextButton"], '
                    'button[data-automation-id="submit"], '
                    'button[type="submit"]'
                )
                if submit_btn:
                    await submit_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=20000)
                    current_url = page.url
                    return BrowserResult(
                        success=True,
                        application_url=current_url,
                    )
                return BrowserResult(
                    success=False,
                    error="No submit button found on Workday page",
                    requires_human_input=True,
                )
            except Exception as exc:
                return BrowserResult(
                    success=False,
                    error=f"Workday browser error: {exc}",
                    requires_human_input=True,
                )
            finally:
                await browser.close()

    async def check_status(self, url: str) -> str:
        return "unknown"
