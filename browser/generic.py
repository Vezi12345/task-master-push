from __future__ import annotations

import asyncio
from typing import Any, Optional

from .base import BrowserAdapter, BrowserResult


class GenericBrowserAdapter(BrowserAdapter):
    name = "generic"

    def can_handle(self, url: str) -> bool:
        return True

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
                inputs = await page.query_selector_all("input, select, textarea")
                for inp in inputs:
                    tag = await inp.evaluate("el => el.tagName.toLowerCase()")
                    name = await inp.get_attribute("name") or ""
                    label_text = ""
                    label_for = await inp.get_attribute("id")
                    if label_for:
                        label_el = await page.query_selector(f'label[for="{label_for}"]')
                        if label_el:
                            label_text = (await label_el.inner_text()).strip()
                    field_type = await inp.get_attribute("type") or tag
                    required = await inp.get_attribute("required") is not None
                    placeholder = await inp.get_attribute("placeholder") or ""
                    options: list[str] = []
                    if tag == "select":
                        option_els = await inp.query_selector_all("option")
                        for opt in option_els:
                            val = await opt.get_attribute("value") or ""
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
                error="Playwright is not installed. Install with: pip install playwright && playwright install",
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
                        f'input[name="{field_name}"]',
                        f'input[placeholder*="{field_name}"]',
                        f'#{field_name}',
                        f'[data-field="{field_name}"]',
                    ]
                    filled = False
                    for sel in selectors:
                        el = await page.query_selector(sel)
                        if el:
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
                        file_input = await page.query_selector(f'input[name="{field_name}"][type="file"]')
                        if file_input:
                            await file_input.set_input_files(file_path)
                submit_btn = await page.query_selector(
                    'button[type="submit"], input[type="submit"], button:has-text("Submit"), button:has-text("Apply")'
                )
                if submit_btn:
                    await submit_btn.click()
                    await page.wait_for_load_state("networkidle", timeout=15000)
                    current_url = page.url
                    return BrowserResult(
                        success=True,
                        application_url=current_url,
                    )
                return BrowserResult(
                    success=False,
                    error="No submit button found on page",
                    requires_human_input=True,
                )
            except Exception as exc:
                return BrowserResult(
                    success=False,
                    error=f"Browser error: {exc}",
                    requires_human_input=True,
                )
            finally:
                await browser.close()

    async def check_status(self, url: str) -> str:
        return "unknown"
