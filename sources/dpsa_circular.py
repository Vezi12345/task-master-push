from __future__ import annotations

import io
import re

import pdfplumber
import requests

from .base import Job, JobSource, JobSourceError

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

HTTP_TIMEOUT = 20

POST_RE = re.compile(r"^POST\s+(?P<ref>\d{1,3}(?:/\d{1,3})+)(?![\d/])\s*(?::\s*)?(?P<title>[^:\s].*)$")

_LABEL_RE = re.compile(r"^([A-Z][A-Za-z0-9 ]*?)\s*:\s*(.*)$")

FIELD_LABELS = {"SALARY", "STIPEND", "CENTRE", "CENTER", "REQUIREMENTS", "DUTIES", "CLOSING DATE"}

NON_DEPARTMENT_HEADINGS = (
    "CIRCULAR",
    "VACANCY CIRCULAR",
    "IMPORTANT NOTES",
    "GENERAL NOTES",
    "INTRODUCTORY NOTES",
    "PURPOSE OF THIS CIRCULAR",
    "ABOUT THIS CIRCULAR",
    "MORE POSTS",
    "THE FOLLOWING",
    "INDEX",
    "LIST OF",
    "SECTION A",
    "SECTION B",
    "SECTION C",
    "NOTES",
    "PLEASE NOTE",
    "HOW TO APPLY",
    "ANNEXURE PAGES",
    "ANNEXURE A",
    "ANNEXURE B",
    "ANNEXURE C",
    "ANNEXURE D",
    "ANNEXURE E",
    "ANNEXURE F",
    "ANNEXURE G",
    "ANNEXURE H",
    "ANNEXURE I",
    "ANNEXURE J",
    "ANNEXURE K",
    "ANNEXURE L",
    "ANNEXURE M",
    "ANNEXURE N",
    "ANNEXURE O",
    "ANNEXURE P",
    "ANNEXURE Q",
    "ANNEXURE R",
    "ANNEXURE S",
    "ANNEXURE T",
    "ANNEXURE U",
    "ANNEXURE V",
    "ANNEXURE W",
    "ANNEXURE X",
    "ANNEXURE Y",
    "ANNEXURE Z",
    "NATIONAL DEPARTMENTS",
    "PROVINCIAL ADMINISTRATIONS",
    "MANAGEMENT ECHELON",
    "OTHER POST",
    "SENIOR MANAGEMENT SERVICE",
    "MEDIUM TERM CONTRACT",
    "ADVERTISED INTERNALLY",
    "ADVERTISED EXTERNALLY",
    "POSTS ADVERTISED",
    "INTERNALLY AND EXTERNALLY",
)


class DpsaCircularSource(JobSource):
    name = "dpsa_circular"

    def search(self, query) -> list[Job]:
        url = (self.config or {}).get("url")
        if not url:
            return []
        text = self.fetch_text(url)
        return parse_circular(
            text,
            source_url=url,
            default_company=(self.config or {}).get("default_company", ""),
        )

    def fetch_text(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise JobSourceError(f"could not download circular: {exc}") from exc
        try:
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as exc:
            raise JobSourceError(f"could not parse circular PDF: {exc}") from exc


def parse_circular(text: str, source_url: str = "", default_company: str = "") -> list[Job]:
    jobs: list[Job] = []
    seen: set[str] = set()
    current: dict = {}
    department: str = ""
    block_closing: str = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            jobs, seen = _flush(jobs, seen, current, source_url, default_company)
            current = {}
            continue

        post = POST_RE.match(line)
        if post:
            jobs, seen = _flush(jobs, seen, current, source_url, default_company)
            current = {
                "title": line,
                "reference": post.group("ref"),
                "department": department,
                "closing_date": block_closing,
            }
            continue

        labeled = _split_label(line)
        if labeled:
            label, value = labeled
            if current:
                current["last_field"] = label
                key = _field_key(label)
                if key:
                    current[key] = value
            elif label == "CLOSING DATE":
                block_closing = value
            continue

        if current and not current.get("last_field"):
            current["title"] = f"{current['title']} {line}".strip()
            continue

        if _is_department_heading(line):
            jobs, seen = _flush(jobs, seen, current, source_url, default_company)
            current = {}
            department = line
            block_closing = ""
            continue

        if current:
            _append_continuation(current, line)

    jobs, seen = _flush(jobs, seen, current, source_url, default_company)
    return jobs


def _split_label(line: str) -> tuple[str, str] | None:
    match = _LABEL_RE.match(line)
    if not match:
        return None
    return match.group(1).strip().upper(), match.group(2).strip()


def _field_key(label: str) -> str:
    if label.startswith("SALARY"):
        return "salary_text"
    return {
        "STIPEND": "salary_text",
        "CENTRE": "centre",
        "CENTER": "centre",
        "REQUIREMENTS": "requirements",
        "DUTIES": "duties",
        "CLOSING DATE": "closing_date",
    }.get(label, "")


def _is_department_heading(line: str) -> bool:
    upper = line.upper()
    if not line or ":" in line:
        return False
    if not line[0].isalpha():
        return False
    if "/" in line:
        return False
    if any(c.islower() for c in line):
        return False
    if any(upper.startswith(label) for label in FIELD_LABELS):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    if any(word in upper for word in NON_DEPARTMENT_HEADINGS):
        return False
    return True


def _append_continuation(current: dict, line: str) -> None:
    key = _field_key(current.get("last_field", ""))
    if not key:
        return
    previous = current.get(key, "")
    current[key] = f"{previous} {line}" if previous else line


def _flush(jobs: list[Job], seen: set[str], current: dict, source_url: str, default_company: str) -> tuple[list[Job], set[str]]:
    if not current:
        return jobs, seen
    job = _finalize(current, source_url, default_company)
    if job:
        return _append(jobs, seen, job)
    return jobs, seen


def _append(jobs: list[Job], seen: set[str], job: Job) -> tuple[list[Job], set[str]]:
    key = job.id or Job.make_id(job.title, job.company, job.url)
    if key in seen:
        return jobs, seen
    seen.add(key)
    jobs.append(job)
    return jobs, seen


def _finalize(entry: dict, source_url: str = "", default_company: str = "") -> Job | None:
    title = entry.get("title", "").strip()
    if not title:
        return None
    salary_text = entry.get("salary_text", "").strip() or None
    salary_min, salary_max = _monthly_salary(salary_text or "")
    requirements = entry.get("requirements", "").strip()
    duties = entry.get("duties", "").strip()
    description_bits = []
    if requirements:
        description_bits.append(f"Requirements: {requirements}")
    if duties:
        description_bits.append(f"Duties: {duties}")
    company = (entry.get("department") or default_company or "").strip()
    closing = entry.get("closing_date", "").strip()
    return Job(
        title=title,
        company=company,
        location=entry.get("centre", "").strip(),
        remote=False,
        description=" ".join(description_bits),
        salary_min=salary_min,
        salary_max=salary_max,
        salary_text=salary_text,
        url=source_url,
        source="dpsa_circular",
        posted_date=closing or None,
        id=(entry.get("reference") or "").strip(),
    )


def _monthly_salary(salary_text: str) -> tuple[int | None, int | None]:
    low: int | None = None
    high: int | None = None
    match = re.search(r"R\s?([\d\s]{4,})\s?[-\u2013]\s?R\s?([\d\s]{4,})", salary_text)
    if match:
        low = _to_number(match.group(1))
        high = _to_number(match.group(2))
    else:
        match = re.search(r"R\s?([\d\s]{4,})", salary_text)
        if match:
            low = _to_number(match.group(1))
    annual = any(word in salary_text.lower() for word in ["per annum", "p.a", "annual"])
    if low is not None and (annual or low > 150000):
        low = round(low / 12)
    if high is not None and (annual or high > 150000):
        high = round(high / 12)
    return low, high


def _to_number(raw: str) -> int | None:
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None
