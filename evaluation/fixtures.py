from __future__ import annotations

"""Offline evaluation-only fixture jobs.

These records exist ONLY to give the offline ranking-quality harnesses
(``evaluation.runner`` / ``evaluation.national_runner``) a deterministic,
private-sector slice to measure against. They are NOT part of the live job
search pipeline: this module is not registered in ``agent.search``'s source
registry and is never served to users by ``search_jobs``.
"""

from sources.base import Job

FIXTURE_SOURCE = "demo"

_FIXTURE_JOBS = [
    {
        "title": "Junior Software Developer",
        "company": "Luno",
        "location": "Remote (South Africa)",
        "remote": True,
        "salary_text": "Not stated",
        "description": (
            "Build and maintain backend services using Python and SQL. "
            "We welcome recent computer science graduates. 0-2 years experience. "
            "Strong problem-solving and willingness to learn. Remote role accepting "
            "South African applicants."
        ),
        "url": "https://careers.luno.com/",
    },
    {
        "title": "Software Engineer (Graduate Programme)",
        "company": "Entersekt",
        "location": "Cape Town, Western Cape",
        "remote": False,
        "salary_text": "Not stated",
        "description": (
            "Two-year graduate programme for computer science graduates. "
            "Learn Java, security engineering and agile delivery. Hybrid work model, "
            "Cape Town based. On-the-job mentorship."
        ),
        "url": "https://www.entersekt.com/careers",
    },
    {
        "title": "Administration Clerk",
        "company": "KZN Department of Health",
        "location": "Durban, KwaZulu-Natal",
        "remote": False,
        "salary_min": 21300,
        "salary_max": 25300,
        "salary_text": "R21,300 - R25,300 per month",
        "description": (
            "Matric plus administrative duties, record keeping and general office "
            "support within the department. Government employment, Durban office."
        ),
        "url": "https://www.kznhealth.gov.za/vacancies",
    },
    {
        "title": "Junior Data Analyst",
        "company": "Takealot",
        "location": "Cape Town, Western Cape",
        "remote": False,
        "salary_min": 30000,
        "salary_max": 35000,
        "salary_text": "R30,000 - R35,000 per month",
        "description": (
            "Turn business data into insight using SQL, Python and Excel. "
            "Bachelor's degree in statistics, computer science or commerce preferred. "
            "Entry level role, Cape Town offices."
        ),
        "url": "https://www.takealot.com/careers",
    },
    {
        "title": "Senior Software Engineer",
        "company": "OfferZen Marketplace",
        "location": "Johannesburg, Gauteng",
        "remote": False,
        "salary_min": 70000,
        "salary_max": 90000,
        "salary_text": "R70,000 - R90,000 per month",
        "description": (
            "Lead development of core marketplace features. 5+ years experience "
            "required with Python and distributed systems. Architectural ownership."
        ),
        "url": "https://www.offerzen.com/careers",
    },
    {
        "title": "Graduate Software Developer",
        "company": "DVT",
        "location": "Remote (South Africa)",
        "remote": True,
        "salary_min": 22000,
        "salary_max": 26000,
        "salary_text": "R22,000 - R26,000 per month",
        "description": (
            "Graduate development programme. We train C# and .NET for client "
            "projects. Fully remote, work from anywhere in South Africa. "
            "Recent graduates encouraged to apply."
        ),
        "url": "https://www.dvt.co.za/careers",
    },
    {
        "title": "Software Developer",
        "company": "Derivco",
        "location": "Durban, KwaZulu-Natal",
        "remote": False,
        "salary_min": 28000,
        "salary_max": 36000,
        "salary_text": "R28,000 - R36,000 per month",
        "description": (
            "Develop and maintain gaming platform services with Python and SQL. "
            "1-2 years experience. Computer science degree or equivalent. "
            "On-site in Durban with occasional hybrid."
        ),
        "url": "https://www.derivco.com/careers",
    },
    {
        "title": "Finance Graduate",
        "company": "Investec",
        "location": "Johannesburg, Gauteng",
        "remote": False,
        "salary_min": 25000,
        "salary_max": 28000,
        "salary_text": "R25,000 - R28,000 per month",
        "description": (
            "Structured graduate programme for BCom finance graduates. "
            "Rotations across banking and wealth management. Johannesburg based."
        ),
        "url": "https://www.investec.com/careers",
    },
    {
        "title": "Junior Web Developer",
        "company": "Deloitte Digital",
        "location": "Johannesburg, Gauteng",
        "remote": False,
        "salary_min": 26000,
        "salary_max": 32000,
        "salary_text": "R26,000 - R32,000 per month",
        "description": (
            "Build client-facing web applications with React and Node.js. "
            "Entry-level role open to recent graduates with a software "
            "development background."
        ),
        "url": "https://www.deloitte.com/careers",
    },
    {
        "title": "IT Support Technician",
        "company": "Bytes Technology Group",
        "location": "Pretoria, Gauteng",
        "remote": False,
        "salary_min": 18000,
        "salary_max": 22000,
        "salary_text": "R18,000 - R22,000 per month",
        "description": (
            "First-line IT support, hardware setup and user troubleshooting. "
            "Matric plus A+ certification. Pretoria office."
        ),
        "url": "https://www.bytes.co.za/careers",
    },
]

FIXTURE_JOBS = [
    Job(
        title=raw["title"],
        company=raw["company"],
        location=raw.get("location", ""),
        remote=raw.get("remote", False),
        description=raw.get("description", ""),
        salary_min=raw.get("salary_min"),
        salary_max=raw.get("salary_max"),
        salary_text=raw.get("salary_text"),
        url=raw.get("url", ""),
        source=FIXTURE_SOURCE,
        posted_date=raw.get("posted_date"),
    )
    for raw in _FIXTURE_JOBS
]


def load_fixture_jobs() -> list[Job]:
    """Return copies of the offline evaluation fixtures."""
    return [
        Job(
            title=j.title,
            company=j.company,
            location=j.location,
            remote=j.remote,
            description=j.description,
            salary_min=j.salary_min,
            salary_max=j.salary_max,
            salary_text=j.salary_text,
            url=j.url,
            source=j.source,
            posted_date=j.posted_date,
        )
        for j in FIXTURE_JOBS
    ]
