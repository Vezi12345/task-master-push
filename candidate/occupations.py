"""Data-driven occupation registry.

This module contains NO candidate-specific logic and NO code branches per
profession: it is a uniform lookup table.  Occupation inference and query
generation (candidate/search_profile.py) consume it generically, so the
system adapts to any candidate whose evidence matches entries here — or
falls back to their own titles/qualifications when nothing matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional


@dataclass(frozen=True)
class Occupation:
    key: str
    label: str
    titles: tuple[str, ...] = ()
    adjacent: tuple[str, ...] = ()
    qualifications: tuple[str, ...] = ()
    keywords: tuple[str, ...] = field(default_factory=tuple)


OCCUPATIONS: list[Occupation] = [
    Occupation(
        key="software_developer",
        label="Software Developer",
        titles=("software developer", "software engineer", "web developer",
                "application developer", "junior developer", "developer",
                "programmer", "full stack developer", "backend developer",
                "frontend developer"),
        adjacent=("qa tester", "test analyst", "technical support",
                  "systems analyst", "solutions engineer", "it technician"),
        qualifications=("software engineering", "computer science",
                        "information technology", "app development",
                        "ict", "information systems"),
        keywords=("python", "java", "javascript", "sql", "git", "api",
                  "agile", "html", "css", "react", "c#", ".net", "php",
                  "rest api", "software development", "programming"),
    ),
    Occupation(
        key="data_analyst",
        label="Data Analyst",
        titles=("data analyst", "data scientist", "business intelligence "
                "analyst", "reporting analyst", "insights analyst"),
        adjacent=("data capturer", "microsoft excel specialist",
                  "research assistant", "operations analyst"),
        qualifications=("data science", "statistics", "analytics",
                        "business intelligence", "mathematics"),
        keywords=("power bi", "tableau", "sql", "excel", "python",
                  "reporting", "dashboards", "statistics", "data analysis"),
    ),
    Occupation(
        key="accountant",
        label="Accountant",
        titles=("accountant", "bookkeeper", "junior accountant",
                "assistant accountant", "accounts clerk",
                "financial accountant", "management accountant",
                "accounts payable clerk", "accounts receivable clerk",
                "finance assistant", "debtors clerk", "creditors clerk",
                "payroll clerk", "payroll administrator"),
        adjacent=("audit clerk", "cashier supervisor", "procurement clerk",
                  "billing clerk", "stock controller", "revenue clerk"),
        qualifications=("accounting", "accountancy", "finance", "financial "
                        "management", "bookkeeping", "cost accounting",
                        "commerce", "bcom", "business management"),
        keywords=("pastel", "sage", "accpac", "vat", "reconciliations",
                  "journals", "trial balance", "debtor", "creditor",
                  "payroll", "ifrs", "tax", "saica", "at(sa)",
                  "finance", "financial", "accounting", "bookkeeping"),
    ),
    Occupation(
        key="auditor",
        label="Auditor",
        titles=("internal auditor", "junior auditor", "audit clerk",
                "audit assistant"),
        adjacent=("risk analyst", "compliance officer", "accountant"),
        qualifications=("internal auditing", "accounting", "auditing",
                        "commerce"),
        keywords=("audit", "sox", "risk assessment", "controls",
                  "working papers", "saica", "iia"),
    ),
    Occupation(
        key="nurse",
        label="Nurse",
        titles=("staff nurse", "professional nurse", "registered nurse",
                "enrolled nurse", "nursing assistant", "clinical nurse",
                "theatre nurse", "nursing sister"),
        adjacent=("healthcare assistant", "care worker", "caregiver",
                  "community health worker", "pharmacy assistant"),
        qualifications=("nursing", "general nursing", "midwifery",
                        "community nursing", "healthcare"),
        keywords=("sanc", "patient care", "clinical", "ward", "theatre",
                  "dispensary", "vital signs", "infection control",
                  "hospice", "clinic", "hospital"),
    ),
    Occupation(
        key="teacher",
        label="Teacher",
        titles=("teacher", "educator", "foundation phase teacher",
                "intermediate phase teacher", "fet teacher", "lecturer",
                "tutor"),
        adjacent=("teaching assistant", "education assistant",
                  "aftercare teacher", "sports coach", "au pair",
                  "facilitator", "trainer"),
        qualifications=("education", "teaching", "bed", "pgce",
                        "early childhood development", "ecd"),
        keywords=("curriculum", "caps", "classroom", "learners", "sace",
                  "lesson planning", "school", "syllabus"),
    ),
    Occupation(
        key="electrician",
        label="Electrician",
        titles=("electrician", "installation electrician",
                "maintenance electrician"),
        adjacent=("electrical assistant", "artisan assistant",
                  "electrical apprentice", "solar installer"),
        qualifications=("electrical engineering", "electrical trade",
                        "electrician n2", "electrical n6"),
        keywords=("wireman", "coc", "certificate of compliance",
                  "single phase", "three phase", "distribution board",
                  "mv lv", "reticulation", "dbe"),
    ),
    Occupation(
        key="engineer",
        label="Engineer",
        titles=("mechanical engineer", "civil engineer",
                "electronic engineer", "industrial engineer",
                "process engineer", "design engineer", "graduate engineer",
                "engineer in training"),
        adjacent=("draughtsman", "cad technician", "site inspector",
                  "quality inspector", "engineering assistant"),
        qualifications=("mechanical engineering", "civil engineering",
                        "chemical engineering", "electronic engineering",
                        "industrial engineering", "beng", "engineering"),
        keywords=("autocad", "solidworks", "project execution", "ecsа",
                  "site supervision", "maintenance planning", "cad"),
    ),
    Occupation(
        key="driver",
        label="Driver",
        titles=("driver", "delivery driver", "code 10 driver",
                "code 14 driver", "truck driver", "courier driver",
                "shuttle driver"),
        adjacent=("storeman", "warehouse assistant", "messenger",
                  "fleet assistant", "yard controller"),
        qualifications=(),
        keywords=("prd", "pdp", "code 10", "code 14", "code 8", "eb licence",
                  "fleet", "deliveries", "long haul", "local runs"),
    ),
    Occupation(
        key="receptionist",
        label="Receptionist",
        titles=("receptionist", "front desk receptionist",
                "front office receptionist", "switchboard operator"),
        adjacent=("office assistant", "meet and greet host",
                  "call centre agent", "customer service agent"),
        qualifications=(),
        keywords=("switchboard", "front desk", "diary management",
                  "visitor access", "meetings", "boardroom"),
    ),
    Occupation(
        key="administrator",
        label="Administrator",
        titles=("administrator", "administrative assistant",
                "office administrator", "admin clerk", "office clerk",
                "data capturer", "general clerk", "records clerk",
                "filing clerk", "operations assistant",
                "customer service administrator", "project administrator"),
        adjacent=("receptionist", "personal assistant", "call centre agent",
                  "sales administrator", "stores assistant", "messenger"),
        qualifications=("business management", "office administration",
                        "management assistant", "business administration"),
        keywords=("ms office", "excel", "word", "outlook", "filing",
                  "typing", "correspondence", "minutes", "diaries",
                  "data entry", "admin support", "pastel", "clerk"),
    ),
    Occupation(
        key="sales_representative",
        label="Sales Representative",
        titles=("sales representative", "sales consultant",
                "field sales representative", "sales agent",
                "area sales manager", "key account manager",
                "business development consultant", "retail sales assistant"),
        adjacent=("call centre agent", "telesales agent", "brand ambassador",
                  "merchandiser", "promoter", "customer service agent"),
        qualifications=("marketing", "business management", "sales",
                        "retail management"),
        keywords=("targets", "cold calling", "pipeline", "crm", "leads",
                  "commission", "client base", "upselling", "in-store"),
    ),
    Occupation(
        key="retail_worker",
        label="Retail Worker",
        titles=("retail assistant", "shop assistant", "cashier",
                "shelf packer", "store assistant", "merchandiser",
                "packaging assistant"),
        adjacent=("warehouse assistant", "stock controller",
                  "receiving clerk", "picker", "packer", "cleaner"),
        qualifications=(),
        keywords=("point of sale", "pos", "till", "stock taking",
                  "housekeeping", "planograms", "customer service"),
    ),
    Occupation(
        key="marketing",
        label="Marketing Specialist",
        titles=("marketing assistant", "marketing coordinator",
                "digital marketing assistant", "social media assistant",
                "content creator", "seo specialist"),
        adjacent=("graphic designer", "copywriter", "brand ambassador",
                  "events coordinator", "sales representative"),
        qualifications=("marketing management", "digital marketing",
                        "communication", "public relations"),
        keywords=("seo", "sem", "google analytics", "meta ads", "canva",
                  "mailchimp", "campaigns", "content calendar", "crm"),
    ),
    Occupation(
        key="graphic_designer",
        label="Graphic Designer",
        titles=("graphic designer", "junior graphic designer",
                "desktop publisher", "dtp operator"),
        adjacent=("photographer", "signage installer", "print operator",
                  "content creator", "artworker"),
        qualifications=("graphic design", "visual communication",
                        "fine art", "multimedia design"),
        keywords=("photoshop", "illustrator", "indesign", "coreldraw",
                  "figma", "prepress", "layout", "branding"),
    ),
    Occupation(
        key="hr",
        label="HR Professional",
        titles=("hr assistant", "hr administrator", "hr clerk",
                "recruitment assistant", "talent acquisition intern",
                "payroll administrator", "skills development facilitator"),
        adjacent=("office administrator", "training coordinator",
                  "safety officer", "receptionist"),
        qualifications=("human resource management", "human resources",
                        "hr management", "labour relations", "psychology"),
        keywords=("recruitment", "onboarding", "sage people", "persal",
                  "b-bbee", "employment equity", "leave administration",
                  "disciplinaries", "ccma"),
    ),
    Occupation(
        key="project_manager",
        label="Project Manager",
        titles=("project manager", "project administrator",
                "project coordinator", "programme coordinator",
                "site administrator"),
        adjacent=("operations administrator", "planning assistant",
                  "scrum master", "pmo assistant", "site clerk"),
        qualifications=("project management", "business management",
                        "construction management", "pmp"),
        keywords=("ms project", "gantt", "stakeholders", "budget tracking",
                  "progress reports", "snag lists", "agile", "scrum"),
    ),
    Occupation(
        key="hospitality",
        label="Hospitality Worker",
        titles=("waiter", "waitress", "food and beverage attendant",
                "barista", "bartender", "housekeeping attendant",
                "kitchen assistant", "commis chef", "chef de partie",
                "front office agent", "concierge"),
        adjacent=("cleaner", "kitchen cleaner", "room attendant",
                  "scullery assistant", "event steward"),
        qualifications=("hospitality management", "food preparation",
                        "professional cookery", "culinary"),
        keywords=("food safety", "haccp", "guest service", "banqueting",
                  "room service", "pos", "stock rotation"),
    ),
    Occupation(
        key="security",
        label="Security Officer",
        titles=("security guard", "security officer", "access controller",
                "site security", "armed response officer",
                "control room operator"),
        adjacent=("cleaner", "gardener", "parking attendant", "bouncer"),
        qualifications=(),
        keywords=("psira", "grade c", "grade b", "grade d", "firearm",
                  "cctv monitoring", "access control", "patrols",
                  "incident report"),
    ),
    Occupation(
        key="cleaner",
        label="Cleaner",
        titles=("cleaner", "general worker", "housekeeping cleaner",
                "domestic worker", "office cleaner", "industrial cleaner"),
        adjacent=("kitchen assistant", "laundry assistant",
                  "grounds keeper", "tea attendant"),
        qualifications=(),
        keywords=("hygiene standards", "chemical handling",
                  "cleaning chemicals", "sweeping", "mopping",
                  "bathroom sanitation"),
    ),
    Occupation(
        key="warehouse_worker",
        label="Warehouse Worker",
        titles=("warehouse assistant", "picker", "packer", "storeman",
                "forklift operator", "despatch clerk", "receiving clerk",
                "stock controller", "inventory clerk"),
        adjacent=("driver", "yard controller", "production operator",
                  "cleaner", "counter sales assistant"),
        qualifications=("logistics management", "supply chain"),
        keywords=("forklift", "reach truck", "picking slips", "despatch",
                  "stock counts", "sap wm", "loading", "offloading"),
    ),
    Occupation(
        key="call_centre",
        label="Call Centre Agent",
        titles=("call centre agent", "contact centre agent",
                "inbound consultant", "outbound consultant",
                "customer care agent", "help desk agent"),
        adjacent=("receptionist", "sales representative", "collections agent",
                  "data capturer"),
        qualifications=(),
        keywords=("inbound calls", "outbound calls", "first call resolution",
                  "crm system", "script adherence", "voice tone"),
    ),
]


def _label(o: Occupation) -> str:
    return o.label


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word containment for (multi-)word phrases."""
    if not phrase or not haystack:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(
        re.escape(part) for part in phrase.split()
    ) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def bucket_for_job_title(title: str) -> Optional[str]:
    """Coarse occupation label for a raw advert title, or None."""
    t = normalize(title)
    best: tuple[int, str] | None = None
    for occ in OCCUPATIONS:
        strength = 0
        if any(contains_phrase(t, p) for p in occ.titles):
            strength = 3
        elif any(contains_phrase(t, p) for p in occ.adjacent):
            strength = 2
        elif any(contains_phrase(t, k) for k in occ.keywords):
            strength = 1
        if strength and (best is None or strength > best[0]):
            best = (strength, occ.label)
    return best[1] if best else None
