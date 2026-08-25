import config
from agent.parse_intent import JobQuery, parse_intent
from agent.rank import rank_jobs
from sources.base import Job
from evaluation.fixtures import load_fixture_jobs


def _query(prompt: str):
    return parse_intent(prompt, config.load_region("za"))


def _jobs():
    return load_fixture_jobs()


def test_durban_example_ranks_remotely_first():
    query = _query(
        "I'm a recent computer science graduate in Durban. Find me entry-level "
        "software engineering jobs, preferably remote or in Durban."
    )
    ranked = rank_jobs(_jobs(), query)
    assert ranked, "expected at least one match"
    titles = [item.job.title for item in ranked]
    assert "Junior Software Developer" in titles[:3]
    assert "Senior Software Engineer" not in titles
    assert "Administration Clerk" not in titles
    assert "Finance Graduate" not in titles


def test_each_match_has_explanations():
    query = _query("entry-level software engineering jobs in Durban, at least R25k")
    ranked = rank_jobs(_jobs(), query)
    for item in ranked:
        assert item.reasons, f"no reasons for {item.job.title}"
        assert 0 <= item.score <= 100
        assert item.summary


def test_unknown_salary_not_rejected():
    query = _query("entry-level software engineering jobs")
    ranked = rank_jobs(_jobs(), query)
    luno = next(item for item in ranked if item.job.company == "Luno")
    salary_reasons = [r for r in luno.reasons if r.startswith("Salary")]
    assert any("⚠ not stated" in r for r in salary_reasons)
    assert luno.job.salary_min is None and luno.score > 0


def test_salary_filter_drops_below_minimum():
    query = _query("entry-level software engineering jobs, at least R50,000")
    ranked = rank_jobs(_jobs(), query)
    for item in ranked:
        assert item.job.salary_min is None or item.job.salary_min >= 50000


def test_remote_required_filters_onsite():
    query = _query("fully remote entry-level software developer jobs")
    ranked = rank_jobs(_jobs(), query)
    assert ranked
    for item in ranked:
        assert item.job.remote


def test_senior_role_filtered_for_entry_level():
    query = _query("entry-level developer jobs")
    ranked = rank_jobs(_jobs(), query)
    titles = [item.job.title for item in ranked]
    assert "Senior Software Engineer" not in titles


def test_scores_are_explainable_and_summarised():
    query = _query("entry-level software engineering jobs in Durban, preferably remote")
    ranked = rank_jobs(_jobs(), query)
    top = ranked[0]
    assert top.reasons
    assert all(r.startswith(("Role", "Seniority", "Location", "Salary", "Skills", "Remote")) for r in top.reasons)


def test_domain_keyword_boosts_matching_job():
    query = _query("aerospace software engineering jobs")
    assert "aerospace" in query.keywords
    generic = Job(
        title="Software Engineer",
        company="GenericCorp",
        location="Cape Town",
        description="Build backend services with Python and SQL.",
    )
    aerospace = Job(
        title="Software Engineer",
        company="AeroCorp",
        location="Cape Town",
        description="Build software systems for the aerospace industry using Python.",
    )
    ranked = rank_jobs([generic, aerospace], query)
    assert len(ranked) == 2
    assert ranked[0].job is aerospace
    assert ranked[0].score > ranked[1].score


def test_admin_clerk_synonyms_not_filtered():
    query = _query("Find me admin clerk jobs in Durban.")
    ranked = rank_jobs(_jobs(), query)
    titles = [item.job.title for item in ranked]
    assert "Administration Clerk" in titles
    admin = next(item for item in ranked if item.job.title == "Administration Clerk")
    role_reasons = [r for r in admin.reasons if r.startswith("Role")]
    assert any("✓" in r for r in role_reasons)


def test_web_developer_matches_software_role():
    query = _query("Find me software developer jobs.")
    ranked = rank_jobs(_jobs(), query)
    titles = [item.job.title for item in ranked]
    assert "Junior Web Developer" in titles
    web = next(item for item in ranked if item.job.title == "Junior Web Developer")
    role_reasons = [r for r in web.reasons if r.startswith("Role")]
    assert any("✓" in r for r in role_reasons)


def test_engineer_surface_form_matches_software_role():
    query = _query("Find me software engineer jobs.")
    backend = Job(
        title="Backend Engineer",
        company="CloudCorp",
        location="Cape Town",
        description="Build APIs and distributed services.",
    )
    ranked = rank_jobs([backend], query)
    assert ranked and ranked[0].job is backend
    assert any("✓" in r for r in ranked[0].reasons if r.startswith("Role"))


def test_role_allowed_and_score_share_synonym_semantics():
    query = _query("Find me data analyst jobs.")
    canonical = Job(
        title="Data Analyst",
        company="CanonicalCo",
        location="Cape Town",
        description="Use SQL to analyse business data.",
    )
    synonym = Job(
        title="Insights Specialist",
        company="SynonymCo",
        location="Cape Town",
        description="Work on analytics projects with SQL and Excel.",
    )
    ranked = rank_jobs([canonical, synonym], query)
    assert len(ranked) == 2
    assert ranked[0].job is canonical
    assert ranked[1].job is synonym


def test_unrelated_roles_still_filtered():
    query = _query("Find me software developer jobs.")
    ranked = rank_jobs(_jobs(), query)
    titles = [item.job.title for item in ranked]
    assert "Administration Clerk" not in titles
    assert "Finance Graduate" not in titles
    assert "IT Support Technician" not in titles


# ---------------------------------------------------------------------------
# Entry-level leadership gate + junk-keyword armour
# ---------------------------------------------------------------------------

def _job(title, description="Generic duties as required.", company="GovCo", location="Pretoria"):
    return Job(title=title, company=company, location=location, description=description)


def test_entry_level_query_filters_leadership_titles():
    query = _query("Find me entry-level finance jobs.")
    jobs = [
        _job("REGIONAL HEAD: FINANCE", "Manage the regional finance office."),
        _job("CHIEF DIRECTOR: FINANCIAL MANAGEMENT", "Executive financial management."),
        _job("SENIOR STATE ACCOUNTANT: BUDGET", "Budget and expenditure accounting."),
        _job("FINANCE CLERK", "Capture and reconcile finance transactions."),
        _job("GRADUATE INTERNSHIP: FINANCIAL MANAGEMENT", "24-month graduate programme."),
    ]
    ranked = rank_jobs(jobs, query)
    titles = [item.job.title for item in ranked]
    assert "REGIONAL HEAD: FINANCE" not in titles
    assert "CHIEF DIRECTOR: FINANCIAL MANAGEMENT" not in titles
    assert "SENIOR STATE ACCOUNTANT: BUDGET" not in titles
    # developmental posts survive even with leadership-ish wording elsewhere
    assert "GRADUATE INTERNSHIP: FINANCIAL MANAGEMENT" in titles
    assert "FINANCE CLERK" in titles


def test_seniority_not_requested_keeps_leadership_titles():
    query = _query("Find me finance jobs.")
    jobs = [_job("REGIONAL HEAD: FINANCE", "Manage the regional finance office.")]
    ranked = rank_jobs(jobs, query)
    assert len(ranked) == 1


def test_typo_keywords_never_reported_as_missing_requirements():
    query = _query("okay then seeach for accounting jobs")
    assert query.roles == ["finance"]
    assert "okay" not in query.keywords
    assert "then" not in query.keywords

    job = _job(
        "ACCOUNTING CLERK: RECONCILIATION",
        "Reconcile accounts and capture finance transactions.",
    )
    ranked = rank_jobs([job], query)
    assert ranked, "accounting clerk must match a finance/accounting request"
    blob = " | ".join(ranked[0].reasons) + " " + ranked[0].summary
    assert "seeach" not in blob
    assert "missing okay" not in blob


def test_skills_reasons_only_reference_explicit_skills():
    query = _query("bookkeeper or accounts clerk jobs")
    job = _job("FINANCE CLERK", "Bookkeeping support and finance administration.")
    ranked = rank_jobs([job], query)
    skill_reasons = [r for r in ranked[0].reasons if r.startswith("Skills")]
    assert skill_reasons
    assert all("missing" not in r.lower() for r in skill_reasons)


def test_leadership_titles_demoted_without_explicit_seniority():
    query = _query("Find me finance jobs.")
    jobs = [
        _job("REGIONAL HEAD: FINANCE", "Manage the regional finance office."),
        _job("FINANCE CLERK", "Capture and reconcile finance transactions."),
    ]
    ranked = rank_jobs(jobs, query)
    by_title = {item.job.title: item for item in ranked}
    assert by_title["FINANCE CLERK"].score > by_title["REGIONAL HEAD: FINANCE"].score
    head = by_title["REGIONAL HEAD: FINANCE"]
    assert any("leadership title" in r for r in head.reasons)


def test_leadership_demotion_not_applied_to_developmental_posts():
    query = _query("Find me finance jobs.")
    job = _job("GRADUATE INTERNSHIP: FINANCIAL MANAGEMENT", "24-month programme.")
    ranked = rank_jobs([job], query)
    assert not any("leadership title" in r for r in ranked[0].reasons)
