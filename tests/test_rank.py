import config
from agent.parse_intent import JobQuery, parse_intent
from agent.rank import rank_jobs
from sources.demo import DEMO_JOBS, DemoSource


def _query(prompt: str):
    return parse_intent(prompt, config.load_region("za"))


def _jobs():
    return DemoSource().search(None)


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
