import json

import config
from agent.parse_intent import parse_intent


def test_durban_example():
    region = config.load_region("za")
    prompt = (
        "I'm a recent computer science graduate in Durban. Find me entry-level "
        "software engineering jobs, preferably remote or in Durban, and show me "
        "the best matches."
    )
    query = parse_intent(prompt, region)
    assert "software engineer" in query.roles
    assert "software developer" in query.roles
    assert query.seniority == "entry-level"
    assert any(loc.city == "Durban" for loc in query.locations)
    assert query.remote == "preferred"
    assert "computer science" in query.skills


def test_salary_extraction():
    region = config.load_region("za")
    query = parse_intent("Find me junior finance jobs paying at least R25k in Johannesburg.", region)
    assert query.min_salary == 25000
    assert query.seniority == "entry-level"
    assert any(loc.city == "Johannesburg" for loc in query.locations)


def test_salary_long_form():
    region = config.load_region("za")
    query = parse_intent("Jobs with a minimum of R20 000 per month.", region)
    assert query.min_salary == 20000


def test_distance_radius():
    region = config.load_region("za")
    query = parse_intent("Cape Town jobs within 30 km.", region)
    loc = query.locations[0]
    assert loc.city == "Cape Town"
    assert loc.radius_km == 30


def test_remote_required_and_onsite():
    region = config.load_region("za")
    assert parse_intent("fully remote developer jobs", region).remote == "required"
    assert parse_intent("onsite jobs in Durban", region).remote == "no"
    assert parse_intent("developer jobs", region).remote == "any"


def test_remote_semantics_matrix():
    region = config.load_region("za")
    assert parse_intent("fully remote software engineer", region).remote == "required"
    assert parse_intent("remote preferred software engineer", region).remote == "preferred"
    assert parse_intent("software engineer, preferably remote", region).remote == "preferred"
    assert parse_intent("remote software engineer jobs", region).remote == "preferred"
    assert parse_intent("on-site software engineer", region).remote == "no"


def test_city_alias():
    region = config.load_region("za")
    query = parse_intent("jobs near Sandton", region)
    assert any(loc.city == "Johannesburg" for loc in query.locations)


def test_no_location_and_no_salary():
    region = config.load_region("za")
    query = parse_intent("find me developer jobs", region)
    assert query.locations == []
    assert query.min_salary is None


def test_currency_from_region():
    region = config.load_region("za")
    query = parse_intent("find me any jobs", region)
    assert query.currency == "ZAR"


def test_query_wording_not_extracted_as_keywords():
    region = config.load_region("za")
    query = parse_intent("Find me software engineering jobs using Python in South Africa.", region)
    assert "using" not in query.keywords
    query = parse_intent("Find me software engineering internships in Johannesburg.", region)
    assert "internship" not in query.keywords
    assert "internships" not in query.keywords
    assert query.seniority == "entry-level"
