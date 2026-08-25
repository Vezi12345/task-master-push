import config
from agent.parse_intent import parse_intent


def _query(prompt: str):
    return parse_intent(prompt, config.load_region("za"))


def _has_role(query, *roles):
    return all(role in query.roles for role in roles)


def _city(query, city):
    return any(loc.city == city for loc in query.locations)


def test_a_entry_level_software_durban():
    q = _query("Find me entry-level software engineering jobs in Durban.")
    assert _has_role(q, "software engineer", "software developer")
    assert q.seniority == "entry-level"
    assert _city(q, "Durban")


def test_b_aerospace_keyword_preserved():
    q = _query("Find me entry-level aerospace software engineering jobs in Durban.")
    assert _has_role(q, "software engineer", "software developer")
    assert q.seniority == "entry-level"
    assert _city(q, "Durban")
    assert "aerospace" in q.keywords


def test_c_remote_junior_python():
    q = _query("Find me remote junior Python developer jobs in South Africa.")
    assert _has_role(q, "software engineer", "software developer")
    assert q.seniority == "entry-level"
    assert q.remote == "preferred"
    assert "python" in q.skills
    assert not q.locations
    assert not any(word in q.keywords for word in ("south", "africa"))


def test_d_salary_cape_town():
    q = _query("Find me software developer jobs in Cape Town paying at least R20,000.")
    assert _has_role(q, "software engineer", "software developer")
    assert _city(q, "Cape Town")
    assert q.min_salary == 20000


def test_e_graduate_johannesburg():
    q = _query("I'm looking for graduate software engineering positions in Johannesburg.")
    assert _has_role(q, "software engineer", "software developer")
    assert q.seniority == "entry-level"
    assert _city(q, "Johannesburg")


def test_f_remote_preferably_cape_town():
    q = _query("Find remote software engineering jobs, preferably in Cape Town.")
    assert q.remote == "preferred"
    assert _city(q, "Cape Town")


def test_g_durban_remote_preferred():
    q = _query("Find software engineering jobs in Durban, remote preferred.")
    assert q.remote == "preferred"
    assert _city(q, "Durban")


def test_h_fully_remote_backend():
    q = _query("Find fully remote entry-level backend developer jobs.")
    assert q.remote == "required"
    assert q.seniority == "entry-level"
    assert _has_role(q, "software engineer", "software developer")
    assert "backend" not in q.keywords


def test_i_fintech_keyword_preserved():
    q = _query("Find fintech graduate developer jobs in South Africa.")
    assert _has_role(q, "software engineer", "software developer")
    assert q.seniority == "entry-level"
    assert "fintech" in q.keywords
    assert not q.locations


def test_j_junior_data_analyst_cape_town():
    q = _query("Find junior data analyst jobs in Cape Town.")
    assert _has_role(q, "data analyst", "data scientist")
    assert q.seniority == "entry-level"
    assert _city(q, "Cape Town")
