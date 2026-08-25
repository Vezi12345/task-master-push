from candidate.matching import (
    ApplicationReadiness,
    CandidateMatch,
    assess_readiness,
    match_candidate_to_job,
    match_jobs_to_candidate,
)
from candidate.profile import CandidateProfile, Education, Experience, Certification, Project
from sources.base import Job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(**kwargs) -> CandidateProfile:
    defaults = {
        "skills": [],
        "education": [],
        "experience": [],
        "certifications": [],
        "projects": [],
        "location": "",
    }
    defaults.update(kwargs)
    return CandidateProfile(**defaults)


def _job(**kwargs) -> Job:
    defaults = {
        "title": "Software Developer",
        "company": "Test Co",
        "location": "",
        "remote": False,
        "description": "",
    }
    defaults.update(kwargs)
    return Job(**defaults)


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------

def test_matching_skills_found():
    profile = _profile(skills=["Python", "Flask", "SQL"])
    job = _job(description="Python, Flask and SQL required.")
    match = match_candidate_to_job(profile, job)
    assert "python" in match.matched_skills
    assert "flask" in match.matched_skills
    assert "sql" in match.matched_skills
    assert match.score > 50


def test_matching_skills_missing():
    profile = _profile(skills=["Python"])
    job = _job(description="Python, Docker and Kubernetes required.")
    match = match_candidate_to_job(profile, job)
    assert "python" in match.matched_skills
    assert "docker" in match.missing_skills
    assert "kubernetes" in match.missing_skills


def test_case_insensitive_matching():
    profile = _profile(skills=["PYTHON", "JavaScript"])
    job = _job(description="python and javascript skills needed.")
    match = match_candidate_to_job(profile, job)
    assert "python" in match.matched_skills
    assert "javascript" in match.matched_skills


def test_skill_normalisation():
    profile = _profile(skills=["JS", "React.js", "Postgres"])
    job = _job(description="JavaScript, React and PostgreSQL experience.")
    match = match_candidate_to_job(profile, job)
    assert "javascript" in match.matched_skills
    assert "react" in match.matched_skills
    assert "postgresql" in match.matched_skills


def test_skills_from_experience():
    exp = Experience(company="X", title="Dev", skills=["Go", "Docker"])
    profile = _profile(experience=[exp])
    job = _job(description="Go and Docker experience required.")
    match = match_candidate_to_job(profile, job)
    assert "go" in match.matched_skills
    assert "docker" in match.matched_skills


def test_skills_from_projects():
    proj = Project(name="P", description="d", technologies=["React", "Node.js"])
    profile = _profile(projects=[proj])
    job = _job(description="React and Node.js developer.")
    match = match_candidate_to_job(profile, job)
    assert "react" in match.matched_skills
    assert "node.js" in match.matched_skills


# ---------------------------------------------------------------------------
# Education matching
# ---------------------------------------------------------------------------

def test_education_relevant():
    edu = Education(institution="UCT", qualification="BSc", field="Computer Science")
    profile = _profile(education=[edu])
    job = _job(description="Computer Science degree required.")
    match = match_candidate_to_job(profile, job)
    assert match.education_match == "relevant"


def test_education_present_but_not_specific():
    edu = Education(institution="UCT", qualification="BCom", field="Finance")
    profile = _profile(education=[edu])
    job = _job(description="Computer Science degree required.")
    match = match_candidate_to_job(profile, job)
    assert match.education_match == "present"


def test_education_unknown_when_no_requirements():
    edu = Education(institution="UCT", qualification="BSc", field="CS")
    profile = _profile(education=[edu])
    job = _job(description="Build great software.")
    match = match_candidate_to_job(profile, job)
    assert match.education_match == "present"


def test_education_missing_when_required():
    profile = _profile()
    job = _job(description="Computer Science degree required.")
    match = match_candidate_to_job(profile, job)
    assert match.education_match == "unknown"


# ---------------------------------------------------------------------------
# Location matching
# ---------------------------------------------------------------------------

def test_location_match_same_city():
    profile = _profile(location="Durban")
    job = _job(location="Durban, KwaZulu-Natal")
    match = match_candidate_to_job(profile, job)
    assert match.location_match is True


def test_location_match_remote():
    profile = _profile(location="Durban")
    job = _job(location="Cape Town", remote=True)
    match = match_candidate_to_job(profile, job)
    assert match.location_match is True


def test_location_no_match():
    profile = _profile(location="Durban")
    job = _job(location="Johannesburg", remote=False)
    match = match_candidate_to_job(profile, job)
    assert match.location_match is False


def test_location_unknown_cand_passes():
    profile = _profile()
    job = _job(location="Durban")
    match = match_candidate_to_job(profile, job)
    assert match.location_match is True


def test_location_unknown_job_passes():
    profile = _profile(location="Durban")
    job = _job()
    match = match_candidate_to_job(profile, job)
    assert match.location_match is True


# ---------------------------------------------------------------------------
# Experience matching
# ---------------------------------------------------------------------------

def test_experience_junior_applies_for_entry():
    exp = Experience(title="Intern", description="junior developer work")
    profile = _profile(experience=[exp])
    job = _job(description="Entry-level developer, 0-2 years experience.")
    match = match_candidate_to_job(profile, job)
    assert match.experience_match in ("strong", "partial")


def test_experience_senior_role():
    profile = _profile()
    job = _job(description="Senior developer, 5+ years experience required.")
    match = match_candidate_to_job(profile, job)
    assert match.experience_match in ("weak", "unknown")


def test_experience_unknown_when_no_markers():
    profile = _profile()
    job = _job(description="Build software for our team.")
    match = match_candidate_to_job(profile, job)
    assert match.experience_match == "unknown"


# ---------------------------------------------------------------------------
# Certification matching
# ---------------------------------------------------------------------------

def test_certification_matched():
    cert = Certification(name="AWS Certified Cloud Practitioner")
    profile = _profile(certifications=[cert])
    job = _job(description="AWS certification preferred.")
    match = match_candidate_to_job(profile, job)
    assert len(match.certification_match) >= 1


def test_no_certifications():
    profile = _profile()
    job = _job(description="AWS certification preferred.")
    match = match_candidate_to_job(profile, job)
    assert match.certification_match == []


# ---------------------------------------------------------------------------
# Application readiness
# ---------------------------------------------------------------------------

def test_readiness_ready():
    profile = _profile(
        email="test@test.com",
        phone="+27 82 123 4567",
        skills=["Python", "Flask"],
    )
    job = _job(description="Python and Flask developer.")
    match = match_candidate_to_job(profile, job)
    readiness = assess_readiness(profile, job, match)
    assert readiness.ready is True
    assert readiness.score > 40


def test_readiness_no_contact():
    profile = _profile(skills=["Python"])
    job = _job(description="Python developer.")
    match = match_candidate_to_job(profile, job)
    readiness = assess_readiness(profile, job, match)
    assert readiness.ready is False
    assert any("contact" in b.lower() for b in readiness.blockers)


def test_readiness_empty_profile():
    profile = _profile()
    job = _job(description="Python developer.")
    match = match_candidate_to_job(profile, job)
    readiness = assess_readiness(profile, job, match)
    assert isinstance(readiness, ApplicationReadiness)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_candidate_match_json():
    profile = _profile(skills=["Python"])
    job = _job(description="Python required.")
    match = match_candidate_to_job(profile, job)
    data = match.model_dump()
    assert isinstance(data["score"], int)
    assert isinstance(data["matched_skills"], list)
    assert isinstance(data["missing_skills"], list)
    restored = CandidateMatch(**data)
    assert restored.score == match.score


def test_readiness_json():
    readiness = ApplicationReadiness(
        ready=True, score=85, reasons=["Good match"], blockers=[], warnings=[]
    )
    data = readiness.model_dump()
    restored = ApplicationReadiness(**data)
    assert restored.ready is True
    assert restored.score == 85


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------

def test_batch_matching():
    from agent.rank import RankedJob

    profile = _profile(skills=["Python"])
    jobs = [
        _job(title="Python Dev", description="Python required."),
        _job(title="Java Dev", description="Java required."),
    ]
    ranked = [RankedJob(job=j, score=70, reasons=[], summary="") for j in jobs]
    results = match_jobs_to_candidate(profile, ranked)
    assert len(results) == 2
    assert "candidate_match" in results[0]
    assert results[0]["candidate_match"].score >= 0
    assert results[0]["rank"] is ranked[0]


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

def test_empty_job_description():
    profile = _profile(skills=["Python"])
    job = _job(description="")
    match = match_candidate_to_job(profile, job)
    assert isinstance(match, CandidateMatch)
    assert match.score >= 0


def test_empty_profile_empty_job():
    profile = _profile()
    job = _job(description="")
    match = match_candidate_to_job(profile, job)
    assert isinstance(match, CandidateMatch)
    assert match.score >= 0


def test_search_still_works_without_cv(monkeypatch):
    import config
    from agent.parse_intent import parse_intent
    from agent.search import search_jobs
    from agent.rank import rank_jobs
    from sources.dpsa_circular import DpsaCircularSource

    monkeypatch.setattr(DpsaCircularSource, "search", lambda self, query: [])
    region = config.load_region("za")
    query = parse_intent("software engineering jobs", region)
    jobs, messages = search_jobs(query, region)
    ranked = rank_jobs(jobs, query)
    assert len(ranked) > 0
    assert all(hasattr(r, "job") for r in ranked)
    assert all(hasattr(r, "score") for r in ranked)


def test_cv_upload_and_search_api(tmp_path, monkeypatch):
    from candidate import storage
    from candidate.profile import CandidateProfile
    from agent.orchestrator import PipelineResult
    from agent.parse_intent import JobQuery
    from agent.rank import RankedJob
    from sources.base import Job

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    profile = CandidateProfile(
        name="Test User",
        email="test@test.com",
        skills=["Python"],
    )
    storage.save_profile(profile)

    loaded = storage.load_profile()
    assert loaded is not None

    dummy_job = Job(
        title="Python Developer",
        company="Test Corp",
        location="Durban",
        remote=False,
        description="Python and Django development.",
    )
    dummy_query = JobQuery(roles=["developer"], seniority="junior")
    dummy_result = PipelineResult(
        query=dummy_query,
        jobs_found=[dummy_job],
        ranked=[RankedJob(job=dummy_job, score=80, reasons=["Good match"], summary="Great fit")],
    )

    from agent import orchestrator
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda prompt, region, llm=None: dummy_result)

    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        resp = client.post(
            "/api/search",
            json={"query": "software engineering jobs"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["has_cv"] is True
        assert len(data["ranked"]) > 0
        first = data["ranked"][0]
        assert "candidate_match" in first
        assert "readiness" in first
        assert "score" in first["candidate_match"]
