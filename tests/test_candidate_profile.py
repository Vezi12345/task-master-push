from candidate.profile import (
    CandidateProfile,
    Certification,
    Education,
    Experience,
    Project,
)


def test_candidate_profile_defaults():
    profile = CandidateProfile()
    assert profile.name == ""
    assert profile.email == ""
    assert profile.phone == ""
    assert profile.skills == []
    assert profile.education == []
    assert profile.experience == []
    assert profile.certifications == []
    assert profile.projects == []
    assert profile.achievements == []


def test_candidate_profile_full():
    profile = CandidateProfile(
        name="Jane Doe",
        email="jane@example.com",
        phone="+27 82 123 4567",
        location="Durban",
        professional_summary="Software engineer with 5 years experience.",
        skills=["Python", "SQL", "React"],
        education=[
            Education(
                institution="University of KwaZulu-Natal",
                qualification="BSc",
                field="Computer Science",
                start_date="2016",
                end_date="2019",
            )
        ],
        experience=[
            Experience(
                company="Luno",
                title="Backend Developer",
                start_date="Jan 2020",
                end_date="Present",
                description="Built microservices.",
                skills=["Python", "PostgreSQL"],
            )
        ],
        certifications=[
            Certification(name="AWS SAA", issuer="Amazon", date="2022")
        ],
        projects=[
            Project(
                name="TaskMaster",
                description="Job search agent",
                technologies=["Python", "Flask"],
            )
        ],
        achievements=["Won hackathon 2023"],
    )
    assert profile.name == "Jane Doe"
    assert len(profile.education) == 1
    assert profile.education[0].institution == "University of KwaZulu-Natal"
    assert len(profile.experience) == 1
    assert profile.experience[0].company == "Luno"
    assert len(profile.certifications) == 1
    assert profile.certifications[0].name == "AWS SAA"
    assert len(profile.projects) == 1
    assert profile.projects[0].name == "TaskMaster"
    assert profile.achievements == ["Won hackathon 2023"]


def test_candidate_profile_json_roundtrip():
    profile = CandidateProfile(
        name="John Smith",
        email="john@test.com",
        skills=["Python", "Java"],
        education=[
            Education(institution="UCT", qualification="BEng", field="Electrical")
        ],
    )
    json_str = profile.model_dump_json()
    restored = CandidateProfile.model_validate_json(json_str)
    assert restored.name == "John Smith"
    assert restored.email == "john@test.com"
    assert restored.skills == ["Python", "Java"]
    assert len(restored.education) == 1
    assert restored.education[0].institution == "UCT"


def test_candidate_profile_optional_fields():
    profile = CandidateProfile(
        name="Test User",
        skills=["Go"],
    )
    data = profile.model_dump()
    assert data["email"] == ""
    assert data["phone"] == ""
    assert data["location"] == ""
    assert data["professional_summary"] == ""
    assert data["education"] == []
    assert data["experience"] == []
    assert data["certifications"] == []
    assert data["projects"] == []
    assert data["achievements"] == []


def test_education_model():
    edu = Education(
        institution="Stellenbosch",
        qualification="MSc",
        field="Data Science",
        start_date="2020",
        end_date="2022",
    )
    data = edu.model_dump()
    assert data["institution"] == "Stellenbosch"
    assert data["field"] == "Data Science"


def test_experience_model():
    exp = Experience(
        company="Takealot",
        title="Data Analyst",
        start_date="2021",
        end_date="2023",
        description="Built dashboards.",
        skills=["SQL", "Tableau"],
    )
    data = exp.model_dump()
    assert data["company"] == "Takealot"
    assert data["skills"] == ["SQL", "Tableau"]


def test_project_model():
    proj = Project(
        name="Portfolio",
        description="Personal website",
        technologies=["React", "Node.js"],
    )
    data = proj.model_dump()
    assert data["technologies"] == ["React", "Node.js"]


def test_certification_model():
    cert = Certification(name="PMP", issuer="PMI", date="2023-06")
    data = cert.model_dump()
    assert data["name"] == "PMP"
    assert data["date"] == "2023-06"


def test_profile_from_dict():
    data = {
        "name": "Dict User",
        "email": "dict@test.com",
        "skills": ["R"],
        "education": [],
        "experience": [],
        "certifications": [],
        "projects": [],
        "achievements": [],
    }
    profile = CandidateProfile(**data)
    assert profile.name == "Dict User"
    assert profile.skills == ["R"]


def test_save_and_load_profile(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    profile = CandidateProfile(
        name="Saved User",
        email="saved@test.com",
        skills=["Python"],
    )
    storage.save_profile(profile)
    loaded = storage.load_profile()
    assert loaded is not None
    assert loaded.name == "Saved User"
    assert loaded.email == "saved@test.com"
    assert loaded.skills == ["Python"]


def test_load_profile_returns_none_when_missing(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    loaded = storage.load_profile()
    assert loaded is None


def test_load_profile_returns_none_on_corrupt(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "corrupt.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    (tmp_path / "corrupt.json").write_text("not valid json {{{")
    loaded = storage.load_profile()
    assert loaded is None
