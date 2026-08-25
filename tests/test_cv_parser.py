from io import BytesIO
from pathlib import Path

import pytest

from candidate.cv_parser import CvExtractionError, extract_pdf_text, parse_cv
from candidate.profile import CandidateProfile

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_extract_pdf_text_sample():
    text = extract_pdf_text(FIXTURES / "sample_cv.pdf")
    assert "John Smith" in text
    assert "john.smith@email.com" in text
    assert "+27 82 555 1234" in text
    assert "Python" in text


def test_extract_pdf_text_sections():
    text = extract_pdf_text(FIXTURES / "sample_cv_sections.pdf")
    assert "Jane Doe" in text
    assert "SKILLS" in text
    assert "EDUCATION" in text
    assert "EXPERIENCE" in text


def test_extract_pdf_text_empty_pdf():
    with pytest.raises(CvExtractionError, match="no extractable text"):
        extract_pdf_text(FIXTURES / "empty.pdf")


def test_extract_pdf_text_missing_file():
    with pytest.raises(CvExtractionError, match="File not found"):
        extract_pdf_text(FIXTURES / "nonexistent.pdf")


def test_extract_pdf_text_invalid_file(tmp_path):
    bad_file = tmp_path / "bad.pdf"
    bad_file.write_text("not a pdf")
    with pytest.raises(CvExtractionError):
        extract_pdf_text(bad_file)


def test_parse_cv_deterministic_email():
    text = (
        "John Smith\n"
        "john.smith@email.com\n"
        "+27 82 555 1234\n"
        "Durban\n"
        "Skills: Python, JavaScript\n"
    )
    profile = parse_cv(text)
    assert isinstance(profile, CandidateProfile)
    assert profile.name == "John Smith"
    assert profile.email == "john.smith@email.com"
    assert profile.phone == "+27 82 555 1234"


def test_parse_cv_deterministic_skills():
    text = (
        "Skills: Python, JavaScript, React, SQL\n"
        "Education: BSc Computer Science\n"
    )
    profile = parse_cv(text)
    assert "python" in [s.lower() for s in profile.skills]


def test_parse_cv_deterministic_education():
    text = (
        "EDUCATION\n"
        "BSc Computer Science, University of Cape Town, 2018 - 2021\n"
    )
    profile = parse_cv(text)
    assert len(profile.education) >= 1


def test_parse_cv_deterministic_certifications():
    text = (
        "CERTIFICATIONS\n"
        "AWS Certified Cloud Practitioner, Amazon, 2023\n"
    )
    profile = parse_cv(text)
    assert len(profile.certifications) >= 1
    assert "aws" in profile.certifications[0].name.lower()


def test_parse_cv_deterministic_experience():
    text = (
        "EXPERIENCE\n"
        "Software Developer at Takealot, Jan 2022 - Present\n"
        "Built and maintained services.\n"
    )
    profile = parse_cv(text)
    assert len(profile.experience) >= 1
    assert profile.experience[0].company == "Takealot"


def test_parse_cv_from_pdf_fixture():
    text = extract_pdf_text(FIXTURES / "sample_cv.pdf")
    profile = parse_cv(text)
    assert profile.name == "John Smith"
    assert profile.email == "john.smith@email.com"
    assert profile.phone == "+27 82 555 1234"


def test_parse_cv_from_pdf_sections_fixture():
    text = extract_pdf_text(FIXTURES / "sample_cv_sections.pdf")
    profile = parse_cv(text)
    assert profile.name == "Jane Doe"
    assert profile.email == "jane.doe@email.com"
    assert len(profile.education) >= 1
    assert len(profile.experience) >= 1
    assert len(profile.certifications) >= 1


def test_parse_cv_empty_text():
    profile = parse_cv("")
    assert isinstance(profile, CandidateProfile)
    assert profile.name == ""


def test_profile_serialization():
    text = (
        "John Smith\n"
        "john@test.com\n"
        "+27 82 111 2222\n"
        "Skills: Python, Flask\n"
    )
    profile = parse_cv(text)
    json_str = profile.model_dump_json()
    restored = CandidateProfile.model_validate_json(json_str)
    assert restored.name == profile.name
    assert restored.email == profile.email


def test_extract_pdf_text_returns_string():
    text = extract_pdf_text(FIXTURES / "sample_cv.pdf")
    assert isinstance(text, str)
    assert len(text) > 0


def _make_app():
    from app import app
    app.config["TESTING"] = True
    return app


def test_upload_cv_success(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    app = _make_app()
    with app.test_client() as client:
        with open(FIXTURES / "sample_cv.pdf", "rb") as f:
            resp = client.post(
                "/api/upload-cv",
                data={"cv": (f, "test_cv.pdf")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["profile"]["name"] == "John Smith"
        assert data["profile"]["email"] == "john.smith@email.com"


def test_upload_cv_missing_file():
    app = _make_app()
    with app.test_client() as client:
        resp = client.post("/api/upload-cv", content_type="multipart/form-data")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


def test_upload_cv_invalid_type(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    app = _make_app()
    with app.test_client() as client:
        resp = client.post(
                "/api/upload-cv",
                data={"cv": (BytesIO(b"not a pdf"), "test.txt")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "Only PDF" in data["error"]


def test_upload_cv_empty_pdf(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    app = _make_app()
    with app.test_client() as client:
        with open(FIXTURES / "empty.pdf", "rb") as f:
            resp = client.post(
                "/api/upload-cv",
                data={"cv": (f, "empty.pdf")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data


def test_api_profile_returns_none_when_empty(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    app = _make_app()
    with app.test_client() as client:
        resp = client.get("/api/profile")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["profile"] is None


def test_api_profile_returns_saved(tmp_path, monkeypatch):
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)

    profile = CandidateProfile(name="API User", email="api@test.com")
    storage.save_profile(profile)

    app = _make_app()
    with app.test_client() as client:
        resp = client.get("/api/profile")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["profile"]["name"] == "API User"
