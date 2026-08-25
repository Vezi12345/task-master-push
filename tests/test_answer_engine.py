from __future__ import annotations

"""Tests for the semantic application answer engine.

Covers: equivalence of unseen question phrasings, completely new questions,
answer provenance (VERIFIED / DERIVED / GENERATED_FROM_EVIDENCE / UNKNOWN),
the never-guess-sensitive-information rule, memory of user answers, and
derivation from verified facts.
"""

from application.answer_engine import (
    AnswerType,
    answer_question,
    classify_question,
    questions_equivalent,
)
from application.question_engine import AnswerStore, QuestionEngine
from candidate.profile import CandidateProfile, Education, Experience, Project
from sources.base import Job


def _engine(tmp_path) -> QuestionEngine:
    return QuestionEngine(AnswerStore(tmp_path / "answers.json"))


def _graduate_profile() -> CandidateProfile:
    return CandidateProfile(
        name="Thandi Mkhize",
        email="thandi@example.com",
        location="Durban",
        skills=["Java", "C#", "SQL", "CSS", "Linux", "MySQL", "Communication"],
        education=[
            Education(
                qualification="Diploma in Information and Communication Technology",
                field="Application Development",
                institution="Durban University of Technology",
                start_date="2022-01-01",
                end_date="2025-11-01",
            ),
        ],
        experience=[
            Experience(
                title="Software Development Intern",
                company="Derivco",
                start_date="2025-01-01",
                end_date="2025-04-01",
                description="Built reporting services with SQL.",
                experience_type="employment",
            ),
            Experience(
                title="Junior Software Developer",
                company="DVT",
                start_date="2025-05-01",
                end_date="2025-11-01",
                description="Developed C# backend features.",
                experience_type="employment",
            ),
        ],
        projects=[
            Project(
                name="Student Portal",
                description="Campus portal built with Java and MySQL.",
                technologies=["Java", "MySQL"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Semantic equivalence (#5)
# ---------------------------------------------------------------------------

LICENCE_PHRASES = [
    "Do you have a valid driver's licence?",
    "Are you in possession of a valid driving licence?",
    "Can you legally drive?",
    "Do you hold a valid driver's license?",
]


def test_licence_questions_are_semantically_equivalent():
    keys = {classify_question(q)[0] for q in LICENCE_PHRASES}
    assert keys == {"drivers_licence"}


def test_equivalent_licence_phrasings_reuse_profile_answer(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(drivers_licence="Yes — Code 8")
    for q in LICENCE_PHRASES:
        result = engine.answer(q, profile)
        assert result.is_answered, q
        assert result.answer == "Yes — Code 8"
        assert result.answer_type == AnswerType.VERIFIED


def test_questions_equivalent_helper():
    assert questions_equivalent(
        "What is your expected salary?",
        "What are your salary expectations?",
    )
    assert not questions_equivalent(
        "Do you have a valid driver's licence?",
        "What is your notice period?",
    )


# ---------------------------------------------------------------------------
# Completely unseen questions (#6, #15)
# ---------------------------------------------------------------------------

def test_unseen_travel_question_uses_stored_preference(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(travel_preference="Yes, within KwaZulu-Natal")
    result = engine.answer("Would you be comfortable travelling to client sites?", profile)
    assert result.is_answered
    assert result.answer == "Yes, within KwaZulu-Natal"
    assert result.field_key == "travel_preference"


def test_unseen_start_date_question_derives_from_availability(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(availability="Immediately")
    result = engine.answer("How soon could you start if selected?", profile)
    assert result.is_answered
    assert result.answer == "Immediately"


def test_unseen_agile_question_generated_from_skill_evidence(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(skills=["Agile", "Scrum", "Python"])
    result = engine.answer("Are you comfortable working in an Agile environment?", profile)
    assert result.is_answered
    assert result.answer_type == AnswerType.GENERATED_FROM_EVIDENCE
    assert "Agile" in result.answer


def test_unseen_distributed_teams_question_from_experience_evidence(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        experience=[
            Experience(
                title="Developer",
                company="DVT",
                description="Delivered features collaborating with distributed teams across time zones.",
            ),
        ],
    )
    result = engine.answer(
        "Do you have experience collaborating with distributed teams?", profile
    )
    assert result.is_answered
    assert result.answer_type == AnswerType.GENERATED_FROM_EVIDENCE


def test_unseen_motivation_question_generated_from_evidence(tmp_path):
    engine = _engine(tmp_path)
    profile = _graduate_profile()
    job = Job(
        title="Graduate Software Developer",
        company="DVT",
        description=(
            "Graduate development programme training C# and .NET. "
            "Recent graduates encouraged to apply."
        ),
    )
    context = {
        "title": job.title,
        "company": job.company,
        "description": job.description,
        "requirements": "",
    }
    for q in (
        "Why are you interested in this position?",
        "What motivates you to pursue this role?",
        "Why are you interested in working for this company?",
    ):
        result = engine.answer(q, profile, job_context=context)
        assert result.is_answered, q
        assert result.answer_type == AnswerType.GENERATED_FROM_EVIDENCE
        assert "DVT" in result.answer
        assert "Graduate Software Developer" in result.answer


def test_genuinely_unknown_information_is_requested_not_invented(tmp_path):
    engine = _engine(tmp_path)
    result = engine.answer("What is your expected salary?", CandidateProfile())
    assert result.answer is None
    assert result.needs_user is True
    assert result.answer_type == AnswerType.UNKNOWN


# ---------------------------------------------------------------------------
# Answer provenance (#7)
# ---------------------------------------------------------------------------

def test_verified_qualification_answer(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        education=[Education(
            qualification="Diploma in Information and Communication Technology",
            field="Application Development",
        )],
    )
    result = engine.answer("What is your highest qualification?", profile)
    assert result.answer_type == AnswerType.VERIFIED
    assert "Diploma in Information and Communication Technology" in result.answer
    assert "Application Development" in result.answer


def test_derived_recent_graduate_yes(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        education=[Education(qualification="Diploma", field="ICT", end_date="2025-11-01")],
    )
    result = engine.answer("Are you a recent graduate (within 2 years)?", profile)
    assert result.is_answered
    assert result.answer == "Yes"
    assert result.answer_type == AnswerType.DERIVED


def test_derived_recent_graduate_no(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        education=[Education(qualification="BSc", field="CS", end_date="2018-12-01")],
    )
    result = engine.answer("Are you a recent graduate (within 2 years)?", profile)
    assert result.answer == "No"
    assert result.answer_type == AnswerType.DERIVED


def test_derived_months_of_experience_from_real_history(tmp_path):
    engine = _engine(tmp_path)
    profile = _graduate_profile()
    result = engine.answer(
        "How many months of professional software development experience do you have?",
        profile,
    )
    assert result.is_answered
    assert result.answer_type == AnswerType.DERIVED
    # 3-month internship + 6-month role = 9 months, calculated not fabricated
    assert "9" in result.answer


def test_generated_from_evidence_has_medium_confidence(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(skills=["Agile"])
    result = engine.answer("Are you comfortable working in an Agile environment?", profile)
    assert result.answer_type == AnswerType.GENERATED_FROM_EVIDENCE


# ---------------------------------------------------------------------------
# Sensitive information is never guessed (#8)
# ---------------------------------------------------------------------------

SENSITIVE_QUESTIONS = [
    "What is your citizenship status?",
    "What is your race / equity group?",
    "What is your gender?",
    "Do you have a disability?",
    "What is your age / date of birth?",
    "Are you a South African citizen?",
]


def test_sensitive_questions_never_inferred_from_hints(tmp_path):
    """Citizenship must NOT be inferred from location, university, name or
    phone number."""
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        name="Thandi Mkhize",
        phone="+27 82 123 4567",
        location="Durban, South Africa",
        education=[Education(
            qualification="Diploma",
            field="ICT",
            institution="Durban University of Technology",
        )],
    )
    for q in SENSITIVE_QUESTIONS:
        result = engine.answer(q, profile)
        assert result.answer is None, q
        assert result.needs_user is True, q
        assert result.answer_type == AnswerType.UNKNOWN, q


def test_sensitive_questions_echo_only_explicitly_stored_values(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(race="Black African", gender="Female")
    assert engine.answer("What is your race / equity group?", profile).answer == "Black African"
    assert engine.answer("What is your gender?", profile).answer == "Female"
    assert engine.answer("Do you have a disability?", profile).answer is None


def test_work_authorisation_derived_from_stored_citizenship_only(tmp_path):
    engine = _engine(tmp_path)
    sa = CandidateProfile(citizenship="South African citizen")
    result = engine.answer("Are you authorised to work in South Africa?", sa)
    assert result.is_answered
    assert result.answer_type == AnswerType.DERIVED

    # Non-SA citizenship proves nothing either way (the candidate may hold a
    # visa or permanent residency) — the system must ask, not guess.
    other = CandidateProfile(citizenship="Zimbabwean national")
    result = engine.answer("Are you authorised to work in South Africa?", other)
    assert result.answer is None
    assert result.needs_user is True

    unknown = CandidateProfile(location="Cape Town")
    result = engine.answer("Are you authorised to work in South Africa?", unknown)
    assert result.answer is None
    assert result.needs_user is True


# ---------------------------------------------------------------------------
# Memory of user answers (#9)
# ---------------------------------------------------------------------------

def test_salary_answer_remembered_for_equivalent_question(tmp_path):
    store = AnswerStore(tmp_path / "answers.json")
    engine = QuestionEngine(store)
    profile = CandidateProfile()

    first = engine.answer("What is your expected salary?", profile)
    assert first.needs_user is True

    # The user answers; the system saves it.
    store.set(first.field_key, "R15,000")

    second = engine.answer("What are your salary expectations?", profile)
    assert second.is_answered
    assert second.answer == "R15,000"
    assert second.needs_user is False


def test_profile_memory_answers_equivalent_phrasing(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile()
    profile.remember_answer(
        "What is your expected salary?", "R15,000", field_key="expected_salary"
    )
    result = engine.answer("What salary range are you looking at?", profile)
    assert result.is_answered
    assert result.answer == "R15,000"
    assert result.source == "memory"


def test_user_answers_persisted_to_candidate_profile(tmp_path, monkeypatch):
    """provide_answers must save answers onto the candidate profile (#9)."""
    import config
    from agent.orchestrator import JobApplicationAgent
    from candidate import storage

    monkeypatch.setattr(storage, "PROFILE_FILE", tmp_path / "profile.json")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "ANSWERS_FILE", tmp_path / "answers.json")

    storage.save_profile(CandidateProfile(name="Test", email="t@t.com"))
    agent = JobApplicationAgent({}, llm=None)
    agent.provide_answers({
        "expected_salary": "R15,000",
        "relocation": "Yes",
        "drivers_licence": "Yes — Code 10",
    })

    saved = storage.load_profile()
    assert saved.get_known_value("expected_salary") == "R15,000"
    assert saved.get_known_value("relocation") == "Yes"
    assert saved.get_known_value("drivers_licence") == "Yes — Code 10"

    # A fresh engine over the saved profile reuses the remembered answers.
    fresh = QuestionEngine(AnswerStore(tmp_path / "answers.json"))
    assert fresh.answer("What are your salary expectations?", saved).answer == "R15,000"
    assert fresh.answer("Are you willing to relocate?", saved).answer == "Yes"


# ---------------------------------------------------------------------------
# DVT-style end-to-end question set (#14)
# ---------------------------------------------------------------------------

DVT_QUESTIONS = [
    "What is your expected salary?",
    "How many years of experience do you have?",
    "Are you willing to relocate?",
    "Do you have a valid driver's licence?",
    "What is your notice period?",
    "Are you authorised to work in South Africa?",
    "What is your availability / start date?",
    "What is your citizenship status?",
    "What is your race / equity group?",
    "What is your gender?",
    "Do you have a disability?",
    "What is your age / date of birth?",
    "Are you a South African citizen?",
    "Are you a recent graduate (within 2 years)?",
]


def test_dvt_question_set_resolved_individually(tmp_path):
    engine = _engine(tmp_path)
    store: AnswerStore = engine.answer_store
    store.set("expected_salary", "R15,000")
    store.set("relocation", "Yes")
    store.set("drivers_licence", "Yes — Code 8")
    store.set("notice_period", "1 month")
    store.set("availability", "Immediately")

    profile = _graduate_profile()
    answered, missing = engine.resolve_common_questions(profile)

    missing_keys = {m.field_key for m in missing}
    answered_keys = set(answered.keys())

    # Automatically answered from stored preferences / profile / derivation.
    for key in (
        "expected_salary",
        "relocation",
        "drivers_licence",
        "notice_period",
        "availability",
        "highest_qualification",
        "years_experience",
        "recent_graduate",
    ):
        assert key in answered_keys, f"{key} should be auto-answered"

    # Sensitive attributes genuinely require user input. Work authorisation
    # is not derived because no citizenship is stored on this profile.
    for key in (
        "citizenship",
        "south_african_citizen",
        "work_authorisation",
        "race",
        "gender",
        "disability",
        "date_of_birth",
    ):
        assert key in missing_keys, f"{key} should need user input"

    # The system must NOT dump every question into needs_information.
    assert len(answered_keys) >= 8
    assert len(missing_keys) <= 7


# ---------------------------------------------------------------------------
# Evidence grounding: unrelated profile data is NEVER evidence (#critical)
# ---------------------------------------------------------------------------

def test_travel_question_never_answered_from_skills(tmp_path):
    """skills=['R'] is NOT evidence of willingness to travel."""
    engine = _engine(tmp_path)
    profile = CandidateProfile(skills=["R"])
    result = engine.answer(
        "Would you be comfortable travelling to client sites?", profile
    )
    assert result.answer is None
    assert result.answer_type == AnswerType.UNKNOWN
    assert result.needs_user is True


def test_distributed_teams_question_never_answered_from_skills(tmp_path):
    """skills=['R'] is NOT evidence of distributed-team experience."""
    engine = _engine(tmp_path)
    profile = CandidateProfile(skills=["R"])
    result = engine.answer(
        "Do you have experience collaborating with distributed teams?", profile
    )
    assert result.answer is None
    assert result.answer_type == AnswerType.UNKNOWN
    assert result.needs_user is True


def test_missing_relocation_preference_is_unknown(tmp_path):
    engine = _engine(tmp_path)
    result = engine.answer("Are you willing to relocate?", CandidateProfile())
    assert result.answer_type == AnswerType.UNKNOWN
    assert result.needs_user is True


def test_missing_travel_preference_is_unknown(tmp_path):
    engine = _engine(tmp_path)
    result = engine.answer(
        "Are you willing to travel as part of this role?", CandidateProfile()
    )
    assert result.answer_type == AnswerType.UNKNOWN
    assert result.needs_user is True


def test_missing_distributed_team_experience_is_unknown(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        experience=[
            Experience(
                title="Developer",
                company="Solo Corp",
                description="Built internal reporting tools independently.",
            ),
        ],
    )
    result = engine.answer(
        "Have you worked with distributed teams?", profile
    )
    assert result.answer_type == AnswerType.UNKNOWN
    assert result.needs_user is True


def test_distributed_teams_answered_from_project_evidence(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        skills=["R"],
        projects=[
            Project(
                name="Campus Portal",
                description="Built with a student team collaborating across three campuses.",
            ),
        ],
    )
    result = engine.answer(
        "Have you worked with distributed teams?", profile
    )
    assert result.is_answered
    assert result.answer_type == AnswerType.GENERATED_FROM_EVIDENCE
    assert result.answer.startswith("Yes")
    assert "Campus Portal" in result.answer


def test_remote_work_answered_from_experience_evidence(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        experience=[
            Experience(
                title="Developer",
                company="RemoteFirst",
                description="Worked remotely with a distributed team for two years.",
            ),
        ],
    )
    result = engine.answer(
        "Do you have remote work experience?", profile
    )
    assert result.is_answered
    assert result.answer_type == AnswerType.GENERATED_FROM_EVIDENCE
    assert "RemoteFirst" in result.answer


def test_skills_are_not_counted_as_experience(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(skills=["Python", "SQL"])
    result = engine.answer(
        "How many years of software development experience do you have?",
        profile,
    )
    assert result.answer_type == AnswerType.UNKNOWN
    assert result.needs_user is True


def test_recent_graduate_derived_from_graduation_year(tmp_path):
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        education=[Education(qualification="Diploma", field="ICT", end_date="2025")],
    )
    result = engine.answer(
        "Are you a recent graduate within two years?", profile
    )
    assert result.is_answered
    assert result.answer == "Yes"
    assert result.answer_type == AnswerType.DERIVED


def test_personal_fact_fields_are_never_generated(tmp_path):
    """Travel, relocation, licences, salary etc. can only come from the
    candidate — even when the profile is full of other data."""
    engine = _engine(tmp_path)
    profile = CandidateProfile(
        skills=["Java", "Python", "SQL"],
        professional_summary="Experienced developer who loves travelling and clients.",
        location="Durban",
    )
    for question in (
        "Would you be comfortable travelling to client sites?",
        "Are you willing to relocate to Johannesburg?",
        "Do you have your own vehicle?",
        "What are your salary expectations?",
    ):
        result = engine.answer(question, profile)
        assert result.answer_type == AnswerType.UNKNOWN, question
        assert result.needs_user is True, question


def test_first_and_last_name_split_from_full_profile_name():
    """Greenhouse dry run: First/Last Name fields must get name parts,
    never the full name in both."""
    first = answer_question("First Name", _graduate_profile())
    last = answer_question("Last Name", _graduate_profile())
    surname = answer_question("Surname", _graduate_profile())
    assert (first.answer, last.answer, surname.answer) == (
        "Thandi", "Mkhize", "Mkhize",
    )


def test_single_word_profile_name_leaves_surname_to_user():
    profile = CandidateProfile(
        name="Stevie", email="s@x.com", skills=["sql"],
    )
    result = answer_question("Last Name", profile)
    assert result.answer is None
    assert result.needs_user is True
