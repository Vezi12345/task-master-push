from __future__ import annotations

import json

from flask import Flask, render_template, request, jsonify

import tempfile

import config
import llm as llm_module
from agent import orchestrator
from agent.rank import RankedJob
from sources.base import Job
from candidate.cv_parser import CvExtractionError, extract_pdf_text, parse_cv
from candidate.matching import assess_readiness, match_candidate_to_job, match_candidate_to_job_detailed
from candidate.profile import CandidateProfile, Project
from candidate.storage import load_profile, save_profile
from application.models import ApplicationStatus

app = Flask(__name__)

config.ensure_data_dir()
region = config.load_region()
llm = llm_module if not llm_module.LLM_OFFLINE else None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/profile", methods=["GET"])
def profile_page():
    return render_template("profile.html")


def _serialize_ranked(ranked_items) -> list[dict]:
    profile = load_profile()
    ranked = []
    for idx, item in enumerate(ranked_items):
        job = item.job
        entry = {
            "index": idx,
            "title": job.title,
            "company": job.company,
            "location": job.location or "Not stated",
            "remote": job.remote,
            "salary_min": job.salary_min,
            "salary_text": job.salary_text or "",
            "url": job.url,
            "source": job.source,
            "score": item.score,
            "reasons": item.reasons,
            "summary": item.summary,
        }
        if profile is not None:
            match = match_candidate_to_job(profile, job)
            detailed = match_candidate_to_job_detailed(profile, job)
            readiness = assess_readiness(profile, job, match)
            entry["candidate_match"] = {
                "score": match.score,
                "matched_skills": match.matched_skills,
                "missing_skills": match.missing_skills,
                "experience_match": match.experience_match,
                "education_match": match.education_match,
                "location_match": match.location_match,
                "certification_match": match.certification_match,
                "strengths": match.strengths,
                "concerns": match.concerns,
            }
            entry["detailed_match"] = {
                "overall_score": detailed.overall_score,
                "dimensions": [d.model_dump() for d in detailed.dimensions],
                "strengths": detailed.strengths,
                "concerns": detailed.concerns,
                "remote_match": detailed.remote_match,
                "salary_match": detailed.salary_match,
                "employment_type_match": detailed.employment_type_match,
                "graduate_eligible": detailed.graduate_eligible,
            }
            entry["readiness"] = {
                "ready": readiness.ready,
                "score": readiness.score,
                "reasons": readiness.reasons,
                "blockers": readiness.blockers,
                "warnings": readiness.warnings,
            }
        ranked.append(entry)
    return ranked


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True)
    prompt = data.get("query", "").strip()
    if not prompt:
        return jsonify({"error": "Empty query"}), 400

    result = orchestrator.run_pipeline(prompt, region, llm)

    ranked = _serialize_ranked(result.ranked)

    from agent.candidate_search import summarize_related

    return jsonify({
        "query": {
            "roles": result.query.roles,
            "seniority": result.query.seniority,
            "locations": [{"city": l.city, "radius_km": l.radius_km} for l in result.query.locations],
            "remote": result.query.remote,
            "min_salary": result.query.min_salary,
            "currency": result.query.currency,
            "skills": result.query.skills,
            "keywords": result.query.keywords,
        },
        "jobs_found": len(result.jobs_found),
        "filtered_out": len(result.jobs_found) - len(ranked),
        "related": summarize_related(result.jobs_found) if not ranked else [],
        "ranked": ranked,
        "messages": result.search_messages,
        "sources": result.source_stats,
        "notes": result.notes,
        "has_cv": load_profile() is not None,
    })


@app.route("/api/keep", methods=["POST"])
def api_keep():
    data = request.get_json(force=True)
    payload = {
        "title": data.get("title", ""),
        "company": data.get("company", ""),
        "location": data.get("location", ""),
        "remote": data.get("remote", False),
        "salary_text": data.get("salary_text", ""),
        "url": data.get("url", ""),
        "source": data.get("source", ""),
    }
    records = _load_kept()
    records.append(payload)
    config.KEPT_JOBS_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonify({"ok": True, "kept_count": len(records)})


@app.route("/api/upload-cv", methods=["POST"])
def api_upload_cv():
    if "cv" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["cv"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            text = extract_pdf_text(tmp.name)
        # keep the real CV file for later uploads to employer forms
        config.ensure_data_dir()
        config.CV_FILE.write_bytes(config.Path(tmp.name).read_bytes())
    except CvExtractionError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Failed to process the uploaded file"}), 400
    try:
        profile = parse_cv(text, llm)
    except Exception:
        return jsonify({"error": "Failed to parse CV content"}), 400
    save_profile(profile)
    return jsonify({"ok": True, "profile": profile.model_dump()})


@app.route("/api/profile", methods=["GET"])
def api_profile():
    profile = load_profile()
    if profile is None:
        return jsonify({"profile": None, "completion": None})
    from candidate.completion import compute_completion, high_value_missing
    data = profile.model_dump()
    # convenience: pre-split name parts so the UI never guesses
    data["first_name"] = profile.first_name
    data["last_name"] = profile.last_name
    return jsonify({
        "profile": data,
        "completion": compute_completion(profile),
        "missing_prompts": high_value_missing(profile),
    })


@app.route("/api/profile", methods=["PUT"])
def api_profile_update():
    """Explicit user updates. This is the ONLY place sensitive/demographic
    values may be written — they are never inferred anywhere else."""
    profile = load_profile()
    if profile is None:
        profile = CandidateProfile()
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object expected"}), 400

    scalar_fields = {
        f for f in CandidateProfile.model_fields
        if f not in _PROFILE_LIST_FIELDS
    }
    updated = []
    for key, value in data.items():
        if key not in scalar_fields:
            continue  # lists/nested sections go through dedicated endpoints
        current_type = type(getattr(profile, key, ""))
        try:
            setattr(profile, key, value)
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid value for {key}"}), 400
        # keep known_fields in sync so the answer engine sees it immediately
        profile.set_known(key, str(value), "user")
        updated.append(key)

    if not updated:
        return jsonify({"error": "No updatable fields provided"}), 400
    save_profile(profile)
    return jsonify({"ok": True, "updated": sorted(updated)})


_PROFILE_LIST_FIELDS = {
    "skills", "skill_details", "education", "experience", "certifications",
    "projects", "achievements", "languages", "documents", "question_memory",
    "known_fields", "preferred_locations", "preferred_roles", "employment_types",
}


@app.route("/api/profile/completion", methods=["GET"])
def api_profile_completion():
    profile = load_profile()
    if profile is None:
        return jsonify({"completion": {"overall": 0, "sections": {}}})
    from candidate.completion import compute_completion, high_value_missing
    return jsonify({
        "completion": compute_completion(profile),
        "missing_prompts": high_value_missing(profile),
    })


@app.route("/api/profile/projects", methods=["GET"])
def api_projects_list():
    profile = load_profile()
    projects = profile.projects if profile else []
    return jsonify({"projects": [p.model_dump() for p in projects]})


@app.route("/api/profile/projects", methods=["POST"])
def api_project_add():
    profile = load_profile()
    if profile is None:
        profile = CandidateProfile()
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    description = (data.get("description") or "").strip()
    if not name and not description:
        return jsonify({"error": "Project needs a name or description"}), 400
    tech_raw = data.get("technologies") or ""
    technologies = (
        [t.strip() for t in tech_raw.split(",") if t.strip()]
        if isinstance(tech_raw, str) else [str(t).strip() for t in tech_raw]
    )
    project = Project(
        name=name, description=description,
        technologies=technologies,
        url=(data.get("url") or "").strip(),
        github_url=(data.get("github_url") or "").strip(),
        role=(data.get("role") or "").strip(),
        achievements=[a.strip() for a in (data.get("achievements") or [])
                      if str(a).strip()],
        is_personal=bool(data.get("is_personal")),
        is_academic=bool(data.get("is_academic")),
        is_work_related=bool(data.get("is_work_related")),
    )
    profile.projects.append(project)
    save_profile(profile)
    return jsonify({"ok": True, "project": project.model_dump(),
                    "index": len(profile.projects) - 1})


@app.route("/api/profile/projects/<int:index>", methods=["PUT"])
def api_project_update(index: int):
    profile = load_profile()
    if profile is None or not (0 <= index < len(profile.projects)):
        return jsonify({"error": "Project not found"}), 404
    data = request.get_json(force=True) or {}
    project = profile.projects[index]
    for key in ("name", "description", "url", "github_url", "role"):
        if key in data:
            setattr(project, key, str(data[key]).strip())
    if "technologies" in data:
        raw = data["technologies"]
        project.technologies = (
            [t.strip() for t in raw.split(",") if t.strip()]
            if isinstance(raw, str) else [str(t).strip() for t in raw]
        )
    for flag in ("is_personal", "is_academic", "is_work_related"):
        if flag in data:
            setattr(project, flag, bool(data[flag]))
    save_profile(profile)
    return jsonify({"ok": True, "project": project.model_dump()})


@app.route("/api/profile/answers", methods=["GET"])
def api_answers_list():
    profile = load_profile()
    answers = profile.question_memory if profile else []
    return jsonify({
        "answers": [
            {
                "question": a.question,
                "answer": a.answer,
                "field_key": a.field_key,
                "confidence": a.confidence,
                "evidence": a.evidence,
                "updated_at": a.updated_at,
            }
            for a in answers
        ]
    })


@app.route("/api/profile/answers", methods=["POST"])
def api_answer_save():
    """Store an application question + the user's answer. If it maps to a
    canonical profile field AND the user ticked 'save to my profile', the
    canonical field itself is updated too."""
    profile = load_profile()
    if profile is None:
        profile = CandidateProfile()
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    if not question or not answer:
        return jsonify({"error": "question and answer are required"}), 400
    field_key = (data.get("field_key") or "").strip()
    confidence = data.get("confidence") or "high"
    evidence = (data.get("evidence") or "").strip()

    profile.remember_answer(
        question, answer, field_key=field_key,
        source="user", confidence=confidence, evidence=evidence,
    )
    saved_to_profile = False
    if data.get("save_to_profile") and field_key:
        stored = profile.get_known_value(field_key)
        if stored != answer:
            saved_to_profile = True
        profile.set_known(field_key, answer, "user")
        # mirror into structured attributes where they exist
        attr_map = {
            "expected_salary": "expected_salary",
            "notice_period": "notice_period",
            "drivers_licence": "drivers_licence",
            "location": "location",
        }
        attr = attr_map.get(field_key)
        if attr and hasattr(profile, attr):
            setattr(profile, attr, answer)
    save_profile(profile)
    return jsonify({"ok": True, "saved_to_profile": saved_to_profile})


@app.route("/api/profile/online-profiles", methods=["POST"])
def api_online_profiles():
    """Verified URLs only. Values that are not http(s) URLs are rejected —
    links are never guessed and never filled with names."""
    profile = load_profile()
    if profile is None:
        profile = CandidateProfile()
    data = request.get_json(force=True) or {}
    updated = []
    for key in ("website", "linkedin", "github", "portfolio"):
        if key not in data:
            continue
        value = str(data[key]).strip()
        if value and not value.lower().startswith(("http://", "https://")):
            return jsonify({
                "error": f"{key} must be a full URL starting with https:// "
                         "(links are stored verbatim, never guessed)"
            }), 400
        setattr(profile.online_profiles, key, value)
        if value:
            profile.set_known(f"online_{key}", value, "user")
        updated.append(key)
    if not updated:
        return jsonify({"error": "No link fields provided"}), 400
    save_profile(profile)
    return jsonify({"ok": True, "updated": sorted(updated)})


@app.route("/api/profile/high-school", methods=["POST"])
def api_high_school():
    """Explicit user-supplied high-school record — nothing here is inferred."""
    from candidate.profile import HighSchoolRecord
    profile = load_profile()
    if profile is None:
        profile = CandidateProfile()
    data = request.get_json(force=True) or {}
    current = profile.high_school.model_dump() if profile.high_school else {}
    allowed = {
        "school", "province", "country", "completion_year",
        "mathematics_result", "mathematics_grade", "native_language",
        "native_language_result", "overall_result", "scoring_system",
    }
    merged = {**current}
    for key, value in data.items():
        if key in allowed:
            merged[key] = str(value).strip()
    profile.high_school = HighSchoolRecord(**merged)
    save_profile(profile)
    return jsonify({"ok": True, "high_school": profile.high_school.model_dump()})


@app.route("/api/profile/education/<int:index>/result", methods=["POST"])
def api_education_result(index: int):
    """Store the ACADEMIC RESULT separately from the qualification name."""
    profile = load_profile()
    if profile is None or not (0 <= index < len(profile.education)):
        return jsonify({"error": "Education record not found"}), 404
    data = request.get_json(force=True) or {}
    edu = profile.education[index]
    result = str(data.get("result", "")).strip()
    if not result:
        return jsonify({"error": "result is required"}), 400
    edu.result = result
    if data.get("grading_system"):
        edu.grading_system = str(data["grading_system"]).strip()
    # keep the engine's known_fields in sync (qualification != result!)
    profile.set_known("education_result", result, "user")
    save_profile(profile)
    return jsonify({"ok": True, "education": edu.model_dump()})


@app.route("/api/profile/missing", methods=["GET"])
def api_profile_missing():
    profile = load_profile()
    if profile is None:
        return jsonify({"missing_prompts": [], "overall": 0})
    from candidate.completion import high_value_missing, compute_completion
    return jsonify({
        "missing_prompts": high_value_missing(profile),
        "overall": compute_completion(profile)["overall"],
    })


@app.route("/api/agent", methods=["POST"])
def api_agent():
    data = request.get_json(force=True)
    prompt = data.get("message", "").strip()
    if not prompt:
        return jsonify({"error": "Empty message"}), 400

    from agent.orchestrator import JobApplicationAgent
    agent = JobApplicationAgent(region, llm)
    result = agent.process_input(prompt)

    response = {
        "state": result.state.value,
        "messages": [
            {"role": m.role, "content": m.content, "type": m.message_type}
            for m in result.messages
        ],
        "has_profile": result.profile is not None,
        "profile_summary": result.profile.summary if result.profile else None,
        "error": result.error,
        "notes": result.notes,
    }

    if result.query:
        response["query"] = {
            "roles": result.query.roles,
            "seniority": result.query.seniority,
            "locations": [{"city": l.city, "radius_km": l.radius_km} for l in result.query.locations],
            "remote": result.query.remote,
            "min_salary": result.query.min_salary,
            "currency": result.query.currency,
            "skills": result.query.skills,
            "keywords": result.query.keywords,
        }

    if result.ranked:
        response["ranked"] = [
            {
                "index": idx,
                "title": item.job.title,
                "company": item.job.company,
                "location": item.job.location or "Not stated",
                "remote": item.job.remote,
                "salary_min": item.job.salary_min,
                "salary_text": item.job.salary_text or "",
                "url": item.job.url,
                "source": item.job.source,
                "score": item.score,
                "reasons": item.reasons,
                "summary": item.summary,
            }
            for idx, item in enumerate(result.ranked)
        ]
        response["jobs_found"] = len(result.jobs_found)

    if result.matched_jobs:
        response["matched_jobs"] = [
            {
                "title": m["job"].title,
                "company": m["job"].company,
                "location": m["job"].location or "Not stated",
                "remote": m["job"].remote,
                "url": m["job"].url,
                "source": m["job"].source,
                "job_preference_score": m["rank"].score,
                "candidate_match_score": m["candidate_match"].score,
                "readiness_score": m["readiness"].score,
                "matched_skills": m["candidate_match"].matched_skills,
                "missing_skills": m["candidate_match"].missing_skills,
                "strengths": m["candidate_match"].strengths,
                "concerns": m["candidate_match"].concerns,
                "readiness_ready": m["readiness"].ready,
            }
            for m in result.matched_jobs
        ]

    if result.applications:
        response["applications"] = [a.to_preview() for a in result.applications]

    if result.missing_information:
        response["missing_information"] = [m.model_dump() for m in result.missing_information]

    if result.application_summaries:
        response["application_summaries"] = result.application_summaries

    return jsonify(response)


@app.route("/api/agent/answer", methods=["POST"])
def api_agent_answer():
    data = request.get_json(force=True)
    answers = data.get("answers", {})
    if not answers:
        return jsonify({"error": "No answers provided"}), 400

    from agent.orchestrator import JobApplicationAgent
    agent = JobApplicationAgent(region, llm)
    result = agent.provide_answers(answers)

    return jsonify({
        "state": result.state.value,
        "messages": [
            {"role": m.role, "content": m.content, "type": m.message_type}
            for m in result.messages
        ],
        "applications": [a.to_preview() for a in result.applications] if result.applications else [],
    })


@app.route("/api/applications", methods=["GET"])
def api_applications():
    from application.tracker import ApplicationTracker
    tracker = ApplicationTracker()
    apps = tracker.all()
    summary = tracker.get_summary()
    return jsonify({
        "applications": [a.to_preview() for a in apps],
        "summary": summary,
    })


@app.route("/api/apply", methods=["POST"])
def api_apply():
    data = request.get_json(force=True)
    job_data = data.get("job", {})
    if not job_data.get("title"):
        return jsonify({"error": "No job data provided"}), 400

    profile = load_profile()
    if profile is None:
        return jsonify({"error": "Please upload your CV first"}), 400

    job = Job(
        id=job_data.get("id", ""),
        title=job_data.get("title", ""),
        company=job_data.get("company", ""),
        location=job_data.get("location", ""),
        url=job_data.get("url", ""),
        source=job_data.get("source", ""),
        description=job_data.get("description", ""),
        salary_min=job_data.get("salary_min"),
        salary_text=job_data.get("salary_text", ""),
        remote=job_data.get("remote", False),
    )

    match_result = match_candidate_to_job(profile, job)
    detailed = match_candidate_to_job_detailed(profile, job)
    readiness = assess_readiness(profile, job, match_result)

    try:
        app = orchestrator.prepare_application(
            job=job,
            candidate_match=match_result,
            detailed_match=detailed,
            readiness=readiness,
            job_preference_score=job_data.get("score", 0),
            llm=llm,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"ok": True, "application": app.to_preview()})


@app.route("/api/applications/<app_id>", methods=["GET"])
def api_application_detail(app_id):
    from application.tracker import ApplicationTracker
    tracker = ApplicationTracker()
    app_obj = tracker.get(app_id)
    if not app_obj:
        app_obj = tracker.find_by_partial_id(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404
    return jsonify({"application": app_obj.to_preview()})


@app.route("/api/applications/<app_id>/approve", methods=["POST"])
def api_approve_application(app_id):
    """Legacy approve endpoint — routes through the centralized submission
    service for safety checks (consent gates, confirmation detection,
    mocked-test tagging).  Prefer POST /api/applications/<id>/confirm for
    new integrations."""
    from application.tracker import ApplicationTracker
    from application.submission import ApplicationAutomationError
    from application.browser import BrowserError, BrowserUnavailable
    from application.models import ApplicationStatus as S

    tracker = ApplicationTracker()
    app_obj = tracker.get(app_id)
    if not app_obj:
        app_obj = tracker.find_by_partial_id(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404

    if app_obj.status not in (S.AWAITING_APPROVAL, S.READY_FOR_REVIEW):
        return jsonify({
            "error": f"Application is {app_obj.status.value}, not awaiting approval or ready for review"
        }), 400

    # Transition legacy AWAITING_APPROVAL → READY_FOR_REVIEW so the
    # centralized service's gate accepts the application.
    if app_obj.status == S.AWAITING_APPROVAL:
        from application.lifecycle import transition
        transition(app_obj, S.READY_FOR_REVIEW, "Legacy approve endpoint routed to centralized service")

    service = _submission_service()

    from application.browser import open_driver
    driver = open_driver(prefer_headless=False)
    try:
        # Legacy applications may lack form_analysis and fill_plan.
        # Use reprepare() to navigate to the real page, analyse the form,
        # build a proper fill plan with CSS selectors, and update
        # app_obj.form_analysis so confirm_and_submit() can locate the
        # submit button.
        plan = None
        if not app_obj.form_analysis and app_obj.application_url:
            profile = load_profile()
            if profile is not None:
                plan = service.reprepare(app_obj, profile, driver)

        if plan is None and app_obj.fill_plan:
            from application.form_filler import FillPlan, PlannedAnswer
            plan = FillPlan(entries=[
                PlannedAnswer(**e) for e in app_obj.fill_plan
            ])

        if plan is None:
            from application.form_filler import FillPlan, PlannedAnswer
            plan_entries = []
            for q, v in (app_obj.answers or {}).items():
                plan_entries.append(PlannedAnswer(
                    question=q, value=v, answer_type="verified",
                    source="user", needs_user=False,
                ))
            plan = FillPlan(entries=plan_entries)

        result = service.confirm_and_submit(
            app_obj, tracker, driver,
            consent_granted=True,
            user_answers={},
            plan=plan,
            force=True,
        )
    except BrowserUnavailable as exc:
        app_obj.update_status(S.REQUIRES_USER_ACTION)
        app_obj.error = str(exc)
        tracker.update(app_obj)
        return jsonify({"ok": True, "application": app_obj.to_preview()})
    except (BrowserError, ApplicationAutomationError) as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        try:
            driver.close()
        except Exception:
            pass

    tracker.update(result)
    return jsonify({"ok": True, "application": result.to_preview()})


@app.route("/api/applications/<app_id>/cancel", methods=["POST"])
def api_cancel_application(app_id):
    from application.tracker import ApplicationTracker

    tracker = ApplicationTracker()
    app_obj = tracker.get(app_id)
    if not app_obj:
        app_obj = tracker.find_by_partial_id(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404

    app_obj.update_status(ApplicationStatus.WITHDRAWN)
    tracker.update(app_obj)
    return jsonify({"ok": True, "application": app_obj.to_preview()})


# ---------------------------------------------------------------------------
# Real-application submission pipeline (review → confirm → submit → confirm email)
# ---------------------------------------------------------------------------

def _get_app_or_none(app_id):
    from application.tracker import ApplicationTracker
    tracker = ApplicationTracker()
    app_obj = tracker.get(app_id) or tracker.find_by_partial_id(app_id)
    return tracker, app_obj


def _submission_service():
    from application.submission import ApplicationAutomationService
    cv_path = config.CV_FILE if config.CV_FILE.exists() else None
    letter_path = config.COVER_LETTER_FILE if config.COVER_LETTER_FILE.exists() else None
    return ApplicationAutomationService(cv_path=cv_path, cover_letter_path=letter_path, llm=llm)


def _mailbox_connector():
    """Email backend chosen by config.resolve_email_backend():
    gmail_api when an OAuth client secret is present, else IMAP."""
    if config.resolve_email_backend() == "gmail_api":
        from application.gmail_api import GmailApiMailboxConnector
        return GmailApiMailboxConnector()
    from application.email_confirmation import ImapMailboxConnector
    return ImapMailboxConnector()


@app.route("/applications")
def applications_page():
    return render_template("applications.html")


@app.route("/api/applications/<app_id>/review", methods=["POST"])
def api_application_review(app_id):
    from application.submission import ApplicationAutomationError

    tracker, app_obj = _get_app_or_none(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404
    try:
        review = _submission_service().build_review(app_obj)
    except ApplicationAutomationError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(review.to_dict())


@app.route("/api/applications/<app_id>/confirm", methods=["POST"])
def api_application_confirm(app_id):
    """EXPLICIT user confirmation gate — nothing is submitted without it."""
    from application.browser import open_driver
    from application.models import ApplicationStatus as S
    from application.submission import (
        ApplicationAutomationError,
        BrowserError,
        BrowserUnavailable,
    )

    data = request.get_json(force=True) or {}
    consent_granted = bool(data.get("consent_granted"))
    user_answers = data.get("answers", {}) or {}

    # "save this answer to my profile" — persist BEFORE the browser flow so
    # the answers survive even if the browser step fails
    save_to_profile = data.get("save_to_profile") or []
    if isinstance(save_to_profile, list) and save_to_profile:
        profile_for_save = load_profile() or CandidateProfile()
        from application.answer_engine import classify_question
        for item in save_to_profile:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if not question or not answer:
                continue
            field_key = (item.get("field_key") or "").strip() \
                or classify_question(question).field_key
            profile_for_save.remember_answer(
                question, answer, field_key=field_key,
                source="user", confidence="high",
                evidence="Answered during application review",
            )
            if item.get("also_update_profile_field") and field_key:
                stored = profile_for_save.get_known_value(field_key)
                if stored != answer:
                    profile_for_save.set_known(field_key, answer, "user")
        save_profile(profile_for_save)

    tracker, app_obj = _get_app_or_none(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404
    if app_obj.status != S.READY_FOR_REVIEW:
        return jsonify({
            "error": f"Application is {app_obj.status.value}, not ready_for_review"
        }), 400

    profile = load_profile()
    if profile is None:
        return jsonify({"error": "Candidate profile not found — upload your CV first"}), 400

    service = _submission_service()
    driver = open_driver(prefer_headless=False)  # visible browser for challenges
    try:
        plan = service.reprepare(app_obj, profile, driver)
        result = service.confirm_and_submit(
            app_obj, tracker, driver,
            consent_granted=consent_granted,
            user_answers=user_answers,
            plan=plan,
        )
    except BrowserUnavailable as exc:
        app_obj.update_status(S.REQUIRES_USER_ACTION)
        app_obj.error = str(exc)
        tracker.update(app_obj)
        return jsonify({"ok": True, "application": app_obj.to_preview()})
    except (BrowserError, ApplicationAutomationError) as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        try:
            driver.close()
        except Exception:
            pass
    tracker.update(result)
    return jsonify({"ok": True, "application": result.to_preview()})


@app.route("/api/applications/<app_id>/check-confirmation", methods=["POST"])
def api_application_check_confirmation(app_id):
    from application.email_confirmation import (
        MailboxUnavailable,
        await_confirmation,
    )

    tracker, app_obj = _get_app_or_none(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404
    connector = _mailbox_connector()
    try:
        updated = await_confirmation(app_obj, tracker, connector=connector)
    except MailboxUnavailable as exc:
        return jsonify({
            "ok": False,
            "error": f"Mailbox unreachable: {exc}",
            "application": app_obj.to_preview(),
        })
    return jsonify({"ok": True, "application": updated.to_preview()})


@app.route("/api/profile/memory", methods=["GET"])
def api_profile_memory():
    """Persistent candidate profile / answer memory with provenance."""
    from application.profile_memory import ProfileMemoryService
    return jsonify(ProfileMemoryService().snapshot())


@app.route("/api/profile/memory/missing", methods=["GET"])
def api_profile_memory_missing():
    from application.profile_memory import ProfileMemoryService
    return jsonify({"questions": ProfileMemoryService().missing_questions()})


@app.route("/api/profile/memory/answer", methods=["POST"])
def api_profile_memory_answer():
    """Save a user answer. Returns HTTP 409 + conflict details when it
    contradicts an existing VERIFIED answer (nothing is overwritten)."""
    from application.profile_memory import ProfileMemoryService
    data = request.get_json(force=True) or {}
    outcome = ProfileMemoryService().save_user_answer(
        str(data.get("key") or ""), str(data.get("answer") or ""),
        question=str(data.get("question") or ""),
    )
    if not outcome.get("ok"):
        return jsonify(outcome), 400
    return jsonify(outcome), (409 if outcome.get("conflict") else 200)


@app.route("/api/profile/memory/conflict/<conflict_id>/resolve", methods=["POST"])
def api_profile_memory_resolve(conflict_id):
    from application.profile_memory import ProfileMemoryService
    data = request.get_json(force=True) or {}
    outcome = ProfileMemoryService().resolve_conflict(
        conflict_id, str(data.get("choice") or ""),
    )
    return jsonify(outcome), (200 if outcome.get("ok") else 400)


@app.route("/api/email/status", methods=["GET"])
def api_email_status():
    """Which email backend is active and whether it is ready."""
    from application.email_confirmation import ImapMailboxConnector
    backend = config.resolve_email_backend()
    info = {"backend": backend}
    if backend == "gmail_api":
        from application.gmail_api import GmailApiMailboxConnector
        conn = GmailApiMailboxConnector()
        info.update({
            "authorized": conn.is_configured(),
            "client_secret_found": config.GMAIL_CLIENT_SECRET_FILE.exists(),
            "token_file": str(config.GMAIL_TOKEN_FILE),
            "connect_hint": "POST /api/email/connect (opens browser) "
                            "or run: python cli.py gmail-auth",
        })
    else:
        imap = ImapMailboxConnector()
        info.update({
            "configured": imap.is_configured(),
            "host": imap.host,
            "blocked_note": "IMAP (TCP 993) is blocked on some networks — "
                            "drop an OAuth client secret into data/ to use "
                            "the gmail_api backend instead",
        })
    return jsonify(info)


@app.route("/api/email/connect", methods=["POST"])
def api_email_connect():
    """One-time interactive Gmail OAuth consent. Blocks until the browser
    redirect lands back on the local callback server (max ~5 min)."""
    if config.resolve_email_backend() != "gmail_api":
        return jsonify({
            "error": "gmail_api backend is not active — save the OAuth "
                     "client-secret JSON to data/gmail_client_secret.json "
                     "or set TASK_MASTER_EMAIL_BACKEND=gmail_api"
        }), 400
    from application.gmail_api import run_authorization_flow
    try:
        token = run_authorization_flow()
    except SystemExit as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "email_address": _mailbox_connector().address})


# ---------------------------------------------------------------------------
# autonomous run — the explicit POST below IS the user's consent gate
# ---------------------------------------------------------------------------

_chat_agent = None

_CHAT_HELP = (
    "I'm your job-search agent. Here's what you can ask me:\n"
    "• Search — \"find entry-level software developer jobs in East London\"\n"
    "• Apply — \"apply to the best 3 matching jobs\" or \"apply where I match at least 80%\"\n"
    "• Track — \"show my applications\" or \"what needs my attention?\"\n"
    "• Decide — \"approve all applications\" / \"cancel application <id>\"\n"
    "• Autopilot — press the Run agent button and I'll search, filter and prepare applications on my own."
)


def _get_session_store():
    from application.session import SessionStore
    return SessionStore()


def _get_chat_agent(session_id=None):
    """Return (agent, SessionState) for a chat interaction.

    A session ties an agent's durable intent state (last query, pending
    applications, questions it is waiting on) to a client-supplied
    idempotent session_id so a multi-turn flow survives server restarts and
    can be resumed from a new browser tab."""
    from agent.orchestrator import JobApplicationAgent
    from application.session import SessionStore

    store = SessionStore()
    state = store.load(session_id) if session_id else None
    if state is None:
        state = store.create()
    agent = JobApplicationAgent(region, llm, session=state)
    return agent, state


def _chat_payload(result) -> dict:
    from agent.candidate_search import summarize_related

    reply_msgs = [
        m for m in result.messages
        if m.role == "agent" and m.message_type not in ("user_input",)
    ]
    reply = "\n".join(m.content for m in reply_msgs).strip()
    state_value = result.state.value if hasattr(result.state, "value") else str(result.state)

    payload = {
        "state": state_value,
        "reply": reply,
        "error": result.error,
        "has_cv": result.profile is not None,
        "awaiting_approval": state_value == "awaiting_approval",
        "needs_answers": bool(result.missing_information),
        "missing_information": [
            {"field_key": m.field_key, "question": m.question}
            for m in result.missing_information
        ],
        "applications": [a.to_preview() for a in result.applications],
        "application_summaries": result.application_summaries,
    }

    if result.ranked or result.jobs_found:
        ranked = _serialize_ranked(result.ranked)
        payload.update({
            "jobs_found": len(result.jobs_found),
            "filtered_out": len(result.jobs_found) - len(ranked),
            "related": summarize_related(result.jobs_found) if not ranked else [],
            "ranked": ranked,
            "notes": result.notes,
        })
    return payload


def _resume_payload(state) -> dict:
    """Payload describing a paused session so the UI can resume it without
    asking the user to re-type a full command."""
    from agent.orchestrator import JobApplicationAgent
    mode = state.mode
    reply = ""
    apps = []

    if mode == "awaiting_approval":
        ids = state.pending_application_ids
        from application.tracker import ApplicationTracker
        tracker = ApplicationTracker()
        apps = [tracker.get(aid) for aid in ids]
        apps = [a for a in apps if a is not None]
        reply = _format_resume_approval(apps)
    elif mode == "autonomous_paused":
        auto = state.autonomous or {}
        questions = auto.get("questions") or []
        reply = ("An autonomous run was paused because it needs information "
                 "from you:\n" +
                 "\n".join(f"  - {q}" for q in questions) +
                 "\n\nReply with your answers (POST /api/autonomous/resume "
                 "with session_id and answers) and I'll continue the run.")
        return {
            "state": "autonomous_paused",
            "reply": reply,
            "error": None,
            "has_cv": True,
            "awaiting_approval": False,
            "needs_answers": bool(questions),
            "missing_information": [
                {"field_key": "", "question": q} for q in questions
            ],
            "applications": [],
            "application_summaries": [],
            "autonomous_paused": True,
            "resumed": True,
        }
    elif mode == "needs_information":
        qs = state.questions_by_app or state.pending_questions or {}
        flattened: list[dict] = []
        for questions in qs.values():
            for q in questions:
                flattened.append({"field_key": q.get("field_key", ""),
                                  "question": q.get("question", "")})
        reply = "You have pending questions I asked earlier. Reply with your answers and I'll continue."
        return {
            "state": "needs_information",
            "reply": reply,
            "error": None,
            "has_cv": True,
            "awaiting_approval": False,
            "needs_answers": True,
            "missing_information": flattened,
            "applications": [],
            "application_summaries": [],
            "resumed": True,
        }

    return {
        "state": mode,
        "reply": reply or _CHAT_HELP,
        "error": None,
        "has_cv": True,
        "awaiting_approval": mode == "awaiting_approval",
        "needs_answers": False,
        "missing_information": [],
        "applications": [a.to_preview() for a in apps] if apps else [],
        "application_summaries": [],
        "resumed": True,
    }


def _format_resume_approval(apps) -> str:
    from application.models import ApplicationStatus
    if not apps:
        return "No applications are currently pending approval."
    ready = [a for a in apps if a.status in (
        ApplicationStatus.AWAITING_APPROVAL, ApplicationStatus.READY_FOR_REVIEW)]
    if not ready:
        return "No applications are currently pending approval."
    parts = [f"{len(ready)} application(s) are ready for your review:"]
    for i, app in enumerate(ready, 1):
        parts.append(f"  {i}. {app.job_title} - {app.job_company}")
        parts.append(f"     Match: {app.candidate_match_score}%")
    parts.append("Reply 'approve' to submit them all, or 'cancel' to abort.")
    return "\n".join(parts)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Conversational endpoint: one user turn in, one structured agent turn out."""
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or data.get("sessionId")

    # A non-command turn on a paused session resumes the pending flow.
    if not message:
        if session_id:
            from application.session import SessionStore
            state = SessionStore().load(session_id)
            if state is not None and state.mode in (
                "awaiting_approval", "needs_information", "autonomous_paused"):
                payload = _resume_payload(state)
                payload["session_id"] = session_id
                return jsonify(payload)
        return jsonify({"error": "Empty message"}), 400

    agent_obj, state = _get_chat_agent(session_id)
    session_id = state.session_id
    try:
        result = agent_obj.process_input(message)
    except Exception as exc:
        return jsonify({
            "state": "failed",
            "reply": f"Something went wrong handling that: {exc}",
            "error": str(exc),
            "session_id": session_id,
        }), 500

    # persist the session that process_input mutated
    from application.session import SessionStore
    SessionStore().save(state)

    payload = _chat_payload(result)
    payload["session_id"] = session_id
    if payload["error"]:
        base = payload["reply"] or payload["error"]
        payload["reply"] = f"{base}\n\n{_CHAT_HELP}"
    elif not payload["reply"]:
        payload["reply"] = _CHAT_HELP
    return jsonify(payload)


@app.route("/api/chat/answers", methods=["POST"])
def api_chat_answers():
    """Answer outstanding agent questions mid-conversation."""
    data = request.get_json(force=True, silent=True) or {}
    answers = data.get("answers")
    if not isinstance(answers, dict) or not answers:
        return jsonify({"error": "No answers provided"}), 400

    session_id = data.get("session_id") or data.get("sessionId")
    agent_obj, state = _get_chat_agent(session_id)
    session_id = state.session_id
    result = agent_obj.provide_answers({str(k): str(v) for k, v in answers.items()})
    from application.session import SessionStore
    SessionStore().save(state)
    payload = _chat_payload(result)
    payload["session_id"] = session_id
    return jsonify(payload)


@app.route("/api/autonomous/run", methods=["POST"])
def api_autonomous_run():
    """Start one autonomous search→apply run. Started ONLY by an explicit
    user action; the request body may carry a free-text query, dry_run, and
    an optional session_id.  When a session_id is supplied the run becomes
    resumable: if a job needs an answer the agent may not guess, the run
    pauses (rather than silently skipping) and records the pending question
    on the session for /api/autonomous/resume to continue."""
    from application.autonomy import AutonomyPolicy, run_autonomous_job_search
    from application.outcome_learning import OutcomeStore
    from application.session import SessionStore
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id") or data.get("sessionId")
    session_state = None
    if session_id:
        store = SessionStore()
        session_state = store.load(session_id) or store.create()
    try:
        policy = None
        if data.get("min_score") is not None:
            policy = AutonomyPolicy(min_score=max(0, min(100, int(data["min_score"]))))
        report = run_autonomous_job_search(
            query_text=(data.get("query") or "").strip(),
            dry_run=bool(data.get("dry_run")),
            policy=policy,
            session=session_state,
            pause_on_input=bool(data.get("pause_on_input", True)),
            outcome_store=OutcomeStore(config.OUTCOMES_FILE),
        )
    except Exception as exc:
        return jsonify({"error": f"Autonomous run failed: {exc}"}), 500

    if session_state is not None:
        SessionStore().save(session_state)
        report["session_id"] = session_state.session_id
    return jsonify(report)


@app.route("/api/autonomous/resume", methods=["POST"])
def api_autonomous_resume():
    """Resume a paused autonomous run after the user answers its questions."""
    from application.autonomy import resume_autonomous_job_search
    from application.outcome_learning import OutcomeStore
    from application.session import SessionStore
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required to resume"}), 400
    answers = data.get("answers")
    if not isinstance(answers, dict) or not answers:
        return jsonify({"error": "No answers provided"}), 400

    store = SessionStore()
    session_state = store.load(session_id)
    if session_state is None or not (session_state.autonomous or {}).get("paused_app_id"):
        return jsonify({"error": "No paused autonomous run found on this session"}), 400
    try:
        report = resume_autonomous_job_search(
            session_state,
            {str(k): str(v) for k, v in answers.items()},
            outcome_store=OutcomeStore(config.OUTCOMES_FILE),
        )
    except Exception as exc:
        return jsonify({"error": f"Resume failed: {exc}"}), 500
    SessionStore().save(session_state)
    report["session_id"] = session_state.session_id
    return jsonify(report)


@app.route("/api/autonomous/report", methods=["GET"])
def api_autonomous_report():
    """Latest saved run report (or empty object)."""
    if not config.AUTONOMOUS_RUNS_DIR.exists():
        return jsonify({})
    reports = sorted(config.AUTONOMOUS_RUNS_DIR.glob("run_*.json"))
    if not reports:
        return jsonify({})
    try:
        return jsonify(json.loads(reports[-1].read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return jsonify({})


def _load_kept() -> list[dict]:
    if not config.KEPT_JOBS_FILE.exists():
        return []
    try:
        return json.loads(config.KEPT_JOBS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
