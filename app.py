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
from candidate.storage import load_profile, save_profile
from application.models import ApplicationStatus

app = Flask(__name__)

config.ensure_data_dir()
region = config.load_region()
llm = llm_module if not llm_module.LLM_OFFLINE else None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True)
    prompt = data.get("query", "").strip()
    if not prompt:
        return jsonify({"error": "Empty query"}), 400

    result = orchestrator.run_pipeline(prompt, region, llm)
    profile = load_profile()

    ranked = []
    for idx, item in enumerate(result.ranked):
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
        "ranked": ranked,
        "messages": result.search_messages,
        "notes": result.notes,
        "has_cv": profile is not None,
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
        return jsonify({"profile": None})
    return jsonify({"profile": profile.model_dump()})


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
    from application.tracker import ApplicationTracker
    from application.form_filler import get_platform

    tracker = ApplicationTracker()
    app_obj = tracker.get(app_id)
    if not app_obj:
        app_obj = tracker.find_by_partial_id(app_id)
    if not app_obj:
        return jsonify({"error": "Application not found"}), 404

    if app_obj.status != ApplicationStatus.AWAITING_APPROVAL:
        return jsonify({"error": f"Application is {app_obj.status.value}, not awaiting approval"}), 400

    platform = get_platform(app_obj.job_url)
    try:
        result = platform.fill_and_submit(
            app_obj.job_url,
            fields=app_obj.answers,
            files={},
        )
        if result.success:
            app_obj.update_status(ApplicationStatus.SUBMITTED)
            app_obj.submitted = True
            app_obj.submission_url = result.application_url or app_obj.job_url
            app_obj.submission_time = app_obj.updated_at
            app_obj.date_submitted = app_obj.updated_at
            app_obj.submission_platform = platform.name
            app_obj.confirmation_id = result.confirmation_id
        else:
            if result.requires_human_input:
                app_obj.update_status(ApplicationStatus.MANUAL_ACTION_REQUIRED)
                app_obj.notes.append("Requires manual submission — browser automation could not complete")
            else:
                app_obj.update_status(ApplicationStatus.FAILED)
            app_obj.errors.append(result.error)
    except Exception as exc:
        app_obj.update_status(ApplicationStatus.FAILED)
        app_obj.errors.append(str(exc))

    tracker.update(app_obj)
    return jsonify({"ok": True, "application": app_obj.to_preview()})


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


def _load_kept() -> list[dict]:
    if not config.KEPT_JOBS_FILE.exists():
        return []
    try:
        return json.loads(config.KEPT_JOBS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    app.run(debug=True, port=5000)
