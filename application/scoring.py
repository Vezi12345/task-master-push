from __future__ import annotations

from .models import Application


def application_priority_score(
    job_preference: int,
    candidate_match: int,
    readiness: int,
) -> int:
    w_job = 0.40
    w_match = 0.40
    w_ready = 0.20
    raw = job_preference * w_job + candidate_match * w_match + readiness * w_ready
    return max(0, min(100, round(raw)))


def compute_all_scores(app: Application) -> Application:
    app.application_priority = application_priority_score(
        app.job_preference_score,
        app.candidate_match_score,
        app.readiness_score,
    )
    return app


def rank_applications(applications: list[Application]) -> list[Application]:
    return sorted(
        applications,
        key=lambda a: a.application_priority,
        reverse=True,
    )
