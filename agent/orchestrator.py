from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from application.models import Application, ApplicationStatus, MissingInfo
from application.question_engine import AnswerStore, QuestionEngine
from application.documents import ApplicationDocuments
from application.tracker import ApplicationTracker
from application.scoring import compute_all_scores, rank_applications
from application.form_filler import get_platform
from candidate.matching import (
    ApplicationReadiness,
    CandidateMatch,
    DetailedMatch,
    assess_readiness,
    match_candidate_to_job,
    match_candidate_to_job_detailed,
)
from candidate.profile import CandidateProfile
from candidate.storage import load_profile, save_profile
from agent.parse_intent import JobQuery, UserIntent, parse_user_intent
from agent.rank import RankedJob, rank_jobs
from agent.search import search_jobs
from sources.base import Job

_SEARCH_VERB_RE = re.compile(
    r"\b(find|search|look(?:ing)? for|show me|hunt|hiring|vacanc\w*|jobs?|"
    r"positions?|openings?|internship|graduate|apply)\b",
    re.IGNORECASE,
)


class AgentState(str, Enum):
    IDLE = "idle"
    RECEIVED = "received"
    PARSING_CV = "parsing_cv"
    UNDERSTANDING_REQUEST = "understanding_request"
    SEARCHING = "searching"
    MATCHING = "matching"
    RANKING = "ranking"
    SELECTING = "selecting"
    PREPARING_APPLICATION = "preparing_application"
    NEEDS_INFORMATION = "needs_information"
    AWAITING_APPROVAL = "awaiting_approval"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class AgentMessage:
    role: str = "agent"
    content: str = ""
    message_type: str = "text"
    data: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: JobQuery
    search_messages: list[str] = field(default_factory=list)
    jobs_found: list[Job] = field(default_factory=list)
    ranked: list[RankedJob] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    state: AgentState = AgentState.IDLE
    messages: list[AgentMessage] = field(default_factory=list)
    profile: Optional[CandidateProfile] = None
    query: Optional[JobQuery] = None
    search_messages: list[str] = field(default_factory=list)
    jobs_found: list[Job] = field(default_factory=list)
    ranked: list[RankedJob] = field(default_factory=list)
    matched_jobs: list[dict] = field(default_factory=list)
    applications: list[Application] = field(default_factory=list)
    missing_information: list[MissingInfo] = field(default_factory=list)
    application_summaries: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None


class JobApplicationAgent:
    def __init__(
        self,
        region: dict,
        llm=None,
        answer_store_path=None,
    ) -> None:
        self.region = region
        self.llm = llm
        self.state = AgentState.IDLE
        self.profile: Optional[CandidateProfile] = None
        self.last_query: Optional[JobQuery] = None
        self.last_ranked: list[RankedJob] = []
        self.last_matched: list[dict] = []
        self.pending_applications: list[Application] = []
        self.messages: list[AgentMessage] = []

        self.question_engine = QuestionEngine(
            AnswerStore(answer_store_path) if answer_store_path else AnswerStore()
        )
        self.document_engine = ApplicationDocuments(llm)
        self.tracker = ApplicationTracker()

    def _msg(self, content: str, message_type: str = "text", data: dict = None) -> AgentMessage:
        msg = AgentMessage(
            role="agent",
            content=content,
            message_type=message_type,
            data=data or {},
        )
        self.messages.append(msg)
        return msg

    def process_input(self, user_input: str) -> AgentResult:
        # each conversational turn stands alone: never replay earlier turns
        self.messages = []
        self.state = AgentState.RECEIVED
        self._msg(user_input, "user_input")

        self.profile = load_profile()

        intent = parse_user_intent(user_input, self.region, self.llm)

        if intent.intent_type == "show_applications":
            return self._handle_show_applications()
        if intent.intent_type == "needs_attention":
            return self._handle_needs_attention()
        if intent.intent_type == "approve":
            return self._handle_approve(intent.target_id)
        if intent.intent_type == "cancel":
            return self._handle_cancel(intent.target_id)
        if intent.intent_type == "apply":
            return self._handle_apply(intent)
        if intent.intent_type == "search":
            query = intent.search_query
            message_text = intent.message or ""
            looks_like_search = bool(
                query is not None and (query.roles or query.skills)
            ) or bool(_SEARCH_VERB_RE.search(message_text))
            if not looks_like_search:
                return AgentResult(
                    state=self.state,
                    messages=list(self.messages),
                    profile=self.profile,
                    error="I didn't understand that. Try searching for jobs or managing applications.",
                )
            return self._handle_search(intent)

        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            error="I didn't understand that. Try searching for jobs or managing applications.",
        )

    def _handle_search(self, intent: UserIntent) -> AgentResult:
        query = intent.search_query
        if query is None:
            return AgentResult(
                state=self.state,
                messages=list(self.messages),
                profile=self.profile,
                error="Could not parse your search request.",
            )

        self.state = AgentState.UNDERSTANDING_REQUEST
        self.last_query = query
        self._msg(self._format_query_understood(query))

        self.state = AgentState.SEARCHING
        jobs, search_messages = search_jobs(query, self.region)
        self._msg(f"Found {len(jobs)} jobs.")
        for msg in search_messages:
            self._msg(msg, "search_detail")

        self.state = AgentState.RANKING
        ranked = rank_jobs(jobs, query, self.llm)
        self.last_ranked = ranked
        if ranked:
            self._msg(f"{len(ranked)} jobs match after filtering.")
        else:
            self._msg("No relevant real jobs were found for this search.")

        matched = []
        if self.profile is not None:
            self.state = AgentState.MATCHING
            matched = self._match_jobs_to_candidate(ranked)
            self.last_matched = matched

        self.state = AgentState.COMPLETED
        self._msg("Search complete.")

        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            query=query,
            search_messages=search_messages,
            jobs_found=jobs,
            ranked=ranked,
            matched_jobs=matched,
            notes=self._build_notes(),
        )

    def _handle_apply(self, intent: UserIntent) -> AgentResult:
        if self.profile is None:
            return AgentResult(
                state=self.state,
                messages=list(self.messages),
                error="Please upload your CV first before applying to jobs.",
            )

        query = intent.search_query
        if query is None:
            query = self.last_query

        if query is None:
            return AgentResult(
                state=self.state,
                messages=list(self.messages),
                profile=self.profile,
                error="No search criteria available. Please search for jobs first.",
            )

        self.last_query = query
        self.state = AgentState.SEARCHING
        jobs, search_messages = search_jobs(query, self.region)
        self._msg(f"Found {len(jobs)} jobs.")

        self.state = AgentState.RANKING
        ranked = rank_jobs(jobs, query, self.llm)
        if not ranked:
            self._msg("No relevant real jobs were found for this search.")
            return AgentResult(
                state=AgentState.COMPLETED,
                messages=list(self.messages),
                profile=self.profile,
                query=query,
                search_messages=search_messages,
                jobs_found=jobs,
                ranked=ranked,
                notes=self._build_notes(),
            )

        self.state = AgentState.MATCHING
        matched = self._match_jobs_to_candidate(ranked)
        self.last_matched = matched
        self.last_ranked = ranked

        self.state = AgentState.SELECTING
        selected = self._select_best_applications(
            matched,
            intent.apply_count,
            intent.min_match_score,
        )

        self.state = AgentState.PREPARING_APPLICATION
        applications, missing = self._prepare_applications(selected)
        self.pending_applications = applications

        if missing:
            self.state = AgentState.NEEDS_INFORMATION
            self._msg(self._format_missing_info(missing))
        else:
            self.state = AgentState.AWAITING_APPROVAL

        self._msg(self._format_approval_request(applications))

        self.state = AgentState.AWAITING_APPROVAL
        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            query=query,
            search_messages=search_messages,
            jobs_found=jobs,
            ranked=ranked,
            matched_jobs=matched,
            applications=applications,
            missing_information=missing,
            notes=self._build_notes(),
        )

    def _handle_show_applications(self) -> AgentResult:
        apps = self.tracker.all()
        summaries = [a.to_preview() for a in apps]
        if not apps:
            self._msg("No applications yet.")
        else:
            self._msg(f"You have {len(apps)} application(s).")
        self.state = AgentState.COMPLETED
        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            application_summaries=summaries,
        )

    def _handle_needs_attention(self) -> AgentResult:
        needs = self.tracker.needs_attention()
        summaries = [a.to_preview() for a in needs]
        if not needs:
            self._msg("No applications need your attention.")
        else:
            self._msg(f"{len(needs)} application(s) need your attention.")
        self.state = AgentState.COMPLETED
        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            application_summaries=summaries,
        )

    def _handle_approve(self, target_id: Optional[str] = None) -> AgentResult:
        if target_id:
            app = self.tracker.find_by_partial_id(target_id)
            if app and app.status == ApplicationStatus.AWAITING_APPROVAL:
                to_submit = [app]
            else:
                self._msg(f"No submittable application found matching '{target_id}'.")
                self.state = AgentState.COMPLETED
                return AgentResult(
                    state=self.state,
                    messages=list(self.messages),
                    profile=self.profile,
                )
        else:
            to_submit = [
                a for a in self.pending_applications
                if a.status == ApplicationStatus.AWAITING_APPROVAL
            ]
        if not to_submit:
            self._msg("No applications pending approval.")
            self.state = AgentState.COMPLETED
            return AgentResult(
                state=self.state,
                messages=list(self.messages),
                profile=self.profile,
            )

        self.state = AgentState.SUBMITTING
        submitted = []
        failed = []

        for app in to_submit:
            result = self._submit_application(app)
            if result:
                submitted.append(app)
            else:
                failed.append(app)

        if submitted:
            self._msg(f"{len(submitted)} application(s) submitted successfully.")
        if failed:
            self._msg(f"{len(failed)} application(s) failed to submit.")

        self.state = AgentState.COMPLETED
        self.pending_applications = []
        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            applications=submitted + failed,
        )

    def _handle_cancel(self, target_id: Optional[str] = None) -> AgentResult:
        if target_id:
            app = self.tracker.find_by_partial_id(target_id)
            if app:
                app.update_status(ApplicationStatus.DRAFT)
                self.tracker.update(app)
                self._msg(f"Application {app.id} cancelled.")
            else:
                self._msg(f"No application found matching '{target_id}'.")
        else:
            for app in self.pending_applications:
                app.update_status(ApplicationStatus.DRAFT)
                self.tracker.update(app)
            self.pending_applications = []
            self._msg("All pending applications cancelled.")
        self.state = AgentState.COMPLETED
        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
        )

    def _match_jobs_to_candidate(self, ranked: list[RankedJob]) -> list[dict]:
        results = []
        for item in ranked:
            match = match_candidate_to_job(self.profile, item.job)
            detailed = match_candidate_to_job_detailed(self.profile, item.job)
            readiness = assess_readiness(self.profile, item.job, match)
            results.append({
                "job": item.job,
                "rank": item,
                "candidate_match": match,
                "detailed_match": detailed,
                "readiness": readiness,
            })
        return results

    def _select_best_applications(
        self,
        matched: list[dict],
        count: Optional[int] = None,
        min_score: Optional[int] = None,
    ) -> list[dict]:
        selected = []
        for item in matched:
            if min_score is not None:
                if item["candidate_match"].score < min_score:
                    continue
            selected.append(item)

        selected.sort(
            key=lambda x: x["rank"].score + x["candidate_match"].score,
            reverse=True,
        )

        if count is not None:
            selected = selected[:count]

        if not selected and matched:
            selected = matched[:min(count or 3, len(matched))]

        return selected

    def _prepare_applications(
        self,
        selected: list[dict],
    ) -> tuple[list[Application], list[MissingInfo]]:
        applications: list[Application] = []
        all_missing: list[MissingInfo] = []

        for item in selected:
            job = item["job"]

            existing = self.tracker.find_by_job_id(job.id)
            if existing and existing.status not in (
                ApplicationStatus.DRAFT,
                ApplicationStatus.WITHDRAWN,
            ):
                continue

            rank = item["rank"]
            match = item["candidate_match"]
            readiness = item["readiness"]
            detailed = item.get("detailed_match")

            app_id = str(uuid.uuid4())[:12]
            app = Application(
                id=app_id,
                job_id=job.id,
                job_title=job.title,
                job_company=job.company,
                job_location=job.location,
                job_url=job.url,
                job_salary_text=job.salary_text or "",
                job_description=job.description,
                job_remote=job.remote,
                job_platform=job.platform.value if hasattr(job, 'platform') else "",
                candidate_name=self.profile.name,
                candidate_email=self.profile.email,
                job_preference_score=rank.score,
                candidate_match_score=match.score,
                readiness_score=readiness.score,
                warnings=list(readiness.warnings),
            )
            compute_all_scores(app)

            self.document_engine.prepare_documents(app, self.profile, job)

            answers, missing = self.question_engine.resolve_common_questions(
                self.profile, job.description
            )
            if answers:
                app.answers.update(answers)
            if missing:
                app.missing_information = missing
                all_missing.extend(missing)

            job_answers, job_missing = self.question_engine.detect_job_requirements(
                job.description, self.profile
            )
            if job_answers:
                app.answers.update(job_answers)
            if job_missing:
                app.missing_information.extend(job_missing)
                all_missing.extend(job_missing)

            if not app.missing_information:
                app.update_status(ApplicationStatus.AWAITING_APPROVAL)
                app.date_prepared = app.updated_at
            else:
                app.update_status(ApplicationStatus.NEEDS_INFORMATION)

            if readiness.blockers:
                app.errors.extend(readiness.blockers)

            if detailed and detailed.concerns:
                app.notes.extend(detailed.concerns)

            self.tracker.add(app)
            applications.append(app)

        return applications, all_missing

    def _submit_application(self, app: Application) -> bool:
        platform = get_platform(app.job_url)
        try:
            result = platform.fill_and_submit(
                app.job_url,
                fields=app.answers,
                files={},
            )
            if result.success:
                app.update_status(ApplicationStatus.SUBMITTED)
                app.submitted = True
                app.submission_url = result.application_url or app.job_url
                app.submission_time = app.updated_at
                app.date_submitted = app.updated_at
                app.submission_platform = platform.name
                app.confirmation_id = result.confirmation_id
                self.tracker.update(app)
                self._msg(
                    f"Application submitted: {app.job_title} at {app.job_company}"
                )
                return True
            else:
                if result.requires_human_input:
                    app.update_status(ApplicationStatus.MANUAL_ACTION_REQUIRED)
                    app.notes.append("Requires manual submission — browser automation could not complete")
                else:
                    app.update_status(ApplicationStatus.FAILED)
                app.errors.append(result.error)
                self.tracker.update(app)
                self._msg(
                    f"Application failed: {app.job_title} at {app.job_company} "
                    f"- {result.error}"
                )
                return False
        except Exception as exc:
            app.update_status(ApplicationStatus.FAILED)
            app.errors.append(str(exc))
            self.tracker.update(app)
            self._msg(
                f"Application error: {app.job_title} at {app.job_company} - {exc}"
            )
            return False

    def provide_answers(self, answers: dict[str, str]) -> AgentResult:
        self.question_engine.answer_store.bulk_set(answers)

        # Remember answers on the candidate profile so future applications
        # reuse them even without the answer store.
        profile = self.profile or load_profile()
        if profile is not None:
            question_for_key: dict[str, str] = {}
            for app in self.pending_applications:
                for m in app.missing_information:
                    question_for_key.setdefault(m.field_key, m.question)
            for key, value in answers.items():
                if not str(value).strip():
                    continue
                profile.set_known(key, str(value), "user")
                question = question_for_key.get(key)
                if question:
                    profile.remember_answer(question, str(value), field_key=key)
            save_profile(profile)

        for app in self.pending_applications:
            if app.status == ApplicationStatus.NEEDS_INFORMATION:
                app.answers.update(answers)
                still_missing = [
                    m for m in app.missing_information
                    if m.field_key not in answers
                ]
                app.missing_information = still_missing
                if not still_missing:
                    app.update_status(ApplicationStatus.AWAITING_APPROVAL)
                self.tracker.update(app)

        still_pending = [
            a for a in self.pending_applications
            if a.status == ApplicationStatus.NEEDS_INFORMATION
        ]

        if not still_pending:
            self.state = AgentState.AWAITING_APPROVAL
            self._msg("All questions answered. Applications ready for review.")
        else:
            self._msg(
                f"{len(still_pending)} application(s) still need information."
            )

        return AgentResult(
            state=self.state,
            messages=list(self.messages),
            profile=self.profile,
            applications=self.pending_applications,
        )

    def _format_query_understood(self, query: JobQuery) -> str:
        parts = ["Understood your request:"]
        if query.roles:
            parts.append(f"  Roles: {', '.join(query.roles)}")
        if query.seniority:
            parts.append(f"  Seniority: {query.seniority}")
        if query.locations:
            locs = ", ".join(f"{l.city} ({l.radius_km}km)" for l in query.locations)
            parts.append(f"  Locations: {locs}")
        if query.remote != "any":
            parts.append(f"  Remote: {query.remote}")
        if query.min_salary:
            parts.append(f"  Min salary: {query.currency} {query.min_salary:,}")
        if query.skills:
            parts.append(f"  Skills: {', '.join(query.skills)}")
        return "\n".join(parts)

    def _format_missing_info(self, missing: list[MissingInfo]) -> str:
        if not missing:
            return ""
        parts = ["I need some information from you:"]
        for m in missing:
            parts.append(f"  - {m.question}")
        return "\n".join(parts)

    def _format_approval_request(self, applications: list[Application]) -> str:
        if not applications:
            return "No applications to review."
        parts = [
            f"{len(applications)} application(s) ready for your review:",
            "",
        ]
        for i, app in enumerate(applications, 1):
            parts.append(
                f"  {i}. {app.job_title} - {app.job_company} "
                f"({app.job_location or 'location not stated'})"
            )
            parts.append(
                f"     Match: {app.candidate_match_score}% | "
                f"Priority: {app.application_priority}%"
            )
            if app.warnings:
                for w in app.warnings:
                    parts.append(f"     Warning: {w}")
            parts.append("")
        parts.append("Reply 'approve' to submit, or 'cancel' to abort.")
        return "\n".join(parts)

    def _build_notes(self) -> list[str]:
        notes: list[str] = []
        if self.llm is not None and not self.llm.is_available():
            notes.append("Ollama is offline - used built-in rules.")
        return notes


def run_pipeline(prompt: str, region: dict, llm=None) -> PipelineResult:
    query = _parse(prompt, region, llm)
    jobs, messages = search_jobs(query, region)
    ranked = rank_jobs(jobs, query, llm)
    notes: list[str] = []
    if llm is not None and not llm.is_available():
        notes.append("Ollama is offline - used built-in rules for intent parsing.")
    return PipelineResult(query=query, search_messages=messages, jobs_found=jobs, ranked=ranked, notes=notes)


def _parse(prompt: str, region: dict, llm=None) -> JobQuery:
    from agent.parse_intent import parse_intent
    return parse_intent(prompt, region, llm)


def prepare_application(
    job: Job,
    candidate_match: CandidateMatch,
    detailed_match: DetailedMatch,
    readiness: ApplicationReadiness,
    job_preference_score: int,
    llm=None,
) -> Application:
    profile = load_profile()
    if profile is None:
        raise ValueError("No candidate profile found. Upload a CV first.")

    tracker = ApplicationTracker()
    existing = tracker.find_by_job_id(job.id)
    if existing and existing.status not in (
        ApplicationStatus.DRAFT,
        ApplicationStatus.WITHDRAWN,
    ):
        return existing

    question_engine = QuestionEngine(AnswerStore())
    document_engine = ApplicationDocuments(llm)

    app_id = str(uuid.uuid4())[:12]
    app = Application(
        id=app_id,
        job_id=job.id,
        job_title=job.title,
        job_company=job.company,
        job_location=job.location,
        job_url=job.url,
        job_salary_text=job.salary_text or "",
        job_description=job.description,
        job_remote=job.remote,
        job_platform=job.platform.value if hasattr(job, "platform") else "",
        candidate_name=profile.name,
        candidate_email=profile.email,
        job_preference_score=job_preference_score,
        candidate_match_score=candidate_match.score,
        readiness_score=readiness.score,
        warnings=list(readiness.warnings),
    )
    compute_all_scores(app)

    document_engine.prepare_documents(app, profile, job)

    answers, missing = question_engine.resolve_common_questions(
        profile, job.description
    )
    if answers:
        app.answers.update(answers)
    if missing:
        app.missing_information = missing

    job_answers, job_missing = question_engine.detect_job_requirements(
        job.description, profile
    )
    if job_answers:
        app.answers.update(job_answers)
    if job_missing:
        app.missing_information.extend(job_missing)

    if not app.missing_information:
        app.update_status(ApplicationStatus.AWAITING_APPROVAL)
        app.date_prepared = app.updated_at
    else:
        app.update_status(ApplicationStatus.NEEDS_INFORMATION)

    if readiness.blockers:
        app.errors.extend(readiness.blockers)

    if detailed_match and detailed_match.concerns:
        app.notes.extend(detailed_match.concerns)

    tracker.add(app)
    return app
