from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from sources.base import Job


class ApplicationStatus(str, Enum):
    DISCOVERED = "discovered"
    SELECTED = "selected"
    DRAFT = "draft"
    PREPARING = "preparing"
    NEEDS_INFORMATION = "needs_information"
    READY_FOR_REVIEW = "ready_for_review"
    AWAITING_APPROVAL = "awaiting_approval"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PENDING = "pending"
    FAILED = "failed"
    MANUAL_ACTION_REQUIRED = "manual_action_required"
    INTERVIEW = "interview"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    OFFER = "offer"


class DocumentStatus(BaseModel):
    cv_ready: bool = False
    cover_letter_ready: bool = False
    cover_letter_text: str = ""
    tailored_summary: str = ""


class MissingInfo(BaseModel):
    question: str
    field_key: str = ""
    category: str = ""
    priority: str = "required"
    answer: Optional[str] = None


class Application(BaseModel):
    id: str = ""
    job_id: str = ""
    job_title: str = ""
    job_company: str = ""
    job_location: str = ""
    job_url: str = ""
    job_salary_text: str = ""
    job_description: str = ""
    job_remote: bool = False
    job_platform: str = ""

    candidate_name: str = ""
    candidate_email: str = ""

    status: ApplicationStatus = ApplicationStatus.DRAFT

    job_preference_score: int = 0
    candidate_match_score: int = 0
    readiness_score: int = 0
    application_priority: int = 0

    documents: DocumentStatus = Field(default_factory=DocumentStatus)
    answers: dict[str, str] = Field(default_factory=dict)
    missing_information: list[MissingInfo] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    submission_url: str = ""
    submission_time: Optional[str] = None
    submitted: bool = False
    submission_platform: str = ""
    confirmation_id: str = ""

    date_prepared: str = ""
    date_submitted: str = ""
    date_responded: str = ""

    created_at: str = ""
    updated_at: str = ""

    def model_post_init(self, __context) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def update_status(self, status: ApplicationStatus) -> None:
        self.status = status
        self.updated_at = datetime.now().isoformat()

    @property
    def is_submittable(self) -> bool:
        return (
            self.status == ApplicationStatus.AWAITING_APPROVAL
            and not self.submitted
            and len(self.missing_information) == 0
            and len(self.errors) == 0
        )

    @property
    def has_missing_info(self) -> bool:
        return len(self.missing_information) > 0

    def to_preview(self) -> dict:
        return {
            "id": self.id,
            "company": self.job_company,
            "role": self.job_title,
            "location": self.job_location or "Not stated",
            "remote": self.job_remote,
            "salary": self.job_salary_text or "Not stated",
            "url": self.job_url,
            "platform": self.job_platform,
            "job_preference_score": self.job_preference_score,
            "candidate_match_score": self.candidate_match_score,
            "readiness_score": self.readiness_score,
            "application_priority": self.application_priority,
            "status": self.status.value,
            "documents": self.documents.model_dump(),
            "answers": self.answers,
            "missing_information": [m.model_dump() for m in self.missing_information],
            "warnings": self.warnings,
            "errors": self.errors,
            "notes": self.notes,
            "submitted": self.submitted,
            "submission_url": self.submission_url,
            "submission_time": self.submission_time,
            "submission_platform": self.submission_platform,
            "confirmation_id": self.confirmation_id,
            "date_prepared": self.date_prepared,
            "date_submitted": self.date_submitted,
            "job_description": self.job_description,
        }
