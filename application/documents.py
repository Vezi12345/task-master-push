from __future__ import annotations

from typing import Optional

from candidate.profile import CandidateProfile
from sources.base import Job

from .cover_letter import CoverLetterGenerator
from .models import Application, DocumentStatus


class ApplicationDocuments:
    def __init__(self, llm=None) -> None:
        self.cover_letter_gen = CoverLetterGenerator(llm)

    def prepare_documents(
        self,
        app: Application,
        profile: CandidateProfile,
        job: Job,
    ) -> Application:
        has_contact = bool(profile.email or profile.phone)
        has_name = bool(profile.name)
        has_education = bool(profile.education)
        has_skills = bool(profile.skills)
        app.documents.cv_ready = has_contact and has_name

        cover_letter = self.cover_letter_gen.generate(profile, job)
        app.documents.cover_letter_text = cover_letter
        app.documents.cover_letter_ready = bool(cover_letter)

        app.documents.tailored_summary = self._tailor_summary(profile, job)

        if not has_contact:
            app.warnings.append(
                "CV may be incomplete — no contact details found"
            )
        if not has_name:
            app.warnings.append(
                "CV may be incomplete — no name found"
            )
        if not has_education:
            app.notes.append(
                "No education details found on CV"
            )
        if not has_skills:
            app.notes.append(
                "No skills found on CV — matching may be less accurate"
            )

        return app

    def _tailor_summary(
        self,
        profile: CandidateProfile,
        job: Job,
    ) -> str:
        if profile.professional_summary:
            return profile.professional_summary

        skills = profile.skills[:5]
        job_text = f"{job.title} {job.description}".lower()
        relevant_skills = [s for s in skills if s.lower() in job_text]
        if not relevant_skills:
            relevant_skills = skills

        if relevant_skills:
            return (
                f"Professional with experience in {', '.join(relevant_skills)}. "
                f"Seeking to contribute to {job.company or 'your organisation'}."
            )

        return ""
