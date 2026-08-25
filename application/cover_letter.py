from __future__ import annotations

from typing import Optional

from candidate.profile import CandidateProfile
from sources.base import Job


_COVER_LETTER_TEMPLATE = """Dear Hiring Manager,

I am writing to express my strong interest in the {job_title} position at {company}.

{body_paragraph}

{skills_paragraph}

{closing_paragraph}

Sincerely,
{candidate_name}
"""


class CoverLetterGenerator:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def generate(
        self,
        profile: CandidateProfile,
        job: Job,
        extra_context: str = "",
    ) -> str:
        if self.llm is not None and self.llm.is_available():
            return self._generate_with_llm(profile, job, extra_context)
        return self._generate_template(profile, job)

    def _generate_with_llm(
        self,
        profile: CandidateProfile,
        job: Job,
        extra_context: str = "",
    ) -> str:
        system = (
            "You write tailored cover letters for job applications. "
            "Use ONLY factual information provided about the candidate. "
            "Never invent qualifications, experience, skills, achievements, "
            "or certifications. Write a professional, concise cover letter "
            "of 3-4 paragraphs. Output ONLY the letter text, no markdown "
            "formatting."
        )
        candidate_info = _build_candidate_summary(profile)
        user_prompt = (
            f"Candidate information:\n{candidate_info}\n\n"
            f"Job title: {job.title}\n"
            f"Company: {job.company}\n"
            f"Location: {job.location}\n"
            f"Job description: {job.description[:1500]}\n"
        )
        if extra_context:
            user_prompt += f"\nAdditional context: {extra_context}\n"

        try:
            result = self.llm.chat_json(
                system,
                user_prompt,
            )
            letter = result.get("cover_letter", "")
            if letter and len(letter) > 50:
                return letter.strip()
        except Exception:
            pass

        return self._generate_template(profile, job)

    def _generate_template(
        self,
        profile: CandidateProfile,
        job: Job,
    ) -> str:
        name = profile.name or "Applicant"
        company = job.company or "your organisation"
        job_title = job.title or "the advertised position"

        body = _build_body_paragraph(profile, job)
        skills = _build_skills_paragraph(profile, job)
        closing = _build_closing_paragraph(profile, job)

        return _COVER_LETTER_TEMPLATE.format(
            job_title=job_title,
            company=company,
            body_paragraph=body,
            skills_paragraph=skills,
            closing_paragraph=closing,
            candidate_name=name,
        ).strip()


def _build_candidate_summary(profile: CandidateProfile) -> str:
    parts: list[str] = []
    if profile.name:
        parts.append(f"Name: {profile.name}")
    if profile.location:
        parts.append(f"Location: {profile.location}")
    if profile.email:
        parts.append(f"Email: {profile.email}")
    if profile.professional_summary:
        parts.append(f"Summary: {profile.professional_summary}")
    if profile.skills:
        parts.append(f"Skills: {', '.join(profile.skills)}")
    if profile.education:
        for edu in profile.education:
            edu_bits = [p for p in (edu.qualification, edu.field, edu.institution) if p]
            if edu_bits:
                parts.append(f"Education: {' - '.join(edu_bits)}")
    if profile.experience:
        for exp in profile.experience:
            desc = f" at {exp.company}" if exp.company else ""
            parts.append(f"Experience: {exp.title}{desc}")
    if profile.certifications:
        for cert in profile.certifications:
            parts.append(f"Certification: {cert.name}")
    if profile.achievements:
        for ach in profile.achievements:
            parts.append(f"Achievement: {ach}")
    return "\n".join(parts)


def _build_body_paragraph(profile: CandidateProfile, job: Job) -> str:
    name = profile.name or "I"
    company = job.company or "your organisation"

    if profile.professional_summary:
        return (
            f"I believe I am a strong fit for this role. "
            f"{profile.professional_summary}"
        )

    skills = profile.skills[:5]
    if skills:
        skill_str = ", ".join(skills)
        return (
            f"With experience in {skill_str}, I am confident in my ability "
            f"to contribute effectively to the {company} team."
        )

    return (
        f"I am enthusiastic about the opportunity to contribute to "
        f"{company} and grow professionally in this role."
    )


def _build_skills_paragraph(profile: CandidateProfile, job: Job) -> str:
    if not profile.skills:
        return ""

    job_text = f"{job.title} {job.description}".lower()
    relevant_skills = [s for s in profile.skills if s.lower() in job_text]
    if not relevant_skills:
        relevant_skills = profile.skills[:6]
    else:
        relevant_skills = relevant_skills[:6]
    skill_str = ", ".join(relevant_skills)

    if profile.experience:
        years = len(profile.experience)
        return (
            f"My technical skill set includes {skill_str}. "
            f"With {years} role(s) of professional experience, "
            f"I have prepared myself to handle the responsibilities "
            f"of this position effectively."
        )

    return (
        f"My technical skill set includes {skill_str}. "
        f"I am eager to apply these skills in a professional environment."
    )


def _build_closing_paragraph(profile: CandidateProfile, job: Job) -> str:
    return (
        f"I would welcome the opportunity to discuss how my background "
        f"and enthusiasm align with your team's needs. "
        f"I am available for an interview at your convenience."
    )
