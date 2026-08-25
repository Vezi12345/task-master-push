from __future__ import annotations

"""Professional display names for technology/skill entries.

Only renames entries that already exist in the candidate profile —
never invents or drops skills.
"""

_SKILL_DISPLAY = {
    ".net": ".NET",
    "ai": "AI",
    "angular": "Angular",
    "api": "API",
    "asp.net": "ASP.NET",
    "aws": "AWS",
    "azure": "Azure",
    "bash": "Bash",
    "c": "C",
    "c#": "C#",
    "c++": "C++",
    "ci/cd": "CI/CD",
    "communication": "Communication",
    "css": "CSS",
    "docker": "Docker",
    "excel": "Excel",
    "git": "Git",
    "go": "Go",
    "gcp": "GCP",
    "golang": "Go",
    "html": "HTML",
    "java": "Java",
    "javascript": "JavaScript",
    "jira": "Jira",
    "js": "JavaScript",
    "kubernetes": "Kubernetes",
    "leadership": "Leadership",
    "linux": "Linux",
    "machine learning": "Machine Learning",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "node": "Node.js",
    "node.js": "Node.js",
    "php": "PHP",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "power bi": "Power BI",
    "problem solving": "Problem Solving",
    "project management": "Project Management",
    "python": "Python",
    "r": "R",
    "react": "React",
    "redis": "Redis",
    "rest": "REST",
    "sap": "SAP",
    "scrum": "Scrum",
    "sql": "SQL",
    "sql server": "SQL Server",
    "tableau": "Tableau",
    "teamwork": "Teamwork",
    "time management": "Time Management",
    "typescript": "TypeScript",
    "vue": "Vue",
    "vue.js": "Vue.js",
}


def format_skill_name(skill: str) -> str:
    cleaned = (skill or "").strip()
    if not cleaned:
        return cleaned
    return _SKILL_DISPLAY.get(cleaned.lower(), cleaned.title())


def format_skill_list(skills) -> str:
    return ", ".join(format_skill_name(s) for s in skills)
