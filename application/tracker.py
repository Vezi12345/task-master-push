from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import DATA_DIR

from .models import Application, ApplicationStatus


TRACKER_FILE = DATA_DIR / "applications.json"


class ApplicationTracker:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or TRACKER_FILE
        self._applications: dict[str, Application] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for item in data:
                    app = Application(**item)
                    self._applications[app.id] = app
            except (json.JSONDecodeError, OSError):
                self._applications = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [app.model_dump() for app in self._applications.values()]
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, app: Application) -> None:
        self._applications[app.id] = app
        self._save()

    def get(self, app_id: str) -> Optional[Application]:
        return self._applications.get(app_id)

    def update(self, app: Application) -> None:
        self._applications[app.id] = app
        self._save()

    def remove(self, app_id: str) -> bool:
        if app_id in self._applications:
            del self._applications[app_id]
            self._save()
            return True
        return False

    def all(self) -> list[Application]:
        return list(self._applications.values())

    def by_status(self, status: ApplicationStatus) -> list[Application]:
        return [a for a in self._applications.values() if a.status == status]

    def find_by_job_id(self, job_id: str) -> Optional[Application]:
        for app in self._applications.values():
            if app.job_id == job_id:
                return app
        return None

    def find_by_partial_id(self, partial_id: str) -> Optional[Application]:
        partial_lower = partial_id.lower()
        for app in self._applications.values():
            if app.id.lower().startswith(partial_lower):
                return app
        return None

    def is_duplicate(self, job_id: str) -> bool:
        existing = self.find_by_job_id(job_id)
        if existing is None:
            return False
        if existing.submitted:
            return True
        if existing.status in (
            ApplicationStatus.SUBMITTED,
            ApplicationStatus.PENDING,
            ApplicationStatus.INTERVIEW,
            ApplicationStatus.OFFER,
        ):
            return True
        return False

    def get_submittable(self) -> list[Application]:
        return [
            a for a in self._applications.values()
            if a.is_submittable
        ]

    def get_by_status_group(self) -> dict[str, list[Application]]:
        groups: dict[str, list[Application]] = {}
        for app in self._applications.values():
            status = app.status.value
            if status not in groups:
                groups[status] = []
            groups[status].append(app)
        return groups

    def get_summary(self) -> dict:
        all_apps = self.all()
        by_status: dict[str, int] = {}
        for app in all_apps:
            status = app.status.value
            by_status[status] = by_status.get(status, 0) + 1
        submittable = self.get_submittable()
        return {
            "total": len(all_apps),
            "by_status": by_status,
            "submittable_count": len(submittable),
            "needs_attention": [
                a.to_preview()
                for a in all_apps
                if a.status in (
                    ApplicationStatus.NEEDS_INFORMATION,
                    ApplicationStatus.FAILED,
                    ApplicationStatus.MANUAL_ACTION_REQUIRED,
                )
            ],
        }

    def needs_attention(self) -> list[Application]:
        return [
            a for a in self._applications.values()
            if a.status in (
                ApplicationStatus.NEEDS_INFORMATION,
                ApplicationStatus.FAILED,
                ApplicationStatus.MANUAL_ACTION_REQUIRED,
            )
        ]
