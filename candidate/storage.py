from __future__ import annotations

import json
from pathlib import Path

from config import DATA_DIR

from .profile import CandidateProfile

PROFILE_FILE = DATA_DIR / "candidate_profile.json"


def save_profile(profile: CandidateProfile) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(
        profile.model_dump_json(indent=2), encoding="utf-8"
    )


def load_profile() -> CandidateProfile | None:
    if not PROFILE_FILE.exists():
        return None
    try:
        data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        return CandidateProfile(**data)
    except (json.JSONDecodeError, Exception):
        return None
