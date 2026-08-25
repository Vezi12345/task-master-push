from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REGIONS_DIR = ROOT_DIR / "config" / "regions"
DATA_DIR = ROOT_DIR / "data"
KEPT_JOBS_FILE = DATA_DIR / "kept_jobs.json"
APPLICATIONS_FILE = DATA_DIR / "applications.json"
ANSWERS_FILE = DATA_DIR / "answers.json"
ANSWER_CONFLICTS_FILE = DATA_DIR / "answer_conflicts.json"
CV_FILE = DATA_DIR / "candidate_cv.pdf"
COVER_LETTER_FILE = DATA_DIR / "cover_letter.txt"

# Email backend: "imap" | "gmail_api" | "auto".
# "auto" picks gmail_api when an OAuth client-secret file exists, else imap.
EMAIL_BACKEND = os.environ.get("TASK_MASTER_EMAIL_BACKEND", "auto").strip().lower()
_env_secret = os.environ.get("TASK_MASTER_GMAIL_CLIENT_SECRET_FILE", "")
GMAIL_CLIENT_SECRET_FILE = Path(_env_secret) if _env_secret else DATA_DIR / "gmail_client_secret.json"
GMAIL_TOKEN_FILE = DATA_DIR / "gmail_token.json"

DEFAULT_REGION = os.environ.get("TASK_MASTER_REGION", "za")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("TASK_MASTER_MODEL", "qwen2.5:7b")
LLM_OFFLINE = os.environ.get("TASK_MASTER_LLM_OFFLINE", "0") == "1"

# --- autonomous application policy (user-started runs only) ---
MIN_APPLICATION_SCORE = int(os.environ.get("TASK_MASTER_MIN_APPLICATION_SCORE", "75"))
MAX_APPLICATIONS_PER_RUN = int(os.environ.get("TASK_MASTER_MAX_APPLICATIONS_PER_RUN", "5"))
MAX_APPLICATIONS_PER_DAY = int(os.environ.get("TASK_MASTER_MAX_APPLICATIONS_PER_DAY", "10"))
AUTONOMOUS_RUNS_DIR = DATA_DIR / "autonomous_runs"


def load_region(name: str = DEFAULT_REGION) -> dict:
    path = REGIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Region config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_email_backend() -> str:
    if EMAIL_BACKEND in ("gmail_api", "imap"):
        return EMAIL_BACKEND
    return "gmail_api" if GMAIL_CLIENT_SECRET_FILE.exists() else "imap"
