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

DEFAULT_REGION = os.environ.get("TASK_MASTER_REGION", "za")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("TASK_MASTER_MODEL", "qwen2.5:7b")
LLM_OFFLINE = os.environ.get("TASK_MASTER_LLM_OFFLINE", "0") == "1"


def load_region(name: str = DEFAULT_REGION) -> dict:
    path = REGIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Region config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
