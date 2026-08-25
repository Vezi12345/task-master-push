from __future__ import annotations

import json
import os
from typing import Optional

import requests

from config import LLM_MODEL, OLLAMA_HOST, LLM_OFFLINE


class LLMUnavailable(RuntimeError):
    pass


class LLMInvalidOutput(ValueError):
    pass


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", OLLAMA_HOST)


def is_available(timeout: float = 2.0) -> bool:
    if LLM_OFFLINE:
        return False
    try:
        resp = requests.get(f"{_host()}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def chat_json(
    system: str,
    user: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    timeout: int = 180,
) -> dict:
    model = model or os.environ.get("TASK_MASTER_MODEL", LLM_MODEL)
    host = host or _host()
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except Exception as exc:
        raise LLMUnavailable(str(exc)) from exc
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMInvalidOutput(f"model returned invalid JSON: {content!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMInvalidOutput(f"model returned non-object JSON: {parsed!r}")
    return parsed
