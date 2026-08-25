from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

from config import LLM_MODEL, OLLAMA_HOST, LLM_OFFLINE


class LLMUnavailable(RuntimeError):
    pass


class LLMInvalidOutput(ValueError):
    pass


_AVAIL_TTL_SECONDS = 60.0
_avail_cache: dict = {"ok": None, "at": 0.0}


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", OLLAMA_HOST)


def is_available(timeout: float = 2.0) -> bool:
    """Probe Ollama at most once per TTL window.

    Every rank/summary/parse call used to re-probe the port; with Ollama down
    (firewalled SYN drops) each probe burned the full timeout, so a single
    search paid 2s x dozens of calls. The verdict is now memoised briefly.
    """
    if LLM_OFFLINE:
        return False
    now = time.monotonic()
    cached = _avail_cache["ok"]
    if cached is not None and (now - _avail_cache["at"]) < _AVAIL_TTL_SECONDS:
        return cached
    try:
        resp = requests.get(f"{_host()}/api/tags", timeout=timeout)
        ok = resp.ok
    except Exception:
        ok = False
    _avail_cache["ok"] = ok
    _avail_cache["at"] = now
    return ok


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
