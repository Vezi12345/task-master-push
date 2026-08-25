import pytest

import llm


def test_is_available_false_when_offline_host():
    assert llm.is_available() is False or llm.is_available() is True


def test_chat_json_raises_unavailable_on_bad_host():
    with pytest.raises(llm.LLMUnavailable):
        llm.chat_json("system", "user", host="http://127.0.0.1:9", timeout=3)


def test_chat_json_raises_invalid_output():
    from unittest.mock import MagicMock, patch

    fake = MagicMock()
    fake.json.return_value = {"message": {"content": "not json"}}
    with patch("llm.requests.post", return_value=fake):
        with pytest.raises(llm.LLMInvalidOutput):
            llm.chat_json("system", "user", host="http://127.0.0.1:11434")


def test_is_available_probes_at_most_once_per_window(monkeypatch):
    import llm as llm_mod

    calls = []

    class _R:
        ok = True

    def fake_get(url, timeout=None):
        calls.append(1)
        return _R()

    monkeypatch.setattr(llm_mod.requests, "get", fake_get)
    monkeypatch.setattr(llm_mod, "LLM_OFFLINE", False)
    monkeypatch.setattr(llm_mod, "_avail_cache", {"ok": None, "at": 0.0})

    assert llm_mod.is_available() is True
    assert llm_mod.is_available() is True
    assert len(calls) == 1, "second probe within TTL must be memoised"

    # after the window expires we probe again
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: llm_mod._avail_cache["at"] + llm_mod._AVAIL_TTL_SECONDS + 1)
    llm_mod.is_available()
    assert len(calls) == 2


def test_is_available_failure_is_memoised_too(monkeypatch):
    import llm as llm_mod

    calls = []

    def boom(url, timeout=None):
        calls.append(1)
        raise ConnectionError("refused")

    monkeypatch.setattr(llm_mod.requests, "get", boom)
    monkeypatch.setattr(llm_mod, "LLM_OFFLINE", False)
    monkeypatch.setattr(llm_mod, "_avail_cache", {"ok": None, "at": 0.0})

    assert llm_mod.is_available() is False
    assert llm_mod.is_available() is False
    assert len(calls) == 1
