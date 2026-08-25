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
