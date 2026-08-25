from __future__ import annotations

"""Gmail REST API connector tests.

Network is never touched: requests.get / requests.post are faked. Token
files live in tmp_path so the real data/ directory stays untouched.
"""

import base64
import json
import time

import pytest
import requests as real_requests

import config
from application.email_confirmation import MailboxNotConfigured, MailboxUnavailable
from application.gmail_api import GmailApiMailboxConnector, _b64url_decode, _to_email_message


def _token_file(tmp_path) -> object:
    path = tmp_path / "gmail_token.json"
    path.write_text(json.dumps({
        "access_token": "cached-access",
        "refresh_token": "refresh-abc",
        "expires_at": time.time() + 3600,
        "client_id": "cid",
        "client_secret": "csec",
    }), encoding="utf-8")
    return path


def _connector(tmp_path) -> GmailApiMailboxConnector:
    return GmailApiMailboxConnector(token_file=tmp_path / "gmail_token.json")


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# configuration honesty (mirrors the IMAP connector contract)
# ---------------------------------------------------------------------------

def test_no_token_file_reports_unconfigured(tmp_path):
    conn = GmailApiMailboxConnector(token_file=tmp_path / "absent.json")
    assert conn.is_configured() is False
    with pytest.raises(MailboxNotConfigured):
        conn.fetch_recent()


def test_token_with_refresh_token_is_configured(tmp_path):
    _token_file(tmp_path)
    assert _connector(tmp_path).is_configured() is True


# ---------------------------------------------------------------------------
# backend resolution
# ---------------------------------------------------------------------------

def test_auto_backend_prefers_gmail_api_when_secret_exists(tmp_path, monkeypatch):
    secret = tmp_path / "secret.json"
    secret.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "EMAIL_BACKEND", "auto")
    monkeypatch.setattr(config, "GMAIL_CLIENT_SECRET_FILE", secret)
    assert config.resolve_email_backend() == "gmail_api"


def test_auto_backend_falls_back_to_imap_without_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMAIL_BACKEND", "auto")
    monkeypatch.setattr(
        config, "GMAIL_CLIENT_SECRET_FILE", tmp_path / "absent-secret.json"
    )
    assert config.resolve_email_backend() == "imap"


def test_explicit_backend_overrides_auto(monkeypatch):
    monkeypatch.setattr(config, "EMAIL_BACKEND", "imap")
    assert config.resolve_email_backend() == "imap"


# ---------------------------------------------------------------------------
# fetch_recent over faked HTTP
# ---------------------------------------------------------------------------

def _gmail_message(msg_id="m1", plain="Application received", internal_ms=None):
    data = base64.urlsafe_b64encode(plain.encode()).decode().rstrip("=")
    return {
        "id": msg_id,
        "internalDate": str(internal_ms),
        "payload": {
            "headers": [
                {"name": "From", "value": "no-reply@lever.co"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": "Thanks for applying"},
            ],
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": data}},
                {"mimeType": "text/html", "body": {"data": "aHRtbA"}},
            ],
        },
    }


class FakeGet:
    def __init__(self, list_ids=("m1",), message=None, first_status_401=False):
        self.list_ids = list_ids
        self.message = message or _gmail_message()
        self.calls: list[str] = []
        self.first_status_401 = first_status_401

    def __call__(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        if url.endswith("/messages"):
            return FakeResponse(200, {"messages": [{"id": i} for i in self.list_ids]})
        if self.first_status_401 and len(self.calls) == 2:
            return FakeResponse(401, {"error": "invalid credentials"})
        return FakeResponse(200, dict(self.message, id=url.rsplit("/", 1)[-1]))


def test_fetch_recent_maps_messages_correctly(tmp_path, monkeypatch):
    _token_file(tmp_path)
    fake = FakeGet(message=_gmail_message(internal_ms=1700000000000))
    monkeypatch.setattr("application.gmail_api.requests.get", fake)
    emails = _connector(tmp_path).fetch_recent(since_days=7)
    assert len(emails) == 1
    email = emails[0]
    assert email.id == "m1"
    assert email.from_addr == "no-reply@lever.co"
    assert email.subject == "Thanks for applying"
    assert "Application received" in email.body
    assert "html" not in email.body  # html part ignored, plain preferred
    assert email.date is not None and email.date.year == 2023


def test_fetch_recent_sends_bearer_and_query(tmp_path, monkeypatch):
    _token_file(tmp_path)
    fake = FakeGet(list_ids=())
    captured = {}

    def get(url, headers=None, params=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        return FakeResponse(200, {})

    monkeypatch.setattr("application.gmail_api.requests.get", get)
    emails = _connector(tmp_path).fetch_recent(since_days=3, limit=50)
    assert emails == []
    assert captured["url"].endswith("/gmail/v1/users/me/messages")
    assert captured["headers"]["Authorization"] == "Bearer cached-access"
    assert captured["params"]["maxResults"] == 50
    assert captured["params"]["labelIds"] == "INBOX"
    assert "after:" in captured["params"]["q"]


def test_expired_token_is_refreshed_then_reused(tmp_path, monkeypatch):
    path = tmp_path / "gmail_token.json"
    path.write_text(json.dumps({
        "access_token": "stale",
        "refresh_token": "refresh-abc",
        "expires_at": time.time() - 10,  # expired
        "client_id": "cid",
        "client_secret": "csec",
    }), encoding="utf-8")

    posts = []

    def fake_post(url, data=None, timeout=None):
        posts.append({"url": url, "data": data})
        return FakeResponse(200, {"access_token": "fresh", "expires_in": 3600})

    seen_bearers = []

    def get(url, headers=None, params=None, timeout=None):
        seen_bearers.append(headers["Authorization"])
        if url.endswith("/messages"):
            return FakeResponse(200, {})
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("application.gmail_api.requests.post", fake_post)
    monkeypatch.setattr("application.gmail_api.requests.get", get)

    emails = _connector(tmp_path).fetch_recent(since_days=14)
    assert emails == []
    assert posts[0]["data"]["grant_type"] == "refresh_token"
    assert posts[0]["data"]["refresh_token"] == "refresh-abc"
    assert seen_bearers[-1] == "Bearer fresh"

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["access_token"] == "fresh"


def test_401_mid_session_forces_single_refresh_retry(tmp_path, monkeypatch):
    _token_file(tmp_path)
    refresh_posts = []
    gets = FakeGet(first_status_401=True)

    def fake_post(url, data=None, timeout=None):
        refresh_posts.append(data)
        return FakeResponse(200, {"access_token": "brand-new", "expires_in": 3600})

    monkeypatch.setattr("application.gmail_api.requests.post", fake_post)
    monkeypatch.setattr("application.gmail_api.requests.get", gets)

    emails = _connector(tmp_path).fetch_recent(since_days=14)
    assert len(refresh_posts) == 1
    assert len(emails) == 1


def test_network_error_raises_mailbox_unavailable(tmp_path, monkeypatch):
    _token_file(tmp_path)

    def boom(*a, **kw):
        raise real_requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr("application.gmail_api.requests.get", boom)
    with pytest.raises(MailboxUnavailable):
        _connector(tmp_path).fetch_recent()


def test_403_raises_helpful_mailbox_unavailable(tmp_path, monkeypatch):
    _token_file(tmp_path)

    def forbidden(url, headers=None, params=None, timeout=None):
        return FakeResponse(403, {"error": "API not enabled"})

    monkeypatch.setattr("application.gmail_api.requests.get", forbidden)
    with pytest.raises(MailboxUnavailable, match="403"):
        _connector(tmp_path).fetch_recent()


# ---------------------------------------------------------------------------
# payload parsing helpers
# ---------------------------------------------------------------------------

def test_b64url_decode_pads_correctly():
    raw = b"hello world"
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert _b64url_decode(encoded) == raw


def test_to_email_message_falls_back_to_date_header():
    msg = {
        "id": "x",
        "payload": {
            "headers": [
                {"name": "Date", "value": "Tue, 14 Nov 2023 08:00:00 +0000"},
                {"name": "Subject", "value": "s"},
            ],
        },
    }
    email = _to_email_message(msg)
    assert email.date is not None and email.date.tzinfo is not None
