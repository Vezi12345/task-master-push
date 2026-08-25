from __future__ import annotations

"""Gmail REST API mailbox connector.

Drop-in alternative to ImapMailboxConnector for networks that block
Gmail's IMAP endpoints (TCP 993) but allow HTTPS to googleapis.com.
Talks ONLY to:
  https://accounts.google.com/o/oauth2/v2/auth   (browser consent)
  https://oauth2.googleapis.com/token            (code/token exchange)
  https://gmail.googleapis.com/gmail/v1/...      (read-only mail API)

Same interface as the IMAP connector: is_configured() + fetch_recent()
returning EmailMessage objects, raising MailboxNotConfigured /
MailboxUnavailable from application.email_confirmation so the workflow
and UI behave identically.

Setup (once):
  1. Google Cloud Console -> OAuth client ID -> "Desktop app",
     download the JSON and save it as data/gmail_client_secret.json
     (or point TASK_MASTER_GMAIL_CLIENT_SECRET_FILE at it).
  2. Run:  python cli.py gmail-auth
     A browser opens; consent grants READ-ONLY Gmail access; the token
     is cached at data/gmail_token.json and auto-refreshed afterwards.

Environment variables:
  TASK_MASTER_EMAIL_BACKEND              imap | gmail_api | auto (default)
  TASK_MASTER_GMAIL_CLIENT_SECRET_FILE   override secret-file path
  TASK_MASTER_GMAIL_OAUTH_PORT           local redirect port, default 8899
  TASK_MASTER_EMAIL_ADDRESS              display/verification only
"""

import base64
import json
import os
import secrets
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests

import config
from .email_confirmation import EmailMessage, MailboxNotConfigured, MailboxUnavailable


_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


class GmailApiMailboxConnector:
    """Read-only Gmail via REST over HTTPS (works where IMAP is blocked)."""

    def __init__(
        self,
        address: Optional[str] = None,
        token_file: Optional[Path] = None,
        client_secret_file: Optional[Path] = None,
    ) -> None:
        self.address = address or os.environ.get("TASK_MASTER_EMAIL_ADDRESS", "")
        self.token_file = Path(token_file) if token_file else config.GMAIL_TOKEN_FILE
        self.client_secret_file = (
            Path(client_secret_file) if client_secret_file
            else config.GMAIL_CLIENT_SECRET_FILE
        )

    # ------------------------------------------------------------------ #
    # configuration honesty (mirrors the IMAP connector contract)
    # ------------------------------------------------------------------ #

    def is_configured(self) -> bool:
        """True only when a cached OAuth token with a refresh_token exists."""
        return self._load_token().get("refresh_token", "") != ""

    def fetch_recent(self, since_days: int = 14, limit: int = 100) -> list[EmailMessage]:
        """Fetch recent inbox messages read-only via the Gmail REST API."""
        token = self._load_token()
        if not token.get("refresh_token"):
            raise MailboxNotConfigured(
                "Gmail API integration is not authorised — run "
                "`python cli.py gmail-auth` once (or set TASK_MASTER_IMAP_* "
                "to use IMAP instead)"
            )
        since_epoch = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp())
        access = self._access_token(token)

        try:
            resp = self._api_get(
                f"{_API_BASE}/messages",
                access,
                params={
                    "q": f"after:{since_epoch}",
                    "maxResults": min(int(limit), 500),
                    "labelIds": "INBOX",
                },
                token=token,
            )
        except requests.exceptions.RequestException as exc:
            raise MailboxUnavailable(
                f"Cannot reach gmail.googleapis.com — {exc.__class__.__name__}: {exc}"
            ) from exc

        ids = [m["id"] for m in resp.get("messages", [])][-limit:]

        messages: list[EmailMessage] = []
        for msg_id in ids:
            try:
                full = self._api_get(
                    f"{_API_BASE}/messages/{msg_id}", self._access_token(token),
                    params={"format": "full"}, token=token,
                )
            except requests.exceptions.RequestException as exc:
                raise MailboxUnavailable(
                    f"Gmail API error while fetching message {msg_id} — "
                    f"{exc.__class__.__name__}: {exc}"
                ) from exc
            messages.append(_to_email_message(full))
        return messages

    # ------------------------------------------------------------------ #
    # OAuth / HTTP plumbing
    # ------------------------------------------------------------------ #

    def _load_token(self) -> dict:
        try:
            return json.loads(self.token_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def _save_token(self, token: dict) -> None:
        config.ensure_data_dir()
        self.token_file.write_text(
            json.dumps(token, indent=2), encoding="utf-8"
        )

    def _access_token(self, token: dict, force_refresh: bool = False) -> str:
        expires_at = float(token.get("expires_at", 0))
        if not force_refresh and time.time() < expires_at - 60:
            return token.get("access_token", "")
        try:
            resp = requests.post(_TOKEN_URL, data={
                "client_id": token["client_id"],
                "client_secret": token["client_secret"],
                "refresh_token": token["refresh_token"],
                "grant_type": "refresh_token",
            }, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise MailboxUnavailable(
                f"Cannot reach {_TOKEN_URL} — {exc.__class__.__name__}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise MailboxUnavailable(
                f"Gmail token refresh failed (HTTP {resp.status_code}) — "
                "re-run `python cli.py gmail-auth`"
            )
        payload = resp.json()
        token["access_token"] = payload["access_token"]
        token["expires_at"] = time.time() + int(payload.get("expires_in", 3600))
        self._save_token(token)
        return token["access_token"]

    def _api_get(
        self, url: str, access_token: str, token: dict,
        params: Optional[dict] = None,
    ) -> dict:
        headers = {"Authorization": f"Bearer {access_token}"}
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 401:
            # expired/revoked mid-flight — refresh once and retry
            headers["Authorization"] = f"Bearer {self._access_token(token, force_refresh=True)}"
            resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 403:
            raise MailboxUnavailable(
                "Gmail API returned 403 — check that the Gmail API is "
                "enabled for this Google Cloud project and the consent "
                "scope is gmail.readonly"
            )
        if resp.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()


# --------------------------------------------------------------------------- #
# Gmail payload -> EmailMessage
# --------------------------------------------------------------------------- #

def _collect_plain_text(payload: dict) -> str:
    parts: list[str] = []

    def walk(node: dict) -> None:
        mime = node.get("mimeType", "")
        body = node.get("body", {})
        data = body.get("data")
        if mime == "text/plain" and data:
            parts.append(_b64url_decode(data).decode("utf-8", "replace"))
        elif mime.startswith("multipart/") or node.get("parts"):
            for child in node.get("parts", []):
                walk(child)

    walk(payload)
    return "\n".join(parts)[:20000]


def _header(payload: dict, name: str) -> str:
    for item in payload.get("headers", []):
        if item.get("name", "").lower() == name.lower():
            return item.get("value", "")
    return ""


def _to_email_message(full: dict) -> EmailMessage:
    payload = full.get("payload", {})
    internal_ms = full.get("internalDate")
    when: Optional[datetime] = None
    if internal_ms:
        try:
            when = datetime.fromtimestamp(int(internal_ms) / 1000, tz=timezone.utc)
        except (ValueError, OverflowError):
            when = None
    if when is None:
        date_raw = _header(payload, "Date")
        if date_raw:
            try:
                when = parsedate_to_datetime(date_raw)
            except (TypeError, ValueError):
                when = None
    return EmailMessage(
        id=full.get("id", ""),
        from_addr=_header(payload, "From"),
        to_addr=_header(payload, "To"),
        subject=_header(payload, "Subject"),
        body=_collect_plain_text(payload),
        date=when,
    )


# --------------------------------------------------------------------------- #
# one-time interactive authorization flow (used by `python cli.py gmail-auth`)
# --------------------------------------------------------------------------- #

def load_client_secret(path: Path) -> tuple[str, str]:
    """Return (client_id, client_secret) from a downloaded OAuth JSON."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Client secret file not found: {path}\n"
            "Google Cloud Console -> APIs & Services -> Credentials -> "
            "OAuth client ID (type: Desktop app) -> download JSON."
        ) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON") from exc
    section = raw.get("installed") or raw.get("web")
    if not section or not section.get("client_id"):
        raise SystemExit(
            f"{path} does not look like an OAuth client-secret JSON "
            "(expected an 'installed' or 'web' section)"
        )
    return section["client_id"], section["client_secret"]


class _CallbackServer:
    """One-shot localhost HTTP server capturing the ?code= redirect."""

    def __init__(self, port: int) -> None:
        self.result: dict = {}
        self.done = Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                query = parse_qs(urlparse(self.path).query)
                outer.result = {k: v[0] for k, v in query.items()}
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                ok = "code" in query
                page = (
                    "<h2>Task Master connected to Gmail.</h2>"
                    "You can close this window." if ok else
                    f"<h2>Authorization failed.</h2>"
                    f"<pre>{query.get('error', ['unknown error'])[0]}</pre>"
                )
                self.wfile.write(page.encode("utf-8"))
                outer.done.set()

            def log_message(self, *args) -> None:  # silence console
                pass

        self.server = HTTPServer(("127.0.0.1", port), Handler)

    def wait(self, timeout_s: float = 300) -> bool:
        self.server.timeout = 1
        deadline = time.time() + timeout_s
        while not self.done.is_set():
            if time.time() > deadline:
                return False
            self.server.handle_request()
        return True

    def close(self) -> None:
        self.server.server_close()


def run_authorization_flow(
    connector: Optional[GmailApiMailboxConnector] = None,
    port: int = 8899,
    open_browser: bool = True,
) -> dict:
    """Interactive installed-app OAuth flow. Returns the saved token dict."""
    import os

    connector = connector or GmailApiMailboxConnector()
    port = int(os.environ.get("TASK_MASTER_GMAIL_OAUTH_PORT", port))
    client_id, client_secret = load_client_secret(connector.client_secret_file)
    redirect_uri = f"http://localhost:{port}"

    state = secrets.token_urlsafe(24)
    auth_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    auth_url = f"{_AUTH_URL}?{urlencode(auth_params)}"

    listener = _CallbackServer(port)
    print(f"\nWaiting for Google's redirect on {redirect_uri} ...")
    print("Opening browser for Google consent (read-only Gmail scope).\n")
    print(f"AUTH_URL: {auth_url}\n")
    if open_browser:
        if not webbrowser.open(auth_url):
            print("Could not launch a browser automatically — "
                  "open the AUTH_URL above manually.")
    else:
        print("Browser not auto-opened — use the AUTH_URL above.\n")

    try:
        completed = listener.wait(timeout_s=300)
    finally:
        listener.close()

    if not completed:
        raise SystemExit("Timed out waiting for the OAuth redirect (5 min).")
    query = listener.result
    if query.get("error"):
        raise SystemExit(f"Authorization failed: {query['error']}")
    if query.get("state") != state:
        raise SystemExit("OAuth state mismatch — aborting (possible CSRF).")
    code = query.get("code")
    if not code:
        raise SystemExit("No authorization code received from Google.")

    resp = requests.post(_TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code": code,
    }, timeout=30)
    if resp.status_code != 200:
        raise SystemExit(
            f"Token exchange failed (HTTP {resp.status_code}): {resp.text[:300]}"
        )
    payload = resp.json()
    if not payload.get("refresh_token"):
        raise SystemExit(
            "Google did not return a refresh_token — revoke the app at "
            "https://myaccount.google.com/permissions and retry "
            "(prompt=consent should normally force its return)."
        )

    token = {
        "access_token": payload["access_token"],
        "refresh_token": payload["refresh_token"],
        "expires_at": time.time() + int(payload.get("expires_in", 3600)),
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": payload.get("scope", _SCOPE),
    }
    connector._save_token(token)

    whoami = ""
    try:
        r = requests.get(f"{_API_BASE}/profile",
                         headers={"Authorization": f"Bearer {token['access_token']}"},
                         timeout=30)
        if r.status_code == 200:
            whoami = r.json().get("emailAddress", "")
    except requests.exceptions.RequestException:
        pass

    print(f"Gmail authorized OK{f' for {whoami}' if whoami else ''}.")
    print(f"Token cached at {connector.token_file}")
    return token
