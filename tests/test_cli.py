import io
import sys

import pytest

import cli
from agent.rank import RankedJob
from sources.base import Job


class _StreamProxy:
    def __init__(self, inner):
        self.inner = inner

    def reconfigure(self, **kwargs):
        self.inner.reconfigure(**kwargs)

    def write(self, text):
        self.inner.write(text)

    def flush(self):
        self.inner.flush()


def test_ranked_output_survives_cp1252_console(monkeypatch):
    wrapped = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", _StreamProxy(wrapped))
    cli._configure_console()
    job = Job(
        title="Junior Software Developer",
        company="Luno",
        location="Remote (South Africa)",
        remote=True,
    )
    ranked = [
        RankedJob(
            job=job,
            score=93,
            reasons=["Role:      ✓ 'software developer' in title", "Remote:    ✓ accepts SA applicants"],
        )
    ]
    cli._print_ranked(ranked)
    wrapped.flush()
    rendered = wrapped.buffer.getvalue().decode("utf-8", errors="replace")
    assert "Junior Software Developer" in rendered
    assert "93%" in rendered
