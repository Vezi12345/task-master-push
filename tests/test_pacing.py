"""Tests for the adaptive rate-limiting / politeness policy."""
from __future__ import annotations

import pytest

from application.pacing import PolitenessPolicy, host_of
from sources._common import PACER, fetch_json


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.slept += seconds
        self.t += seconds

    def advance(self, seconds):
        self.t += seconds


def make_policy(**kw):
    clock = FakeClock()
    kw.setdefault("monotonic", clock.monotonic)
    kw.setdefault("sleep", clock.sleep)
    return PolitenessPolicy(**kw), clock


def test_first_request_waits_min_interval():
    p, clock = make_policy(min_interval=0.5)
    slept = p.wait_for("example.com")
    assert slept == 0.5
    assert clock.slept == 0.5


def test_host_is_isolated():
    p, clock = make_policy(min_interval=0.5)
    p.reported_failure("host-a")
    p.reported_failure("host-a")
    # host-b does not inherit host-a's backoff.
    assert p.host_stats("host-b")["consecutive_failures"] == 0
    assert p.host_stats("host-b")["enforced_delay"] == pytest.approx(0.5)
    assert p.host_stats("host-a")["enforced_delay"] > 0.5


def test_second_request_same_host_no_extra_sleep_within_interval():
    # After the first request slept the full interval, the clock has advanced
    # past it, so an immediate second request needs no further wait.
    p, clock = make_policy(min_interval=0.5)
    p.wait_for("h")
    # already advanced 0.5 -> no extra delay needed
    assert p.wait_for("h") == 0.0


def test_failure_increases_enforced_delay():
    p, clock = make_policy(min_interval=0.2, initial_backoff=1.0,
                           backoff_factor=2.0, max_backoff=60.0)
    p.reported_failure("h")
    d1 = p.host_stats("h")["enforced_delay"]
    p.reported_failure("h")
    d2 = p.host_stats("h")["enforced_delay"]
    p.reported_failure("h")
    d3 = p.host_stats("h")["enforced_delay"]
    assert d1 < d2 < d3


def test_success_resets_failures():
    p, clock = make_policy(min_interval=0.2, initial_backoff=1.0,
                           backoff_factor=2.0)
    p.reported_failure("h")
    p.reported_failure("h")
    p.reported_success("h")
    stats = p.host_stats("h")
    assert stats["consecutive_failures"] == 0
    assert stats["enforced_delay"] == 0.2  # back to min interval


def test_backoff_bounded_by_max():
    p, clock = make_policy(min_interval=0.2, initial_backoff=1.0,
                           backoff_factor=10.0, max_backoff=5.0)
    for _ in range(20):
        p.reported_failure("h")
    d = p.host_stats("h")["enforced_delay"]
    assert d <= 5.0


def test_backoff_decays_after_quiet_period():
    p, clock = make_policy(min_interval=0.2, initial_backoff=2.0,
                           backoff_factor=2.0, decay_after=20.0)
    p.reported_failure("h")
    assert p.host_stats("h")["enforced_delay"] > 0.5
    clock.advance(20.0)
    # After quiet >= decay_after, enforced delay returns to the minimum.
    assert p.host_stats("h")["enforced_delay"] == pytest.approx(0.2)


def test_clear_and_reset():
    p, clock = make_policy(min_interval=0.2)
    p.wait_for("h")
    assert p.host_stats("h")["enforced_delay"] == 0.2
    p.clear_host("h")
    assert "h" not in p._hosts
    p.wait_for("h2")
    p.reset()
    assert p._hosts == {}


def test_host_of():
    assert host_of("https://api.lever.co/v0/postings/x") == "api.lever.co"
    assert host_of("not a url") == ""


# ------------------------------------------------------------- integration

def test_fetch_json_uses_pacer_but_tests_patch_it_out(monkeypatch):
    """The shared PACER is real; but callers replace ``fetch_json`` (as the
    ATS sources do), so no politeness delay is imposed on mocked tests."""
    # Sanity probe of the shared policy only — no requests are made here.
    assert isinstance(PACER, PolitenessPolicy)
    assert PACER.min_interval > 0
