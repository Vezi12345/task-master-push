"""Adaptive rate limiting and politeness for outbound requests.

A single shared :class:`PolitenessPolicy` enforces a minimum gap between
requests to the same host and enlarges that gap adaptively when a host
throttles (HTTP 429) or fails intermittently (5xx). It is dependency-free on
the rest of the application so it can be unit-tested in isolation and reused
across sources and browser calls.

Model
-----
Each host keeps a *current gap* — the minimum wall-clock separation enforced
between consecutive requests to that host. It starts at ``min_interval``,
grows geometrically on every failure (bounded by ``max_backoff``), is reset
by a success, and decays back toward ``min_interval`` after a quiet period.
``wait_for`` only sleeps the *remaining* gap since the last request, so a
burst within the gap is throttled but a mature quiet period adds nothing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

DEFAULT_MIN_INTERVAL = 0.5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_BACKOFF = 60.0
DEFAULT_DECAY_AFTER = 30.0


@dataclass
class _HostState:
    current_gap: float = 0.0
    last_request_time: float = 0.0
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        if self.current_gap <= 0:
            self.current_gap = DEFAULT_MIN_INTERVAL


class PolitenessPolicy:
    """Enforce politeness and adaptive backoff per host.

    Parameters
    ----------
    min_interval:
        Minimum seconds between requests to the same host.
    initial_backoff:
        Backoff (seconds) applied after the first failure.
    backoff_factor:
        Multiplier applied to the backoff for each consecutive failure.
    max_backoff:
        Ceiling (seconds) a single backoff gap can reach.
    decay_after:
        Seconds of quiet after which the enforced gap decays toward
        ``min_interval``.
    monotonic / sleep:
        Injectable clock for deterministic testing (defaults to the real
        ``time.monotonic`` / ``time.sleep``).
    """

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        decay_after: float = DEFAULT_DECAY_AFTER,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.decay_after = decay_after
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._hosts: dict[str, _HostState] = {}

    # -- host helpers -------------------------------------------------------
    def _host(self, netloc: str) -> _HostState:
        key = (netloc or "").lower()
        st = self._hosts.get(key)
        if st is None:
            st = _HostState(current_gap=self.min_interval)
            self._hosts[key] = st
        return st

    def _backoff_for(self, failures: int) -> float:
        if failures <= 0:
            return self.min_interval
        return min(
            self.max_backoff,
            self.initial_backoff * (self.backoff_factor ** (failures - 1)),
        )

    def _computed_gap(self, st: _HostState) -> float:
        if st.consecutive_failures <= 0:
            return self.min_interval
        elapsed = max(0.0, self._monotonic() - st.last_request_time)
        if elapsed >= self.decay_after:
            return self.min_interval
        base = max(self.min_interval, st.current_gap)
        fraction = 1.0 - (elapsed / self.decay_after)
        return self.min_interval + (base - self.min_interval) * fraction

    # -- public API ---------------------------------------------------------
    def wait_for(self, netloc: str) -> float:
        """Block until the next request to *netloc* is permitted.

        Returns the number of seconds actually slept."""
        with self._lock:
            st = self._host(netloc)
            now = self._monotonic()
            gap = self._computed_gap(st)
            elapsed = max(0.0, now - st.last_request_time)
            delay = max(0.0, gap - elapsed)
            st.last_request_time = now
        if delay > 0:
            self._sleep(delay)
        return delay

    def reported_failure(self, netloc: str) -> None:
        """Record a failure (or throttle signal) against *netloc*."""
        with self._lock:
            st = self._host(netloc)
            st.consecutive_failures += 1
            backoff = self._backoff_for(st.consecutive_failures)
            st.current_gap = max(st.current_gap, backoff)

    def reported_success(self, netloc: str) -> None:
        """Record a success against *netloc* (resets its backoff)."""
        with self._lock:
            st = self._host(netloc)
            st.consecutive_failures = 0
            st.current_gap = self.min_interval

    def host_stats(self, netloc: str) -> dict:
        with self._lock:
            st = self._host(netloc)
            return {
                "netloc": (netloc or "").lower(),
                "consecutive_failures": st.consecutive_failures,
                "enforced_delay": round(self._computed_gap(st), 3),
                "last_request_ago": round(max(0.0, self._monotonic() - st.last_request_time), 3),
            }

    def clear_host(self, netloc: str) -> None:
        with self._lock:
            self._hosts.pop((netloc or "").lower(), None)

    def reset(self) -> None:
        with self._lock:
            self._hosts.clear()


def host_of(url: str) -> str:
    """Return the netloc (host[:port]) of a URL, or '' if unparseable."""
    from urllib.parse import urlparse

    return urlparse(url or "").netloc
