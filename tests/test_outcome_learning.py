"""Tests for the outcome learning loop."""
from __future__ import annotations

import pytest

from application.models import Application, ApplicationStatus
from application.outcome_learning import (
    OutcomeStore,
    adjust_priority_with_payoff,
    age_application,
    application_payoff_bonus,
    band_for_status,
    compute_payoff_bonus,
    role_family,
)


def make_app(**kw):
    defaults = dict(
        id="app1", job_company="Acme Ltd", job_title="Junior Accountant",
        application_priority=70, status=ApplicationStatus.OFFER,
        date_submitted="2026-01-01T00:00:00+00:00",
    )
    defaults.update(kw)
    return Application(**defaults)


# --------------------------------------------------------------- store

def test_store_persists_and_reloads(tmp_path):
    p = tmp_path / "outcomes.json"
    s = OutcomeStore(p)
    s.record("source:real", "offer")
    s.record("source:real", "interview")
    s.save()
    s2 = OutcomeStore(p)
    assert s2.get("source:real").offer == 1
    assert s2.get("source:real").interview == 1


def test_store_missing_file_starts_empty(tmp_path):
    s = OutcomeStore(tmp_path / "nope.json")
    assert s.get("source:real") is not None
    assert s.get("source:real").offer == 0


def test_record_unknown_outcome_raises(tmp_path):
    s = OutcomeStore(tmp_path / "o.json")
    with pytest.raises(ValueError):
        s.record("k", "bogus")


def test_record_application_uses_aggregate_keys(tmp_path):
    p = tmp_path / "o.json"
    s = OutcomeStore(p)
    app = make_app(job_company="Acme (Pty) Ltd", job_title="Junior Accountant",
                   status=ApplicationStatus.OFFER)
    keys = s.record_application(app)
    assert "company:acme_(pty)_ltd" in keys
    assert "role:finance_accounting" in keys
    assert s.get("company:acme_(pty)_ltd").offer == 1
    assert s.get("role:finance_accounting").offer == 1
    assert p.exists()


def test_band_for_status_mapping():
    assert band_for_status(ApplicationStatus.OFFER) == "offer"
    assert band_for_status(ApplicationStatus.INTERVIEW) == "interview"
    assert band_for_status(ApplicationStatus.REJECTED) == "rejected"
    assert band_for_status(ApplicationStatus.SUBMITTED) == "pending"
    assert band_for_status(ApplicationStatus.DRAFT) is None


# ------------------------------------------------------------- payoff

def test_payoff_zero_without_evidence():
    assert compute_payoff_bonus(OutcomeStore(None).get("x")) == 0


def test_payoff_positive_for_offers():
    s = OutcomeStore(None)
    s.record("k", "offer")
    s.record("k", "offer")
    assert compute_payoff_bonus(s.get("k")) > 0


def test_payoff_negative_for_no_response():
    s = OutcomeStore(None)
    for _ in range(5):
        s.record("k", "no_response")
    assert compute_payoff_bonus(s.get("k")) < 0


def test_payoff_bounded():
    s = OutcomeStore(None)
    for _ in range(50):
        s.record("k", "offer")
    bonus = compute_payoff_bonus(s.get("k"))
    assert -10 <= bonus <= 10


def test_application_payoff_uses_best_key():
    store = OutcomeStore(None)
    store.record("company:acme_ltd", "offer")
    store.record("company:acme_ltd", "offer")
    app = make_app(job_company="Acme Ltd", status=ApplicationStatus.INTERVIEW)
    assert application_payoff_bonus(app, store) > 0


def test_application_payoff_zero_without_store():
    app = make_app()
    assert application_payoff_bonus(app, None) == 0


def test_adjust_priority_default_noop():
    app = make_app(application_priority=70)
    assert adjust_priority_with_payoff(app, None) == 70


def test_adjust_priority_clamps_to_bounds():
    store = OutcomeStore(None)
    for _ in range(50):
        store.record("company:acme_ltd", "no_response")
    app = make_app(job_company="Acme Ltd", application_priority=5)
    adjusted = adjust_priority_with_payoff(app, store)
    assert 0 <= adjusted <= 100


# ------------------------------------------------------------- aging

def test_age_marks_no_response_after_silence(tmp_path):
    from datetime import datetime, timedelta
    old = (datetime.now() - timedelta(days=40)).isoformat()
    app = make_app(status=ApplicationStatus.SUBMITTED, date_submitted=old)
    store = OutcomeStore(tmp_path / "o.json")
    assert age_application(app, store, silence_days=21) is True
    assert app.date_responded == "no_response"
    assert store.get("company:acme_ltd").no_response >= 1
    assert store.get("role:finance_accounting").no_response >= 1


def test_age_does_not_touch_recent(tmp_path):
    from datetime import datetime, timedelta
    recent = (datetime.now() - timedelta(days=2)).isoformat()
    app = make_app(status=ApplicationStatus.SUBMITTED, date_submitted=recent)
    store = OutcomeStore(tmp_path / "o.json")
    assert age_application(app, store, silence_days=21) is False
    assert store.get("company:acme_ltd").no_response == 0


def test_age_skips_resolved_apps(tmp_path):
    app = make_app(status=ApplicationStatus.INTERVIEW, date_submitted="2020-01-01T00:00:00+00:00")
    store = OutcomeStore(tmp_path / "o.json")
    assert age_application(app, store) is False


# ---------------------------------------------------------- role family

def test_role_family():
    assert role_family("Senior Financial Accountant") == "finance_accounting"
    assert role_family("Software Engineer") == "engineering"
    assert role_family("Weird Title 42") == ""


def test_summary_shape(tmp_path):
    s = OutcomeStore(tmp_path / "o.json")
    s.record("source:real", "offer")
    from application.outcome_learning import summary as mod_summary
    out = mod_summary(s)
    assert "source:real" in out["bonus_by_key"]
    assert out["bonus_by_key"]["source:real"]["payoff_bonus"] > 0


# ------------------------------------------------------ wiring integration

def test_transition_records_learnable_outcome(tmp_path):
    from application.lifecycle import transition
    store = OutcomeStore(tmp_path / "o.json")
    app = make_app(status=ApplicationStatus.CONFIRMED)
    transition(app, ApplicationStatus.OFFER, outcome_store=store)
    assert store.get("company:acme_ltd").offer == 1
    assert store.get("role:finance_accounting").offer == 1


def test_transition_records_rejection(tmp_path):
    from application.lifecycle import transition
    store = OutcomeStore(tmp_path / "o.json")
    app = make_app(status=ApplicationStatus.CONFIRMED)
    transition(app, ApplicationStatus.REJECTED, outcome_store=store)
    assert store.get("company:acme_ltd").rejected == 1


def test_transition_without_store_does_not_record(tmp_path):
    from application.lifecycle import transition
    store = OutcomeStore(tmp_path / "o.json")
    app = make_app(status=ApplicationStatus.DRAFT)
    transition(app, ApplicationStatus.SELECTED)
    assert store.keys() == []


def test_transition_with_store_ignores_nonlearnable(tmp_path):
    from application.lifecycle import transition
    store = OutcomeStore(tmp_path / "o.json")
    app = make_app(status=ApplicationStatus.SUBMITTED)
    transition(app, ApplicationStatus.AWAITING_CONFIRMATION,
               outcome_store=store)
    assert store.keys() == []
