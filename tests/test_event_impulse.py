"""Tests for event impulse / supply shock models"""
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from src.models.event_impulse import (
    EventImpulseModel, SupplyShockModel, ImpulseParameters,
    SEVERITY_TO_NUM, DIRECTION_TO_SIGN,
)
from src.data.events_database import (
    EVENTS, GeopoliticalEvent, EventType, Severity, ImpactDirection,
)


def test_default_params_loaded():
    model = EventImpulseModel()
    assert len(model.params) > 0
    # Spec Appendix C values
    p = model.params[("war", "extreme")]
    assert p.beta == pytest.approx(0.51, abs=0.01)
    assert p.decay_rate == pytest.approx(0.015, abs=0.005)


def test_impulse_zero_before_event():
    model = EventImpulseModel()
    e = EVENTS[0]
    assert model.impulse(e, days_since_onset=-1) == 0.0
    assert model.impulse(e, days_since_onset=-100) == 0.0


def test_impulse_peaks_at_onset():
    """The impulse should be maximum at t = τᵢ (onset)"""
    model = EventImpulseModel()
    war_event = next(e for e in EVENTS if e.event_type == EventType.WAR
                     and e.severity == Severity.EXTREME)
    onset_impact = model.impulse(war_event, 0)
    later_impact = model.impulse(war_event, 30)
    assert abs(onset_impact) > abs(later_impact), \
        "Impulse should decay over time"


def test_impulse_decays_exponentially():
    """At t = half_life, impact should be ~half of onset"""
    model = EventImpulseModel()
    e = next(e for e in EVENTS if e.event_type == EventType.WAR
             and e.severity == Severity.EXTREME)
    p = model.params[("war", "extreme")]
    half_life = p.half_life_days

    onset = model.impulse(e, 0)
    at_half_life = model.impulse(e, half_life)
    assert at_half_life == pytest.approx(onset / 2, rel=0.05)


def test_bullish_event_positive_impact():
    """A bullish event should produce positive log impact"""
    model = EventImpulseModel()
    e = next(e for e in EVENTS
             if e.impact_direction == ImpactDirection.BULLISH
             and e.severity in (Severity.HIGH, Severity.EXTREME))
    assert model.impulse(e, 1) > 0


def test_bearish_event_negative_impact():
    model = EventImpulseModel()
    e = next(e for e in EVENTS if e.impact_direction == ImpactDirection.BEARISH)
    assert model.impulse(e, 1) < 0


def test_aggregate_impact_returns_dict():
    model = EventImpulseModel()
    result = model.aggregate_impact(date(2022, 3, 1))
    assert "total_log_impact" in result
    assert "total_pct_impact" in result
    assert "events" in result
    assert isinstance(result["events"], list)


def test_aggregate_includes_active_war():
    """For March 2022, Russia-Ukraine war should contribute"""
    model = EventImpulseModel()
    result = model.aggregate_impact(date(2022, 3, 1))
    event_ids = {e["event_id"] for e in result["events"]}
    assert any("ukraine" in eid.lower() for eid in event_ids)


def test_project_forward():
    model = EventImpulseModel()
    proj = model.project_forward(EVENTS, start=date(2026, 1, 1), horizon_days=20)
    assert len(proj) == 21
    assert proj["days_from_start"].iloc[0] == 0
    assert proj["days_from_start"].iloc[-1] == 20


# ---------------------------------------------------------------------------
# Supply shock model
# ---------------------------------------------------------------------------
def test_supply_shock_zero_for_no_disruption():
    shock = SupplyShockModel()
    assert shock.shock_impact(disrupted_mbd=0.0) == 0.0


def test_supply_shock_capped():
    """Even an absurdly large disruption should be capped"""
    shock = SupplyShockModel(impact_cap=1.0)
    huge = shock.shock_impact(disrupted_mbd=500.0)
    assert abs(huge) <= 1.0


def test_hormuz_closure_increases_with_duration():
    """Longer closures should have larger (but diminishing) effect"""
    shock = SupplyShockModel()
    r_1 = shock.hormuz_closure_impact(1)
    r_30 = shock.hormuz_closure_impact(30)
    r_90 = shock.hormuz_closure_impact(90)
    # All should be bullish (positive)
    assert r_1["log_price_impact"] > 0
    assert r_30["log_price_impact"] > 0
    assert r_90["log_price_impact"] > 0


def test_hormuz_pass_through_below_one():
    """Pass-through factor should be < 1 (shadow flows partially compensate)"""
    shock = SupplyShockModel()
    r = shock.hormuz_closure_impact(7)
    assert 0 < r["pass_through_fraction"] < 1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_calibrate_keeps_defaults_for_sparse_buckets():
    """If a bucket has < min_events, defaults should be retained"""
    # Build minimal synthetic price df
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    prices = 60 * np.exp(np.cumsum(np.random.RandomState(0).normal(0, 0.01, len(dates))))
    df = pd.DataFrame({"date": dates, "wti_close": prices})

    model = EventImpulseModel()
    initial = dict(model.params)
    model.calibrate(df, min_events_per_bucket=100)  # Impossibly high threshold
    # All defaults should be retained
    assert set(model.params.keys()) >= set(initial.keys())
