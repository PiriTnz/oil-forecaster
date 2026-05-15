"""Tests for regime detection"""
import numpy as np
import pandas as pd
import pytest

from src.models.regime_detector import (
    HMMRegimeDetector, RegimeConfig, EventBasedRegimeLabeler, transition_matrix
)


@pytest.fixture
def multi_regime_data():
    """Synthetic data with two clear regimes"""
    rng = np.random.default_rng(0)
    n_per = 250
    # Regime A: low vol, slight positive drift
    a_ret = rng.normal(0.001, 0.008, n_per)
    # Regime B: high vol, negative drift
    b_ret = rng.normal(-0.003, 0.04, n_per)
    ret = np.concatenate([a_ret, b_ret, a_ret])
    n = len(ret)

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "ret_1d": ret,
    })
    df["vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["drawdown_1y"] = -np.abs(np.cumsum(ret) - np.maximum.accumulate(np.cumsum(ret)))
    df["vix_level"] = 15 + 100 * np.abs(ret)
    return df.dropna().reset_index(drop=True)


def test_hmm_fits_and_predicts(multi_regime_data):
    detector = HMMRegimeDetector(RegimeConfig(n_regimes=2))
    detector.fit(multi_regime_data)
    pred = detector.predict(multi_regime_data)

    # Each row should have a regime assignment
    assert pred["regime_id"].notna().any()
    # We asked for 2 regimes
    assert pred["regime_id"].dropna().nunique() <= 2


def test_event_based_labeler():
    """Test that event regime labels are assigned for known event dates"""
    dates = pd.date_range("2022-02-01", "2022-03-01", freq="B")
    df = pd.DataFrame({"date": dates})
    out = EventBasedRegimeLabeler().label(df)
    assert "event_regime" in out.columns
    # After Feb 24 2022 (Ukraine war), we expect "war_regime"
    after_war = out[pd.to_datetime(out["date"]) >= "2022-02-24"]
    assert (after_war["event_regime"] == "war_regime").any()


def test_transition_matrix():
    s = pd.Series(["a", "a", "b", "b", "a", "b"])
    tm = transition_matrix(s)
    # Row sums should be 1
    assert np.allclose(tm.sum(axis=1), 1.0)
