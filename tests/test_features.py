"""Tests for feature engineering"""
import numpy as np
import pandas as pd
import pytest
from src.features.builder import (
    add_return_features, add_volatility_features, add_momentum_features,
    add_macro_features, add_seasonality_features, make_target,
    build_feature_matrix, get_feature_columns,
)


@pytest.fixture
def synthetic_df():
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    prices = 60 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "date": dates,
        "wti_close": prices,
        "brent_close": prices + 3,
        "dxy_close": 95 + np.cumsum(rng.normal(0, 0.1, n)),
        "spx_close": 3000 + np.cumsum(rng.normal(1, 5, n)),
        "vix_close": np.abs(20 + rng.normal(0, 5, n)),
        "gold_close": 1500 + np.cumsum(rng.normal(0.5, 2, n)),
        "tnx_close": 2 + rng.normal(0, 0.05, n),
    })


def test_add_return_features(synthetic_df):
    out = add_return_features(synthetic_df)
    assert "ret_1d" in out.columns
    assert "ret_5d" in out.columns
    assert "ret_252d" in out.columns
    # First row of 1d return should be NaN
    assert pd.isna(out["ret_1d"].iloc[0])
    # Later rows should be finite
    assert not pd.isna(out["ret_1d"].iloc[10])


def test_add_volatility_features(synthetic_df):
    df = add_return_features(synthetic_df)
    out = add_volatility_features(df)
    assert "vol_21d" in out.columns
    assert "drawdown_1y" in out.columns
    # Drawdown must be non-positive
    assert (out["drawdown_1y"].dropna() <= 0).all()


def test_add_momentum_features(synthetic_df):
    out = add_momentum_features(synthetic_df)
    assert "rsi_14" in out.columns
    assert "sma_20" in out.columns
    # RSI bounded 0-100
    rsi = out["rsi_14"].dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_seasonality_cyclic(synthetic_df):
    out = add_seasonality_features(synthetic_df)
    assert "month_sin" in out.columns
    assert "month_cos" in out.columns
    # sin^2 + cos^2 = 1
    s2 = out["month_sin"]**2 + out["month_cos"]**2
    assert np.allclose(s2, 1.0, atol=1e-6)


def test_make_target_shifts_correctly(synthetic_df):
    out = make_target(synthetic_df, horizon_days=5)
    assert "target_logret_5d" in out.columns
    # The last 5 rows should have NaN target (no future data)
    assert out["target_logret_5d"].iloc[-5:].isna().all()


def test_make_target_direction_is_categorical(synthetic_df):
    out = make_target(synthetic_df, horizon_days=5)
    direction = out["target_direction_5d"].dropna().unique()
    assert set(direction).issubset({-1, 0, 1})


def test_full_pipeline_runs(synthetic_df):
    out = build_feature_matrix(synthetic_df, horizon_days=5, include_events=False)
    # Should have substantially more columns than input
    assert len(out.columns) > 30
    # Should have a target column
    assert "target_logret_5d" in out.columns


def test_get_feature_columns_excludes_target_and_date():
    df = pd.DataFrame({
        "date": [1], "wti_close": [60], "ret_1d": [0.01],
        "target_logret_5d": [0.02], "rsi_14": [50],
    })
    cols = get_feature_columns(df)
    assert "date" not in cols
    assert "target_logret_5d" not in cols
    assert "rsi_14" in cols
