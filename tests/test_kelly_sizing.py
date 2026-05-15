"""Tests for Kelly position sizing"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.kelly_sizing import (
    KellySizer, KellyConfig, KellyBacktester, KellyBacktestConfig
)


def test_basic_kelly_formula():
    """Without modifiers, f = kelly_fraction · μ/σ²"""
    sizer = KellySizer(KellyConfig(
        kelly_fraction=0.5,
        confidence_threshold=0.0,    # disable gate
        use_disagreement_in_denom=False,
        min_position=0.0,
        regime_downscale={},          # no regime downscale
    ))
    result = sizer.size(
        predicted_log_return=0.01,
        forecast_variance=0.01,
        confidence=1.0,
        regime_name=None,
    )
    # raw_kelly = 0.01/0.01 = 1.0; half-Kelly = 0.5
    assert result["raw_kelly"] == pytest.approx(1.0)
    assert result["position"] == pytest.approx(0.5)


def test_confidence_gate():
    """Below threshold → zero position"""
    sizer = KellySizer(KellyConfig(confidence_threshold=0.5))
    result = sizer.size(
        predicted_log_return=0.05,
        forecast_variance=0.001,
        confidence=0.3,
    )
    assert result["position"] == 0.0
    assert any("confidence" in r for r in result["rejected_reasons"])


def test_position_cap():
    """High edge / low vol should be capped"""
    sizer = KellySizer(KellyConfig(kelly_fraction=1.0, max_position=1.0,
                                    confidence_threshold=0.0,
                                    use_disagreement_in_denom=False))
    result = sizer.size(
        predicted_log_return=0.1,
        forecast_variance=0.001,    # raw_kelly = 100
        confidence=1.0,
    )
    assert abs(result["position"]) <= 1.0
    assert result["raw_kelly"] > 50


def test_regime_downscale():
    """War regime should produce smaller position than normal regime"""
    # Use small edge so no capping occurs
    sizer = KellySizer(KellyConfig(
        kelly_fraction=1.0,
        max_position=10.0,
        confidence_threshold=0.0,
        use_disagreement_in_denom=False,
        regime_downscale={"war_regime": 0.4, "normal_up": 1.0},
    ))
    # μ=0.001, σ²=0.001 ⇒ raw_kelly=1.0, well below any cap
    normal = sizer.size(
        predicted_log_return=0.001, forecast_variance=0.001,
        confidence=1.0, regime_name="normal_up",
    )["position"]
    war = sizer.size(
        predicted_log_return=0.001, forecast_variance=0.001,
        confidence=1.0, regime_name="war_regime",
    )["position"]
    assert abs(war) < abs(normal)
    assert abs(war / normal - 0.4) < 0.05


def test_disagreement_penalty():
    """High disagreement should shrink position via variance inflation"""
    base_cfg = KellyConfig(
        kelly_fraction=0.5,
        max_position=10.0,
        confidence_threshold=0.0,
        use_disagreement_in_denom=True,
        disagreement_weight=100.0,
    )
    sizer = KellySizer(base_cfg)
    low_d = sizer.size(0.02, 0.001, confidence=1.0, disagreement=0.001)["position"]
    high_d = sizer.size(0.02, 0.001, confidence=1.0, disagreement=0.03)["position"]
    assert abs(high_d) < abs(low_d)


def test_trump_volatility_penalty():
    sizer = KellySizer(KellyConfig(
        kelly_fraction=0.5, max_position=10.0,
        confidence_threshold=0.0, use_disagreement_in_denom=False,
        trump_volatility_penalty=0.5,
    ))
    no_tvol = sizer.size(0.02, 0.001, confidence=1.0, trump_volatility_factor=0.0)["position"]
    high_tvol = sizer.size(0.02, 0.001, confidence=1.0, trump_volatility_factor=1.0)["position"]
    # With tvol=1.0 and penalty=0.5, multiplier = 1 - 0.5*1 = 0.5
    assert abs(high_tvol / no_tvol - 0.5) < 0.05


def test_sign_preserved():
    """Bearish predictions produce short positions"""
    sizer = KellySizer(KellyConfig(confidence_threshold=0.0,
                                    use_disagreement_in_denom=False))
    result = sizer.size(
        predicted_log_return=-0.02,
        forecast_variance=0.001,
        confidence=1.0,
    )
    assert result["position"] < 0


def test_backtester_runs():
    """Smoke test of the full pipeline"""
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="5B"),
        "pred_ensemble": rng.normal(0, 0.015, n),
        "actual": rng.normal(0, 0.02, n),
        "vol_21d": np.abs(rng.normal(0.3, 0.05, n)),
        "confidence": rng.uniform(0.4, 0.9, n),
        "pred_disagreement": np.abs(rng.normal(0.005, 0.002, n)),
        "regime_name": rng.choice(["normal_up", "war_regime", "calm_drift"], n),
        "trump_volatility_factor": rng.uniform(0, 0.5, n),
    })

    sizer = KellySizer()
    bt = KellyBacktester(sizer)
    result = bt.run(df)
    assert "equity_curve" in result
    assert "metrics" in result
    assert "sharpe_ratio" in result["metrics"]
    assert len(result["equity_curve"]) == n
