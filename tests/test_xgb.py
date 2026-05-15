"""Tests for XGBoost forecaster"""
import numpy as np
import pandas as pd
import pytest

try:
    import xgboost as xgb  # noqa
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


pytestmark = pytest.mark.skipif(not HAS_XGB, reason="xgboost not installed")


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    f3 = rng.normal(0, 1, n)

    # Synthetic target with real signal
    target = 0.3 * f1 - 0.2 * f2 + 0.05 * rng.normal(0, 1, n)
    target_logret = target * 0.02  # scale to realistic log-return range

    df = pd.DataFrame({
        "date": dates,
        "f1": f1, "f2": f2, "f3": f3,
        "target_logret_5d": target_logret,
        "target_direction_5d": np.sign(target).astype(int),
    })
    return df


def test_xgb_regressor_fits(training_data):
    from src.models.xgb_model import XGBoostForecaster, XGBConfig
    model = XGBoostForecaster(XGBConfig(
        horizon_days=5, task="regression",
        n_estimators=50, n_splits=3,
    ))
    model.fit(training_data, feature_cols=["f1", "f2", "f3"])
    assert model.model is not None
    assert len(model.cv_metrics) == 3
    assert model.feature_importance is not None


def test_xgb_predicts(training_data):
    from src.models.xgb_model import XGBoostForecaster, XGBConfig
    model = XGBoostForecaster(XGBConfig(
        horizon_days=5, task="regression",
        n_estimators=30, n_splits=2,
    ))
    model.fit(training_data, feature_cols=["f1", "f2", "f3"])
    preds = model.predict(training_data)
    assert len(preds) == len(training_data)
    assert preds.dtype == np.float32 or preds.dtype == np.float64


def test_xgb_save_load(training_data, tmp_path):
    from src.models.xgb_model import XGBoostForecaster, XGBConfig
    model = XGBoostForecaster(XGBConfig(
        horizon_days=5, task="regression",
        n_estimators=20, n_splits=2,
    ))
    model.fit(training_data, feature_cols=["f1", "f2", "f3"])
    p = tmp_path / "saved"
    model.save(str(p))
    loaded = XGBoostForecaster.load(str(p))
    assert loaded.feature_cols == ["f1", "f2", "f3"]
    p1 = model.predict(training_data)
    p2 = loaded.predict(training_data)
    np.testing.assert_array_almost_equal(p1, p2)
