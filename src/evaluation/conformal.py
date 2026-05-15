"""
Conformal Prediction for distribution-free uncertainty intervals.

Implementation of section 11.3 of MATHEMATICAL_MODEL.md.

Given:
  - A trained predictor f̂(·)
  - A calibration set {(xᵢ, yᵢ)}ᵢ₌₁ⁿ disjoint from training data
  - Desired miscoverage level α ∈ (0,1) (e.g., α=0.1 for 90% coverage)

The conformal prediction interval at a new point xₜ is:

    [ f̂(xₜ) − q_{1-α},  f̂(xₜ) + q_{1-α} ]

where q_{1-α} is the (1-α)·(n+1)/n empirical quantile of the calibration
residuals |yᵢ − f̂(xᵢ)|.

This provides MARGINAL COVERAGE GUARANTEE:
    P( yₜ ∈ interval ) ≥ 1 − α

under the exchangeability assumption. For time series, exchangeability holds
approximately within stationary segments, so we apply it within rolling windows.

For sharper intervals we additionally support:
  - LOCALLY WEIGHTED variant where the interval width depends on x (heteroscedastic)
  - QUANTILE conformal where the base predictor is a quantile regressor
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConformalConfig:
    miscoverage_level: float = 0.1  # α; 0.1 ⇒ 90% intervals
    method: str = "split"           # "split" or "weighted"
    rolling_window: Optional[int] = None  # None ⇒ use entire calibration set


class SplitConformalPredictor:
    """
    Standard split-conformal prediction.

    Workflow:
        1. Split training data into proper-train and calibration sets.
        2. Train base predictor on proper-train.
        3. Compute residuals αᵢ = |yᵢ − ŷᵢ| on calibration set.
        4. At test time, return [ŷ − q, ŷ + q] where q is the (1-α) quantile of {αᵢ}.
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        self.config = config or ConformalConfig()
        self.calibration_residuals: Optional[np.ndarray] = None
        self.q_hat: Optional[float] = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformalPredictor":
        """
        Compute the conformal quantile from calibration data.

        Args:
            y_true: shape (n,) — true targets
            y_pred: shape (n,) — base model predictions

        Returns: self (for chaining)
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 30:
            raise ValueError(f"Need at least 30 valid calibration samples, got {mask.sum()}")

        residuals = np.abs(y_true[mask] - y_pred[mask])
        self.calibration_residuals = residuals

        n = len(residuals)
        # Finite-sample correction: rank ⌈(n+1)(1-α)⌉ / n
        adjusted_level = min(1.0, np.ceil((n + 1) * (1 - self.config.miscoverage_level)) / n)
        self.q_hat = float(np.quantile(residuals, adjusted_level, method="higher"))

        logger.info(
            f"Conformal calibrated: n={n}, α={self.config.miscoverage_level}, "
            f"q̂={self.q_hat:.5f}, median residual={np.median(residuals):.5f}"
        )
        return self

    def predict_interval(self, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Return prediction intervals around base predictions.

        Args:
            y_pred: shape (n,) — base model predictions

        Returns: DataFrame with columns [lower, point, upper, width]
        """
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() first")

        y_pred = np.asarray(y_pred).ravel()
        return pd.DataFrame({
            "lower": y_pred - self.q_hat,
            "point": y_pred,
            "upper": y_pred + self.q_hat,
            "width": np.full_like(y_pred, 2 * self.q_hat),
        })

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Check empirical coverage on held-out test data.

        Returns a dict with coverage rate, average width, conditional coverage by quantile.
        """
        intervals = self.predict_interval(y_pred)
        y_true = np.asarray(y_true).ravel()

        mask = np.isfinite(y_true)
        covered = (y_true[mask] >= intervals["lower"].values[mask]) & \
                  (y_true[mask] <= intervals["upper"].values[mask])

        target = 1 - self.config.miscoverage_level
        empirical = float(covered.mean())

        return {
            "target_coverage": target,
            "empirical_coverage": empirical,
            "coverage_gap": empirical - target,
            "mean_width": float(intervals["width"].mean()),
            "median_width": float(intervals["width"].median()),
            "n_test": int(mask.sum()),
        }


class WeightedConformalPredictor(SplitConformalPredictor):
    """
    Heteroscedastic ("locally weighted") variant.

    Instead of |y − ŷ|, the non-conformity score is |y − ŷ| / σ̂(x), where
    σ̂ is an auxiliary uncertainty estimate (e.g., XGBoost predicting absolute
    residual magnitude). This produces NARROWER intervals when the model is
    confident and WIDER when uncertain — sharper average coverage.
    """

    def __init__(self, config: Optional[ConformalConfig] = None):
        super().__init__(config)
        self.q_hat = None

    def calibrate_weighted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_hat: np.ndarray,
    ) -> "WeightedConformalPredictor":
        """
        Args:
            y_true: true targets on calibration set
            y_pred: base predictions on calibration set
            sigma_hat: per-sample uncertainty estimates (must be > 0)
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        sigma_hat = np.asarray(sigma_hat).ravel()

        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (sigma_hat > 1e-8)
        scores = np.abs(y_true[mask] - y_pred[mask]) / sigma_hat[mask]
        self.calibration_residuals = scores

        n = len(scores)
        adjusted = min(1.0, np.ceil((n + 1) * (1 - self.config.miscoverage_level)) / n)
        self.q_hat = float(np.quantile(scores, adjusted, method="higher"))
        logger.info(f"Weighted conformal calibrated: q̂={self.q_hat:.4f}")
        return self

    def predict_interval_weighted(
        self,
        y_pred: np.ndarray,
        sigma_hat: np.ndarray,
    ) -> pd.DataFrame:
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate_weighted() first")

        y_pred = np.asarray(y_pred).ravel()
        sigma_hat = np.asarray(sigma_hat).ravel()
        half_width = self.q_hat * sigma_hat
        return pd.DataFrame({
            "lower": y_pred - half_width,
            "point": y_pred,
            "upper": y_pred + half_width,
            "width": 2 * half_width,
        })


# =============================================================================
# Rolling-window conformal for non-stationary time series
# =============================================================================
class RollingConformalPredictor:
    """
    For time series, the exchangeability assumption breaks when distribution
    drifts. We adapt by recomputing q̂ on a rolling window of recent residuals.

    This is a simplified version of Adaptive Conformal Inference (Gibbs &
    Candès, 2021).
    """

    def __init__(self, miscoverage: float = 0.1, window: int = 252):
        self.miscoverage = miscoverage
        self.window = window
        self.residual_history: list[float] = []

    def update(self, y_true: float, y_pred: float) -> None:
        """Record a new (true, pred) pair. Call after each forecast becomes observable."""
        self.residual_history.append(abs(y_true - y_pred))
        if len(self.residual_history) > 4 * self.window:
            self.residual_history = self.residual_history[-2 * self.window:]

    def current_q_hat(self) -> Optional[float]:
        """The (1-α) quantile over the rolling window of recent residuals."""
        if len(self.residual_history) < 30:
            return None
        recent = self.residual_history[-self.window:]
        n = len(recent)
        adjusted = min(1.0, np.ceil((n + 1) * (1 - self.miscoverage)) / n)
        return float(np.quantile(recent, adjusted, method="higher"))

    def predict_interval(self, y_pred: float) -> Optional[tuple[float, float]]:
        q = self.current_q_hat()
        if q is None:
            return None
        return (y_pred - q, y_pred + q)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic test
    rng = np.random.default_rng(0)
    n_cal, n_test = 500, 200

    # True data: y = sin(x) + noise
    x_cal = rng.uniform(0, 10, n_cal)
    y_cal = np.sin(x_cal) + rng.normal(0, 0.3, n_cal)
    pred_cal = np.sin(x_cal) + rng.normal(0, 0.1, n_cal)  # imperfect predictor

    x_test = rng.uniform(0, 10, n_test)
    y_test = np.sin(x_test) + rng.normal(0, 0.3, n_test)
    pred_test = np.sin(x_test) + rng.normal(0, 0.1, n_test)

    print("Split Conformal (90% target):")
    cp = SplitConformalPredictor(ConformalConfig(miscoverage_level=0.1))
    cp.calibrate(y_cal, pred_cal)
    coverage = cp.evaluate_coverage(y_test, pred_test)
    print(f"  Empirical coverage: {coverage['empirical_coverage']:.3f} "
          f"(target {coverage['target_coverage']:.3f})")
    print(f"  Mean interval width: {coverage['mean_width']:.4f}")
