"""
Kelly Criterion Position Sizing for the trading layer.

Standard Kelly:
    f* = μ / σ²
where μ is the expected return, σ² is the variance.

For a forecasting model that produces ŷ_t = predicted log return:
    f*_t = ŷ_t / σ̂²_t
where σ̂² is the conditional return variance.

WHY THIS MATTERS (vs the naïve "long if up, short if down" strategy):
  - Same predictions, but better Sharpe through right-sized positions
  - Penalize predictions when model is uncertain (high disagreement → small position)
  - Penalize during high realized volatility (large σ → small position)
  - Account for transaction costs and prevent over-trading

SAFETY MODIFICATIONS for fat-tailed oil markets:

  1. FRACTIONAL KELLY: use λ · f* with λ ∈ [0.25, 0.5]
     Full Kelly is theoretically optimal but assumes accurate μ, σ. For oil
     with regime shifts and tail events, full Kelly leads to ruin during
     model errors. Half-Kelly is a common compromise.

  2. POSITION CAPS:
     |f| ≤ f_max  (default 1.0 = no leverage)
     prevents the formula from suggesting 5x positions during low-vol periods

  3. CONFIDENCE GATE: position is zeroed when model confidence c_t < c_min
     "if we don't believe the prediction, don't trade"

  4. REGIME-AWARE DOWNSCALING: in crisis regimes (war, supply disruption),
     ALL position sizes are scaled down by a constant β_regime < 1.
     The model may have edge but tail risks dominate.

  5. KELLY WITH UNCERTAINTY (Markowitz-blended):
       f* = ŷ / (σ² + λ_unc · disagreement²)
     adds the ensemble disagreement to the denominator so high-disagreement
     forecasts get smaller positions automatically.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KellyConfig:
    """Configuration for Kelly position sizing."""
    # Fractional Kelly multiplier
    kelly_fraction: float = 0.5     # 0.5 = half-Kelly

    # Hard caps
    max_position: float = 1.0        # |f| ≤ this (1.0 = no leverage)
    min_position: float = 0.0        # zero out tiny positions (set to ε > 0)

    # Confidence gating
    confidence_threshold: float = 0.30   # if c_t < this, position = 0

    # Uncertainty penalty
    disagreement_weight: float = 100.0    # λ_unc in the formula
    use_disagreement_in_denom: bool = True

    # Regime-aware downscaling
    regime_downscale: dict = field(default_factory=lambda: {
        "war_regime":       0.40,
        "crisis_regime":    0.40,
        "sanctions_regime": 0.70,
        "opec_regime":      0.85,
        "normal_regime":    1.00,
        "calm_uptrend":     1.00,
        "calm_drift":       1.00,
        "normal_up":        1.00,
        "normal_down":      1.00,
        "spike":            0.50,
        "crisis":           0.40,
    })

    # Volatility floor (avoid divide-by-zero / explosive sizing during quiet periods)
    min_vol: float = 0.01   # annualized; ~1% floor

    # Trump-volatility penalty: if trump_volatility_factor is high, downscale
    trump_volatility_penalty: float = 0.5  # multiplier when tvol = 1.0

    # Horizon for converting daily vol to forecast horizon
    horizon_days: int = 5


# =============================================================================
# Core sizer
# =============================================================================
class KellySizer:
    """
    Translate forecast (μ, σ², context) into position size ∈ [-f_max, +f_max].
    """

    def __init__(self, config: Optional[KellyConfig] = None):
        self.config = config or KellyConfig()

    def size(
        self,
        predicted_log_return: float,
        forecast_variance: float,
        confidence: float = 0.5,
        disagreement: Optional[float] = None,
        regime_name: Optional[str] = None,
        trump_volatility_factor: float = 0.0,
    ) -> dict:
        """
        Compute position size for a single forecast.

        Returns a dict with:
          - position: final signed position ∈ [-f_max, f_max]
          - raw_kelly: the pre-cap Kelly fraction
          - sizing_reasons: list of strings explaining adjustments
          - rejected_reasons: if position=0, why
        """
        c = self.config
        reasons = []
        rejected_reasons = []

        # --- Step 1: Confidence gate ---
        if confidence < c.confidence_threshold:
            return {
                "position": 0.0,
                "raw_kelly": 0.0,
                "sizing_reasons": [],
                "rejected_reasons": [f"confidence {confidence:.2f} < threshold {c.confidence_threshold}"],
            }

        # --- Step 2: Volatility floor & disagreement-augmented variance ---
        sigma2 = max(forecast_variance, c.min_vol ** 2)
        if c.use_disagreement_in_denom and disagreement is not None:
            sigma2_eff = sigma2 + c.disagreement_weight * (disagreement ** 2)
            reasons.append(f"variance inflated by disagreement: σ²+λ·d² = {sigma2_eff:.5f}")
        else:
            sigma2_eff = sigma2

        # --- Step 3: Raw Kelly fraction ---
        raw_kelly = predicted_log_return / sigma2_eff

        # --- Step 4: Apply fractional Kelly ---
        f = c.kelly_fraction * raw_kelly
        reasons.append(f"raw_kelly={raw_kelly:.3f} → fractional Kelly={f:.3f} (×{c.kelly_fraction})")

        # --- Step 5: Regime downscale ---
        regime_factor = 1.0
        if regime_name and regime_name in c.regime_downscale:
            regime_factor = c.regime_downscale[regime_name]
            f *= regime_factor
            reasons.append(f"regime '{regime_name}' downscale ×{regime_factor:.2f}")

        # --- Step 6: Trump-volatility penalty ---
        if trump_volatility_factor > 0.05:
            penalty = 1 - c.trump_volatility_penalty * trump_volatility_factor
            penalty = max(0.0, min(penalty, 1.0))
            f *= penalty
            reasons.append(
                f"trump_volatility {trump_volatility_factor:.2f} ⇒ multiplier {penalty:.2f}"
            )

        # --- Step 7: Hard cap ---
        if abs(f) > c.max_position:
            reasons.append(f"capped from {f:+.3f} to ±{c.max_position}")
            f = np.sign(f) * c.max_position

        # --- Step 8: Min-position dead zone ---
        if abs(f) < c.min_position:
            rejected_reasons.append(f"|f|={abs(f):.4f} < min_position {c.min_position}")
            f = 0.0

        return {
            "position": float(f),
            "raw_kelly": float(raw_kelly),
            "sizing_reasons": reasons,
            "rejected_reasons": rejected_reasons,
        }

    def size_series(
        self,
        df: pd.DataFrame,
        pred_col: str = "pred_ensemble",
        variance_col: str = "vol_21d",
        confidence_col: str = "confidence",
        disagreement_col: str = "pred_disagreement",
        regime_col: str = "regime_name",
        trump_vol_col: str = "trump_volatility_factor",
    ) -> pd.DataFrame:
        """
        Compute position sizes for a full DataFrame of forecasts.

        Returns a new DataFrame with 'position', 'raw_kelly', and reason columns.
        """
        # Convert annualized vol to per-horizon variance
        # σ²_horizon = σ²_annual · (horizon / 252)
        positions = []
        raw_kellys = []
        reasons_list = []
        rejected_list = []

        for _, row in df.iterrows():
            pred = float(row.get(pred_col, 0)) if not pd.isna(row.get(pred_col, np.nan)) else 0.0
            if pd.isna(pred) or abs(pred) < 1e-8:
                positions.append(0.0)
                raw_kellys.append(0.0)
                reasons_list.append([])
                rejected_list.append(["no prediction"])
                continue

            ann_vol = float(row.get(variance_col, 0.3)) if not pd.isna(row.get(variance_col, np.nan)) else 0.3
            sigma2_horizon = (ann_vol ** 2) * (self.config.horizon_days / 252)

            conf = float(row[confidence_col]) if confidence_col in row and not pd.isna(row[confidence_col]) else 0.5
            disagree = float(row[disagreement_col]) if disagreement_col in row and not pd.isna(row[disagreement_col]) else None
            regime = row[regime_col] if regime_col in row and not pd.isna(row[regime_col]) else None
            tvol = float(row[trump_vol_col]) if trump_vol_col in row and not pd.isna(row[trump_vol_col]) else 0.0

            result = self.size(
                predicted_log_return=pred,
                forecast_variance=sigma2_horizon,
                confidence=conf,
                disagreement=disagree,
                regime_name=str(regime) if regime else None,
                trump_volatility_factor=tvol,
            )
            positions.append(result["position"])
            raw_kellys.append(result["raw_kelly"])
            reasons_list.append(result["sizing_reasons"])
            rejected_list.append(result["rejected_reasons"])

        out = df.copy()
        out["position"] = positions
        out["raw_kelly"] = raw_kellys
        out["sizing_reasons"] = reasons_list
        out["rejected_reasons"] = rejected_list
        return out


# =============================================================================
# Strategy-level backtester using Kelly
# =============================================================================
@dataclass
class KellyBacktestConfig:
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0   # 5 basis points per change in position
    leverage_cost_bps: float = 2.0      # per-day cost when |position| > 1 (margin interest)
    funding_cost_bps: float = 1.0       # daily cost for any non-zero position
    horizon_days: int = 5


class KellyBacktester:
    """
    Walk-forward backtest using Kelly-sized positions.

    Inputs: a DataFrame of out-of-sample predictions with 'pred_ensemble',
    'actual' (the realized return), volatility, confidence, regime, etc.

    Outputs: equity curve + performance metrics, with detailed P&L attribution.
    """

    def __init__(
        self,
        sizer: KellySizer,
        config: Optional[KellyBacktestConfig] = None,
    ):
        self.sizer = sizer
        self.config = config or KellyBacktestConfig()

    def run(
        self,
        predictions_df: pd.DataFrame,
        actual_col: str = "actual",
        date_col: str = "date",
    ) -> dict:
        """
        Returns a dict with equity_curve, metrics, and position log.
        """
        if predictions_df.empty:
            return {"equity_curve": pd.DataFrame(), "metrics": {}}

        # Compute positions
        sized = self.sizer.size_series(predictions_df)

        # Order by date
        sized = sized.sort_values(date_col).reset_index(drop=True)

        # P&L computation
        cap = self.config.initial_capital
        cost_per_change = self.config.transaction_cost_bps / 10_000
        leverage_cost = self.config.leverage_cost_bps / 10_000
        funding_cost = self.config.funding_cost_bps / 10_000

        positions = sized["position"].values
        actuals = sized[actual_col].fillna(0).values

        # gross return = position × realized log return
        # (positions are held over the forecast horizon, so we attribute the
        # entire horizon return to the day the position was opened)
        gross_returns = positions * actuals

        # Transaction cost: paid on absolute change in position
        position_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
        tx_costs = position_changes * cost_per_change

        # Leverage interest (per horizon — multiply by horizon_days proxy)
        leverage_used = np.maximum(np.abs(positions) - 1.0, 0.0)
        lev_costs = leverage_used * leverage_cost * self.config.horizon_days

        # Funding cost on any non-zero position
        fund_costs = (np.abs(positions) > 0.001).astype(float) * funding_cost * self.config.horizon_days

        net_returns = gross_returns - tx_costs - lev_costs - fund_costs
        equity = cap * np.exp(np.cumsum(net_returns))  # using log-return cumprod

        # Metrics
        metrics = self._compute_metrics(net_returns, positions, actuals)

        equity_curve = pd.DataFrame({
            "date": sized[date_col].values,
            "position": positions,
            "actual_return": actuals,
            "gross_return": gross_returns,
            "tx_cost": tx_costs,
            "lev_cost": lev_costs,
            "fund_cost": fund_costs,
            "net_return": net_returns,
            "equity": equity,
            "drawdown": self._drawdown(equity),
            "raw_kelly": sized["raw_kelly"].values,
        })

        return {
            "equity_curve": equity_curve,
            "metrics": metrics,
            "positions_log": sized[[date_col, "position", "raw_kelly",
                                    "sizing_reasons", "rejected_reasons"]],
        }

    def _compute_metrics(self, net_returns, positions, actuals) -> dict:
        valid = ~np.isnan(net_returns)
        nr = net_returns[valid]
        pos = positions[valid]
        act = actuals[valid]
        if len(nr) == 0:
            return {}

        # Annualization factor: number of forecast periods per year
        periods_per_year = 252 / self.config.horizon_days

        total_log_return = float(np.sum(nr))
        ann_return = float(np.mean(nr) * periods_per_year)
        ann_vol = float(np.std(nr) * np.sqrt(periods_per_year))
        sharpe = ann_return / (ann_vol + 1e-9)

        equity_curve = np.exp(np.cumsum(nr))
        max_dd = float(self._drawdown(equity_curve).min())

        traded_mask = np.abs(pos) > 0.001
        hit_rate_when_traded = float(np.mean(nr[traded_mask] > 0)) if traded_mask.any() else 0.0
        avg_position = float(np.mean(np.abs(pos)))
        trade_count = int(np.sum(traded_mask))

        # Directional accuracy when we DID trade
        dir_acc = float(np.mean(np.sign(pos[traded_mask]) == np.sign(act[traded_mask]))) if traded_mask.any() else 0.0

        return {
            "total_log_return": total_log_return,
            "total_pct_return": float(np.expm1(total_log_return) * 100),
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "hit_rate_when_traded": hit_rate_when_traded,
            "directional_accuracy_when_traded": dir_acc,
            "avg_position_size": avg_position,
            "n_forecasts": len(nr),
            "n_trades": trade_count,
            "fraction_traded": trade_count / len(nr) if len(nr) else 0,
        }

    @staticmethod
    def _drawdown(equity: np.ndarray) -> np.ndarray:
        running_max = np.maximum.accumulate(equity)
        return (equity - running_max) / running_max


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print(" KELLY SIZER DEMO")
    print("=" * 70)

    sizer = KellySizer(KellyConfig(kelly_fraction=0.5, max_position=1.0))

    # Test cases that illustrate the formula
    test_cases = [
        {
            "name": "Normal market, moderate edge, high confidence",
            "predicted_log_return": 0.02,    # 2% expected over horizon
            "forecast_variance": 0.001,       # σ_horizon ~ 3.2%
            "confidence": 0.8, "disagreement": 0.005,
            "regime_name": "normal_up", "trump_volatility_factor": 0.0,
        },
        {
            "name": "War regime, same edge — downscaled",
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.8, "disagreement": 0.005,
            "regime_name": "war_regime", "trump_volatility_factor": 0.0,
        },
        {
            "name": "High Trump volatility — penalized",
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.8, "disagreement": 0.005,
            "regime_name": "normal_up", "trump_volatility_factor": 0.8,
        },
        {
            "name": "Low confidence — rejected",
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.2, "disagreement": 0.005,
            "regime_name": "normal_up", "trump_volatility_factor": 0.0,
        },
        {
            "name": "High disagreement — variance inflated",
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.8, "disagreement": 0.030,
            "regime_name": "normal_up", "trump_volatility_factor": 0.0,
        },
        {
            "name": "Strong bearish signal in crisis",
            "predicted_log_return": -0.05, "forecast_variance": 0.005,
            "confidence": 0.9, "disagreement": 0.008,
            "regime_name": "crisis_regime", "trump_volatility_factor": 0.0,
        },
    ]

    for tc in test_cases:
        name = tc.pop("name")
        result = sizer.size(**tc)
        print(f"\n{name}")
        print(f"  Inputs: pred={tc['predicted_log_return']:+.3f}, σ²={tc['forecast_variance']:.4f}, "
              f"conf={tc['confidence']:.2f}, regime={tc['regime_name']}")
        print(f"  → position = {result['position']:+.3f}")
        print(f"    raw_kelly = {result['raw_kelly']:+.2f}")
        if result["sizing_reasons"]:
            for r in result["sizing_reasons"]:
                print(f"    · {r}")
        if result["rejected_reasons"]:
            for r in result["rejected_reasons"]:
                print(f"    ✗ {r}")
