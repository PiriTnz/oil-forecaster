"""
Event Impulse Response Model.

Direct implementation of section 5 of MATHEMATICAL_MODEL.md.

For each historical event i with onset τᵢ, severity sᵢ, direction δᵢ ∈ {-1, +1},
the impact on log-price at day t is:

    ΔL⁽ⁱ⁾_t = δᵢ · β_{s,θ} · exp(-λ_θ (t - τᵢ)) · 1[t ≥ τᵢ]

where:
    β_{s,θ} = peak impact magnitude (depends on severity s and event type θ)
    λ_θ     = exponential decay rate
    1[·]    = indicator function

The aggregate event contribution to log price at day t is the sum over all events:

    ΔL^event_t = Σᵢ ΔL⁽ⁱ⁾_t

Calibration: for each (severity, type) pair, fit (β, λ) by minimizing squared
error between observed log-price moves and the model's predicted impulse, over
a 60-day window after each historical event.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.events_database import (
    EVENTS, GeopoliticalEvent, EventType, Severity, ImpactDirection,
)

logger = logging.getLogger(__name__)


# --- Severity numeric mapping (matches MATHEMATICAL_MODEL.md §3.5) ---
SEVERITY_TO_NUM = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.EXTREME: 4,
}

# --- Direction sign convention ---
DIRECTION_TO_SIGN = {
    ImpactDirection.BULLISH: +1.0,
    ImpactDirection.BEARISH: -1.0,
    ImpactDirection.MIXED:    0.0,
}

# Default calibrated constants (Appendix C of the spec).
# These get overwritten by .calibrate() when run with real price data.
DEFAULT_CALIBRATION: dict[tuple[str, str], tuple[float, float]] = {
    # (event_type, severity): (beta, lambda)
    ("supply_disruption", "high"):    (0.14, 0.050),
    ("supply_disruption", "extreme"): (0.55, 0.030),
    ("war",              "high"):     (0.18, 0.020),
    ("war",              "extreme"):  (0.51, 0.015),
    ("opec_action",      "high"):     (0.12, 0.010),
    ("opec_action",      "extreme"):  (0.45, 0.025),  # magnitude only, sign from direction
    ("financial_crisis", "extreme"):  (0.65, 0.008),
    ("pandemic",         "extreme"):  (0.70, 0.012),
    ("sanctions",        "medium"):   (0.06, 0.005),
    ("sanctions",        "high"):     (0.10, 0.004),
    # Conservative defaults for unseen combinations
    ("default", "low"):     (0.02, 0.030),
    ("default", "medium"):  (0.05, 0.020),
    ("default", "high"):    (0.12, 0.015),
    ("default", "extreme"): (0.30, 0.010),
}


@dataclass
class ImpulseParameters:
    """Calibrated parameters for a (severity, event_type) bucket."""
    beta: float           # Peak impact magnitude in log-price units (always positive)
    decay_rate: float     # λ, per day (always positive)
    n_observations: int = 0  # How many historical events fit this bucket
    rmse: float = 0.0        # Fit RMSE

    @property
    def half_life_days(self) -> float:
        """Days until impulse decays to half its peak."""
        return float(np.log(2) / self.decay_rate)


# =============================================================================
# Core model
# =============================================================================
class EventImpulseModel:
    """
    Closed-form geopolitical event impulse response model.

    Used for:
      1. Sanity-checking ML forecasts ("does Hormuz closure really only add 0.3%?")
      2. Stress-test priors for hypothetical scenarios (§12.1)
      3. Fallback predictions when the ML model is unavailable
    """

    def __init__(self, calibration: Optional[dict] = None):
        """
        calibration: dict mapping (event_type, severity) → (beta, lambda)
        """
        self.params: dict[tuple[str, str], ImpulseParameters] = {}
        cal = calibration or DEFAULT_CALIBRATION
        for key, (beta, lam) in cal.items():
            self.params[key] = ImpulseParameters(beta=beta, decay_rate=lam)

    # ------------------------------------------------------------------
    # Single-event impulse
    # ------------------------------------------------------------------
    def impulse(
        self,
        event: GeopoliticalEvent,
        days_since_onset: float,
    ) -> float:
        """
        Compute ΔL⁽ⁱ⁾_t = δ · β · exp(-λ · Δτ) for a single event.

        Returns log-price impact (e.g., 0.10 = +10% in log-price, ~+10.5% in price).
        Returns 0 if the event has not started yet.
        """
        if days_since_onset < 0:
            return 0.0

        params = self._get_params(event.event_type.value, event.severity.value)
        sign = DIRECTION_TO_SIGN[event.impact_direction]

        return sign * params.beta * np.exp(-params.decay_rate * days_since_onset)

    # ------------------------------------------------------------------
    # Aggregate impulse at a date
    # ------------------------------------------------------------------
    def aggregate_impact(
        self,
        as_of: date,
        events: Optional[list[GeopoliticalEvent]] = None,
        lookback_days: int = 180,
    ) -> dict:
        """
        Sum impulses from all events active or recently expired (within lookback_days).

        Returns a dict with total impact and breakdown by event.
        """
        events = events if events is not None else EVENTS
        total_log_impact = 0.0
        breakdown = []

        for e in events:
            days_since = (as_of - e.start_date).days
            if 0 <= days_since <= lookback_days:
                contribution = self.impulse(e, days_since)
                if abs(contribution) > 1e-6:
                    breakdown.append({
                        "event_id": e.event_id,
                        "name": e.name,
                        "days_since_onset": days_since,
                        "log_impact": float(contribution),
                        "pct_impact": float(np.expm1(contribution) * 100),
                    })
                    total_log_impact += contribution

        # Sort breakdown by absolute impact size
        breakdown.sort(key=lambda x: abs(x["log_impact"]), reverse=True)

        return {
            "as_of": as_of.isoformat(),
            "total_log_impact": float(total_log_impact),
            "total_pct_impact": float(np.expm1(total_log_impact) * 100),
            "n_contributing_events": len(breakdown),
            "events": breakdown,
        }

    # ------------------------------------------------------------------
    # Forward projection for stress testing
    # ------------------------------------------------------------------
    def project_forward(
        self,
        events: list[GeopoliticalEvent],
        start: date,
        horizon_days: int,
    ) -> pd.DataFrame:
        """
        Project the trajectory of event-driven log-price impact over a horizon.

        Useful for "given these events are active starting today, what does the
        impulse-model expected price path look like over the next N days?"
        """
        rows = []
        for offset in range(horizon_days + 1):
            d = start + timedelta(days=offset)
            agg = self.aggregate_impact(d, events=events, lookback_days=365)
            rows.append({
                "date": d,
                "days_from_start": offset,
                "cumulative_log_impact": agg["total_log_impact"],
                "cumulative_pct_impact": agg["total_pct_impact"],
                "n_active_events": agg["n_contributing_events"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        price_df: pd.DataFrame,
        events: Optional[list[GeopoliticalEvent]] = None,
        window_days: int = 60,
        price_col: str = "wti_close",
        date_col: str = "date",
        min_events_per_bucket: int = 2,
    ) -> dict[tuple[str, str], ImpulseParameters]:
        """
        Fit (β, λ) parameters per (event_type, severity) bucket on historical data.

        For each event, observe the log-price path L_t - L_{τ-1} for t ∈ [τ, τ+W],
        then solve:

            (β*, λ*) = argmin Σ_i Σ_t [(L_t - L_{τᵢ-1}) - δᵢ·β·exp(-λ(t-τᵢ))]²

        per (severity, type) bucket, pooling events of that bucket.
        """
        events = events or EVENTS
        df = price_df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df = df.set_index(date_col)

        # Group events by bucket
        buckets: dict[tuple[str, str], list[GeopoliticalEvent]] = {}
        for e in events:
            key = (e.event_type.value, e.severity.value)
            buckets.setdefault(key, []).append(e)

        new_params = {}

        for (etype, sev), bucket_events in buckets.items():
            if len(bucket_events) < min_events_per_bucket:
                logger.info(f"Bucket ({etype}, {sev}): only {len(bucket_events)} events, "
                            f"keeping default params")
                if (etype, sev) in self.params:
                    new_params[(etype, sev)] = self.params[(etype, sev)]
                continue

            # Collect (delta_days, observed_log_move, sign) triples across all events in bucket
            obs_days, obs_moves, obs_signs = [], [], []
            for e in bucket_events:
                if e.start_date not in df.index:
                    # Find the previous trading day
                    valid_dates = df.index[df.index < e.start_date]
                    if len(valid_dates) == 0:
                        continue
                    pre_date = valid_dates[-1]
                else:
                    pre_date = e.start_date

                if pre_date not in df.index:
                    continue
                L_pre = np.log(df.loc[pre_date, price_col])
                sign = DIRECTION_TO_SIGN[e.impact_direction]
                if sign == 0:
                    continue

                end_date = e.start_date + timedelta(days=window_days)
                window_df = df.loc[pre_date:end_date]
                if len(window_df) < 5:
                    continue

                for d_idx, row in window_df.iterrows():
                    delta_d = (d_idx - e.start_date).days
                    if delta_d < 0:
                        continue
                    obs_days.append(delta_d)
                    obs_moves.append(float(np.log(row[price_col]) - L_pre))
                    obs_signs.append(sign)

            if len(obs_days) < 20:
                logger.info(f"Bucket ({etype}, {sev}): only {len(obs_days)} observations, "
                            f"keeping default")
                if (etype, sev) in self.params:
                    new_params[(etype, sev)] = self.params[(etype, sev)]
                continue

            obs_days_arr = np.array(obs_days)
            obs_moves_arr = np.array(obs_moves)
            obs_signs_arr = np.array(obs_signs)

            def loss(params, days=obs_days_arr, moves=obs_moves_arr, signs=obs_signs_arr):
                beta, lam = params
                if beta < 0 or lam < 0:
                    return 1e10
                predicted = signs * beta * np.exp(-lam * days)
                return float(np.mean((moves - predicted) ** 2))

            initial = self.params.get((etype, sev), ImpulseParameters(0.1, 0.02))
            result = minimize(
                loss,
                x0=[initial.beta, initial.decay_rate],
                method="Nelder-Mead",
                options={"xatol": 1e-6, "maxiter": 500},
            )

            beta_fit, lambda_fit = result.x
            beta_fit = max(0.001, beta_fit)
            lambda_fit = max(0.001, lambda_fit)
            rmse = float(np.sqrt(result.fun))

            new_params[(etype, sev)] = ImpulseParameters(
                beta=beta_fit,
                decay_rate=lambda_fit,
                n_observations=len(obs_days),
                rmse=rmse,
            )
            logger.info(f"Calibrated ({etype:>20s}, {sev:>8s}): "
                        f"β={beta_fit:.4f}, λ={lambda_fit:.4f}, "
                        f"half_life={np.log(2)/lambda_fit:.1f}d, "
                        f"n={len(obs_days)}, rmse={rmse:.4f}")

        self.params.update(new_params)
        return new_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_params(self, event_type: str, severity: str) -> ImpulseParameters:
        """Look up calibrated params, with sensible fallback."""
        if (event_type, severity) in self.params:
            return self.params[(event_type, severity)]
        if ("default", severity) in self.params:
            return self.params[("default", severity)]
        return ImpulseParameters(beta=0.05, decay_rate=0.02)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all calibrated parameters."""
        rows = []
        for (etype, sev), p in self.params.items():
            rows.append({
                "event_type": etype,
                "severity": sev,
                "beta": p.beta,
                "decay_rate": p.decay_rate,
                "half_life_days": p.half_life_days,
                "n_obs": p.n_observations,
                "rmse": p.rmse,
            })
        return pd.DataFrame(rows).sort_values(
            ["event_type", "severity"]
        ).reset_index(drop=True)


# =============================================================================
# Supply-shock model (§5.4 of the spec)
# =============================================================================
@dataclass
class SupplyShockModel:
    """
    Short-run supply-shock impact on log price.

    Standard model:
        ΔL^supply = -η · (ΔQ / Q_world)

    where η is the short-run price elasticity of supply. Literature anchors:
      - η ≈ -10 for short-run oil (Hamilton 2009, Kilian 2009)
      - This reflects very low short-run elasticity of demand (~0.1)
    """
    elasticity: float = -10.0       # η in the formula
    world_output_mbd: float = 100.0 # Million barrels per day, world
    impact_cap: float = 1.0          # Cap |ΔL^supply| ≤ this (price doubling cap)

    def shock_impact(self, disrupted_mbd: float) -> float:
        """
        disrupted_mbd: barrels per day removed from market (in millions)
        Returns: log-price impact (positive = bullish).
        """
        fraction = disrupted_mbd / self.world_output_mbd
        log_impact = -self.elasticity * fraction
        return float(np.sign(log_impact) * min(abs(log_impact), self.impact_cap))

    def hormuz_closure_impact(self, days_closed: int) -> dict:
        """
        Estimate impact of full Strait of Hormuz closure for N days.

        Hormuz handles ~20 mb/d (~20% of world seaborne oil).
        Shadow flows partially compensate, so we model effective disruption
        as a function of duration.
        """
        # Effective disruption tapers due to SPR releases and shadow flows
        # Empirical: 1-day shock ≈ 80% pass-through, 30-day ≈ 50%, 90-day ≈ 35%
        pass_through = 0.8 * np.exp(-days_closed / 30) + 0.35
        effective_mbd = 20.0 * pass_through

        impulse = self.shock_impact(effective_mbd)
        return {
            "days_closed": days_closed,
            "effective_disruption_mbd": float(effective_mbd),
            "pass_through_fraction": float(pass_through),
            "log_price_impact": float(impulse),
            "pct_price_impact": float(np.expm1(impulse) * 100),
        }


# =============================================================================
# Demo
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print(" EVENT IMPULSE MODEL DEMO")
    print("=" * 70)

    model = EventImpulseModel()
    print("\n[1] Calibrated parameters (defaults from spec Appendix C):")
    print(model.summary().to_string(index=False))

    # Aggregate impact today
    print("\n[2] Aggregate event-driven log-price impact today (2026-05-15):")
    today_impact = model.aggregate_impact(date(2026, 5, 15))
    print(f"    Total log impact: {today_impact['total_log_impact']:.4f}")
    print(f"    Total pct impact: {today_impact['total_pct_impact']:+.2f}%")
    print(f"    Contributing events ({today_impact['n_contributing_events']}):")
    for e in today_impact["events"][:5]:
        print(f"      • {e['name']} ({e['days_since_onset']}d ago): "
              f"{e['pct_impact']:+.2f}%")

    # Hormuz scenarios
    print("\n[3] Hormuz closure scenarios (supply shock model):")
    shock = SupplyShockModel()
    for n_days in [1, 7, 30, 90]:
        r = shock.hormuz_closure_impact(n_days)
        print(f"    {n_days:>3d} days closed: "
              f"effective {r['effective_disruption_mbd']:.1f} mb/d disrupted, "
              f"price impact ≈ {r['pct_price_impact']:+.1f}%")

    # Forward projection
    print("\n[4] Forward 30-day projection of event impact:")
    proj = model.project_forward(EVENTS, start=date(2026, 5, 15), horizon_days=30)
    print(proj.iloc[::5].to_string(index=False))

    print("\n" + "=" * 70)
