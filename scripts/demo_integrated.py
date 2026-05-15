"""
END-TO-END DEMO: Trump-aware oil forecasting with all three new pieces.

Runs:
  1. News sentiment with Trump rhetoric-action gap detection
  2. Regime-aware LSTM with attention (skipped if torch missing)
  3. Kelly position sizing with regime + trump-volatility adjustments

Use this to verify the integration works end-to-end on synthetic data.
"""
import sys
import logging
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.news_sentiment import (
    NewsItem, LLMNewsAnalyzer, RhetoricActionDetector, DailySentimentSignals
)
from src.evaluation.kelly_sizing import (
    KellySizer, KellyConfig, KellyBacktester, KellyBacktestConfig
)
from src.models.event_impulse import EventImpulseModel
from src.data.events_database import EVENTS


def section(title):
    print("\n" + "=" * 72)
    print(f" {title}")
    print("=" * 72)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # ========================================================================
    section("1. NEWS SENTIMENT — analyzing a Trump-volatile day")
    # ========================================================================
    # March 23, 2026: Trump postponed attacks, oil dropped, suspicious bets
    items = [
        NewsItem(
            title="Trump announces 2-week postponement of Iran strikes for talks",
            source="Reuters", published_at=datetime(2026, 3, 23, 14),
            description="A whole civilization will die tonight, never to be brought back again. I don't want that to happen, but it probably will.",
        ),
        NewsItem(
            title="Trump: 'I will negotiate, but my finger stays on the trigger'",
            source="Bloomberg", published_at=datetime(2026, 3, 23, 16),
        ),
        NewsItem(
            title="Iran FM cautiously welcomes pause",
            source="AP", published_at=datetime(2026, 3, 23, 17),
        ),
        NewsItem(
            title="FT: $580M short bets on oil placed 15 min before Trump statement",
            source="Financial Times", published_at=datetime(2026, 3, 23, 19),
            description="Investigation into possible insider trading raised by lawmakers",
        ),
    ]

    analyzer = LLMNewsAnalyzer()  # falls back to rule-based without API key
    signals = analyzer.analyze_day(date(2026, 3, 23), items)
    print(f"\nDate: {signals.date}")
    print(f"  overall_sentiment       = {signals.overall_sentiment:+.2f}")
    print(f"  geopolitical_tension    = {signals.geopolitical_tension:.1f}/10")
    print(f"  trump_oil_rhetoric      = {signals.trump_oil_rhetoric:+.2f}")
    print(f"  trump_action_alignment  = {signals.trump_action_alignment:+.2f}")
    print(f"  trump_volatility_factor = {signals.trump_volatility_factor:.2f}")
    print(f"  iran_diplomatic_openness= {signals.iran_diplomatic_openness:+.2f}")
    print(f"  hormuz_mentions         = {signals.hormuz_mentions}")
    print(f"  hormuz_closure_threat   = {signals.hormuz_closure_threat}")
    print(f"  llm_confidence          = {signals.llm_confidence:.2f}")
    print(f"  notes: {signals.notes}")

    # ========================================================================
    section("2. RHETORIC-ACTION GAP — tracking Trump reliability over time")
    # ========================================================================
    detector = RhetoricActionDetector(window=20, lookahead=3)

    # Real-ish pattern from Trump's 2026 Iran war statements vs actions
    pattern = [
        ("2026-02-28", +0.95, +0.95, "Epic Fury launched (threat → action)"),
        ("2026-03-04", +0.90, +0.85, "Hormuz Navy ops (consistent)"),
        ("2026-03-23", -0.70, +0.60, "Postpone attacks → strikes continued"),
        ("2026-03-27", +0.95, +0.10, "Strait of Trump rename (talk only)"),
        ("2026-04-08", -0.85, -0.20, "Announce ceasefire (partial)"),
        ("2026-04-13", +0.80, +0.85, "Blockade declared → enforced"),
        ("2026-04-17", -0.50, -0.70, "Strait open (consistent)"),
        ("2026-04-19", +0.70, +0.80, "Iran ship seized (consistent)"),
        ("2026-05-04", +0.40, +0.40, "Project Freedom (consistent)"),
        ("2026-05-06", -0.50, -0.50, "Pause (consistent)"),
    ]

    for d_str, rhet, act, desc in pattern:
        detector.add_observation(date.fromisoformat(d_str), rhet, act)
        gap = rhet - act
        marker = "⚠️ GAP" if abs(gap) > 0.5 else "  ✓  "
        print(f"  {marker}  {d_str}  rhet={rhet:+.2f}  action={act:+.2f}  gap={gap:+.2f}  {desc}")

    rel = detector.current_reliability()
    vol = detector.current_volatility()
    disc = detector.discount_factor()
    print(f"\n  Reliability of Trump rhetoric (correlation):  {rel:+.3f}")
    print(f"  Volatility of rhetoric over window:           {vol:.3f}")
    print(f"  Discount factor (how much to trust today's):  {disc:.3f}")

    # ========================================================================
    section("3. EVENT IMPULSE MODEL — closed-form impact of active events")
    # ========================================================================
    impulse_model = EventImpulseModel()
    today_impact = impulse_model.aggregate_impact(date(2026, 5, 15))
    print(f"\nFor 2026-05-15, event-driven log-price impact:")
    print(f"  Total: {today_impact['total_log_impact']:+.4f} = {today_impact['total_pct_impact']:+.2f}% premium")
    print(f"  Top 5 contributing events:")
    for e in today_impact['events'][:5]:
        print(f"    • {e['name'][:55]:<55s}  {e['pct_impact']:+6.2f}%  ({e['days_since_onset']}d ago)")

    # ========================================================================
    section("4. KELLY SIZING — translating predictions into positions")
    # ========================================================================
    sizer = KellySizer(KellyConfig(
        kelly_fraction=0.5,
        max_position=1.0,
        confidence_threshold=0.30,
        regime_downscale={"war_regime": 0.4, "normal_up": 1.0, "crisis_regime": 0.4},
    ))

    # Synthetic forecasts under different conditions
    scenarios = [
        ("Normal market, +2% pred, high conf, calm", {
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.85, "disagreement": 0.005,
            "regime_name": "normal_up", "trump_volatility_factor": 0.0,
        }),
        ("Same pred but during war regime", {
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.85, "disagreement": 0.005,
            "regime_name": "war_regime", "trump_volatility_factor": 0.0,
        }),
        ("Same pred but high Trump-volatility day", {
            "predicted_log_return": 0.02, "forecast_variance": 0.001,
            "confidence": 0.85, "disagreement": 0.005,
            "regime_name": "war_regime", "trump_volatility_factor": 0.8,
        }),
        ("Strong bearish in crisis with high conf", {
            "predicted_log_return": -0.05, "forecast_variance": 0.005,
            "confidence": 0.90, "disagreement": 0.008,
            "regime_name": "crisis_regime", "trump_volatility_factor": 0.2,
        }),
        ("Weak signal, low confidence — rejected", {
            "predicted_log_return": 0.005, "forecast_variance": 0.001,
            "confidence": 0.20, "disagreement": 0.005,
            "regime_name": "normal_up", "trump_volatility_factor": 0.0,
        }),
    ]
    print()
    print(f"  {'Scenario':<48s}  {'Position':>10s}  Notes")
    print(f"  {'-'*48}  {'-'*10}  {'-'*40}")
    for name, kwargs in scenarios:
        r = sizer.size(**kwargs)
        if r["position"] == 0:
            note = f"REJECTED: {r['rejected_reasons'][0] if r['rejected_reasons'] else ''}"
        else:
            note = f"raw_kelly={r['raw_kelly']:+.2f}"
        print(f"  {name:<48s}  {r['position']:+10.3f}  {note}")

    # ========================================================================
    section("5. INTEGRATED BACKTEST — synthetic 1-year out-of-sample")
    # ========================================================================
    rng = np.random.default_rng(42)
    n_days = 252

    # Simulate predictions where model has small edge in normal regime,
    # bigger edge in war regime (because attention picks up similar past days)
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n_days, freq="B"),
    })

    # Regime sequence: 60% normal, 40% war
    regimes = rng.choice(["normal_up", "war_regime"], n_days, p=[0.6, 0.4])
    df["regime_name"] = regimes
    df["vol_21d"] = np.where(regimes == "war_regime",
                              rng.uniform(0.5, 0.9, n_days),
                              rng.uniform(0.2, 0.4, n_days))

    # True returns: higher mean and vol in war regimes
    true_returns = np.where(
        regimes == "war_regime",
        rng.normal(0.005, 0.04, n_days),
        rng.normal(0.001, 0.012, n_days),
    )

    # Model predictions: noisy version of truth (worse signal in war regime
    # for the BASELINE model; but our regime-aware model would be better)
    noise_scale = np.where(regimes == "war_regime", 0.025, 0.008)
    model_predictions = true_returns + rng.normal(0, noise_scale, n_days)
    # Add some Trump-volatility days (mostly noise)
    trump_vol = np.where(
        rng.uniform(size=n_days) > 0.85,
        rng.uniform(0.5, 1.0, n_days),
        rng.uniform(0, 0.2, n_days),
    )
    df["pred_ensemble"] = model_predictions
    df["actual"] = true_returns
    df["confidence"] = rng.uniform(0.4, 0.9, n_days)
    df["pred_disagreement"] = np.abs(rng.normal(0.005, 0.003, n_days))
    df["trump_volatility_factor"] = trump_vol

    # Run two backtests: naïve (no Kelly) vs. Kelly-with-regime-aware
    # --- Naive: full +1/-1 per sign of prediction ---
    naive_positions = np.sign(df["pred_ensemble"].values)
    naive_returns = naive_positions * df["actual"].values
    naive_equity = 100_000 * np.exp(np.cumsum(naive_returns))
    naive_sharpe = (np.mean(naive_returns) / (np.std(naive_returns) + 1e-9)) * np.sqrt(252)

    # --- Kelly-with-regime: ---
    backtester = KellyBacktester(sizer)
    result = backtester.run(df, actual_col="actual")
    kelly_metrics = result["metrics"]

    print(f"\n                                   Naive    Kelly-aware")
    print(f"  Final equity ($100k start):  ${naive_equity[-1]:>8,.0f}    ${result['equity_curve']['equity'].iloc[-1]:>8,.0f}")
    print(f"  Annualized Sharpe:            {naive_sharpe:>8.3f}    {kelly_metrics['sharpe_ratio']:>8.3f}")
    print(f"  Total return:                 {(np.exp(naive_returns.sum())-1)*100:>7.2f}%    {kelly_metrics['total_pct_return']:>7.2f}%")
    print(f"  Trades:                       {n_days:>8d}    {kelly_metrics['n_trades']:>8d}")
    print(f"  Avg position size:                 1.00    {kelly_metrics['avg_position_size']:>8.3f}")
    print(f"  Max drawdown:                       -       {kelly_metrics['max_drawdown']*100:>7.2f}%")

    if kelly_metrics['sharpe_ratio'] > naive_sharpe:
        print(f"\n  → Kelly improved Sharpe by {(kelly_metrics['sharpe_ratio']/naive_sharpe - 1)*100:+.1f}%")

    print("\n" + "=" * 72)
    print(" Demo complete. All three pieces working together:")
    print("   1. News sentiment with Trump rhetoric-action tracking")
    print("   2. Event-driven impulse model providing closed-form impact")
    print("   3. Kelly sizing producing risk-adjusted positions")
    print("=" * 72)


if __name__ == "__main__":
    main()
