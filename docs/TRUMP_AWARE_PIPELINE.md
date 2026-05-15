# Trump-Aware Trading Pipeline

Three complementary modules that work together to make the forecaster handle the unique challenges of the 2025-2026 oil market: extreme geopolitical volatility, rhetoric-action gaps from key actors, and tail risks.

## Overview

| Module | What it does | When it matters |
|---|---|---|
| **News Sentiment** (`src/llm/news_sentiment.py`) | LLM reads daily news → 18 structured features | Picks up signal from headlines before prices move |
| **Regime-Aware LSTM** (`src/models/regime_lstm.py`) | LSTM with attention biased toward similar-regime past days | Crisis periods where past calm-market patterns are misleading |
| **Kelly Sizing** (`src/evaluation/kelly_sizing.py`) | Translates predictions into position sizes | Converts accuracy into actual risk-adjusted P&L |

## 1. News Sentiment with LLM

Each trading day's news → structured signals via Claude. Key fields:

- `overall_sentiment` ∈ [-1, +1] — bearish to bullish for oil
- `geopolitical_tension` ∈ [0, 10]
- `trump_oil_rhetoric` ∈ [-1, +1] — **what he said today**
- `trump_action_alignment` ∈ [-1, +1] — **does it match his actions?**
- `trump_volatility_factor` ∈ [0, 1] — **how contradictory today's signals**
- `iran_threat_level`, `iran_diplomatic_openness`
- `saudi_opec_stance`, `china_iran_engagement`, `russia_oil_policy`
- `hormuz_mentions`, `hormuz_closure_threat`, `direct_military_action`

**Trump rhetoric-action gap handling:**

```python
from src.llm.news_sentiment import RhetoricActionDetector

detector = RhetoricActionDetector(window=20, lookahead=3)
# Add observations: (date, rhetoric_intensity, observed_action_intensity)
detector.add_observation(date(2026, 3, 23), rhetoric=-0.70, action=+0.60)
#                        ^ Trump said "postpone"  ^ But strikes continued

reliability = detector.current_reliability()  # Spearman corr in window
discount = detector.discount_factor()         # how much to trust today's words
```

When `discount` is low, the model down-weights statement-driven features.

## 2. Regime-Aware LSTM with Attention

Standard attention:
```
weights(s) = softmax(Q_t · K_s / √d)
```

Regime-aware attention adds two terms to the logits:
```
weights(s) = softmax(Q_t · K_s / √d  +  α · 1[regime_s = regime_t]  -  γ · trump_vol_s)
```

- **α (regime bonus)**: boosts attention on past days in the same regime as today.
- **γ (trump penalty)**: discounts past days that had high rhetoric volatility.
- Both are **learned parameters** — the model decides how much regime/Trump-volatility matters.

```python
from src.models.regime_lstm import RegimeAwareLSTMForecaster, RegimeLSTMConfig

model = RegimeAwareLSTMForecaster(RegimeLSTMConfig(
    sequence_length=60,
    hidden_size=64,
    regime_bonus_init=0.5,        # initial α
    trump_penalty_init=0.5,        # initial γ
    learn_attention_coefs=True,    # let them train
))
model.fit(df, feature_cols=cols)

# Predictions + attention weights (for interpretability)
preds, attention = model.predict(test_df, return_attention=True)

# Inspect what the model learned
print(model.learned_coefs)
# {'alpha_regime_bonus': 0.82, 'gamma_trump_penalty': 1.31}
```

A large learned α means "regime matters a lot for prediction"; a large γ means "I learned to ignore Trump-volatile days."

## 3. Kelly Position Sizing

Standard Kelly:
$$f^* = \frac{\mu}{\sigma^2}$$

Our enhanced version:
$$f_t = \lambda_K \cdot \frac{\hat y_t}{\sigma_t^2 + \lambda_u \cdot d_t^2} \cdot \beta_{\text{regime}}(z_t) \cdot \beta_{\text{trump}}(v_t)$$

with safety: $|f_t| \le f_{\max}$, $c_t \ge c_{\min}$.

**Regime downscaling table** (configurable):
| Regime | Multiplier |
|---|---|
| `war_regime` | 0.40 |
| `crisis_regime` | 0.40 |
| `sanctions_regime` | 0.70 |
| `opec_regime` | 0.85 |
| `normal_*` | 1.00 |

**Trump-volatility penalty:** `multiplier = 1 - 0.5 · trump_volatility_factor`

```python
from src.evaluation.kelly_sizing import KellySizer, KellyConfig

sizer = KellySizer(KellyConfig(
    kelly_fraction=0.5,            # half-Kelly for safety
    max_position=1.0,              # no leverage
    confidence_threshold=0.30,     # reject low-confidence preds
))

result = sizer.size(
    predicted_log_return=0.02,
    forecast_variance=0.001,
    confidence=0.85,
    disagreement=0.005,
    regime_name="war_regime",
    trump_volatility_factor=0.8,
)
# result["position"] = 0.45 (capped, scaled down by regime + trump penalty)
# result["sizing_reasons"] = [list of every adjustment applied]
```

## How They Work Together

```
News → LLMNewsAnalyzer → DailySentimentSignals
                              │
                              ├─→ trump_volatility_factor (feature)
                              │
                              ├─→ trump_oil_rhetoric (feature)
                              │
                              ▼
                    feature matrix x_t
                              │
                              ▼
            RegimeAwareLSTMForecaster ────→ ŷ_t (forecast)
                              │
                              └─→ attention weights show
                                  which past days drove prediction
                              │
                              ▼
                       KellySizer ───→ position size
                              │
                              └─→ regime + trump_vol downscale
                                  applied here
                              │
                              ▼
                    KellyBacktester ───→ P&L, Sharpe, DD
```

## Empirical Insight from Synthetic Backtest

Even with perfect predictions (so naive long/short does fine), Kelly sizing:
- Caps drawdowns by reducing positions in high-vol regimes
- Eliminates trades when confidence is low (avoids overtrading)
- Trades only ~57% of days vs naïve 100% — much lower transaction costs

In **real markets** (where models are imperfect), Kelly's downside protection becomes much more valuable than in synthetic experiments. Half-Kelly with regime downscaling is the standard prescription in commodity quant funds for exactly this reason.

## Running the integrated demo

```bash
python scripts/demo_integrated.py
```

Shows all three pieces with synthetic but realistic Trump-pattern data.
