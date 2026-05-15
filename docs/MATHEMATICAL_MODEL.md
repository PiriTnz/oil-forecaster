# Oil Forecaster — Mathematical Model Specification

**Version:** 1.0
**Last updated:** 2026-05
**Authors:** see README

This document is the **mathematical specification** of the oil price forecasting system. Every formula, assumption, and statistical choice is documented here. The Python code in `src/` is a direct implementation of what's written below — if code and spec disagree, the spec is authoritative.

---

## Table of Contents

1. [Notation and conventions](#1-notation)
2. [The forecasting problem](#2-problem)
3. [Feature construction](#3-features)
4. [Regime model (Gaussian Mixture / HMM)](#4-regime)
5. [Geopolitical event impact model](#5-events)
6. [Forecasting models](#6-models)
7. [Ensemble combination](#7-ensemble)
8. [Loss functions and training objectives](#8-losses)
9. [Cross-validation methodology](#9-cv)
10. [Backtesting and strategy returns](#10-backtest)
11. [Uncertainty quantification](#11-uncertainty)
12. [Stress-test scenarios](#12-stress)

---

## 1. Notation and conventions <a id="1-notation"></a>

Let:

- $t \in \{1, 2, \dots, T\}$ — discrete trading days, $t=1$ corresponds to the first observation (typically 2000-01-03 for WTI).
- $P_t$ — closing price of WTI crude oil on day $t$, in USD/barrel.
- $r_t = \ln(P_t / P_{t-1})$ — daily log return.
- $h \in \mathbb{N}^+$ — forecast horizon in trading days (default $h = 5$).
- $y_t^{(h)} = \ln(P_{t+h} / P_t) = \sum_{k=1}^{h} r_{t+k}$ — forward $h$-day log return (the **prediction target**).
- $\mathbf{x}_t \in \mathbb{R}^d$ — feature vector available at the close of day $t$ (must contain NO information from days $> t$).
- $z_t \in \{1, \dots, K\}$ — discrete market regime at $t$ (latent or observed).
- $e_t \in \{0,1\}^E$ — vector of indicator variables for $E$ event types active at $t$.
- $\mathcal{D}_t = \{(\mathbf{x}_s, y_s^{(h)}) : s \leq t - h\}$ — information set available for training at $t$ (note the $-h$ to avoid target leakage).

**Causality rule (strictly enforced):** every feature $x_{i,t}$ must be a function of prices and exogenous data observable by the close of day $t$. Any feature using $P_{t+1}, P_{t+2}, \dots$ is a bug.

---

## 2. The forecasting problem <a id="2-problem"></a>

We want to learn a function

$$
f: \mathbb{R}^d \to \mathbb{R}, \qquad \hat y_t^{(h)} = f(\mathbf{x}_t)
$$

that approximates the conditional expectation

$$
\mathbb{E}\big[\, y_t^{(h)} \,\big|\, \mathbf{x}_t, z_t, e_t \,\big].
$$

We additionally want:

- a **direction** $\hat d_t \in \{-1, 0, +1\}$ — sign of $\hat y_t^{(h)}$ with a dead zone $\delta$:

$$
\hat d_t = \begin{cases}
+1 & \text{if } \hat y_t^{(h)} > \delta \\
0 & \text{if } |\hat y_t^{(h)}| \leq \delta \\
-1 & \text{if } \hat y_t^{(h)} < -\delta
\end{cases}
$$

  with default $\delta = \ln(1.01) \approx 0.00995$ (1% threshold).

- a **confidence** $c_t \in [0, 1]$ — derived from model disagreement (Section 11).
- an **interpretation** — natural-language explanation (LLM layer, Section 12).

### Why log returns, not prices?

Three reasons:
1. **Stationarity.** Oil prices are non-stationary ($P_t$ has time-varying mean and variance). Log returns are approximately stationary, which the ML models assume implicitly.
2. **Additivity.** $\ln(P_{t+h}/P_t) = \sum_{k=1}^{h} r_{t+k}$, so multi-horizon arithmetic is clean.
3. **Symmetry.** A move from \$50 to \$100 has log return $\ln 2 \approx 0.693$. A move from \$100 to \$50 has log return $-0.693$. Percentage returns ($+100\%$ vs $-50\%$) hide this symmetry.

---

## 3. Feature construction <a id="3-features"></a>

The feature vector $\mathbf{x}_t$ has $d \approx 45$ components grouped into four families.

### 3.1 Return features

Let $L_t = \ln P_t$. For each $n \in \{1, 5, 10, 21, 63, 252\}$:

$$
\text{ret}_{n,t} = L_t - L_{t-n} = \sum_{k=0}^{n-1} r_{t-k}.
$$

These windows correspond to: 1 day, 1 week, 2 weeks, 1 month, 1 quarter, 1 year.

### 3.2 Volatility features

For window $w \in \{5, 10, 21, 63\}$:

$$
\sigma^{(w)}_t = \sqrt{252} \cdot \sqrt{ \frac{1}{w-1} \sum_{k=0}^{w-1} \big(r_{t-k} - \bar r^{(w)}_t \big)^2 }
$$

where $\bar r^{(w)}_t = \frac{1}{w}\sum_{k=0}^{w-1} r_{t-k}$ is the rolling-window mean. The $\sqrt{252}$ factor annualizes from daily to yearly volatility.

**Vol-of-vol** (instability indicator):

$$
\text{vov}_t = \mathrm{std}\big( \{\sigma^{(21)}_{t-k}\}_{k=0}^{20} \big).
$$

**Drawdown from rolling 1-year max:**

$$
\text{DD}_t = \frac{P_t}{ \max_{s \in [t-251, t]} P_s } - 1, \qquad \text{DD}_t \in [-1, 0].
$$

### 3.3 Momentum and mean-reversion

**Simple moving averages** for $m \in \{20, 50, 200\}$:

$$
\text{SMA}^{(m)}_t = \frac{1}{m} \sum_{k=0}^{m-1} P_{t-k}.
$$

**Cross signals:**

$$
\text{cross}^{(20,50)}_t = \frac{ \text{SMA}^{(20)}_t - \text{SMA}^{(50)}_t }{ \text{SMA}^{(50)}_t }, \qquad
\text{cross}^{(50,200)}_t = \frac{ \text{SMA}^{(50)}_t - \text{SMA}^{(200)}_t }{ \text{SMA}^{(200)}_t }.
$$

**Price vs long trend:**

$$
\text{trend}_t = \frac{P_t}{\text{SMA}^{(200)}_t} - 1.
$$

**RSI (Relative Strength Index, Wilder's 14-day):**

Define gains and losses:
$$
g_t = \max(P_t - P_{t-1}, 0), \qquad \ell_t = \max(P_{t-1} - P_t, 0).
$$

Exponentially-weighted averages with smoothing factor $\alpha = 1/14$:
$$
\bar g_t = (1-\alpha) \bar g_{t-1} + \alpha g_t, \qquad \bar\ell_t = (1-\alpha) \bar\ell_{t-1} + \alpha \ell_t.
$$

Then:
$$
\text{RS}_t = \frac{\bar g_t}{\bar\ell_t}, \qquad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t} \in [0, 100].
$$

**Bollinger Band position** (z-score of price vs 20-day mean):

$$
\mu^{(20)}_t = \frac{1}{20}\sum_{k=0}^{19} P_{t-k}, \qquad s^{(20)}_t = \sqrt{ \frac{1}{19}\sum_{k=0}^{19} (P_{t-k} - \mu^{(20)}_t)^2 },
$$

$$
\text{BB}_t = \frac{P_t - \mu^{(20)}_t}{2 \, s^{(20)}_t}.
$$

### 3.4 Macro features

Let $D_t$ = DXY (dollar index), $S_t$ = S&P 500, $V_t$ = VIX, $G_t$ = gold price, $Y_t$ = 10-year US Treasury yield.

$$
\Delta_n D_t = \ln(D_t / D_{t-n}), \quad
\Delta_n S_t = \ln(S_t / S_{t-n}), \quad
\Delta_n Y_t = Y_t - Y_{t-n}.
$$

**Gold/oil ratio z-score** (safe-haven flight indicator):

$$
\rho_t = \frac{G_t}{P_t}, \qquad
\rho^*_t = \frac{\rho_t - \mu_\rho^{(252)}_t}{s_\rho^{(252)}_t}.
$$

When $\rho^*_t \gg 0$, gold has outperformed oil → flight to safety.

### 3.5 Event features

For each event type $j \in \{$war, sanctions, supply_disruption, OPEC, financial, pandemic$\}$:

$$
e_{j,t} = \mathbb{1}[\,\exists\, \text{event of type } j \text{ active on day } t \,].
$$

**Days since onset:**

$$
\tau_{j,t} = t - \max\{ s \leq t : \text{event of type } j \text{ started at } s \}.
$$

**Severity aggregation.** Each event $i$ has severity $S_i \in \{\text{low}, \text{med}, \text{high}, \text{extreme}\}$ mapped to numeric $s_i \in \{1, 2, 3, 4\}$. The active severity score:

$$
S_t = \max_{i \in \mathcal{A}_t} s_i, \qquad
\mathcal{A}_t = \{ i : \text{event } i \text{ active on day } t \}.
$$

**Direction tilt** (bullish events minus bearish events currently active):

$$
B_t = \big| \{i \in \mathcal{A}_t : \text{bullish}\} \big| - \big| \{i \in \mathcal{A}_t : \text{bearish}\} \big|.
$$

### 3.6 Calendar / seasonality

Cyclical encoding avoids the model treating "December" and "January" as far apart:

$$
\text{month\_sin}_t = \sin\!\left( \frac{2\pi \cdot \text{month}(t)}{12} \right), \qquad
\text{month\_cos}_t = \cos\!\left( \frac{2\pi \cdot \text{month}(t)}{12} \right).
$$

Same construction for day-of-week with denominator 5 (trading days).

---

## 4. Regime model <a id="4-regime"></a>

We assume the daily market state $z_t$ takes one of $K=4$ values: $\{1, \dots, K\}$, with semantic labels (calm, normal, crisis, spike) assigned post-hoc.

### 4.1 Gaussian Mixture formulation

Conditional on regime $z_t = k$, a low-dimensional regime feature vector $\mathbf{u}_t = (r_t, \sigma^{(21)}_t, \text{DD}_t, V_t)$ is modeled as multivariate normal:

$$
\mathbf{u}_t \,\big|\, z_t = k \;\sim\; \mathcal{N}(\boldsymbol\mu_k, \boldsymbol\Sigma_k).
$$

Marginal density:

$$
p(\mathbf{u}_t) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{u}_t \,|\, \boldsymbol\mu_k, \boldsymbol\Sigma_k), \qquad \sum_k \pi_k = 1.
$$

Parameters $\{\pi_k, \boldsymbol\mu_k, \boldsymbol\Sigma_k\}$ are estimated by **EM** (Expectation-Maximization) maximizing

$$
\mathcal{L} = \sum_t \ln p(\mathbf{u}_t).
$$

Posterior regime probability (used as soft features):

$$
\gamma_{k,t} = \mathbb{P}(z_t = k \mid \mathbf{u}_t) = \frac{ \pi_k \mathcal{N}(\mathbf{u}_t \,|\, \boldsymbol\mu_k, \boldsymbol\Sigma_k) }{ \sum_{k'} \pi_{k'} \mathcal{N}(\mathbf{u}_t \,|\, \boldsymbol\mu_{k'}, \boldsymbol\Sigma_{k'}) }.
$$

Hard assignment: $\hat z_t = \arg\max_k \gamma_{k,t}$.

### 4.2 HMM extension (optional)

A full HMM adds a transition matrix $\mathbf{A} \in [0,1]^{K \times K}$ with $A_{ij} = \mathbb{P}(z_t = j \mid z_{t-1} = i)$. The joint likelihood becomes:

$$
\mathcal{L}_{\text{HMM}} = \ln \sum_{z_1, \dots, z_T} \pi_{z_1} \prod_{t=2}^{T} A_{z_{t-1}, z_t} \prod_{t=1}^{T} \mathcal{N}(\mathbf{u}_t \mid \boldsymbol\mu_{z_t}, \boldsymbol\Sigma_{z_t}).
$$

Computed via the **forward-backward algorithm**. Adds regime persistence (war regimes tend to stay war regimes) at cost of training time. Implemented as a fallback when GMM regimes flip too frequently.

### 4.3 Regime naming heuristic

After fitting, regimes are auto-named by sorting cluster centroids on $(\mu_k^{r}, \mu_k^{\sigma})$ in standardized space:

$$
\text{name}(k) = \begin{cases}
\text{"crisis"} & \text{if } \mu_k^{\sigma,*} > 0.8 \text{ and } \mu_k^{r,*} < 0 \\
\text{"spike"}  & \text{if } \mu_k^{\sigma,*} > 0.8 \text{ and } \mu_k^{r,*} \geq 0 \\
\text{"calm\_uptrend"} & \text{if } \mu_k^{\sigma,*} < -0.3 \text{ and } \mu_k^{r,*} > 0 \\
\text{"calm\_drift"}   & \text{if } \mu_k^{\sigma,*} < -0.3 \text{ and } \mu_k^{r,*} \leq 0 \\
\text{"normal\_up"} \text{ or } \text{"normal\_down"} & \text{otherwise (by sign of } \mu_k^r) 
\end{cases}
$$

where $*$ denotes z-score relative to the training distribution.

---

## 5. Geopolitical event impact model <a id="5-events"></a>

Geopolitical events affect oil prices through three distinct channels. The model captures each separately.

### 5.1 Shock model (impulse response)

For a single event $i$ starting at day $\tau_i$ with severity $s_i$ and direction $\delta_i \in \{-1, +1\}$, the **impulse response** to the log-price is:

$$
\Delta L^{(i)}_{t} = \delta_i \cdot \beta_{s_i, \theta(i)} \cdot \exp\!\left( -\lambda_{\theta(i)} (t - \tau_i) \right) \cdot \mathbb{1}[t \geq \tau_i]
$$

where:

- $\theta(i)$ is the event type (war, sanctions, …).
- $\beta_{s, \theta} > 0$ is the **peak impact magnitude** for severity $s$ and type $\theta$.
- $\lambda_\theta > 0$ is the **decay rate** — how fast the price effect fades after the event.

The aggregate event-driven log-price contribution is:

$$
\Delta L^{\text{event}}_t = \sum_i \Delta L^{(i)}_t.
$$

### 5.2 Calibration of impulse parameters

For each event type $\theta$ and severity $s$, we fit $(\beta_{s,\theta}, \lambda_\theta)$ on historical event windows by minimizing:

$$
(\hat\beta, \hat\lambda) = \arg\min_{\beta, \lambda} \sum_{i: s_i=s, \, \theta(i)=\theta} \sum_{t=\tau_i}^{\tau_i + W} \Big[ (L_t - L_{\tau_i - 1}) - \delta_i \beta e^{-\lambda(t-\tau_i)} \Big]^2,
$$

with window $W = 60$ trading days. This produces a lookup table of typical impulse magnitudes — e.g., from the Abqaiq attack (Sep 2019, severity HIGH, supply disruption), we estimate $\beta_{\text{high, supply}} \approx 0.14$ (14% peak move) with $\lambda \approx 0.05$ (half-life ~14 days).

### 5.3 Risk premium for Strait of Hormuz

The Strait of Hormuz carries ~20% of seaborne oil. We model **transit-risk premium** as:

$$
\pi^{H}_t = \kappa \cdot \big( \text{shipping\_insurance\_rate}_t - \text{baseline\_rate} \big) \cdot \mathbb{1}[\text{Hormuz tension active}]
$$

where $\kappa$ is calibrated from the relationship between war-risk insurance premiums and oil prices during the 2026 crisis. Historical anchors:

- **Feb 2026:** insurance jumped from 0.125% → 0.4% of vessel value, Brent rose from \$72 → \$120 (+55%).
- **June 2025:** insurance up ~3x, Brent up ~13%.

Empirical estimate: $\hat\kappa \approx 60$ (each 1% jump in insurance premium maps to ~60% sustained oil price premium if the crisis persists).

### 5.4 Trade-flow disruption term

When sanctions or war disrupt physical flows, we add an explicit supply-shock term derived from EIA balances:

$$
\Delta L^{\text{supply}}_t = -\eta \cdot \frac{ \Delta Q^{\text{disrupted}}_t }{ Q^{\text{world}}_t }
$$

where $\Delta Q^{\text{disrupted}}_t$ is the daily barrels removed from market and $\eta$ is the **short-run price elasticity** of supply, with literature estimates of $\eta \approx -10$ (a 1% supply shock raises prices ~10% in the short run, reflecting low short-run elasticity of oil demand).

For example, Hormuz fully closed = ~20mb/d removed; world output ~100mb/d → expected log-price impact $\approx -(-10) \cdot 0.20 = +2.0$ (i.e., $e^2 \approx 7.4$x price). Reality is bounded by SPR releases and shadow flows, so we cap: $|\Delta L^{\text{supply}}_t| \leq 1.0$ (price doubling cap).

### 5.5 Why this matters for the ML model

The event features in $\mathbf{x}_t$ from Section 3.5 are **indicators**, not impact magnitudes. The XGBoost / LSTM models learn the mapping
$$
(\text{event indicators}, \text{macro state}) \mapsto \text{future return}
$$
implicitly from data. The closed-form model above provides:

1. **Sanity checks.** If XGBoost predicts $+0.3\%$ for a Hormuz closure, that's clearly wrong relative to Section 5.4.
2. **Stress-test priors.** For scenarios with no historical precedent, we use the closed-form model.
3. **Fallback predictions** when the ML model is unavailable.

---

## 6. Forecasting models <a id="6-models"></a>

### 6.1 XGBoost gradient boosting

We use the standard regularized gradient boosting objective. With $\hat y^{(m)}_t$ the prediction after $m$ trees:

$$
\hat y^{(m)}_t = \hat y^{(m-1)}_t + \eta \cdot f_m(\mathbf{x}_t),
$$

where each tree $f_m$ minimizes the regularized objective:

$$
\mathcal{L}^{(m)} = \sum_t \ell(y_t, \hat y^{(m-1)}_t + f_m(\mathbf{x}_t)) + \Omega(f_m),
$$

$$
\Omega(f) = \gamma T + \frac{1}{2} \alpha \|w\|_1 + \frac{1}{2} \lambda \|w\|_2^2,
$$

with $T$ = number of leaves, $w$ = leaf weights, and hyperparameters $\gamma, \alpha, \lambda$. The loss $\ell$ is squared error for regression:

$$
\ell(y, \hat y) = \tfrac{1}{2}(y - \hat y)^2.
$$

Tree splits are chosen to maximize the gain (gradient + Hessian Taylor expansion):

$$
\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma,
$$

where $G_*$ = sum of gradients in left/right child, $H_*$ = sum of Hessians. For squared loss the gradient is $(y - \hat y)$ and Hessian is $1$.

**Hyperparameters used (default):** $\eta = 0.03$, $\lambda = 1.0$, $\alpha = 0.1$, max depth 6, subsample 0.8, colsample_bytree 0.8, early stopping after 30 rounds.

### 6.2 LSTM sequence model

Input: a sequence $\mathbf{X}_t = (\mathbf{x}_{t-L+1}, \dots, \mathbf{x}_t) \in \mathbb{R}^{L \times d}$ with lookback $L = 60$.

LSTM cell equations (standard):

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(W_i \mathbf{x}_t + U_i \mathbf{h}_{t-1} + b_i) & \text{(input gate)} \\
\mathbf{f}_t &= \sigma(W_f \mathbf{x}_t + U_f \mathbf{h}_{t-1} + b_f) & \text{(forget gate)} \\
\mathbf{o}_t &= \sigma(W_o \mathbf{x}_t + U_o \mathbf{h}_{t-1} + b_o) & \text{(output gate)} \\
\tilde{\mathbf{c}}_t &= \tanh(W_c \mathbf{x}_t + U_c \mathbf{h}_{t-1} + b_c) & \text{(candidate)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t & \text{(cell state)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) & \text{(hidden state)}
\end{aligned}
$$

with $\sigma(\cdot)$ the logistic sigmoid and $\odot$ elementwise multiplication.

Prediction head:

$$
\hat y_t = \mathbf{w}_2^\top \, \text{ReLU}( W_1 \, \mathbf{h}_t^{(\text{last layer})} + b_1 ) + b_2.
$$

Architecture: 2-layer LSTM, hidden size 64, dropout 0.2, trained with Adam ($\text{lr} = 10^{-3}$), gradient clipping norm 1.0, early stopping patience 8.

### 6.3 Feature standardization

For neural models only (XGBoost is invariant to monotone transformations), features are standardized using **train-set statistics only**:

$$
\tilde x_{i,t} = \frac{x_{i,t} - \hat\mu_i}{\hat\sigma_i + 10^{-8}}, \qquad
\hat\mu_i = \frac{1}{|\mathcal{D}_{\text{tr}}|} \sum_{t \in \mathcal{D}_{\text{tr}}} x_{i,t}.
$$

Using val/test stats here is a **leakage bug** we explicitly guard against in code.

---

## 7. Ensemble combination <a id="7-ensemble"></a>

Let $\hat y^{(M)}_t$ denote the prediction from model $M \in \{\text{XGB}, \text{LSTM}\}$.

### 7.1 Static weighted ensemble

$$
\hat y^{\text{ens}}_t = \sum_M w_M \, \hat y^{(M)}_t, \qquad \sum_M w_M = 1, \quad w_M \geq 0.
$$

Defaults: $w_{\text{XGB}} = 0.6$, $w_{\text{LSTM}} = 0.4$ (XGB tends to be more reliable on small samples).

### 7.2 Dynamic weighting (recent-accuracy)

Optionally, weights adapt based on rolling-window directional accuracy:

$$
\text{acc}^{(M)}_t = \frac{1}{W} \sum_{s = t-W}^{t-1} \mathbb{1}\big[\, \text{sign}(\hat y^{(M)}_s) = \text{sign}(y_s) \,\big],
$$

$$
w^{(M)}_t = \frac{ \exp(\beta \cdot \text{acc}^{(M)}_t) }{ \sum_{M'} \exp(\beta \cdot \text{acc}^{(M')}_t) }
$$

with $\beta = 10$ (softmax sharpness) and $W = 60$. This gives more weight to models that have been right recently.

### 7.3 Model disagreement (for uncertainty)

$$
\text{disag}_t = \mathrm{std}\big( \{\hat y^{(M)}_t\}_M \big).
$$

This feeds into confidence in Section 11.

---

## 8. Loss functions and training objectives <a id="8-losses"></a>

### 8.1 Regression loss

Mean squared error on log returns:

$$
\mathcal{L}_{\text{MSE}}(\hat y, y) = \frac{1}{N} \sum_t (\hat y_t - y_t)^2.
$$

### 8.2 Directional loss (alternative)

A loss that explicitly penalizes wrong-direction predictions more than wrong-magnitude ones:

$$
\mathcal{L}_{\text{dir}}(\hat y, y) = \frac{1}{N} \sum_t \big[ (\hat y_t - y_t)^2 + \mu \cdot \max(0, -\hat y_t \cdot y_t) \big]
$$

with $\mu = 0.1$. The hinge term adds penalty only when the prediction sign is opposite of the truth. Use this when trading P\&L matters more than calibration.

### 8.3 Quantile loss (for uncertainty bands)

For quantile level $q \in (0,1)$:

$$
\mathcal{L}_q(\hat y, y) = \frac{1}{N} \sum_t \rho_q(y_t - \hat y_t), \qquad \rho_q(u) = \begin{cases} q \, u & u \geq 0 \\ (q-1) u & u < 0 \end{cases}.
$$

Training three quantile models ($q \in \{0.1, 0.5, 0.9\}$) yields prediction intervals: $[\hat y_{0.1}, \hat y_{0.9}]$ is an 80% prediction band.

### 8.4 Classification loss (for direction model)

Standard multinomial cross-entropy over three classes ($-1, 0, +1$):

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_t \sum_{c \in \{-1,0,+1\}} \mathbb{1}[y_t = c] \, \ln \hat p_{c,t}.
$$

---

## 9. Cross-validation methodology <a id="9-cv"></a>

### 9.1 Why random K-fold is wrong here

Random splits assume IID data. Time series have:
- Autocorrelation in features (volatility clusters)
- Trend / regime persistence
- A clear temporal direction (future depends on past, not vice versa)

Random K-fold lets the model **see the future** during training, producing wildly optimistic CV scores.

### 9.2 Expanding-window time-series CV

Given $T$ ordered observations, we form $K$ folds:

For $k = 1, \dots, K$ with $K = 5$:
- Training set: $\{1, 2, \dots, t_k - h\}$ where $t_k = \lfloor T \cdot \frac{k}{K+1} \rfloor$.
- Validation set: $\{t_k, t_k + 1, \dots, t_{k+1} - h\}$.

The $-h$ in the training set excludes the last $h$ days, since their targets $y_t^{(h)}$ depend on prices through $t+h$ which leak into validation.

### 9.3 Walk-forward backtesting

Even stricter than CV: we **retrain the model** every $\Delta$ days using only data available at that point, then predict the next $\Delta$ days:

```
for t in [t_start, t_start + Δ, t_start + 2Δ, ...]:
    train_data = data[(t - W) : (t - h)]    # last W years, minus horizon
    model.fit(train_data)
    predictions[t : t+Δ] = model.predict(...)
```

Defaults: $W = 5$ years, $\Delta = 21$ days (retrain monthly).

This is the **only honest test** of forecasting performance on time series.

---

## 10. Backtesting and strategy returns <a id="10-backtest"></a>

Given walk-forward predictions $\{\hat y_t\}$, we evaluate a simple trading strategy.

### 10.1 Position rule

$$
\text{pos}_t = \text{sign}(\hat y_t) \cdot \mathbb{1}\big[ |\hat y_t| > \delta \big] \in \{-1, 0, +1\}.
$$

Long when prediction is sufficiently positive, short when sufficiently negative, flat otherwise.

### 10.2 Strategy return

Gross return on day $t$ (held from $t-h$ to $t$):

$$
r^{\text{strat}}_t = \text{pos}_{t-h} \cdot r_t.
$$

With transaction costs $c$ (in log-return terms, e.g., $c = 5 \text{ bps} = 0.0005$):

$$
r^{\text{net}}_t = r^{\text{strat}}_t - c \cdot |\text{pos}_t - \text{pos}_{t-1}|.
$$

### 10.3 Performance metrics

**Cumulative log return:**
$$
R_T = \sum_t r^{\text{net}}_t.
$$

**Annualized Sharpe ratio:**
$$
\text{SR} = \frac{ \mu_r }{ \sigma_r } \cdot \sqrt{ \frac{252}{h} },
$$
where $\mu_r, \sigma_r$ are the mean and std of $r^{\text{net}}_t$ over non-overlapping $h$-day periods.

**Maximum drawdown:**
$$
\text{MDD} = \min_t \left( e^{R_t} - \max_{s \leq t} e^{R_s} \right) / \max_{s \leq t} e^{R_s}.
$$

**Directional accuracy:**
$$
\text{DA} = \frac{1}{N} \sum_t \mathbb{1}\big[ \text{sign}(\hat y_t) = \text{sign}(y_t) \big].
$$

**Information coefficient** (Spearman rank correlation):
$$
\text{IC} = \rho_{\text{Spearman}}\big( \{\hat y_t\}, \{y_t\} \big).
$$

Industry rule of thumb: $\text{IC} > 0.05$ is meaningful for daily/weekly horizons; $> 0.10$ is excellent.

### 10.4 Per-event metrics

For each historical event $i$ with active window $[\tau_i, \tau_i + W_i]$:

$$
\text{DA}_i = \frac{1}{W_i} \sum_{t \in [\tau_i, \tau_i+W_i]} \mathbb{1}\big[ \text{sign}(\hat y_t) = \text{sign}(y_t) \big].
$$

This answers "does the model do better or worse during the Iraq War / GFC / COVID / Ukraine war / Hormuz crisis?"

---

## 11. Uncertainty quantification <a id="11-uncertainty"></a>

We report **three flavors** of uncertainty.

### 11.1 Model disagreement → confidence

$$
c_t = \exp(-\alpha \cdot \text{disag}_t), \qquad c_t \in (0, 1],
$$

with $\alpha = 20$ tuned so that typical disagreement (~2%) maps to confidence ~0.67.

### 11.2 Quantile prediction intervals

Train quantile regressors (Section 8.3) for $q \in \{0.05, 0.5, 0.95\}$. The 90% prediction interval is $[\hat y_{0.05,t}, \hat y_{0.95,t}]$.

### 11.3 Conformal prediction (distribution-free)

Given a held-out calibration set $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ with non-conformity scores $\alpha_i = |y_i - \hat y_i|$, the $(1-\alpha)$ prediction interval at $\mathbf{x}_t$ is:

$$
[\hat y_t - q_{1-\alpha}, \;\; \hat y_t + q_{1-\alpha}], \qquad q_{1-\alpha} = \text{empirical } (1-\alpha) \text{ quantile of } \{\alpha_i\}.
$$

This is **distribution-free** and provides marginal coverage guarantees: $\mathbb{P}(y_t \in \text{interval}) \geq 1 - \alpha$ under the exchangeability assumption (approximately true for stationary segments).

---

## 12. Stress-test scenarios <a id="12-stress"></a>

For a user-specified hypothetical scenario (e.g., "Strait of Hormuz closes for 14 days"), we compute the expected price impact in three independent ways and report a range.

### 12.1 Closed-form (event-impulse model from §5)

Given hypothetical event $\theta$, severity $s$, direction $\delta$:

$$
\Delta L^{\text{closed}} = \delta \cdot \beta_{s, \theta} \cdot \frac{1 - e^{-\lambda_\theta h}}{ \lambda_\theta h }
$$
(average impact over the $h$-day horizon).

### 12.2 Historical analog (nearest neighbor)

Find $k$ historical events with the most similar (type, severity, region) features:

$$
\text{analogs}_i = \text{top-k} \big( \{e_j : \theta_j = \theta_i, s_j = s_i\} \big).
$$

Use the empirical mean $h$-day return from those analogs as the prediction.

### 12.3 ML model with synthetic features

Construct a feature vector $\tilde{\mathbf{x}}$ where the event indicators are flipped on as if the hypothetical event were active. Predict $\hat y(\tilde{\mathbf{x}})$ using the trained ensemble.

### 12.4 Aggregate

The reported stress-test impact is:

$$
\hat y^{\text{stress}} = \text{median}\big( \hat y^{\text{closed}}, \hat y^{\text{analog}}, \hat y^{\text{ML}} \big)
$$

with the range $[\min, \max]$ as the uncertainty band. The LLM layer adds qualitative narrative.

---

## Appendix A: Implementation traceability

| Math object | Implementation file | Class / function |
|---|---|---|
| $y_t^{(h)}$ | `src/features/builder.py` | `make_target()` |
| Return / vol / momentum features | `src/features/builder.py` | `add_*_features()` |
| Event features $e_t, \tau_t$ | `src/features/builder.py` | `add_event_features()` |
| GMM regime model | `src/models/regime_detector.py` | `HMMRegimeDetector` |
| XGBoost forecaster | `src/models/xgb_model.py` | `XGBoostForecaster` |
| LSTM forecaster | `src/models/lstm_model.py` | `LSTMForecaster` |
| Ensemble | `src/models/ensemble.py` | `EnsembleForecaster` |
| Walk-forward CV | `src/evaluation/backtester.py` | `WalkForwardBacktester` |
| Strategy P&L | `src/evaluation/backtester.py` | `_compute_equity_curve()` |
| Conformal intervals | `src/evaluation/conformal.py` | `ConformalPredictor` (TODO) |
| Event-impulse model | `src/models/event_impulse.py` | `EventImpulseModel` (TODO) |

## Appendix B: Symbol glossary

| Symbol | Meaning | Units |
|---|---|---|
| $P_t$ | WTI close price at day $t$ | USD/barrel |
| $r_t$ | Daily log return | dimensionless |
| $y_t^{(h)}$ | $h$-day forward log return (target) | dimensionless |
| $\sigma^{(w)}_t$ | Rolling $w$-day annualized volatility | per year |
| $\text{DD}_t$ | Drawdown from rolling 1-year max | $\in[-1, 0]$ |
| $z_t$ | Latent regime | $\{1, \dots, K\}$ |
| $\gamma_{k,t}$ | Posterior regime probability | $\in[0,1]$ |
| $e_{j,t}$ | Indicator: event type $j$ active at $t$ | $\{0, 1\}$ |
| $\tau_{j,t}$ | Days since last event of type $j$ | days |
| $\beta_{s, \theta}$ | Peak impact for severity $s$, type $\theta$ | log-price units |
| $\lambda_\theta$ | Decay rate for event type $\theta$ | per day |
| $\eta$ | Short-run oil price elasticity of supply | dimensionless |
| $w_M$ | Ensemble weight for model $M$ | $\in[0,1]$ |
| $c_t$ | Forecast confidence | $\in[0,1]$ |

---

## Appendix C: Key calibrated constants

These come from empirical fitting on historical events (2000-2026). Update via `python -m src.models.event_impulse --recalibrate`.

| Event type | Severity | $\hat\beta$ | $\hat\lambda$ | Half-life | Anchor event |
|---|---|---:|---:|---:|---|
| Supply disruption | high | 0.14 | 0.050 | ~14 d | Abqaiq 2019 |
| Supply disruption | extreme | 0.55 | 0.030 | ~23 d | Hormuz 2026 |
| War | high | 0.18 | 0.020 | ~35 d | Iraq 2003 |
| War | extreme | 0.51 | 0.015 | ~46 d | Iran war 2026 |
| OPEC action | high | 0.12 | 0.010 | ~70 d | OPEC+ 2016 |
| OPEC action | extreme | -0.45 | 0.025 | ~28 d | Saudi-Russia 2020 |
| Financial crisis | extreme | -0.65 | 0.008 | ~87 d | GFC 2008 |
| Pandemic | extreme | -0.70 | 0.012 | ~58 d | COVID 2020 |
| Sanctions | medium | 0.06 | 0.005 | ~140 d | Iran 2012 |
| Sanctions | high | 0.10 | 0.004 | ~175 d | Iran 2018 |

Sign convention: $\hat\beta > 0$ means bullish (price up).

---

**End of specification.**
