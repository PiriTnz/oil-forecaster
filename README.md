# 🛢️ Oil Forecaster

Production-grade oil price forecasting system combining ML, deep learning, and LLM analysis. Trained on 25 years of historical data (2000–2025) with explicit modeling of geopolitical events: wars, sanctions, OPEC actions, financial crises, and pandemics.

[![CI](https://github.com/YOUR_USERNAME/oil-forecaster/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/oil-forecaster/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)

## What it does

Given current market data and the geopolitical context, the system produces:

1. **Quantitative forecast** — h-day forward log-return prediction from an ensemble (XGBoost + LSTM)
2. **Regime classification** — what market regime are we in (war, sanctions, OPEC action, crisis, normal)
3. **LLM narrative** — explanation of *why* the model predicts what it does, with historical analogs
4. **Backtest evidence** — performance during specific historical crises (Iraq War, 2008 GFC, Arab Spring, COVID, Ukraine, Israel-Iran, etc.)
5. **Stress test** — hypothetical scenario analysis via the LLM analyst layer

## Key Design Decisions

| Decision | Why |
|---|---|
| **Curated events database** | Pure price-based models miss regime shifts. Hand-labeled events let the model condition on real-world context. |
| **Walk-forward backtesting** | The only honest test for time-series models. Random K-fold leaks future info into training. |
| **Ensemble (XGB + LSTM)** | XGB handles non-linear feature interactions; LSTM captures sequential patterns. Disagreement = uncertainty. |
| **Regime detection** | Same predictor variables behave differently in war vs calm markets. Letting the model see regime context improves accuracy. |
| **LLM as narrator, not predictor** | LLMs hallucinate numbers. Quantitative models forecast; LLM explains and contextualizes. |
| **Per-event backtest metrics** | "How does our model do during oil price wars?" matters more than overall accuracy. |

## Architecture

```
                    ┌───────────────────────┐
                    │   Geopolitical Events │
                    │   Database (2000-25)  │
                    └───────────┬───────────┘
                                │
   Yahoo  ┐                     ▼
   EIA   ─┼──→ Data Loader → Feature Builder → Regime Detector
   FRED  ┘                          │                  │
                                    ▼                  ▼
                            XGBoost  + LSTM  →  Ensemble Forecast
                                                       │
                                                       ▼
                                              LLM Interpreter (Claude)
                                                       │
                                                       ▼
                                               FastAPI + Dashboard
```

## Tech Stack

| Layer | Tools |
|---|---|
| **Data** | Yahoo Finance, EIA, FRED, pandas, pyarrow |
| **ML** | scikit-learn, XGBoost, PyTorch (LSTM), Prophet |
| **MLOps** | MLflow (experiment tracking + model registry) |
| **LLM** | Anthropic Claude (Sonnet 4.5) |
| **Serving** | FastAPI, uvicorn, prometheus-fastapi-instrumentator |
| **Container** | Docker (multi-stage, non-root) |
| **Orchestration** | Kubernetes (EKS), HPA, CronJob for retraining |
| **IaC** | Terraform (VPC, EKS, ECR, S3) |
| **CI/CD** | GitHub Actions (lint, test, security, build, deploy) |
| **Observability** | Prometheus, Grafana, custom alerts |
| **Security** | Trivy, Bandit, NetworkPolicy, IAM IRSA |

## Quick Start

### Local Development

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/oil-forecaster.git
cd oil-forecaster
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY, optionally EIA_API_KEY and FRED_API_KEY

# 2. Install
pip install -r requirements-dev.txt

# 3. Run demo (downloads ~15 years of data, trains XGB, makes a forecast)
python scripts/quickstart_demo.py

# 4. Full training pipeline (creates artifacts/ for the API to serve)
python -m src.pipelines.train --horizon 5

# 5. Start API
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs
# → http://localhost:8000/static/index.html  (dashboard)
```

### With Docker Compose

```bash
docker compose up -d                                # API + MLflow + Prometheus + Grafana
docker compose --profile training run --rm training  # One-off training run
```

Services:
- API:        http://localhost:8000
- Dashboard:  http://localhost:8000/static/index.html
- Docs:       http://localhost:8000/docs
- MLflow:     http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana:    http://localhost:3000  (admin/admin)

## Project Structure

```
oil-forecaster/
├── src/
│   ├── data/
│   │   ├── events_database.py     # Curated geopolitical events 2000-2025
│   │   └── loader.py              # Yahoo/EIA/FRED data loaders with caching
│   ├── features/
│   │   └── builder.py             # Returns, vol, momentum, macro, event features
│   ├── models/
│   │   ├── regime_detector.py     # GMM/HMM market regime classification
│   │   ├── xgb_model.py           # Gradient-boosted forecaster
│   │   ├── lstm_model.py          # PyTorch LSTM forecaster
│   │   └── ensemble.py            # Weighted ensemble of forecasters
│   ├── evaluation/
│   │   └── backtester.py          # Walk-forward backtesting framework
│   ├── llm/
│   │   └── interpreter.py         # Claude-powered forecast narratives
│   ├── pipelines/
│   │   ├── train.py               # End-to-end training pipeline
│   │   └── inference.py           # Inference pipeline
│   └── api/
│       └── main.py                # FastAPI serving layer
├── frontend/                      # HTML+JS dashboard
├── k8s/                           # Kubernetes manifests
├── terraform/                     # AWS infrastructure as code
├── monitoring/                    # Prometheus + Grafana configs
├── tests/                         # Pytest suite
├── scripts/quickstart_demo.py     # End-to-end demo script
├── Dockerfile                     # Serving image (multi-stage)
├── Dockerfile.training            # Training image
├── docker-compose.yml             # Local dev stack
└── requirements.txt
```

## API Examples

```bash
# Get current forecast with LLM analysis
curl http://localhost:8000/api/v1/forecast | jq

# Stress test a hypothetical scenario
curl -X POST http://localhost:8000/api/v1/forecast/stress-test \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Strait of Hormuz closes for 7 days", "horizon_days": 30}'

# List geopolitical events
curl 'http://localhost:8000/api/v1/data/events?event_type=war&start_year=2020'

# Recent regime classifications
curl http://localhost:8000/api/v1/data/regimes?days=180

# Backtest summary (per-event performance)
curl http://localhost:8000/api/v1/backtest/summary | jq '.by_event'
```

## How the model handles war vs peace

The training data covers multiple regime types from 2000-2025. The model learns these patterns:

| Regime | Typical features | Model behavior |
|---|---|---|
| **War (acute)** | high vol, war event active, news count high | Predictions weight risk-premium, mean-revert slower |
| **Sanctions** | sustained high prices, OPEC capacity tight | Slower momentum decay, predicts grinding gains |
| **OPEC action** | sudden supply shock, big single-day moves | Larger absolute predictions but lower confidence |
| **Financial crisis** | demand collapse, vol spike, equity correlation high | Bearish bias, regime-dependent volatility |
| **Normal** | range-bound, vol < 30% | Mean-reversion signals dominate |

Per-regime metrics are tracked in `backtest_by_regime`.

## Deploy to AWS EKS

```bash
# 1. Provision infrastructure
cd terraform
terraform init
terraform apply -var environment=staging

# 2. Configure kubectl
$(terraform output -raw configure_kubectl)

# 3. Create secrets
kubectl create namespace oil-forecaster
kubectl create secret generic oil-forecaster-secrets \
  --from-literal=ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  --from-literal=EIA_API_KEY=$EIA_API_KEY \
  -n oil-forecaster

# 4. Deploy
kubectl apply -f k8s/

# 5. Verify
kubectl get pods -n oil-forecaster
kubectl logs -f deployment/oil-forecaster-api -n oil-forecaster
```

## Retraining

- **Scheduled**: Weekly via Kubernetes `CronJob` (`k8s/04-training-cronjob.yaml`)
- **On-demand**: `kubectl create job --from=cronjob/oil-forecaster-train manual-$(date +%s) -n oil-forecaster`
- **Local**: `python -m src.pipelines.train`

All training runs log to MLflow with hyperparameters, CV metrics, and feature importances. The model registry stores the latest blessed model; the API loads it on startup.

## Limitations & Disclaimers

⚠️ **This is a research/portfolio project.** It is NOT financial advice. Models WILL be wrong, especially during unprecedented events (e.g., COVID-induced negative WTI was not predictable from any prior data).

Specific known limitations:
- Yahoo Finance free tier rate-limits aggressive backfills
- Events database is curated by hand and may miss recent events
- LSTM is disabled by default in `train.py` for speed; enable in production
- News sentiment is not yet integrated (architecturally ready, needs NewsAPI key)
- Crude futures roll-yield is not modeled explicitly

## License

MIT — see [LICENSE](LICENSE)
