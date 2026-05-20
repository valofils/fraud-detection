# Banking Fraud Detection — End-to-End ML System

Production-grade fraud detection system trained on the ULB Credit Card dataset.
XGBoost classifier with engineered features, a FastAPI scoring endpoint, Streamlit
monitoring dashboard, MLflow experiment tracking, automated retraining pipeline,
and a GitHub Actions CI/CD workflow.

---

## Results

| Model | ROC-AUC | PR-AUC | F1 | MCC |
|-------|---------|--------|-----|-----|
| **XGBoost (production, Phase 8)** | **0.9821** | **0.8625** | 0.8511 | 0.8509 |
| XGBoost (Phase 4 baseline) | 0.9747 | 0.8599 | 0.8701 | 0.8722 |
| Random Forest | 0.9543 | 0.8497 | 0.8000 | 0.8006 |
| Logistic Regression | 0.9781 | 0.7111 | 0.8042 | 0.8039 |

**Production threshold: 0.192** (cost-optimal, FN cost = 10× FP)
- False Positives: 13 | False Negatives: 15
- Precision: 0.8602 | Recall: 0.8421
- Brier Score: 0.000410 (near-perfect calibration)

---

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|----------|-------|
| Raw rows | 284,807 |
| After dedup | 280,922 |
| Fraud transactions | 473 (0.168%) |
| Class imbalance | 599:1 |
| Model features | 49 (28 PCA + 21 engineered) |

Primary metric: **PR-AUC** (accuracy is meaningless at 599:1 imbalance).

---

## Project Structure

```
fraud-detection/
├── fraud_eda_phase2.py                  # EDA, duplicate removal, imbalance analysis
├── fraud_feature_engineering_phase3.py  # 21 engineered features
├── fraud_modelling_phase4.py            # Model training, holdout evaluation
├── fraud_interpretability_phase5.py     # SHAP, threshold tuning, calibration
├── fraud_api_phase6.py                  # FastAPI scoring endpoint
├── fraud_mlflow_phase7a.py              # MLflow experiment logging & registry
├── fraud_dashboard_phase7b.py           # Streamlit monitoring dashboard
├── fraud_retrain_phase8a.py             # Automated retraining pipeline
│
├── models/
│   ├── xgboost.joblib                   # Production model (Phase 8, 49 features)
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── feature_names_phase8.json        # Canonical feature list
│   └── backup/                          # Timestamped champion backups
│
├── results/
│   ├── test_split.csv                   # Phase 4 holdout split
│   ├── test_split_phase8.csv            # Phase 8 holdout split
│   ├── final_metrics.csv                # Phase 4-5 metrics
│   └── retrain_phase8_summary.csv       # Phase 8 retrain comparison
│
├── tests/
│   └── test_fraud_api.py                # pytest suite (38 tests)
│
├── .github/
│   └── workflows/
│       └── fraud_ci.yml                 # CI/CD: lint → test → retrain → Docker → smoke
│
├── creditcard.csv                       # Raw dataset (not committed)
├── creditcard_clean.csv                 # After EDA cleaning
├── creditcard_features.csv              # After feature engineering
├── Dockerfile                           # Multi-stage, non-root user
├── docker-compose.yml                   # API + Dashboard + MLflow
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone <repo>
cd fraud-detection
python -m venv fraud_env
fraud_env\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Run the pipeline (first time)

```bash
# EDA
python fraud_eda_phase2.py

# Feature engineering
python fraud_feature_engineering_phase3.py

# Train models
python fraud_modelling_phase4.py

# SHAP + threshold tuning
python fraud_interpretability_phase5.py
```

### 3. Start services

```bash
# Terminal 1 — API
python fraud_api_phase6.py
# → http://localhost:8000/docs

# Terminal 2 — MLflow (optional)
mlflow ui --port 5000
python fraud_mlflow_phase7a.py      # log models (run once)
# → http://localhost:5000

# Terminal 3 — Dashboard (optional)
streamlit run fraud_dashboard_phase7b.py
# → http://localhost:8501
```

### 4. Docker (all services)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API + Swagger | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## API

### `POST /predict` — single transaction

```powershell
$body = '{"V1":-2.31,"V2":1.95,...,"Risk_score":0.82}'
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" -Body $body
```

```json
{
  "fraud_probability": 0.999939,
  "is_fraud": true,
  "decision": "REVIEW",
  "threshold_used": 0.192,
  "latency_ms": 10.02
}
```

### `POST /predict/batch` — up to 1,000 transactions

```json
{ "transactions": [ {...}, {...} ] }
```

### Decision logic

| Score | Decision | Action |
|-------|----------|--------|
| < 0.192 | `PASS` | Approve |
| 0.192 – 0.90 | `BLOCK` | Auto-decline |
| ≥ 0.90 | `REVIEW` | Human review queue |

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.192,
  "uptime_seconds": 3600,
  "total_requests": 1500,
  "total_fraud_flagged": 12
}
```

---

## Features

### Original (28)
PCA-transformed components `V1`–`V28` (anonymised per dataset license).

### Engineered (21, Phase 3)

| Group | Features |
|-------|----------|
| Amount | `Amount_scaled`, `Amount_log`, `Amount_bin`, `Is_micro`, `Is_large` |
| Time (cyclic) | `Hour_sin`, `Hour_cos`, `Is_night`, `Is_peak` |
| Interactions | `V14_x_V12`, `V4_x_V11`, `V14_x_V17`, `Amount_x_V14`, `Amount_x_V4`, `Amount_x_V12`, `V2_x_Amount` |
| Aggregates | `Top_V_mean`, `Top_V_std`, `Top_V_max_abs`, `Top_V_l2_norm`, `Risk_score` |

`V10_x_V16` and `V3_x_V9` were dropped in Phase 8 (near-zero SHAP values).
`Amount_bin_te` (target encoding) excluded from training to prevent leakage.

### Top SHAP features (Phase 5)

| Feature | Mean \|SHAP\| | Note |
|---------|--------------|------|
| `Risk_score` | 1.60 | Dominant — Phase 3 composite |
| `V14` | 0.91 | Decision boundary at −5 |
| `V4_x_V11` | 0.67 | Interaction term — retained |
| `V12` | 0.54 | |
| `Top_V_l2_norm` | 0.48 | FP trigger at value 34 |

---

## Retraining

```bash
python fraud_retrain_phase8a.py             # retrain + auto-promote if PR-AUC improves ≥ 0.002
python fraud_retrain_phase8a.py --dry-run   # evaluate only, no files written
python fraud_retrain_phase8a.py --force     # promote regardless of gate
```

The pipeline:
1. Loads `creditcard_features.csv`, drops `V10_x_V16` and `V3_x_V9`
2. Trains XGBoost + Random Forest on 80/20 stratified split
3. Runs 5-fold CV PR-AUC (in-process, `n_jobs=1` — avoids Windows XGBoost crash)
4. Applies promotion gate: challenger PR-AUC must exceed champion by ≥ 0.002
5. Backs up champion to `models/backup/xgboost_YYYYMMDD_HHMMSS.joblib`
6. Saves new model, logs full run to MLflow, sets `alias:production`

---

## CI/CD

Five-job GitHub Actions pipeline (`.github/workflows/fraud_ci.yml`):

```
push/PR → Lint → Test → Retrain* → Docker → Smoke test
                                ↑
              *only when creditcard_features.csv changes
               or manual dispatch
```

**Required secrets:** `DOCKER_USERNAME`, `DOCKER_PASSWORD`
**Optional:** `MLFLOW_TRACKING_URI`

On promotion, the Docker image is tagged `latest` + git SHA.
On no promotion, only the git SHA tag is pushed.
Every retrain posts a commit comment with the PR-AUC delta.

---

## Tests

```bash
pytest tests/ -v                    # all tests (mock model, no .joblib needed)
pytest tests/ -v -k "not slow"      # skip latency SLA tests
pytest tests/ -v -k "TestRealModel" # regression tests (requires models/xgboost.joblib)
```

38 tests covering: schema validation, decision thresholds, batch scoring,
model determinism, fraud vector regression pin (`prob > 0.99`), latency SLA (p99 < 200ms).

---

## EDA Findings (Phase 2)

- 1,081 duplicate rows removed
- Fraud median amount: **$9.82** — card-testing pattern
- Fraud distributed evenly across all hours; legitimate transactions trough 03:00–06:00
- Top discriminative features by Cohen's d: V14, V4, V12, V11, V10

---

## Stack

| Layer | Library |
|-------|---------|
| Data | pandas, numpy |
| ML | scikit-learn, xgboost, imbalanced-learn |
| Interpretability | shap |
| Serving | fastapi, uvicorn |
| Dashboard | streamlit |
| Tracking | mlflow |
| Serialisation | joblib |
| CI/CD | GitHub Actions, Docker |

---

## Notes

- **LightGBM** was trained in Phase 4 but dropped — `scale_pos_weight` bug produced PR-AUC 0.0956 with `is_unbalance=True`. Excluded from production.
- **False Negative analysis (Phase 5):** one FN had `prob=0.0001` — the fraudster perfectly mimicked legitimate behaviour across all PCA features. Undetectable without external signals (device fingerprint, IP, velocity).
- **False Positive analysis:** one FP had `prob=0.9772` — legitimate transaction with extreme `Top_V_l2_norm=34` inflated `Risk_score`. Routed to `REVIEW` (human queue) rather than `BLOCK`.
- The model accepts `V10_x_V16` and `V3_x_V9` in the request body for backward compatibility but excludes them from inference.
