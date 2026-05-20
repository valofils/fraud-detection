"""
Phase 7a — MLflow Experiment Tracking & Model Registry
Banking Fraud Detection | Logs all Phase 4-5 models + metrics + artifacts
Run once after training; idempotent (re-run safe via run naming).

Usage:
    python fraud_mlflow_phase7a.py
    mlflow ui --port 5000          # view at http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"   # local MLflow server
EXPERIMENT_NAME = "fraud-detection-ulb"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
TEST_SPLIT = RESULTS_DIR / "test_split.csv"
PRODUCTION_THRESHOLD = 0.192

# Models to register: (filename, flavor, registered_name)
MODELS = [
    ("xgboost.joblib",          "xgboost",  "fraud-xgboost"),
    ("random_forest.joblib",    "sklearn",  "fraud-random-forest"),
    ("logistic_regression.joblib", "sklearn", "fraud-logistic-regression"),
]

# Phase 4 holdout metrics (from final_metrics.csv — logged as params for traceability)
PHASE4_METRICS: dict[str, dict] = {
    "xgboost": {
        "roc_auc": 0.9747, "pr_auc": 0.8599, "f1": 0.8701, "mcc": 0.8722,
    },
    "random_forest": {
        "roc_auc": 0.9739, "pr_auc": 0.8412, "f1": 0.8462, "mcc": 0.8467,
    },
    "logistic_regression": {
        "roc_auc": 0.9781, "pr_auc": 0.7111, "f1": 0.8042, "mcc": 0.8039,
    },
}

# Phase 5 threshold-tuned metrics for XGBoost (production model)
PHASE5_XGBOOST: dict = {
    "brier_score": 0.000428,
    "production_threshold": 0.192,
    "threshold_fp": 13,
    "threshold_fn": 14,
    "threshold_precision": 0.8617,
    "threshold_recall": 0.8526,
    "f1_optimal_threshold": 0.847,
    "f1_optimal_fp": 5,
    "f1_optimal_fn": 18,
}

# Phase 3 feature engineering metadata
FEATURE_ENGINEERING: dict = {
    "n_original_features": 31,   # V1-V28 + Amount + Time + Class
    "n_engineered_features": 21,
    "n_model_features": 51,      # Amount/Time replaced by derivatives
    "top_shap_feature": "Risk_score",
    "risk_score_mean_shap": 1.6,
    "dropped_interactions": "V10_x_V16, V3_x_V9",
    "encoding": "Amount_bin_te excluded (leakage prevention)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load holdout test split produced by Phase 4."""
    df = pd.read_csv(TEST_SPLIT)
    y = df["Class"]
    X = df.drop(columns=["Class"])
    return X, y


def recompute_metrics(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """Recompute live metrics from the saved model on the test split."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc":          round(roc_auc_score(y, proba), 4),
        "pr_auc":           round(average_precision_score(y, proba), 4),
        "f1":               round(f1_score(y, preds), 4),
        "mcc":              round(matthews_corrcoef(y, preds), 4),
        "precision":        round(precision_score(y, preds, zero_division=0), 4),
        "recall":           round(recall_score(y, preds, zero_division=0), 4),
        "brier_score":      round(brier_score_loss(y, proba), 6),
        "threshold_used":   threshold,
    }


def log_model_run(
    client: MlflowClient,
    experiment_id: str,
    model_file: str,
    flavor: str,
    registered_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_production: bool = False,
) -> str:
    """Log one model as an MLflow run; return run_id."""
    stem = Path(model_file).stem
    model = joblib.load(MODEL_DIR / model_file)

    threshold = PRODUCTION_THRESHOLD if is_production else 0.5
    live_metrics = recompute_metrics(model, X_test, y_test, threshold)
    phase4 = PHASE4_METRICS.get(stem.replace("_fixed", ""), {})

    run_name = f"{stem}-phase4-5"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id

        # --- Tags ---
        mlflow.set_tags({
            "project": "fraud-detection-ulb",
            "phase": "4-5",
            "dataset": "ULB Credit Card (284807 rows, 473 fraud after dedup)",
            "imbalance_ratio": "599:1",
            "is_production": str(is_production),
            "model_type": stem,
        })

        # --- Params ---
        mlflow.log_params({
            "train_test_split": "80/20 stratified",
            "metric_primary": "PR-AUC",
            "class_imbalance_strategy": "scale_pos_weight / cost-sensitive threshold",
            "production_threshold": PRODUCTION_THRESHOLD,
            **{f"phase4_{k}": v for k, v in phase4.items()},
            **FEATURE_ENGINEERING,
        })

        if is_production:
            mlflow.log_params({f"phase5_{k}": v for k, v in PHASE5_XGBOOST.items()})

        # --- Live metrics ---
        mlflow.log_metrics(live_metrics)

        # --- Model artifact ---
        sample = X_test.iloc[:5]
        sig = infer_signature(sample, model.predict_proba(sample))

        if flavor == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=sig,
                registered_model_name=registered_name,
                input_example=sample,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=sig,
                registered_model_name=registered_name,
                input_example=sample,
            )

        # --- Extra artifacts ---
        # Log feature list as JSON
        feature_list = {"features": list(X_test.columns), "n_features": len(X_test.columns)}
        features_path = Path("_tmp_features.json")
        features_path.write_text(json.dumps(feature_list, indent=2))
        mlflow.log_artifact(str(features_path), artifact_path="metadata")
        features_path.unlink()

        # Log final_metrics.csv if present
        if (RESULTS_DIR / "final_metrics.csv").exists():
            mlflow.log_artifact(str(RESULTS_DIR / "final_metrics.csv"), artifact_path="results")

        print(f"  ✓ {stem} | run_id={run_id} | PR-AUC={live_metrics['pr_auc']}")
        return run_id


def promote_to_production(client: MlflowClient, registered_name: str) -> None:
    """Tag latest model version with 'production' alias (MLflow 2.9+ API, no stages)."""
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        print(f"  ! No versions found for {registered_name}")
        return
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(
        name=registered_name,
        alias="production",
        version=latest.version,
    )
    print(f"  ✓ {registered_name} v{latest.version} → alias:production")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Create or fetch experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"dataset": "ULB", "task": "binary-classification"},
        )
        print(f"Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using experiment '{EXPERIMENT_NAME}' (id={experiment_id})")

    # Load test data
    print("\nLoading test split...")
    X_test, y_test = load_test_data()
    print(f"  {X_test.shape[0]} rows | {y_test.sum()} fraud | {len(X_test.columns)} features")

    # Log all models
    print("\nLogging models...")
    production_run_id = None
    for model_file, flavor, reg_name in MODELS:
        path = MODEL_DIR / model_file
        if not path.exists():
            print(f"  ! Skipping {model_file} (not found)")
            continue
        is_prod = "xgboost" in model_file
        run_id = log_model_run(
            client, experiment_id, model_file, flavor, reg_name,
            X_test, y_test, is_production=is_prod,
        )
        if is_prod:
            production_run_id = run_id

    # Promote XGBoost to Production stage
    print("\nPromoting production model...")
    promote_to_production(client, "fraud-xgboost")

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MLflow logging complete.
Experiment : {EXPERIMENT_NAME}
Production : fraud-xgboost @ threshold=0.192
Run ID     : {production_run_id}

View UI    : http://127.0.0.1:5000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    main()
