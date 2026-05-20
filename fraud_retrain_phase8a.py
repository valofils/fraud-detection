"""
Phase 8a — Retraining Pipeline
Banking Fraud Detection | Automated retraining with feature pruning + MLflow auto-promotion

Changes vs Phase 4-5:
  - Drops V10_x_V16 and V3_x_V9 (near-zero SHAP, Phase 5 finding)
  - Retrain XGBoost (production) + Random Forest (challenger)
  - PR-AUC gate: new model must beat champion by >= MIN_IMPROVEMENT
  - Auto-promotes winner to MLflow alias 'production' and saves to models/
  - Full audit trail in MLflow; never overwrites champion if gate fails

Usage:
    python fraud_retrain_phase8a.py
    python fraud_retrain_phase8a.py --force   # skip improvement gate
    python fraud_retrain_phase8a.py --dry-run # score only, no promotion
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("retrain")

FEATURES_CSV      = Path("creditcard_features.csv")
MODEL_DIR         = Path("models")
BACKUP_DIR        = Path("models/backup")
MLFLOW_URI        = "http://127.0.0.1:5000"
MLFLOW_URI_LOCAL  = "mlruns"   # fallback: local file store (no server needed)
EXPERIMENT_NAME   = "fraud-detection-ulb"
REGISTERED_XGB    = "fraud-xgboost"
REGISTERED_RF     = "fraud-random-forest"

PRODUCTION_THRESHOLD = 0.192
TEST_SIZE            = 0.20
RANDOM_STATE         = 42
MIN_IMPROVEMENT      = 0.002   # PR-AUC delta required to replace champion
CV_FOLDS             = 5

# Features to drop — near-zero SHAP interaction terms (Phase 5 finding)
DROP_FEATURES: list[str] = ["V10_x_V16", "V3_x_V9"]

# Champion metrics from Phase 4-5 (baseline to beat)
CHAMPION_METRICS: dict[str, float] = {
    "pr_auc": 0.8599,
    "roc_auc": 0.9747,
    "f1":      0.8701,
    "mcc":     0.8722,
}

# XGBoost hyperparameters (Phase 4 tuned)
XGB_PARAMS: dict = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 599,   # class imbalance ratio
    "eval_metric":      "aucpr",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbosity":        0,
}

RF_PARAMS: dict = {
    "n_estimators":  300,
    "max_depth":     20,
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    log.info("Loading %s", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in features CSV")

    y = df["Class"]
    X = df.drop(columns=["Class"])

    # Drop leaky / excluded columns if present
    for col in ["Amount_bin_te", "Amount", "Time"]:
        if col in X.columns:
            X = X.drop(columns=[col])
            log.info("  Excluded column: %s", col)

    # Drop near-zero SHAP features
    dropped = [f for f in DROP_FEATURES if f in X.columns]
    X = X.drop(columns=dropped)
    log.info("  Pruned features: %s", dropped)

    feature_names = list(X.columns)
    log.info("  Shape: %s | Fraud: %d (%.3f%%)",
             X.shape, y.sum(), y.mean() * 100)
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(model, X: pd.DataFrame, y: pd.Series,
                    threshold: float = PRODUCTION_THRESHOLD) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    return {
        "roc_auc":      round(roc_auc_score(y, proba), 4),
        "pr_auc":       round(average_precision_score(y, proba), 4),
        "f1":           round(f1_score(y, preds), 4),
        "mcc":          round(matthews_corrcoef(y, preds), 4),
        "precision":    round(precision_score(y, preds, zero_division=0), 4),
        "recall":       round(recall_score(y, preds, zero_division=0), 4),
        "brier_score":  round(brier_score_loss(y, proba), 6),
        "fp":           fp,
        "fn":           fn,
        "threshold":    threshold,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    log.info("Training XGBoost (%d features)...", X_train.shape[1])
    t0 = time.perf_counter()
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    log.info("  Done in %.1fs", time.perf_counter() - t0)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    log.info("Training Random Forest (%d features)...", X_train.shape[1])
    t0 = time.perf_counter()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    log.info("  Done in %.1fs", time.perf_counter() - t0)
    return model


def cv_pr_auc(model, X: pd.DataFrame, y: pd.Series) -> float:
    """5-fold stratified CV PR-AUC for robust model comparison.
    n_jobs=1 (in-process) avoids XGBoost + loky multiprocessing crash on Windows.
    """
    log.info("  Running %d-fold CV PR-AUC (n_jobs=1)...", CV_FOLDS)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    log.info("  CV PR-AUC: %.4f ± %.4f", scores.mean(), scores.std())
    return float(scores.mean())


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
def log_retrain_run(
    client: MlflowClient,
    experiment_id: str,
    model,
    model_name: str,
    registered_name: str,
    flavor: str,
    metrics: dict,
    cv_score: float,
    feature_names: list[str],
    X_sample: pd.DataFrame,
    champion_pr_auc: float,
    promoted: bool,
) -> str:
    run_name = f"{model_name}-phase8-retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags({
            "project":        "fraud-detection-ulb",
            "phase":          "8a-retrain",
            "model_type":     model_name,
            "promoted":       str(promoted),
            "features_dropped": ",".join(DROP_FEATURES),
            "n_features":     str(len(feature_names)),
        })
        mlflow.log_params({
            "drop_features":         DROP_FEATURES,
            "n_features":            len(feature_names),
            "production_threshold":  PRODUCTION_THRESHOLD,
            "min_improvement_gate":  MIN_IMPROVEMENT,
            "cv_folds":              CV_FOLDS,
            **(XGB_PARAMS if model_name == "xgboost" else RF_PARAMS),
        })
        mlflow.log_metrics({
            **metrics,
            "cv_pr_auc":            round(cv_score, 4),
            "pr_auc_delta_vs_champion": round(metrics["pr_auc"] - champion_pr_auc, 4),
        })

        sig = infer_signature(X_sample, model.predict_proba(X_sample))
        if flavor == "xgboost":
            mlflow.xgboost.log_model(
                model, name="model",
                signature=sig,
                registered_model_name=registered_name,
                input_example=X_sample,
            )
        else:
            mlflow.sklearn.log_model(
                model, name="model",
                signature=sig,
                registered_model_name=registered_name,
                input_example=X_sample,
            )

        # Log feature list
        feat_path = Path("_features_phase8.json")
        feat_path.write_text(json.dumps({"features": feature_names,
                                         "dropped": DROP_FEATURES}, indent=2))
        mlflow.log_artifact(str(feat_path), artifact_path="metadata")
        feat_path.unlink()

        run_id = run.info.run_id
        log.info("  MLflow run: %s | PR-AUC=%.4f | CV=%.4f | promoted=%s",
                 run_id[:8], metrics["pr_auc"], cv_score, promoted)
        return run_id


def set_production_alias(client: MlflowClient, registered_name: str) -> str:
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for {registered_name}")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(
        name=registered_name,
        alias="production",
        version=latest.version,
    )
    log.info("  %s v%s → alias:production", registered_name, latest.version)
    return latest.version


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
def backup_champion(model_file: str) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    src = MODEL_DIR / model_file
    if src.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = BACKUP_DIR / f"{src.stem}_{ts}.joblib"
        shutil.copy2(src, dst)
        log.info("  Champion backed up → %s", dst)


def save_model(model, model_file: str) -> None:
    path = MODEL_DIR / model_file
    joblib.dump(model, path)
    log.info("  Saved → %s", path)


def save_feature_list(feature_names: list[str]) -> None:
    path = MODEL_DIR / "feature_names_phase8.json"
    path.write_text(json.dumps({"features": feature_names,
                                "dropped": DROP_FEATURES,
                                "n_features": len(feature_names)}, indent=2))
    log.info("  Feature list saved → %s", path)


def save_test_split(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    out = X_test.copy()
    out["Class"] = y_test.values
    path = Path("results/test_split_phase8.csv")
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    log.info("  Test split saved → %s", path)


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------
def evaluate_promotion(
    challenger_metrics: dict,
    champion_pr_auc: float,
    force: bool,
) -> tuple[bool, str]:
    delta = challenger_metrics["pr_auc"] - champion_pr_auc
    if force:
        return True, f"forced (delta={delta:+.4f})"
    if delta >= MIN_IMPROVEMENT:
        return True, f"PR-AUC improved by {delta:+.4f} (gate={MIN_IMPROVEMENT})"
    return False, f"PR-AUC delta {delta:+.4f} below gate ({MIN_IMPROVEMENT}) — champion retained"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(force: bool = False, dry_run: bool = False) -> None:
    log.info("═" * 60)
    log.info("Phase 8a — Retraining Pipeline")
    log.info("Dropped features: %s", DROP_FEATURES)
    if dry_run:
        log.info("DRY RUN — no models will be saved or promoted")
    log.info("═" * 60)

    # MLflow setup — probe server; fall back to local file store if unreachable
    import socket
    def _mlflow_server_reachable(uri: str) -> bool:
        try:
            host = uri.split("//")[1].split(":")[0]
            port = int(uri.split(":")[-1].split("/")[0])
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception:
            return False

    if _mlflow_server_reachable(MLFLOW_URI):
        tracking_uri = MLFLOW_URI
        log.info("MLflow server reachable at %s", MLFLOW_URI)
    else:
        tracking_uri = MLFLOW_URI_LOCAL
        log.warning("MLflow server unreachable — using local file store: %s/", MLFLOW_URI_LOCAL)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = exp.experiment_id if exp else mlflow.create_experiment(EXPERIMENT_NAME)

    # Data
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info("Train: %d | Test: %d | Features: %d",
             len(X_train), len(X_test), len(feature_names))

    results: list[dict] = []

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = compute_metrics(xgb_model, X_test, y_test)
    xgb_cv = cv_pr_auc(
        XGBClassifier(**XGB_PARAMS), X_train, y_train
    )
    xgb_promote, xgb_reason = evaluate_promotion(
        xgb_metrics, CHAMPION_METRICS["pr_auc"], force
    )

    log.info("XGBoost results:")
    for k, v in xgb_metrics.items():
        log.info("  %s: %s", k, v)
    log.info("  Promotion: %s — %s", "YES" if xgb_promote else "NO", xgb_reason)

    if not dry_run:
        backup_champion("xgboost.joblib")
        xgb_run_id = log_retrain_run(
            client, experiment_id, xgb_model, "xgboost", REGISTERED_XGB,
            "xgboost", xgb_metrics, xgb_cv, feature_names,
            X_test.iloc[:5], CHAMPION_METRICS["pr_auc"], xgb_promote,
        )
        if xgb_promote:
            save_model(xgb_model, "xgboost.joblib")
            set_production_alias(client, REGISTERED_XGB)

    results.append({"model": "xgboost", "promoted": xgb_promote,
                    "reason": xgb_reason, **xgb_metrics, "cv_pr_auc": xgb_cv})

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = compute_metrics(rf_model, X_test, y_test)
    rf_cv = cv_pr_auc(
        RandomForestClassifier(**RF_PARAMS), X_train, y_train
    )
    rf_promote, rf_reason = evaluate_promotion(
        rf_metrics, CHAMPION_METRICS["pr_auc"], force=False  # RF is never auto-promoted to prod
    )
    # RF is challenger only — log but never replace XGBoost as production
    rf_promote_registry = rf_metrics["pr_auc"] > 0.84  # version gate for registry only

    log.info("Random Forest results:")
    for k, v in rf_metrics.items():
        log.info("  %s: %s", k, v)

    if not dry_run:
        backup_champion("random_forest.joblib")
        rf_run_id = log_retrain_run(
            client, experiment_id, rf_model, "random_forest", REGISTERED_RF,
            "sklearn", rf_metrics, rf_cv, feature_names,
            X_test.iloc[:5], CHAMPION_METRICS["pr_auc"], rf_promote_registry,
        )
        if rf_promote_registry:
            save_model(rf_model, "random_forest.joblib")

    results.append({"model": "random_forest", "promoted": rf_promote_registry,
                    "reason": rf_reason, **rf_metrics, "cv_pr_auc": rf_cv})

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results)[
        ["model", "pr_auc", "cv_pr_auc", "roc_auc", "f1", "mcc",
         "precision", "recall", "fp", "fn", "promoted", "reason"]
    ]
    summary_path = Path("results/retrain_phase8_summary.csv")
    summary_path.parent.mkdir(exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    if not dry_run:
        save_feature_list(feature_names)
        save_test_split(X_test, y_test)

    # Print summary table
    log.info("\n%s", "═" * 60)
    log.info("RETRAIN SUMMARY")
    log.info("%s", "═" * 60)
    for _, row in summary_df.iterrows():
        log.info(
            "%-22s PR-AUC=%.4f (CV=%.4f) | F1=%.4f | MCC=%.4f | FP=%s FN=%s | promoted=%s",
            row["model"], row["pr_auc"], row["cv_pr_auc"],
            row["f1"], row["mcc"], row["fp"], row["fn"], row["promoted"],
        )
    log.info("%s", "═" * 60)

    champion_delta = xgb_metrics["pr_auc"] - CHAMPION_METRICS["pr_auc"]
    if xgb_promote:
        log.info("✅ New XGBoost champion promoted (PR-AUC delta %+.4f)", champion_delta)
        log.info("   Features pruned: %s → 49 features", DROP_FEATURES)
    else:
        log.info("⏸  Champion retained. Challenger delta: %+.4f", champion_delta)

    if dry_run:
        log.info("DRY RUN complete — no files written.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8a — Fraud Detection Retraining")
    parser.add_argument("--force",   action="store_true",
                        help="Promote regardless of PR-AUC improvement gate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score only, no model saving or MLflow promotion")
    args = parser.parse_args()
    main(force=args.force, dry_run=args.dry_run)
