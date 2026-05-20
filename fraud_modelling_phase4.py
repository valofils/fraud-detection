"""
Project 01 — Banking Fraud Detection
Phase 4: Modelling

Input  : creditcard_features.csv  (output of Phase 3)
Usage  : python fraud_modelling_phase4.py
Output : models/                  — saved model artefacts (.joblib)
         model_plots/             — evaluation plots
         results/metrics.csv      — consolidated metrics table
"""

import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

PLOTS_DIR = "model_plots"
MODELS_DIR = "models"
RESULTS_DIR = "results"
for d in [PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
FRAUD_COLOR = "#E24B4A"
LEGIT_COLOR = "#378ADD"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ── 1. Load & Split ────────────────────────────────────────────────────────────

def load_and_split(path: str = "creditcard_features.csv"):
    df = pd.read_csv(path)
    print(f"Loaded : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Fraud  : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    # Drop target-encoded column from feature matrix —
    # it was built on the full dataset so we exclude it to avoid any leakage
    # during cross-validation (the CV-safe version is already baked in).
    # We keep it but rebuild it inside CV folds properly in production.
    # For this phase, exclude it for a clean baseline.
    drop_cols = ["Class", "Amount_bin_te"]
    X = df.drop(columns=drop_cols)
    y = df["Class"]

    print(f"Features : {X.shape[1]}")
    return X, y


# ── 2. Metrics Helper ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "ROC-AUC"  : roc_auc_score(y_true, y_proba),
        "PR-AUC"   : average_precision_score(y_true, y_proba),
        "F1"       : f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred),
        "MCC"      : matthews_corrcoef(y_true, y_pred),
    }


def find_best_threshold(y_true, y_proba) -> float:
    """F1-optimal threshold on precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    return thresholds[np.argmax(f1_scores[:-1])]


# ── 3. Model Definitions ───────────────────────────────────────────────────────

def get_models(X, y) -> dict:
    """
    Four models covering the complexity spectrum:
    - Logistic Regression   : linear baseline, interpretable
    - Random Forest         : ensemble, handles non-linearity
    - XGBoost               : gradient boosting, typically best on tabular fraud data
    - LightGBM              : faster XGBoost alternative, good on large datasets

    Class imbalance strategy:
    - LR / RF : class_weight='balanced' (built-in, no data augmentation needed)
    - XGB / LGBM : scale_pos_weight = n_legit / n_fraud (native support)
    - Additionally wrap LR in a SMOTE pipeline for comparison
    """
    n_legit = (y == 0).sum()
    n_fraud = (y == 1).sum()
    scale = n_legit / n_fraud

    models = {
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                C=0.1,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }
    return models


# ── 4. Cross-Validation ────────────────────────────────────────────────────────

def run_cross_validation(models: dict, X, y) -> pd.DataFrame:
    """
    Stratified K-Fold CV for all models.
    Reports PR-AUC and ROC-AUC per fold + mean ± std.
    """
    print(f"\n── Cross-Validation ({N_SPLITS}-fold Stratified) ────────────────")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = []
    for name, model in models.items():
        t0 = time.time()
        scores = cross_validate(
            model, X, y,
            cv=skf,
            scoring=["roc_auc", "average_precision"],
            n_jobs=1,  # avoid nested parallelism issues
            return_train_score=False,
        )
        elapsed = time.time() - t0
        row = {
            "Model"       : name,
            "ROC-AUC mean": scores["test_roc_auc"].mean(),
            "ROC-AUC std" : scores["test_roc_auc"].std(),
            "PR-AUC mean" : scores["test_average_precision"].mean(),
            "PR-AUC std"  : scores["test_average_precision"].std(),
            "Time (s)"    : round(elapsed, 1),
        }
        cv_results.append(row)
        print(f"  {name:<22} ROC-AUC={row['ROC-AUC mean']:.4f}±{row['ROC-AUC std']:.4f}  "
              f"PR-AUC={row['PR-AUC mean']:.4f}±{row['PR-AUC std']:.4f}  [{elapsed:.1f}s]")

    cv_df = pd.DataFrame(cv_results).set_index("Model")
    return cv_df


# ── 5. Final Training & Holdout Evaluation ─────────────────────────────────────

def train_and_evaluate(models: dict, X, y) -> tuple[pd.DataFrame, dict]:
    """
    80/20 stratified split → train all models → evaluate on holdout.
    Saves models to disk.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n── Holdout Evaluation (80/20 split) ────────────────")
    print(f"Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
    print(f"Train fraud: {y_train.sum():,}  |  Test fraud: {y_test.sum():,}")

    all_metrics = []
    artifacts = {}  # store (model, y_proba, threshold) per model name

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["Model"] = name
        metrics["Threshold"] = round(threshold, 4)
        metrics["Train time (s)"] = round(elapsed, 1)
        all_metrics.append(metrics)

        artifacts[name] = {
            "model"    : model,
            "y_proba"  : y_proba,
            "y_pred"   : y_pred,
            "threshold": threshold,
        }

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, f"{MODELS_DIR}/{safe_name}.joblib")

        print(f"  {name:<22} ROC-AUC={metrics['ROC-AUC']:.4f}  "
              f"PR-AUC={metrics['PR-AUC']:.4f}  "
              f"F1={metrics['F1']:.4f}  "
              f"MCC={metrics['MCC']:.4f}  "
              f"threshold={threshold:.3f}  [{elapsed:.1f}s]")

    # Save test split for Phase 5
    X_test_df = X_test.copy()
    X_test_df["Class"] = y_test.values
    X_test_df.to_csv(f"{RESULTS_DIR}/test_split.csv", index=False)

    metrics_df = (
        pd.DataFrame(all_metrics)
        .set_index("Model")
        [["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "MCC", "Threshold", "Train time (s)"]]
    )
    metrics_df.to_csv(f"{RESULTS_DIR}/metrics.csv")
    print(f"\nSaved: {RESULTS_DIR}/metrics.csv")

    return metrics_df, artifacts, X_test, y_test


# ── 6. Plots ───────────────────────────────────────────────────────────────────

COLORS = ["#378ADD", "#1D9E75", "#E24B4A", "#BA7517"]


def plot_roc_curves(artifacts: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, art), color in zip(artifacts.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, art["y_proba"])
        auc = roc_auc_score(y_test, art["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — all models", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/11_roc_curves.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/11_roc_curves.png")


def plot_pr_curves(artifacts: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline (fraud rate={baseline:.4f})")
    for (name, art), color in zip(artifacts.items(), COLORS):
        precision, recall, _ = precision_recall_curve(y_test, art["y_proba"])
        pr_auc = average_precision_score(y_test, art["y_proba"])
        ax.plot(recall, precision, label=f"{name} (AP={pr_auc:.4f})",
                color=color, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — all models", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/12_pr_curves.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/12_pr_curves.png")


def plot_confusion_matrices(artifacts: dict, y_test) -> None:
    n = len(artifacts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, art) in zip(axes, artifacts.items()):
        cm = confusion_matrix(y_test, art["y_pred"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontweight="bold", fontsize=10)

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color=color, fontweight="bold", fontsize=11)

    plt.suptitle("Confusion matrices (F1-optimal threshold)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/13_confusion_matrices.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/13_confusion_matrices.png")


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    metrics_to_plot = ["ROC-AUC", "PR-AUC", "F1", "MCC"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        values = metrics_df[metric]
        bars = ax.barh(values.index, values.values,
                       color=COLORS[:len(values)], edgecolor="white", height=0.5)
        for bar, val in zip(bars, values.values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.set_title(metric, fontweight="bold")
        ax.set_xlim(0, min(1.0, values.max() + 0.1))
        ax.tick_params(labelsize=9)

    plt.suptitle("Model comparison — holdout metrics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/14_metrics_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/14_metrics_comparison.png")


def plot_feature_importance(artifacts: dict, X, top_n: int = 20) -> None:
    """Plot feature importance for tree-based models."""
    tree_models = {k: v for k, v in artifacts.items()
                   if k in ["Random Forest", "XGBoost", "LightGBM"]}

    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 8))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, art) in zip(axes, tree_models.items()):
        model = art["model"]
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        importances = pd.Series(clf.feature_importances_, index=X.columns)
        top = importances.nlargest(top_n).sort_values()

        colors = [FRAUD_COLOR if i >= len(top) - 5 else LEGIT_COLOR
                  for i in range(len(top))]
        top.plot.barh(ax=ax, color=colors, edgecolor="white")
        ax.set_title(f"{name}\nTop {top_n} features", fontweight="bold", fontsize=10)
        ax.set_xlabel("Importance")
        ax.tick_params(labelsize=8)

    plt.suptitle("Feature importance — tree models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/15_feature_importance.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/15_feature_importance.png")


def plot_threshold_analysis(artifacts: dict, y_test) -> None:
    """Show precision/recall/F1 vs threshold for the best model (XGBoost)."""
    name = "XGBoost"
    if name not in artifacts:
        name = list(artifacts.keys())[-1]

    y_proba = artifacts[name]["y_proba"]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1[:-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precision[:-1], color="#378ADD", linewidth=2, label="Precision")
    ax.plot(thresholds, recall[:-1], color="#1D9E75", linewidth=2, label="Recall")
    ax.plot(thresholds, f1[:-1], color="#E24B4A", linewidth=2, label="F1")
    ax.axvline(thresholds[best_idx], color="black", linestyle="--", linewidth=1.5,
               label=f"Best threshold = {thresholds[best_idx]:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{name} — Precision / Recall / F1 vs threshold", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/16_threshold_analysis.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/16_threshold_analysis.png")


# ── 7. Summary ─────────────────────────────────────────────────────────────────

def print_summary(metrics_df: pd.DataFrame, cv_df: pd.DataFrame) -> None:
    best = metrics_df["PR-AUC"].idxmax()
    print("\n" + "=" * 60)
    print("  MODELLING SUMMARY")
    print("=" * 60)
    print("\nCross-validation results:")
    print(cv_df[["ROC-AUC mean", "ROC-AUC std", "PR-AUC mean", "PR-AUC std"]].to_string())
    print("\nHoldout evaluation:")
    print(metrics_df.to_string())
    print(f"\nBest model by PR-AUC : {best}")
    print(f"  ROC-AUC   : {metrics_df.loc[best, 'ROC-AUC']:.4f}")
    print(f"  PR-AUC    : {metrics_df.loc[best, 'PR-AUC']:.4f}")
    print(f"  F1        : {metrics_df.loc[best, 'F1']:.4f}")
    print(f"  MCC       : {metrics_df.loc[best, 'MCC']:.4f}")
    print(f"  Precision : {metrics_df.loc[best, 'Precision']:.4f}")
    print(f"  Recall    : {metrics_df.loc[best, 'Recall']:.4f}")
    print(f"  Threshold : {metrics_df.loc[best, 'Threshold']:.4f}")
    print(f"\nAll models saved to : {MODELS_DIR}/")
    print(f"Test split saved    : {RESULTS_DIR}/test_split.csv")
    print(f"Metrics saved       : {RESULTS_DIR}/metrics.csv")
    print("\nNext : Phase 5 — SHAP interpretability + threshold tuning")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    X, y = load_and_split("creditcard_features.csv")
    models = get_models(X, y)

    cv_df = run_cross_validation(models, X, y)

    # Re-instantiate models (CV exhausts pipeline state in some sklearn versions)
    models = get_models(X, y)
    metrics_df, artifacts, X_test, y_test = train_and_evaluate(models, X, y)

    print("\n── Generating plots ────────────────────────────────")
    plot_roc_curves(artifacts, y_test)
    plot_pr_curves(artifacts, y_test)
    plot_confusion_matrices(artifacts, y_test)
    plot_metrics_comparison(metrics_df)
    plot_feature_importance(artifacts, X)
    plot_threshold_analysis(artifacts, y_test)

    print_summary(metrics_df, cv_df)


if __name__ == "__main__":
    main()
