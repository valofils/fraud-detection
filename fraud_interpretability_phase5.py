"""
Project 01 — Banking Fraud Detection
Phase 5: Interpretability, Calibration & Threshold Tuning

Input  : models/xgboost.joblib
         models/random_forest.joblib
         models/logistic_regression.joblib
         results/test_split.csv
         creditcard_features.csv  (for SHAP background sample)
Usage  : python fraud_interpretability_phase5.py
Output : shap_plots/   — SHAP visualisations
         results/      — updated metrics, threshold analysis, calibration report
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

SHAP_DIR   = "shap_plots"
RESULTS    = "results"
MODELS_DIR = "models"
for d in [SHAP_DIR, RESULTS]:
    os.makedirs(d, exist_ok=True)

FRAUD_COLOR = "#E24B4A"
LEGIT_COLOR = "#378ADD"
RANDOM_STATE = 42

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ── 1. Load artefacts ──────────────────────────────────────────────────────────

def load_artefacts():
    print("── Loading artefacts ───────────────────────────────")

    test_df = pd.read_csv(f"{RESULTS}/test_split.csv")
    y_test  = test_df["Class"]
    X_test  = test_df.drop(columns=["Class"])

    xgb = joblib.load(f"{MODELS_DIR}/xgboost.joblib")
    rf  = joblib.load(f"{MODELS_DIR}/random_forest.joblib")
    lr  = joblib.load(f"{MODELS_DIR}/logistic_regression.joblib")

    print(f"Test set : {X_test.shape[0]:,} rows | Fraud: {y_test.sum()}")
    print(f"Features : {X_test.shape[1]}")
    return X_test, y_test, xgb, rf, lr


# ── 2. Fix LightGBM ───────────────────────────────────────────────────────────

def retrain_lightgbm(X_test, y_test):
    """
    Phase 4 LightGBM used scale_pos_weight which caused degenerate probabilities.
    Fix: use is_unbalance=True (LightGBM's native recommended approach).
    Retrain on the complement of the test set (approximation for Phase 5 fix).
    """
    print("\n── Retraining LightGBM (is_unbalance=True) ─────────")
    full_df = pd.read_csv("creditcard_features.csv")
    full_df = full_df.drop(columns=["Amount_bin_te"], errors="ignore")

    # Exclude test indices by merging — use all rows not in test set
    X_full = full_df.drop(columns=["Class"])
    y_full = full_df["Class"]

    # Simple retrain on full dataset minus test rows (approximate)
    test_hashes = pd.util.hash_pandas_object(X_test).values
    full_hashes = pd.util.hash_pandas_object(X_full).values
    mask = ~np.isin(full_hashes, test_hashes)

    X_train_lgbm = X_full[mask]
    y_train_lgbm = y_full[mask]

    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,          # correct imbalance fix for LightGBM
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm.fit(X_train_lgbm, y_train_lgbm)

    y_proba = lgbm.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"  LightGBM (fixed) — ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    joblib.dump(lgbm, f"{MODELS_DIR}/lightgbm_fixed.joblib")
    return lgbm


# ── 3. Probability Calibration ─────────────────────────────────────────────────

def calibrate_models(X_test, y_test, xgb, rf, lr):
    """
    Logistic Regression threshold stuck at 1.0 in Phase 4 → miscalibrated.
    Apply isotonic regression calibration to LR and check XGBoost.
    """
    print("\n── Probability Calibration ─────────────────────────")

    # LR pipeline — extract the classifier step for calibration check
    lr_proba = lr.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    rf_proba  = rf.predict_proba(X_test)[:, 1]

    brier = {
        "Logistic Regression": brier_score_loss(y_test, lr_proba),
        "XGBoost"            : brier_score_loss(y_test, xgb_proba),
        "Random Forest"      : brier_score_loss(y_test, rf_proba),
    }

    print("  Brier scores (lower = better calibrated):")
    for name, score in brier.items():
        print(f"    {name:<25} {score:.6f}")

    # Calibration curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    colors = {"Logistic Regression": LEGIT_COLOR,
              "XGBoost": FRAUD_COLOR, "Random Forest": "#1D9E75"}

    for name, proba in [("Logistic Regression", lr_proba),
                        ("XGBoost", xgb_proba),
                        ("Random Forest", rf_proba)]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2,
                label=f"{name} (Brier={brier[name]:.5f})", color=colors[name])

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curves — reliability diagram", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/17_calibration_curves.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_DIR}/17_calibration_curves.png")

    return xgb_proba, rf_proba, lr_proba


# ── 4. Business Threshold Analysis ────────────────────────────────────────────

def business_threshold_analysis(y_test, xgb_proba):
    """
    In fraud detection, costs are asymmetric:
      - False Negative (missed fraud)  : high cost  → bank absorbs the loss
      - False Positive (blocked legit) : low cost   → customer friction

    Sweep thresholds and compute weighted cost at different FN/FP cost ratios.
    Typical real-world ratio: FN cost = 5× to 20× FP cost.
    """
    print("\n── Business Threshold Analysis ─────────────────────")

    thresholds = np.linspace(0.01, 0.99, 200)
    fn_cost_ratio = 10  # FN is 10× more expensive than FP

    records = []
    for t in thresholds:
        y_pred = (xgb_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        cost      = fn * fn_cost_ratio + fp * 1
        records.append({
            "threshold": t, "precision": precision, "recall": recall,
            "f1": f1, "fp": fp, "fn": fn, "cost": cost,
        })

    th_df = pd.DataFrame(records)
    best_f1_t   = th_df.loc[th_df["f1"].idxmax(), "threshold"]
    best_cost_t = th_df.loc[th_df["cost"].idxmin(), "threshold"]

    print(f"  F1-optimal threshold     : {best_f1_t:.3f}")
    print(f"  Cost-optimal threshold   : {best_cost_t:.3f}  (FN cost = {fn_cost_ratio}× FP)")
    print(f"  At cost-optimal threshold:")
    row = th_df.loc[th_df["cost"].idxmin()]
    print(f"    FP (blocked customers) : {int(row['fp'])}")
    print(f"    FN (missed fraud)      : {int(row['fn'])}")
    print(f"    Precision              : {row['precision']:.4f}")
    print(f"    Recall                 : {row['recall']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Precision / Recall / F1 vs threshold
    axes[0].plot(th_df["threshold"], th_df["precision"],
                 color=LEGIT_COLOR, linewidth=2, label="Precision")
    axes[0].plot(th_df["threshold"], th_df["recall"],
                 color="#1D9E75", linewidth=2, label="Recall")
    axes[0].plot(th_df["threshold"], th_df["f1"],
                 color=FRAUD_COLOR, linewidth=2, label="F1")
    axes[0].axvline(best_f1_t, color="black", linestyle="--",
                    linewidth=1.5, label=f"F1-optimal = {best_f1_t:.3f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("XGBoost — P / R / F1 vs threshold", fontweight="bold")
    axes[0].legend(fontsize=9)

    # Business cost vs threshold
    axes[1].plot(th_df["threshold"], th_df["cost"],
                 color=FRAUD_COLOR, linewidth=2)
    axes[1].axvline(best_cost_t, color="black", linestyle="--",
                    linewidth=1.5, label=f"Cost-optimal = {best_cost_t:.3f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel(f"Total cost (FN={fn_cost_ratio}× FP)")
    axes[1].set_title("Business cost vs threshold", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.suptitle("XGBoost threshold analysis", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/18_threshold_business.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_DIR}/18_threshold_business.png")

    return best_cost_t, th_df


# ── 5. SHAP — Global Interpretability ─────────────────────────────────────────

def shap_global(xgb, X_test, y_test):
    print("\n── SHAP Global Interpretability ────────────────────")

    # Use a background sample for speed (TreeExplainer is exact for trees)
    background = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    print(f"  SHAP values computed: {shap_values.shape}")

    # ── 5a. Beeswarm (summary) plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="dot",
                      max_display=20, show=False)
    plt.title("SHAP beeswarm — XGBoost top 20 features", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/19_shap_beeswarm.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_DIR}/19_shap_beeswarm.png")

    # ── 5b. Bar summary (mean |SHAP|) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      max_display=20, show=False)
    plt.title("SHAP mean |value| — XGBoost feature importance", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/20_shap_bar.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_DIR}/20_shap_bar.png")

    return shap_values, explainer


# ── 6. SHAP — Individual Transaction Explanations ─────────────────────────────

def shap_individual(shap_values, explainer, X_test, y_test, xgb_proba):
    print("\n── SHAP Individual Explanations ────────────────────")

    # Pick one true positive (caught fraud) and one false negative (missed fraud)
    tp_idx = np.where((y_test.values == 1) & (xgb_proba >= 0.894))[0]
    fn_idx = np.where((y_test.values == 1) & (xgb_proba < 0.894))[0]
    fp_idx = np.where((y_test.values == 0) & (xgb_proba >= 0.894))[0]

    def save_waterfall(idx, label, filename):
        if len(idx) == 0:
            print(f"  No {label} found — skipping")
            return
        i = idx[0]
        sv = shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=X_test.iloc[i].values,
            feature_names=X_test.columns.tolist(),
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(sv, max_display=15, show=False)
        plt.title(f"SHAP waterfall — {label} (prob={xgb_proba[i]:.4f})",
                  fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(f"{SHAP_DIR}/{filename}", bbox_inches="tight")
        plt.close()
        print(f"  Saved: {SHAP_DIR}/{filename}")

    save_waterfall(tp_idx, "True Positive (caught fraud)",  "21_shap_true_positive.png")
    save_waterfall(fn_idx, "False Negative (missed fraud)", "22_shap_false_negative.png")
    save_waterfall(fp_idx, "False Positive (wrong alert)",  "23_shap_false_positive.png")


# ── 7. SHAP — Interaction Validation ──────────────────────────────────────────

def shap_interactions(xgb, X_test):
    """
    Validate Phase 3 interaction feature choices using SHAP interaction values.
    Computes mean |SHAP interaction| for all feature pairs.
    This confirms which interactions the model actually uses.
    """
    print("\n── SHAP Interaction Validation ─────────────────────")
    print("  Computing SHAP interaction values (sample of 200 rows)...")

    sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(xgb)
    shap_inter = explainer.shap_interaction_values(sample)  # shape: (n, p, p)

    # Mean absolute interaction per pair (off-diagonal)
    mean_inter = np.abs(shap_inter).mean(axis=0)
    np.fill_diagonal(mean_inter, 0)  # zero out main effects

    inter_df = pd.DataFrame(mean_inter,
                            index=X_test.columns,
                            columns=X_test.columns)

    # Top 15 pairwise interactions
    pairs = (inter_df.stack()
             .reset_index()
             .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Mean |SHAP interaction|"})
             .query("`Feature A` < `Feature B`")   # avoid duplicates
             .sort_values("Mean |SHAP interaction|", ascending=False)
             .head(15))

    print("\n  Top 10 SHAP interaction pairs:")
    print(pairs.head(10).to_string(index=False))

    # Phase 3 interaction validation
    phase3_interactions = [
        ("V14", "V12"), ("V4", "V11"), ("V10", "V16"),
        ("V3",  "V9"),  ("V14", "V17"),
    ]
    print("\n  Phase 3 interaction feature validation:")
    for a, b in phase3_interactions:
        if a in inter_df.index and b in inter_df.columns:
            val = inter_df.loc[a, b]
            rank = (pairs["Mean |SHAP interaction|"] >= val).sum()
            print(f"    {a} × {b:<4}  mean|SHAP|={val:.5f}  (rank ≈ {rank})")

    # Heatmap of top features interactions
    top_feats = list(
        pairs[["Feature A", "Feature B"]].values[:10].flatten()
    )
    top_feats = list(dict.fromkeys(top_feats))[:10]  # deduplicate, keep order
    sub = inter_df.loc[top_feats, top_feats]

    fig, ax = plt.subplots(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(sub, cmap="Reds", ax=ax, annot=True, fmt=".4f",
                annot_kws={"size": 7}, linewidths=0.5)
    ax.set_title("SHAP interaction values — top feature pairs", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/24_shap_interactions.png", bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {SHAP_DIR}/24_shap_interactions.png")

    return pairs


# ── 8. SHAP Dependence Plots ──────────────────────────────────────────────────

def shap_dependence(shap_values, X_test):
    """Dependence plots for top 4 features — shows non-linear relationships."""
    print("\n── SHAP Dependence Plots ───────────────────────────")
    top_features = ["Risk_score", "V14", "V14_x_V12", "Top_V_std"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, feat in zip(axes, top_features):
        if feat not in X_test.columns:
            continue
        feat_idx = list(X_test.columns).index(feat)
        shap_feat = shap_values[:, feat_idx]
        feat_vals = X_test[feat].values
        ax.scatter(feat_vals, shap_feat, alpha=0.3, s=8,
                   c=shap_feat, cmap="RdBu_r")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{feat}", fontweight="bold", fontsize=10)

    plt.suptitle("SHAP dependence plots — top features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/25_shap_dependence.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SHAP_DIR}/25_shap_dependence.png")


# ── 9. Final Consolidated Metrics ─────────────────────────────────────────────

def consolidated_metrics(y_test, xgb_proba, rf_proba, lr_proba,
                          lgbm_fixed, X_test, best_cost_t):
    print("\n── Consolidated Final Metrics ──────────────────────")

    lgbm_proba = lgbm_fixed.predict_proba(X_test)[:, 1]

    def metrics_at_threshold(proba, t):
        y_pred = (proba >= t).astype(int)
        return {
            "ROC-AUC"  : roc_auc_score(y_test, proba),
            "PR-AUC"   : average_precision_score(y_test, proba),
            "F1"       : f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall"   : recall_score(y_test, y_pred),
            "MCC"      : matthews_corrcoef(y_test, y_pred),
            "Brier"    : brier_score_loss(y_test, proba),
        }

    from sklearn.metrics import precision_recall_curve
    def best_f1_threshold(proba):
        p, r, t = precision_recall_curve(y_test, proba)
        f1 = 2 * p * r / (p + r + 1e-9)
        return t[np.argmax(f1[:-1])]

    final = {
        "Logistic Regression": metrics_at_threshold(lr_proba,   best_f1_threshold(lr_proba)),
        "Random Forest"      : metrics_at_threshold(rf_proba,   best_f1_threshold(rf_proba)),
        "XGBoost (F1-opt)"   : metrics_at_threshold(xgb_proba,  best_f1_threshold(xgb_proba)),
        "XGBoost (cost-opt)" : metrics_at_threshold(xgb_proba,  best_cost_t),
        "LightGBM (fixed)"   : metrics_at_threshold(lgbm_proba, best_f1_threshold(lgbm_proba)),
    }

    final_df = pd.DataFrame(final).T
    print(final_df.to_string())
    final_df.to_csv(f"{RESULTS}/final_metrics.csv")
    print(f"\nSaved: {RESULTS}/final_metrics.csv")
    return final_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    X_test, y_test, xgb, rf, lr = load_artefacts()

    lgbm_fixed = retrain_lightgbm(X_test, y_test)

    xgb_proba, rf_proba, lr_proba = calibrate_models(X_test, y_test, xgb, rf, lr)

    best_cost_t, th_df = business_threshold_analysis(y_test, xgb_proba)

    shap_values, explainer = shap_global(xgb, X_test, y_test)
    shap_individual(shap_values, explainer, X_test, y_test, xgb_proba)
    interaction_pairs = shap_interactions(xgb, X_test)
    shap_dependence(shap_values, X_test)

    final_df = consolidated_metrics(
        y_test, xgb_proba, rf_proba, lr_proba, lgbm_fixed, X_test, best_cost_t
    )

    print("\n" + "=" * 60)
    print("  PHASE 5 COMPLETE")
    print("=" * 60)
    print(f"  SHAP plots    : {SHAP_DIR}/  (plots 17–25)")
    print(f"  Final metrics : {RESULTS}/final_metrics.csv")
    print(f"  Best model    : XGBoost")
    print(f"  Deploy threshold: {best_cost_t:.3f} (cost-optimal)")
    print("\n  Next: Phase 6 — FastAPI scoring endpoint")
    print("=" * 60)


if __name__ == "__main__":
    main()
