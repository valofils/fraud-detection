"""
Project 01 — Banking Fraud Detection
Phase 3: Feature Engineering

Input  : creditcard_clean.csv  (output of Phase 2)
Usage  : python fraud_feature_engineering_phase3.py
Output : creditcard_features.csv  — enriched feature matrix ready for modelling
         feature_plots/           — feature engineering diagnostic plots
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

LEGIT_COLOR = "#378ADD"
FRAUD_COLOR = "#E24B4A"
PLOTS_DIR = "feature_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# Top discriminative features confirmed in Phase 2
TOP_V = ["V14", "V4", "V12", "V11", "V10", "V16", "V3", "V9", "V17", "V2"]


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_data(path: str = "creditcard_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    # Reconstruct raw Time proxy from cyclic encoding (not needed — we use index order)
    return df


# ── 2. Amount Features ─────────────────────────────────────────────────────────

def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Amount_scaled is already in the dataset (RobustScaler from Phase 2).
    We add binned and log-transformed variants.
    """
    # Recover approximate raw Amount from RobustScaler context is not possible
    # without the scaler object — use Amount_scaled directly for bins via quantile
    df["Amount_log"] = np.log1p(
        np.clip(df["Amount_scaled"] + df["Amount_scaled"].abs().median(), 0, None)
    )

    # Quantile-based bins on the scaled amount (5 bins)
    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    df["Amount_bin"] = kbd.fit_transform(df[["Amount_scaled"]]).astype(int)

    # Micro-transaction flag: bottom 10th percentile of Amount_scaled
    micro_thresh = df["Amount_scaled"].quantile(0.10)
    df["Is_micro"] = (df["Amount_scaled"] <= micro_thresh).astype(int)

    # Large-transaction flag: top 5th percentile
    large_thresh = df["Amount_scaled"].quantile(0.95)
    df["Is_large"] = (df["Amount_scaled"] >= large_thresh).astype(int)

    print("Amount features     : Amount_log, Amount_bin, Is_micro, Is_large")
    return df


# ── 3. Time / Cyclical Features ────────────────────────────────────────────────

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hour_sin and Hour_cos already exist from Phase 2.
    Add a proxy day index and night/day flag from the cyclic features.
    """
    # Reconstruct hour angle from sin/cos (atan2 gives angle in [-pi, pi])
    hour_angle = np.arctan2(df["Hour_sin"], df["Hour_cos"])
    hour_reconstructed = (hour_angle / (2 * np.pi) * 24) % 24

    # Night flag: 00:00–06:00
    df["Is_night"] = ((hour_reconstructed >= 0) & (hour_reconstructed < 6)).astype(int)

    # Peak hours flag: 10:00–20:00 (high legit volume, relatively lower fraud rate)
    df["Is_peak"] = ((hour_reconstructed >= 10) & (hour_reconstructed < 20)).astype(int)

    print("Time features       : Is_night, Is_peak  (Hour_sin/cos retained from Phase 2)")
    return df


# ── 4. Interaction Features ────────────────────────────────────────────────────

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise products of the top discriminative V features.
    V14 × V12 and V4 × V11 showed the strongest cross-class separation in Phase 2.
    Amount × V features capture amount-conditioned fraud patterns.
    """
    # Top V-feature interactions (confirmed by Cohen's d ranking in Phase 2)
    df["V14_x_V12"] = df["V14"] * df["V12"]
    df["V4_x_V11"]  = df["V4"]  * df["V11"]
    df["V10_x_V16"] = df["V10"] * df["V16"]
    df["V3_x_V9"]   = df["V3"]  * df["V9"]
    df["V14_x_V17"] = df["V14"] * df["V17"]

    # Amount × top risk features (captures high-amount fraud signal)
    df["Amount_x_V14"] = df["Amount_scaled"] * df["V14"]
    df["Amount_x_V4"]  = df["Amount_scaled"] * df["V4"]
    df["Amount_x_V12"] = df["Amount_scaled"] * df["V12"]

    # V2 × Amount: notable −0.53 correlation spotted in Phase 2 heatmap
    df["V2_x_Amount"] = df["V2"] * df["Amount_scaled"]

    print("Interaction features: V14×V12, V4×V11, V10×V16, V3×V9, V14×V17,")
    print("                      Amount×V14, Amount×V4, Amount×V12, V2×Amount")
    return df


# ── 5. Aggregate / Statistical Features ───────────────────────────────────────

def engineer_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-level aggregates across the top V features:
    - Mean, std, max absolute value across top discriminative features
    - Fraud risk score: weighted sum using Phase 2 Cohen's d effect sizes as weights
    """
    effect_sizes = {
        "V14": 2.2281, "V4": 1.9850, "V12": 1.8476, "V11": 1.8441,
        "V10": 1.6025, "V16": 1.4449, "V3":  1.3503, "V9":  1.3253,
        "V17": 1.3069, "V2": 1.0894,
    }

    top_vals = df[TOP_V]

    df["Top_V_mean"]   = top_vals.mean(axis=1)
    df["Top_V_std"]    = top_vals.std(axis=1)
    df["Top_V_max_abs"] = top_vals.abs().max(axis=1)
    df["Top_V_l2_norm"] = np.sqrt((top_vals ** 2).sum(axis=1))

    # Weighted risk score: features where fraud > legit get positive weight,
    # features where fraud < legit get negative weight (sign-adjusted)
    fraud_direction = {
        "V14": -1, "V4": +1, "V12": -1, "V11": +1,
        "V10": -1, "V16": -1, "V3":  -1, "V9":  -1,
        "V17": -1, "V2":  +1,
    }
    risk_score = sum(
        df[feat] * effect_sizes[feat] * fraud_direction[feat]
        for feat in TOP_V
    )
    # Normalise to [0, 1]
    df["Risk_score"] = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

    print("Aggregate features  : Top_V_mean, Top_V_std, Top_V_max_abs, Top_V_l2_norm, Risk_score")
    return df


# ── 6. Target Encoding (CV-safe) ───────────────────────────────────────────────

def engineer_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target-encode Amount_bin using 5-fold CV to prevent leakage.
    Each fold uses the out-of-fold fraud rate as the encoded value.
    Global mean is used as a fallback for unseen bins at inference.
    """
    global_mean = df["Class"].mean()
    te_col = np.zeros(len(df))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(df, df["Class"]):
        fold_map = df.iloc[train_idx].groupby("Amount_bin")["Class"].mean()
        te_col[val_idx] = df.iloc[val_idx]["Amount_bin"].map(fold_map).fillna(global_mean)

    df["Amount_bin_te"] = te_col
    print("Target encoding     : Amount_bin_te  (5-fold CV, leakage-free)")
    return df


# ── 7. Diagnostic Plots ────────────────────────────────────────────────────────

def plot_risk_score(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cls, label, color in [(0, "Legit", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
        subset = df[df["Class"] == cls]["Risk_score"]
        subset.plot.kde(ax=axes[0], label=label, color=color, linewidth=2)
    axes[0].set_title("Risk score distribution by class", fontweight="bold")
    axes[0].set_xlabel("Risk score (0–1)")
    axes[0].legend()

    axes[1].hist(df[df["Class"] == 0]["Risk_score"], bins=60,
                 color=LEGIT_COLOR, alpha=0.7, label="Legit", density=True)
    axes[1].hist(df[df["Class"] == 1]["Risk_score"], bins=30,
                 color=FRAUD_COLOR, alpha=0.85, label="Fraud", density=True)
    axes[1].set_title("Risk score histogram (density)", fontweight="bold")
    axes[1].set_xlabel("Risk score (0–1)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_risk_score.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/07_risk_score.png")


def plot_interaction_features(df: pd.DataFrame) -> None:
    interactions = ["V14_x_V12", "V4_x_V11", "V10_x_V16", "Amount_x_V14", "V2_x_Amount"]

    fig, axes = plt.subplots(1, len(interactions), figsize=(18, 4))
    for ax, feat in zip(axes, interactions):
        for cls, label, color in [(0, "Legit", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
            subset = df[df["Class"] == cls][feat]
            lo, hi = subset.quantile([0.02, 0.98])
            subset.clip(lo, hi).plot.kde(ax=ax, label=label, color=color, linewidth=1.8)
        ax.set_title(feat.replace("_x_", " × "), fontweight="bold", fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    plt.suptitle("Interaction feature distributions — Legit vs Fraud",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_interaction_features.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/08_interaction_features.png")


def plot_feature_correlation_heatmap(df: pd.DataFrame, new_features: list) -> None:
    cols = new_features + ["Class"]
    corr = df[cols].corr()["Class"].drop("Class").sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [FRAUD_COLOR if c > 0 else LEGIT_COLOR for c in corr]
    corr.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("New feature correlation with Class (fraud=1)", fontweight="bold")
    ax.set_xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/09_new_feature_correlations.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/09_new_feature_correlations.png")


def plot_night_fraud_rate(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Fraud rate by night/day
    night_rate = df.groupby("Is_night")["Class"].mean() * 100
    bars = axes[0].bar(["Day (06:00–00:00)", "Night (00:00–06:00)"],
                       [night_rate.get(0, 0), night_rate.get(1, 0)],
                       color=[LEGIT_COLOR, FRAUD_COLOR], width=0.5, edgecolor="white")
    for bar, val in zip(bars, [night_rate.get(0, 0), night_rate.get(1, 0)]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{val:.3f}%", ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Fraud rate: day vs night", fontweight="bold")
    axes[0].set_ylabel("Fraud rate (%)")

    # Fraud rate by amount bin
    bin_rate = df.groupby("Amount_bin")["Class"].mean() * 100
    axes[1].bar(range(len(bin_rate)), bin_rate.values,
                color=FRAUD_COLOR, alpha=0.85, edgecolor="white")
    axes[1].set_xticks(range(len(bin_rate)))
    axes[1].set_xticklabels([f"Bin {i}\n(Q{i*20}–Q{(i+1)*20})" for i in range(len(bin_rate))],
                            fontsize=9)
    axes[1].set_title("Fraud rate by amount quantile bin", fontweight="bold")
    axes[1].set_ylabel("Fraud rate (%)")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/10_fraud_rate_by_segment.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/10_fraud_rate_by_segment.png")


# ── 8. Feature Summary ─────────────────────────────────────────────────────────

def print_feature_summary(df: pd.DataFrame, original_cols: list) -> list:
    new_features = [c for c in df.columns if c not in original_cols and c != "Class"]

    print("\n── New features engineered ─────────────────────────")
    groups = {
        "Amount"     : ["Amount_log", "Amount_bin", "Is_micro", "Is_large"],
        "Time"       : ["Is_night", "Is_peak"],
        "Interaction": ["V14_x_V12", "V4_x_V11", "V10_x_V16", "V3_x_V9",
                        "V14_x_V17", "Amount_x_V14", "Amount_x_V4",
                        "Amount_x_V12", "V2_x_Amount"],
        "Aggregate"  : ["Top_V_mean", "Top_V_std", "Top_V_max_abs",
                        "Top_V_l2_norm", "Risk_score"],
        "Encoding"   : ["Amount_bin_te"],
    }
    for group, feats in groups.items():
        corrs = [f"{f} ({df[f].corr(df['Class']):.3f})" for f in feats if f in df.columns]
        print(f"  {group:<12}: {', '.join(corrs)}")

    print(f"\nTotal features: {df.shape[1] - 1}  ({len(new_features)} new + {len(original_cols) - 1} from Phase 2)")
    return new_features


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = load_data("creditcard_clean.csv")
    original_cols = df.columns.tolist()

    print("\n── Engineering features ────────────────────────────")
    df = engineer_amount_features(df)
    df = engineer_time_features(df)
    df = engineer_interaction_features(df)
    df = engineer_aggregate_features(df)
    df = engineer_target_encoding(df)

    new_features = print_feature_summary(df, original_cols)

    print("\n── Generating diagnostic plots ─────────────────────")
    plot_risk_score(df)
    plot_interaction_features(df)
    plot_feature_correlation_heatmap(df, new_features)
    plot_night_fraud_rate(df)

    # Final checks
    assert df.isnull().sum().sum() == 0, "Missing values in feature matrix!"
    assert df.duplicated().sum() == 0, "Duplicates in feature matrix!"

    df.to_csv("creditcard_features.csv", index=False)
    print(f"\nSaved : creditcard_features.csv  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    print("Ready : Phase 4 — Modelling")


if __name__ == "__main__":
    main()
