"""
Phase 7b — Streamlit Monitoring Dashboard
Banking Fraud Detection | Live scoring UI + API metrics + MLflow run comparison

Usage:
    streamlit run fraud_dashboard_phase7b.py
    # Requires fraud_api_phase6.py running on localhost:8000
    # Optional: MLflow server on localhost:5000
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"
MLFLOW_BASE = "http://localhost:5000"
THRESHOLD = 0.192
FEATURE_COLS: list[str] = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount_scaled","Hour_sin","Hour_cos","Amount_log","Amount_bin",
    "Is_micro","Is_large","Is_night","Is_peak",
    "V14_x_V12","V4_x_V11","V10_x_V16","V3_x_V9",
    "V14_x_V17","Amount_x_V14","Amount_x_V4","Amount_x_V12","V2_x_Amount",
    "Top_V_mean","Top_V_std","Top_V_max_abs","Top_V_l2_norm","Risk_score",
]

# Known-fraud test vector (Phase 7 validation payload)
FRAUD_EXAMPLE: dict = {
    "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.41,
    "V5": -0.95, "V6": -1.15, "V7": -2.81, "V8": 0.39,
    "V9": -0.82, "V10": -2.75, "V11": 2.74, "V12": -2.24,
    "V13": 0.17, "V14": -6.50, "V15": 0.22, "V16": -2.19,
    "V17": -2.02, "V18": -1.71, "V19": 0.38, "V20": 0.13,
    "V21": 0.49, "V22": -0.14, "V23": -0.08, "V24": 0.05,
    "V25": 0.21, "V26": 0.13, "V27": 0.24, "V28": 0.06,
    "Amount_scaled": -0.18, "Hour_sin": 0.98, "Hour_cos": 0.17,
    "Amount_log": 2.28, "Amount_bin": 0,
    "Is_micro": 1, "Is_large": 0, "Is_night": 0, "Is_peak": 1,
    "V14_x_V12": 14.56, "V4_x_V11": 9.34,
    "V10_x_V16": 6.02, "V3_x_V9": 1.32,
    "V14_x_V17": 13.13, "Amount_x_V14": -63.83,
    "Amount_x_V4": 33.49, "Amount_x_V12": -21.99,
    "V2_x_Amount": 19.15,
    "Top_V_mean": -1.24, "Top_V_std": 1.89,
    "Top_V_max_abs": 6.50, "Top_V_l2_norm": 9.73,
    "Risk_score": 0.82,
}

LEGIT_EXAMPLE: dict = {
    "V1": 1.19, "V2": 0.26, "V3": 0.17, "V4": 0.45,
    "V5": -0.32, "V6": -0.07, "V7": 0.17, "V8": 0.09,
    "V9": 0.25, "V10": 0.22, "V11": 0.16, "V12": 0.08,
    "V13": -0.09, "V14": 0.14, "V15": -0.21, "V16": 0.05,
    "V17": -0.03, "V18": 0.07, "V19": -0.02, "V20": 0.01,
    "V21": -0.01, "V22": 0.06, "V23": -0.03, "V24": 0.01,
    "V25": 0.05, "V26": 0.04, "V27": 0.01, "V28": 0.00,
    "Amount_scaled": 0.31, "Hour_sin": 0.50, "Hour_cos": 0.87,
    "Amount_log": 4.61, "Amount_bin": 2,
    "Is_micro": 0, "Is_large": 0, "Is_night": 0, "Is_peak": 1,
    "V14_x_V12": 0.01, "V4_x_V11": 0.07,
    "V10_x_V16": 0.01, "V3_x_V9": 0.04,
    "V14_x_V17": -0.00, "Amount_x_V14": 13.82,
    "Amount_x_V4": 45.05, "Amount_x_V12": 7.98,
    "V2_x_Amount": 26.10,
    "Top_V_mean": 0.21, "Top_V_std": 0.18,
    "Top_V_max_abs": 0.45, "Top_V_l2_norm": 1.31,
    "Risk_score": 0.06,
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=5)
def fetch_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def score_single(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def score_batch(payloads: list[dict]) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/predict/batch",
            json={"transactions": payloads},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=30)
def fetch_mlflow_runs() -> pd.DataFrame | None:
    try:
        r = requests.get(
            f"{MLFLOW_BASE}/api/2.0/mlflow/experiments/search",
            params={"filter": "name='fraud-detection-ulb'"},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        exps = r.json().get("experiments", [])
        if not exps:
            return None
        exp_id = exps[0]["experiment_id"]

        r2 = requests.get(
            f"{MLFLOW_BASE}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": [exp_id], "max_results": 20},
            timeout=5,
        )
        if r2.status_code != 200:
            return None

        runs = r2.json().get("runs", [])
        rows = []
        for run in runs:
            info = run.get("info", {})
            metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
            rows.append({
                "run_name": info.get("run_name", ""),
                "status": info.get("status", ""),
                "pr_auc": metrics.get("pr_auc", None),
                "roc_auc": metrics.get("roc_auc", None),
                "f1": metrics.get("f1", None),
                "mcc": metrics.get("mcc", None),
                "brier_score": metrics.get("brier_score", None),
            })
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "score_history": [],   # list of {ts, prob, decision}
        "batch_results": None,
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Page layout helpers
# ---------------------------------------------------------------------------
def render_health_bar(health: dict | None) -> None:
    if health is None:
        st.error("⚠️ API unreachable — start `fraud_api_phase6.py` on port 8000")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("API Status", "🟢 Online")
    c2.metric("Model Loaded", "✅ Yes" if health["model_loaded"] else "❌ No")
    c3.metric("Total Requests", health["total_requests"])
    c4.metric("Fraud Flagged", health["total_fraud_flagged"])

    uptime = health["uptime_seconds"]
    h, m, s = int(uptime // 3600), int((uptime % 3600) // 60), int(uptime % 60)
    st.caption(f"Threshold: **{health['threshold']}** | Uptime: {h:02d}:{m:02d}:{s:02d}")


def decision_badge(decision: str) -> str:
    return {"BLOCK": "🔴 BLOCK", "REVIEW": "🟡 REVIEW", "PASS": "🟢 PASS"}.get(decision, decision)


def render_result(result: dict) -> None:
    if "error" in result:
        st.error(f"API error: {result['error']}")
        return

    prob = result["fraud_probability"]
    decision = result["decision"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{prob:.4f}")
    col2.metric("Decision", decision_badge(decision))
    col3.metric("Latency", f"{result['latency_ms']} ms")

    st.progress(min(prob, 1.0))
    st.caption(f"Threshold used: {result['threshold_used']}")

    # Persist to history
    st.session_state.score_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "probability": prob,
        "decision": decision,
    })


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_single_score() -> None:
    st.subheader("Single Transaction Scorer")

    col_preset, _ = st.columns([2, 3])
    preset = col_preset.selectbox(
        "Load example", ["Custom", "Known Fraud (V14=−6.5)", "Legitimate Transaction"]
    )

    if preset == "Known Fraud (V14=−6.5)":
        defaults = FRAUD_EXAMPLE
    elif preset == "Legitimate Transaction":
        defaults = LEGIT_EXAMPLE
    else:
        defaults = {f: 0.0 for f in FEATURE_COLS}
        defaults.update({"Amount_bin": 0, "Is_micro": 0, "Is_large": 0,
                         "Is_night": 0, "Is_peak": 0})

    with st.form("score_form"):
        st.markdown("**Key features** (edit as needed):")
        c1, c2, c3 = st.columns(3)
        v14  = c1.number_input("V14",         value=float(defaults["V14"]),         format="%.4f")
        v4   = c2.number_input("V4",          value=float(defaults["V4"]),          format="%.4f")
        v12  = c3.number_input("V12",         value=float(defaults["V12"]),         format="%.4f")
        rs   = c1.number_input("Risk_score",  value=float(defaults["Risk_score"]),  format="%.4f", min_value=0.0, max_value=1.0)
        norm = c2.number_input("Top_V_l2_norm", value=float(defaults["Top_V_l2_norm"]), format="%.4f", min_value=0.0)
        micro = c3.selectbox("Is_micro", [0, 1], index=int(defaults["Is_micro"]))

        st.markdown("**Full payload (JSON)** — overrides sliders above if edited:")
        payload_str = st.text_area(
            "JSON payload",
            value=json.dumps(defaults, indent=2),
            height=200,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Score Transaction", type="primary")

    if submitted:
        try:
            payload = json.loads(payload_str)
            # Override with slider values
            payload.update({"V14": v14, "V4": v4, "V12": v12,
                            "Risk_score": rs, "Top_V_l2_norm": norm, "Is_micro": micro})
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return

        with st.spinner("Scoring..."):
            result = score_single(payload)

        st.session_state.last_result = result
        render_result(result)

    # History chart
    if st.session_state.score_history:
        st.markdown("---")
        st.subheader("Session History")
        hist_df = pd.DataFrame(st.session_state.score_history)
        st.line_chart(hist_df.set_index("timestamp")["probability"])
        st.dataframe(hist_df, use_container_width=True)

        if st.button("Clear history"):
            st.session_state.score_history = []
            st.rerun()


def tab_batch_score() -> None:
    st.subheader("Batch Scorer")
    st.markdown(
        "Upload a CSV where each row is one transaction with the 51 model features. "
        "The `Class` column is optional (used for accuracy comparison if present)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)}** rows × **{df.shape[1]}** columns")
        st.dataframe(df.head(3), use_container_width=True)

        has_labels = "Class" in df.columns
        feature_df = df[FEATURE_COLS] if all(c in df.columns for c in FEATURE_COLS) else df

        if not all(c in df.columns for c in FEATURE_COLS):
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            st.error(f"Missing {len(missing)} features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            return

        if st.button("Run Batch Scoring", type="primary"):
            payloads = feature_df.to_dict(orient="records")
            with st.spinner(f"Scoring {len(payloads)} transactions..."):
                result = score_batch(payloads)

            if result and "error" not in result:
                st.session_state.batch_results = result
                preds_df = pd.DataFrame(result["predictions"])

                # Attach labels if available
                if has_labels:
                    preds_df["actual"] = df["Class"].values

                st.success(
                    f"✅ Done | Total: {result['total']} | "
                    f"Fraud flagged: {result['fraud_flagged']} | "
                    f"Review queue: {result['review_flagged']} | "
                    f"Latency: {result['latency_ms']} ms"
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Fraud Rate", f"{result['fraud_flagged']/result['total']*100:.2f}%")
                col2.metric("Review Rate", f"{result['review_flagged']/result['total']*100:.2f}%")
                col3.metric("Pass Rate",
                    f"{(result['total']-result['fraud_flagged'])/result['total']*100:.2f}%")

                st.dataframe(preds_df, use_container_width=True)

                # Download
                csv_out = preds_df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    data=csv_out,
                    file_name="batch_fraud_scores.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Batch error: {result}")

    # Demo with 10 random-ish transactions
    st.markdown("---")
    st.markdown("**No file? Run a quick demo batch (10 synthetic transactions):**")
    if st.button("Run Demo Batch"):
        rng = np.random.default_rng(42)
        demo_rows = []
        for i in range(10):
            row = {f: float(rng.normal()) for f in FEATURE_COLS if f.startswith("V")}
            row.update({
                "Amount_scaled": float(rng.normal(0, 0.5)),
                "Amount_log": float(rng.uniform(1, 7)),
                "Amount_bin": int(rng.integers(0, 5)),
                "Is_micro": int(rng.integers(0, 2)),
                "Is_large": int(rng.integers(0, 2)),
                "Hour_sin": float(np.sin(rng.uniform(0, 2 * np.pi))),
                "Hour_cos": float(np.cos(rng.uniform(0, 2 * np.pi))),
                "Is_night": int(rng.integers(0, 2)),
                "Is_peak": int(rng.integers(0, 2)),
                "V14_x_V12": row["V14"] * row["V12"],
                "V4_x_V11":  row["V4"] * row["V11"],
                "V10_x_V16": row["V10"] * row["V16"],
                "V3_x_V9":   row["V3"] * row["V9"],
                "V14_x_V17": row["V14"] * row["V17"],
                "Amount_x_V14": row.get("Amount_scaled", 0) * row["V14"],
                "Amount_x_V4":  row.get("Amount_scaled", 0) * row["V4"],
                "Amount_x_V12": row.get("Amount_scaled", 0) * row["V12"],
                "V2_x_Amount":  row["V2"] * row.get("Amount_scaled", 0),
                "Top_V_mean": float(np.mean([abs(row[f"V{i}"]) for i in [4, 10, 11, 12, 14]])),
                "Top_V_std":  float(np.std([abs(row[f"V{i}"]) for i in [4, 10, 11, 12, 14]])),
                "Top_V_max_abs": float(max(abs(row[f"V{i}"]) for i in [4, 10, 11, 12, 14])),
                "Top_V_l2_norm": float(np.sqrt(sum(row[f"V{i}"]**2 for i in [4, 10, 11, 12, 14]))),
                "Risk_score": float(np.clip(rng.beta(1, 5), 0, 1)),
            })
            demo_rows.append(row)

        with st.spinner("Scoring demo batch..."):
            result = score_batch(demo_rows)

        if result and "error" not in result:
            preds_df = pd.DataFrame(result["predictions"])
            st.dataframe(preds_df, use_container_width=True)
            st.info(
                f"Fraud flagged: {result['fraud_flagged']} / {result['total']} | "
                f"Latency: {result['latency_ms']} ms"
            )


def tab_mlflow() -> None:
    st.subheader("MLflow Run Comparison")
    st.caption(f"Pulling from {MLFLOW_BASE} — run `mlflow ui --port 5000` to enable")

    runs_df = fetch_mlflow_runs()
    if runs_df is None:
        st.warning(
            "MLflow server not reachable or no runs logged yet. "
            "Run `python fraud_mlflow_phase7a.py` then `mlflow ui --port 5000`."
        )
        # Show static Phase 4 metrics as fallback
        st.markdown("**Phase 4 results (static, from training logs):**")
        static = pd.DataFrame([
            {"Model": "XGBoost",             "ROC-AUC": 0.9747, "PR-AUC": 0.8599, "F1": 0.8701, "MCC": 0.8722},
            {"Model": "Random Forest",        "ROC-AUC": 0.9739, "PR-AUC": 0.8412, "F1": 0.8462, "MCC": 0.8467},
            {"Model": "Logistic Regression",  "ROC-AUC": 0.9781, "PR-AUC": 0.7111, "F1": 0.8042, "MCC": 0.8039},
        ])
        st.dataframe(static, use_container_width=True)
        st.bar_chart(static.set_index("Model")[["PR-AUC", "F1", "MCC"]])
        return

    st.dataframe(runs_df, use_container_width=True)
    numeric_cols = [c for c in ["pr_auc", "roc_auc", "f1", "mcc"] if c in runs_df.columns]
    if numeric_cols and not runs_df[numeric_cols].empty:
        st.bar_chart(runs_df.set_index("run_name")[numeric_cols])

    st.markdown(f"[Open MLflow UI →]({MLFLOW_BASE})")


def tab_model_info() -> None:
    st.subheader("Model & Deployment Info")

    st.markdown("### Production Model")
    info = {
        "Algorithm": "XGBoost",
        "Training set": "224,738 rows (80% stratified)",
        "Test set": "56,184 rows (20% stratified)",
        "Features": "51 (V1–V28 + 23 engineered)",
        "ROC-AUC": "0.9747",
        "PR-AUC": "0.8599",
        "F1": "0.8701",
        "MCC": "0.8722",
        "Brier Score": "0.000428 (near-perfect calibration)",
    }
    for k, v in info.items():
        st.markdown(f"**{k}:** {v}")

    st.markdown("### Threshold Tuning (Phase 5)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Production threshold = 0.192** (cost-optimal, FN cost = 10× FP)")
        st.markdown("- FP: 13 | FN: 14")
        st.markdown("- Precision: 0.8617 | Recall: 0.8526")
    with col2:
        st.markdown("**F1-optimal threshold = 0.847** (too conservative for production)")
        st.markdown("- FP: 5 | FN: 18 → too many missed frauds")

    st.markdown("### Decision Logic")
    st.markdown("""
| Score | Decision | Meaning |
|-------|----------|---------|
| < 0.192 | 🟢 **PASS** | Transaction approved |
| 0.192 – 0.90 | 🔴 **BLOCK** | Auto-declined |
| ≥ 0.90 | 🟡 **REVIEW** | Human review queue |
    """)

    st.markdown("### Top SHAP Features")
    shap_data = pd.DataFrame([
        {"Feature": "Risk_score",     "Mean |SHAP|": 1.60, "Note": "Dominant — Phase 3 composite"},
        {"Feature": "V14",            "Mean |SHAP|": 0.91, "Note": "Decision boundary at −5"},
        {"Feature": "V4_x_V11",       "Mean |SHAP|": 0.67, "Note": "Interaction — keep"},
        {"Feature": "V12",            "Mean |SHAP|": 0.54, "Note": ""},
        {"Feature": "Top_V_l2_norm",  "Mean |SHAP|": 0.48, "Note": "FP trigger at 34"},
        {"Feature": "V10_x_V16",      "Mean |SHAP|": 0.03, "Note": "Near-zero — drop next iter"},
        {"Feature": "V3_x_V9",        "Mean |SHAP|": 0.02, "Note": "Near-zero — drop next iter"},
    ])
    st.dataframe(shap_data, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        page_icon="🔍",
        layout="wide",
    )
    init_state()

    st.title("🔍 Fraud Detection Dashboard")
    st.caption("ULB Credit Card Dataset | XGBoost | Phase 6–7")

    # Health bar (always visible)
    health = fetch_health()
    render_health_bar(health)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Score",
        "📦 Batch Score",
        "📊 MLflow Runs",
        "ℹ️ Model Info",
    ])

    with tab1:
        tab_single_score()
    with tab2:
        tab_batch_score()
    with tab3:
        tab_mlflow()
    with tab4:
        tab_model_info()


if __name__ == "__main__":
    main()
