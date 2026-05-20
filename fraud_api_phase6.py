"""
Phase 6 — FastAPI Scoring Endpoint
Banking Fraud Detection | Production XGBoost @ threshold=0.192
"""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/xgboost.joblib")
THRESHOLD = 0.192          # cost-optimal: FN cost = 10× FP
FP_REVIEW_PROB = 0.90      # score ≥ this triggers human-review flag
LOG_LEVEL = logging.INFO

# Feature order derived from model.get_booster().feature_names — 49 features
# Amount, Time excluded (replaced by derivatives); V10_x_V16, V3_x_V9 dropped (near-zero SHAP)
FEATURE_ORDER: list[str] = [
    # Original PCA components
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    # Engineered — Phase 3 (order matches training pipeline output)
    "Amount_scaled","Hour_sin","Hour_cos","Amount_log","Amount_bin",
    "Is_micro","Is_large","Is_night","Is_peak",
    "V14_x_V12","V4_x_V11",
    "V14_x_V17","Amount_x_V14","Amount_x_V4","Amount_x_V12","V2_x_Amount",
    "Top_V_mean","Top_V_std","Top_V_max_abs","Top_V_l2_norm","Risk_score",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=LOG_LEVEL,
)
logger = logging.getLogger("fraud_api")

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
class AppState:
    model: Any = None
    request_count: int = 0
    fraud_count: int = 0
    start_time: float = 0.0

app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading model from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    app_state.model = joblib.load(MODEL_PATH)
    app_state.start_time = time.time()
    logger.info("Model loaded. Threshold=%.3f", THRESHOLD)
    yield
    # Shutdown
    logger.info(
        "Shutdown | requests=%d fraud_flagged=%d",
        app_state.request_count,
        app_state.fraud_count,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "XGBoost scoring endpoint — ULB Credit Card dataset. "
        "Production threshold=0.192 (cost-optimal, FN cost=10×FP)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """Single transaction. 51 features — Amount and Time excluded (replaced by derivatives)."""
    # Original PCA components
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    # Engineered features (order matches model training pipeline)
    Amount_scaled: float
    Hour_sin: float = Field(..., ge=-1, le=1)
    Hour_cos: float = Field(..., ge=-1, le=1)
    Amount_log: float
    Amount_bin: int = Field(..., ge=0, le=4)
    Is_micro: int = Field(..., ge=0, le=1)
    Is_large: int = Field(..., ge=0, le=1)
    Is_night: int = Field(..., ge=0, le=1)
    Is_peak: int = Field(..., ge=0, le=1)
    V14_x_V12: float
    V4_x_V11: float
    V10_x_V16: float
    V3_x_V9: float
    V14_x_V17: float
    Amount_x_V14: float
    Amount_x_V4: float
    Amount_x_V12: float
    V2_x_Amount: float
    Top_V_mean: float
    Top_V_std: float
    Top_V_max_abs: float
    Top_V_l2_norm: float = Field(..., ge=0)
    Risk_score: float = Field(..., ge=0, le=1)

    @field_validator("Amount_bin")
    @classmethod
    def check_amount_bin(cls, v: int) -> int:
        if v not in range(5):
            raise ValueError("Amount_bin must be 0-4")
        return v

    model_config = {"json_schema_extra": {"example": {
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
    }}}


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    decision: str          # "BLOCK" | "REVIEW" | "PASS"
    threshold_used: float
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    fraud_flagged: int
    review_flagged: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
    uptime_seconds: float
    total_requests: int
    total_fraud_flagged: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_frame(tx: TransactionRequest) -> pd.DataFrame:
    row = {f: getattr(tx, f) for f in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def _build_batch_frame(txs: list[TransactionRequest]) -> pd.DataFrame:
    rows = [{f: getattr(tx, f) for f in FEATURE_ORDER} for tx in txs]
    return pd.DataFrame(rows, columns=FEATURE_ORDER)


def _decision(prob: float) -> str:
    if prob >= THRESHOLD:
        return "REVIEW" if prob >= FP_REVIEW_PROB else "BLOCK"
    return "PASS"


def _score_frame(df: pd.DataFrame) -> np.ndarray:
    return app_state.model.predict_proba(df)[:, 1]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=app_state.model is not None,
        threshold=THRESHOLD,
        uptime_seconds=round(time.time() - app_state.start_time, 1),
        total_requests=app_state.request_count,
        total_fraud_flagged=app_state.fraud_count,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
)
def predict(tx: TransactionRequest, request: Request):
    t0 = time.perf_counter()
    try:
        df = _build_frame(tx)
        prob = float(_score_frame(df)[0])
    except Exception as exc:
        logger.exception("Scoring error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    is_fraud = prob >= THRESHOLD
    app_state.request_count += 1
    if is_fraud:
        app_state.fraud_count += 1

    latency = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "predict | prob=%.4f is_fraud=%s latency=%.1fms client=%s",
        prob, is_fraud, latency, request.client.host if request.client else "unknown",
    )
    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        decision=_decision(prob),
        threshold_used=THRESHOLD,
        latency_ms=latency,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
)
def predict_batch(batch: BatchRequest, request: Request):
    t0 = time.perf_counter()
    try:
        df = _build_batch_frame(batch.transactions)
        probs = _score_frame(df)
    except Exception as exc:
        logger.exception("Batch scoring error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    predictions = []
    fraud_count = 0
    review_count = 0
    for prob in probs:
        prob = float(prob)
        is_fraud = prob >= THRESHOLD
        dec = _decision(prob)
        if is_fraud:
            fraud_count += 1
        if dec == "REVIEW":
            review_count += 1
        predictions.append(PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=is_fraud,
            decision=dec,
            threshold_used=THRESHOLD,
            latency_ms=0.0,  # per-item not meaningful in batch
        ))

    app_state.request_count += len(batch.transactions)
    app_state.fraud_count += fraud_count

    latency = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "batch | n=%d fraud=%d review=%d latency=%.1fms",
        len(batch.transactions), fraud_count, review_count, latency,
    )
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(batch.transactions),
        fraud_flagged=fraud_count,
        review_flagged=review_count,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "fraud_api_phase6:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
