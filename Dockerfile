# ── Fraud Detection API ──────────────────────────────────────────────────────
# Multi-stage build: keeps final image lean by separating dependency install
# from the runtime layer.
# ---------------------------------------------------------------------------

# Stage 1 — dependency builder
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (lightgbm, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2 — runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and artifacts
COPY fraud_api_phase6.py .
COPY models/ models/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Uvicorn in production mode: single worker (scale via replicas in compose)
CMD ["uvicorn", "fraud_api_phase6:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
