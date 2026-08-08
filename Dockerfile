# ──────────────────────────────────────────────────────────────
# Stage 1: Build environment & Pre-download AI/NLP Models
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Sync project dependencies into /app/.venv
RUN uv sync --frozen

# Place virtualenv binaries on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download Aajil-Labs ONNX PII model during build for self-contained image
RUN python -c "import apii; apii.default_pipeline()"

# ──────────────────────────────────────────────────────────────
# Stage 2: Production Runtime (Clean, Minimal, Secure)
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy virtualenv and cached HuggingFace models from builder stage
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.cache /root/.cache

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy application source code and configuration
COPY config/ config/
COPY data/ data/
COPY src/ src/
COPY README.md .env.example ./

EXPOSE 8000

# Direct uvicorn execution as PID 1 for optimal container OS signal handling
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
