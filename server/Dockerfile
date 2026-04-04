ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/env
WORKDIR /app/env

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-editable

# Install uvicorn into uv venv
RUN uv pip install uvicorn

# -------- FINAL IMAGE --------
FROM ${BASE_IMAGE}

WORKDIR /app/env

# Copy app + venv
COPY --from=builder /app/env /app/env

# Env variables
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]