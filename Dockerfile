FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . .

# Install all dependencies
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "openai>=1.35.0" \
    "requests>=2.32.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.7.0"

# /app is the root — server/ lives inside it so "server.app" resolves correctly
ENV PYTHONPATH="/app"
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
