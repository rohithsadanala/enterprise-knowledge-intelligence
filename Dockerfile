# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# System deps: FAISS needs libgomp; curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY frontend/ ./frontend/
COPY docs/ ./docs/
COPY eval_data/ ./eval_data/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app/src
ENV EKI_DATA_DIR=/app/data
ENV EKI_API_HOST=0.0.0.0
ENV EKI_API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "eki.api:app", "--host", "0.0.0.0", "--port", "8000"]
