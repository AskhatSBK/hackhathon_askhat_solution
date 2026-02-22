# ── CPU build (default) ──────────────────────────────────────────────────────
# Build:  docker build -t qazcode .
# Run:    docker run -e GPT_OSS_BASE_URL=... -e GPT_OSS_API_KEY=... -p 8080:8080 qazcode
#
# ── NVIDIA GPU build ─────────────────────────────────────────────────────────
# Use a CUDA-enabled base image and install faiss-gpu instead of faiss-cpu:
#   FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
#   RUN apt-get update && apt-get install -y python3.12 python3.12-dev ...
#   RUN pip install faiss-gpu torch --extra-index-url https://download.pytorch.org/whl/cu124
#   ENV DEVICE=cuda
# Run:  docker run --gpus all -e DEVICE=cuda ...
#
# ── AMD ROCm build ───────────────────────────────────────────────────────────
# Use a ROCm-enabled PyTorch base image (FAISS stays on CPU, only embeddings use GPU):
#   FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
#   RUN pip install faiss-cpu sentence-transformers ...
#   ENV DEVICE=cuda   # ROCm exposes itself as 'cuda' in PyTorch
# Run:  docker run --device /dev/kfd --device /dev/dri -e DEVICE=cuda ...
#
# ── Force CPU regardless of GPU availability ─────────────────────────────────
#   ENV DEVICE=cpu
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (needed by pymystem3 and build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies first (layer cache)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# Copy application source
COPY src/ src/
COPY scripts/ scripts/
COPY data/corpus/ data/corpus/

# Copy pre-built indexes (baked into image — not tracked in git due to size)
COPY data/indexes/ data/indexes/
COPY data/trees/ data/trees/

ENV PYTHONUNBUFFERED=1
# All secrets and config come from .env (via docker compose env_file) or -e flags.
# See .env.example for available variables.
ENV GPT_OSS_BASE_URL=""
ENV GPT_OSS_API_KEY=""
ENV GOOGLE_AI_API_KEY=""
ENV LLM_PROVIDER="oss"
ENV LLM_MODEL=""
# DEVICE auto-detected at runtime (cuda/mps/cpu). Override with -e DEVICE=cpu


EXPOSE 8080

CMD ["uv", "run", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
