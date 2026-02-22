#!/usr/bin/env bash
# =============================================================================
# run.sh — one-click setup and launch for Qazcode Clinical Diagnosis Assistant
#
# What it does:
#   1. Reads API credentials from .env (or env vars)
#   2. Downloads the embedding model if not already present (~610 MB, one-time)
#   3. Pulls the latest pre-built image from GHCR
#   4. Stops any running container with the same name
#   5. Starts the server on port 8080
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
# Override credentials inline (skips .env):
#   GPT_OSS_API_KEY=sk-xxx ./run.sh
# =============================================================================

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────

IMAGE="ghcr.io/askhatsbk/qazcode-nu:latest"
CONTAINER="qazcode"
PORT=8080

MODEL_DIR="$(pwd)/data/models"
MODEL_FILE="Qwen3-Embedding-0.6B-Q8_0.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/${MODEL_FILE}"

# ── Load .env if present ────────────────────────────────────────────────────

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source <(grep -v '^\s*#' .env | grep '=')
    set +a
fi

# ── Validate required credentials ───────────────────────────────────────────

GPT_OSS_BASE_URL="${GPT_OSS_BASE_URL:-}"
GPT_OSS_API_KEY="${GPT_OSS_API_KEY:-}"

if [[ -z "$GPT_OSS_BASE_URL" || -z "$GPT_OSS_API_KEY" ]]; then
    echo ""
    echo "ERROR: GPT_OSS_BASE_URL and GPT_OSS_API_KEY must be set."
    echo "  Copy .env.example → .env and fill in your credentials, then re-run."
    echo ""
    exit 1
fi

# ── Step 1: Download embedding model ────────────────────────────────────────

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_DIR/$MODEL_FILE" ]]; then
    echo "[1/4] Embedding model already present — skipping download."
else
    echo "[1/4] Downloading embedding model (~610 MB, one-time) ..."
    echo "      Source: $MODEL_URL"
    echo "      Dest:   $MODEL_DIR/$MODEL_FILE"
    echo ""
    curl -L --progress-bar \
        --retry 3 --retry-delay 5 \
        -o "$MODEL_DIR/$MODEL_FILE" \
        "$MODEL_URL"
    echo ""
    echo "      Done."
fi

# ── Step 2: Pull latest image ────────────────────────────────────────────────

echo "[2/4] Pulling image: $IMAGE"
docker pull "$IMAGE"

# ── Step 3: Remove existing container ───────────────────────────────────────

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[3/4] Stopping and removing existing container '${CONTAINER}' ..."
    docker rm -f "$CONTAINER"
else
    echo "[3/4] No existing container to remove."
fi

# ── Step 4: Run ─────────────────────────────────────────────────────────────

echo "[4/4] Starting container '${CONTAINER}' on port ${PORT} ..."
docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    -p "${PORT}:8080" \
    -e GPT_OSS_BASE_URL="$GPT_OSS_BASE_URL" \
    -e GPT_OSS_API_KEY="$GPT_OSS_API_KEY" \
    -v "$MODEL_DIR:/app/data/models" \
    "$IMAGE"

# ── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "Server is starting up. Waiting for health check..."
sleep 5

for i in {1..12}; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "  Server is ready!"
        echo "  Web UI:  http://localhost:${PORT}/"
        echo "  API:     http://localhost:${PORT}/diagnose"
        echo "  Logs:    docker logs -f ${CONTAINER}"
        echo ""
        exit 0
    fi
    echo "  Waiting... (${i}/12)"
    sleep 5
done

echo ""
echo "  Server did not respond within 60 s. Check logs:"
echo "  docker logs ${CONTAINER}"
echo ""
exit 1
