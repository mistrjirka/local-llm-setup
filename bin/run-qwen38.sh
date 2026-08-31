#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <frontend-port>" >&2; exit 2; }
PORT=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
# shellcheck source=/dev/null
source "$ROOT/config/config.env"

SERVER="$ROOT/llama.cpp/build-qwen/bin/llama-server"
WRAPPER="$ROOT/bin/llama_cache_proxy.py"
BACKEND_PORT=${QWEN38_BACKEND_PORT:-$((PORT + 10000))}
SNAPSHOT_DIR=${LLAMA_CACHE_ROOT}/qwen38
mkdir -p "$SNAPSHOT_DIR"

exec python3 "$WRAPPER" \
  --listen-port "$PORT" \
  --backend-port "$BACKEND_PORT" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --slot-count 1 \
  --parallel-tool-calls-default \
  -- "$SERVER" \
  --model "$QWEN38_MODEL" \
  --alias qwen3.8-27b \
  --ctx-size 262144 \
  --parallel 1 \
  --split-mode layer \
  --fit off \
  --gpu-layers all \
  --tensor-split 64,2 \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size 4096 \
  --prefill-reuse 1024 \
  --pipeline-copies 2 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-draft f16 \
  --cache-type-v-draft f16 \
  --cache-ram "${QWEN38_CACHE_RAM_MIB:-65536}" \
  --cache-idle-slots \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 8192 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-ubatch 1024 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --slots \
  --perf \
  --no-warmup
