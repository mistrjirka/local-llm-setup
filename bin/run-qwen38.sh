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
CTX_SIZE=${QWEN38_CTX_SIZE:-409600}
NATIVE_CTX=${QWEN38_NATIVE_CTX:-262144}
DEVICE_ORDER=${QWEN38_DEVICE_ORDER:-CUDA1,CUDA0}
TENSOR_SPLIT=${QWEN38_TENSOR_SPLIT:-4,5}
CACHE_TYPE_K=${QWEN38_CACHE_TYPE_K:-q8_0}
CACHE_TYPE_V=${QWEN38_CACHE_TYPE_V:-q8_0}
DRAFT_CACHE_TYPE_K=${QWEN38_DRAFT_CACHE_TYPE_K:-f16}
DRAFT_CACHE_TYPE_V=${QWEN38_DRAFT_CACHE_TYPE_V:-f16}
UBATCH_SIZE=${QWEN38_UBATCH_SIZE:-2048}
DRAFT_UBATCH_SIZE=${QWEN38_DRAFT_UBATCH_SIZE:-512}
SHORTLIST=${GGML_CUDA_QWEN35_MTP_SHORTLIST:-"$ROOT/llama.cpp/data/mtp-shortlists/qwen38-27b-exact-131072.i32"}
[[ -s $SHORTLIST ]] || { echo "missing Qwen3.8 MTP shortlist: $SHORTLIST" >&2; exit 1; }

# Qwen3.8 is native at 262144. Above that, use static YaRN with a factor
# matching the requested window. Qwen recommends target/native rather than
# always forcing the full 4x 1M profile. The metadata override is required by
# llama-server so a slot may actually exceed the GGUF's native context field.
YARN_ARGS=()
if (( CTX_SIZE > NATIVE_CTX )); then
  YARN_SCALE=${QWEN38_YARN_SCALE:-$(awk -v n="$CTX_SIZE" -v d="$NATIVE_CTX" 'BEGIN { printf "%.8g", n/d }')}
  YARN_ARGS=(
    --override-kv "qwen35.context_length=int:${CTX_SIZE}"
    --rope-scaling yarn
    --rope-scale "$YARN_SCALE"
    --yarn-orig-ctx "$NATIVE_CTX"
  )
  # Slot-state files are context/cache-layout specific. Keep the extended
  # profile away from legacy 262k/F16 snapshots.
  SNAPSHOT_DIR=${LLAMA_CACHE_ROOT}/qwen38/ctx${CTX_SIZE}-${CACHE_TYPE_K}-${CACHE_TYPE_V}
fi
mkdir -p "$SNAPSHOT_DIR"

# Validated V100 32 GB + RTX 2080 Ti 22 GB tensor-parallel profile.
# llama.cpp enumerated V100=CUDA0 and RTX 2080 Ti=CUDA1 during tuning, hence
# CUDA1 first here and the 4:5 split is RTX:V100. Keep these configurable for
# hosts whose CUDA enumeration differs.
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-internal}
export GGML_CUDA_AR_COPY_THRESHOLD=${GGML_CUDA_AR_COPY_THRESHOLD:-131072}
export GGML_CUDA_TURING_CUBLAS_MIN_BATCH=${GGML_CUDA_TURING_CUBLAS_MIN_BATCH:-256}
export GGML_CUDA_VOLTA_Q8_FATTN_TC=${GGML_CUDA_VOLTA_Q8_FATTN_TC:-1}
export GGML_CUDA_VOLTA_Q5_X4=${GGML_CUDA_VOLTA_Q5_X4:-1}
export GGML_CUDA_VOLTA_Q6_W4R4=${GGML_CUDA_VOLTA_Q6_W4R4:-1}
export GGML_CUDA_QWEN35_MTP_SHORTLIST=$SHORTLIST

exec python3 "$WRAPPER" \
  --listen-port "$PORT" \
  --backend-port "$BACKEND_PORT" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --slot-count 1 \
  --parallel-tool-calls-default \
  -- "$SERVER" \
  --model "$QWEN38_MODEL" \
  --alias qwen3.8-27b \
  --ctx-size "$CTX_SIZE" \
  "${YARN_ARGS[@]}" \
  --parallel 1 \
  --split-mode tensor \
  --fit off \
  --gpu-layers all \
  --device "$DEVICE_ORDER" \
  --tensor-split "$TENSOR_SPLIT" \
  --flash-attn on \
  --batch-size 4096 \
  --ubatch-size "$UBATCH_SIZE" \
  --prefill-reuse 1024 \
  --pipeline-copies 1 \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --cache-type-k-draft "$DRAFT_CACHE_TYPE_K" \
  --cache-type-v-draft "$DRAFT_CACHE_TYPE_V" \
  --cache-ram "${QWEN38_CACHE_RAM_MIB:-65536}" \
  --cache-idle-slots \
  --ctx-checkpoints 32 \
  --checkpoint-min-step 8192 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-ubatch "$DRAFT_UBATCH_SIZE" \
  --spec-mtp-defer-prompt \
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
