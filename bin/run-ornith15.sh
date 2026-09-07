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
BACKEND_PORT=${ORNITH15_BACKEND_PORT:-$((PORT + 10000))}
SLOTS=${ORNITH15_PARALLEL:-4}
NATIVE_CTX=${ORNITH15_NATIVE_CTX:-262144}
SHARED_400K=${ORNITH15_SHARED_400K:-1}
if [[ $SHARED_400K == 1 ]]; then
  CTX_PER_SLOT=${ORNITH15_SHARED_CTX_PER_SLOT:-400000}
  CTX_TOTAL=${ORNITH15_SHARED_KV_POOL:-1400000}
  YARN_SCALE_OVERRIDE=${ORNITH15_SHARED_YARN_SCALE:-}
  [[ $SLOTS -eq 4 ]] || { echo "ORNITH15_SHARED_400K is validated only with ORNITH15_PARALLEL=4" >&2; exit 2; }
  (( CTX_TOTAL >= CTX_PER_SLOT && CTX_TOTAL <= SLOTS * CTX_PER_SLOT )) || {
    echo "invalid shared KV pool: need CTX_PER_SLOT <= pool <= SLOTS*CTX_PER_SLOT" >&2
    exit 2
  }
  REQUIRED_SHARED_PREFIX=$(( (SLOTS * CTX_PER_SLOT - CTX_TOTAL + SLOTS - 2) / (SLOTS - 1) ))
  KV_MODE_TAG="kvu-pool${CTX_TOTAL}"
  KV_ARGS=(
    --kv-unified
    --kv-unified-per-slot "$CTX_PER_SLOT"
    --slot-fork-prefix
  )
  CACHE_ARGS=(--cache-ram 0 --no-cache-idle-slots)
else
  CTX_PER_SLOT=${ORNITH15_CTX_PER_SLOT:-350000}
  CTX_TOTAL=$((SLOTS * CTX_PER_SLOT))
  YARN_SCALE_OVERRIDE=${ORNITH15_YARN_SCALE:-}
  KV_MODE_TAG="fixed"
  KV_ARGS=(--no-kv-unified)
  CACHE_ARGS=(--cache-ram "${ORNITH15_CACHE_RAM_MIB:-32768}" --cache-idle-slots)
fi
DEVICE_ORDER=${ORNITH15_DEVICE_ORDER:-CUDA1,CUDA0}
TENSOR_SPLIT=${ORNITH15_TENSOR_SPLIT:-14,35}
CACHE_TYPE_K=${ORNITH15_CACHE_TYPE_K:-q8_0}
CACHE_TYPE_V=${ORNITH15_CACHE_TYPE_V:-q8_0}
DRAFT_CACHE_TYPE_K=${ORNITH15_DRAFT_CACHE_TYPE_K:-q8_0}
DRAFT_CACHE_TYPE_V=${ORNITH15_DRAFT_CACHE_TYPE_V:-q8_0}
UBATCH_SIZE=${ORNITH15_UBATCH_SIZE:-256}
DRAFT_UBATCH_SIZE=${ORNITH15_DRAFT_UBATCH_SIZE:-128}
MTP_N_MAX=${ORNITH15_MTP_N_MAX:-3}
MTP_DEVICE=${ORNITH15_MTP_DEVICE:-CUDA1}
REASONING_MAP=${ORNITH15_REASONING_MAP:-'{"none":0,"low":2048,"medium":8192,"high":32768,"xhigh":-1}'}

# Hybrid/recurrent slot states depend on the model/context/parallel geometry.
# Keep incompatible snapshots apart so a profile change cannot poison startup.
MODEL_TAG=$(basename -- "${ORNITH15_MODEL%.gguf}")
MTP_MODEL_TAG=$(basename -- "${ORNITH15_MTP_MODEL%.gguf}")
if [[ $SHARED_400K == 1 ]]; then
  # Shared-layout snapshots include draft identity because .draft/.spec companions
  # are only valid for the exact MTP configuration that produced them.
  SNAPSHOT_DIR=${LLAMA_CACHE_ROOT}/ornith15/${MODEL_TAG}/ctx${CTX_PER_SLOT}-p${SLOTS}-${CACHE_TYPE_K}-${CACHE_TYPE_V}-${KV_MODE_TAG}-d${DRAFT_CACHE_TYPE_K}-${DRAFT_CACHE_TYPE_V}-mtp${MTP_N_MAX}-${MTP_MODEL_TAG}
  echo "Ornith shared-400k: 4x${CTX_PER_SLOT} logical over ${CTX_TOTAL} physical KV; full-capacity use needs >=${REQUIRED_SHARED_PREFIX} shared-prefix tokens" >&2
else
  # Preserve the historical fixed-slot path so enabling the new optional profile
  # does not invalidate existing 350k snapshots.
  SNAPSHOT_DIR=${LLAMA_CACHE_ROOT}/ornith15/${MODEL_TAG}/ctx${CTX_PER_SLOT}-p${SLOTS}-${CACHE_TYPE_K}-${CACHE_TYPE_V}
fi
mkdir -p "$SNAPSHOT_DIR"

YARN_ARGS=()
if (( CTX_PER_SLOT > NATIVE_CTX )); then
  YARN_SCALE=${YARN_SCALE_OVERRIDE:-$(awk -v n="$CTX_PER_SLOT" -v d="$NATIVE_CTX" 'BEGIN { printf "%.10g", n/d }')}
  YARN_ARGS=(
    --override-kv "qwen35moe.context_length=int:${CTX_PER_SLOT}"
    --rope-scaling yarn
    --rope-scale "$YARN_SCALE"
    --yarn-orig-ctx "$NATIVE_CTX"
  )
fi

MMPROJ=${ORNITH15_MMPROJ:-"$MODEL_ROOT/ornith15/mmproj-Ornith-1.5-35B-BF16.gguf"}
[[ -s $MMPROJ ]] || { echo "missing Ornith vision projector: $MMPROJ" >&2; exit 1; }
[[ -s $ORNITH15_MTP_MODEL ]] || { echo "missing Ornith MTP draft: $ORNITH15_MTP_MODEL" >&2; exit 1; }
MMPROJ_ARGS=(--mmproj "$MMPROJ")
if [[ ${ORNITH15_MMPROJ_OFFLOAD:-0} == 1 ]]; then
  MMPROJ_ARGS+=(--mmproj-offload)
  [[ -z ${ORNITH15_MMPROJ_DEVICE:-} ]] || MMPROJ_ARGS+=(--mmproj-device "$ORNITH15_MMPROJ_DEVICE")
else
  MMPROJ_ARGS+=(--no-mmproj-offload)
fi

# Normal build + runtime selective Volta MoE MMQ keeps dense/attention work off
# the globally-forced MMQ path. V100=CUDA0, RTX 2080 Ti=CUDA1 on the tuned host.
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-internal}
export GGML_CUDA_AR_COPY_THRESHOLD=${GGML_CUDA_AR_COPY_THRESHOLD:-131072}
export GGML_CUDA_TURING_CUBLAS_MIN_BATCH=${GGML_CUDA_TURING_CUBLAS_MIN_BATCH:-256}
export GGML_CUDA_VOLTA_Q8_FATTN_TC=${GGML_CUDA_VOLTA_Q8_FATTN_TC:-1}
export GGML_CUDA_VOLTA_Q5_X4=${GGML_CUDA_VOLTA_Q5_X4:-1}
export GGML_CUDA_VOLTA_Q6_W4R4=${GGML_CUDA_VOLTA_Q6_W4R4:-1}
export GGML_CUDA_VOLTA_FORCE_MMQ=${GGML_CUDA_VOLTA_FORCE_MMQ:-moe}
export GGML_CUDA_VOLTA_GQA8_NCOLS2=${GGML_CUDA_VOLTA_GQA8_NCOLS2:-2}

PLACEMENT=(
  --split-mode layer
  --fit off
  --gpu-layers all
  --device "$DEVICE_ORDER"
  --tensor-split "$TENSOR_SPLIT"
)
if [[ -n ${ORNITH15_EXTRA_ARGS:-} ]]; then
  # shellcheck disable=SC2206
  PLACEMENT=( ${ORNITH15_EXTRA_ARGS} )
fi

exec python3 "$WRAPPER" \
  --listen-port "$PORT" \
  --backend-port "$BACKEND_PORT" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --slot-count "$SLOTS" \
  --parallel-tool-calls-default \
  --reasoning-budget-map "$REASONING_MAP" \
  -- "$SERVER" \
  --model "$ORNITH15_MODEL" \
  --alias ornith-1.5-35b-a3b \
  "${MMPROJ_ARGS[@]}" \
  --ctx-size "$CTX_TOTAL" \
  --parallel "$SLOTS" \
  "${KV_ARGS[@]}" \
  "${YARN_ARGS[@]}" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  "${CACHE_ARGS[@]}" \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size "$UBATCH_SIZE" \
  --pipeline-copies 1 \
  "${PLACEMENT[@]}" \
  --spec-type draft-mtp \
  --spec-draft-model "$ORNITH15_MTP_MODEL" \
  --spec-draft-device "$MTP_DEVICE" \
  --spec-draft-ngl all \
  --spec-draft-type-k "$DRAFT_CACHE_TYPE_K" \
  --spec-draft-type-v "$DRAFT_CACHE_TYPE_V" \
  --spec-draft-ubatch "$DRAFT_UBATCH_SIZE" \
  --spec-draft-n-max "$MTP_N_MAX" \
  --spec-mtp-defer-prompt \
  --temp 0.6 \
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
