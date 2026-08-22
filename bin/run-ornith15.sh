#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <frontend-port>" >&2; exit 2; }
PORT=$1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
# shellcheck source=/dev/null
source "$ROOT/config/config.env"

SERVER="$ROOT/llama.cpp/build-ornith-mmq/bin/llama-server"
WRAPPER="$ROOT/bin/llama_cache_proxy.py"
BACKEND_PORT=${ORNITH15_BACKEND_PORT:-$((PORT + 10000))}
SNAPSHOT_DIR=${LLAMA_CACHE_ROOT}/ornith15
SLOTS=${ORNITH15_PARALLEL:-4}
CTX_PER_SLOT=${ORNITH15_CTX_PER_SLOT:-250112}
CTX_TOTAL=$((SLOTS * CTX_PER_SLOT))
REASONING_MAP=${ORNITH15_REASONING_MAP:-'{"none":0,"low":2048,"medium":8192,"high":32768,"xhigh":-1}'}
mkdir -p "$SNAPSHOT_DIR"

# This placement is tuned for V100 32 GB (CUDA0 in llama.cpp) + 3060 Ti 8 GB
# (CUDA1), AD-Q5_K-Q4_K target, four 250112-token Q8/Q8 slots and the
# Shisa MTP draft on CUDA1. For a different GPU topology, override
# ORNITH15_EXTRA_ARGS in config.env.
DEFAULT_PLACEMENT=(
  --gpu-layers 41
  --tensor-split 40,1
  --override-tensor 'blk\.36\.ffn_(gate|up|gate_up|down).*=CUDA1,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU'
  --fit off
)
if [[ -n ${ORNITH15_EXTRA_ARGS:-} ]]; then
  # Intentional word splitting: this is an advanced local override.
  # shellcheck disable=SC2206
  DEFAULT_PLACEMENT=( ${ORNITH15_EXTRA_ARGS} )
fi

exec python3 "$WRAPPER" \
  --listen-port "$PORT" \
  --backend-port "$BACKEND_PORT" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --slot-count "$SLOTS" \
  --reasoning-budget-map "$REASONING_MAP" \
  -- "$SERVER" \
  --model "$ORNITH15_MODEL" \
  --alias ornith-1.5-35b-a3b \
  --ctx-size "$CTX_TOTAL" \
  --parallel "$SLOTS" \
  --no-kv-unified \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram "${ORNITH15_CACHE_RAM_MIB:-32768}" \
  --cache-idle-slots \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 512 \
  "${DEFAULT_PLACEMENT[@]}" \
  --spec-type draft-mtp \
  --spec-draft-model "$ORNITH15_MTP_MODEL" \
  --spec-draft-device CUDA1 \
  --spec-draft-ngl all \
  --spec-draft-type-k q8_0 \
  --spec-draft-type-v q8_0 \
  --spec-draft-ubatch 512 \
  --spec-draft-n-max 3 \
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
