#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/.." && pwd)
# shellcheck source=/dev/null
source "$ROOT/config/config.env"
exec "$ROOT/bin/llama-swap" --config "$ROOT/config/llama-swap.yaml" --listen "${LLAMA_SWAP_LISTEN:-127.0.0.1:8080}"
