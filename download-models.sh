#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -f $HERE/../config/config.env ]]; then
  ROOT=$(cd -- "$HERE/.." && pwd)
else
  ROOT=${PREFIX:-$HOME/.local/share/local-llm-setup}
fi
# shellcheck source=/dev/null
source "$ROOT/config/config.env"
SRC=$ROOT/llama.cpp
ORNITH15_MMPROJ=${ORNITH15_MMPROJ:-"$MODEL_ROOT/ornith15/mmproj-Ornith-1.5-35B-BF16.gguf"}
mkdir -p "$MODEL_ROOT/qwen38" "$MODEL_ROOT/ornith15" "$(dirname -- "$ORNITH15_MMPROJ")"

HF=""
need_hf_cli() {
  if [[ -n $HF ]]; then
    return
  fi
  local hf_venv=$ROOT/venv-hf
  if [[ ! -x $hf_venv/bin/hf ]]; then
    python3 -m venv "$hf_venv"
    "$hf_venv/bin/pip" install -U pip 'huggingface_hub[hf_xet]'
  fi
  HF=$hf_venv/bin/hf
}

echo "==> Qwen3.8-27B UD-Q5_K_XL"
if [[ ! -s $QWEN38_MODEL ]]; then
  need_hf_cli
  "$HF" download unsloth/Qwen3.8-27B-GGUF \
    Qwen3.8-27B-UD-Q5_K_XL.gguf --local-dir "$MODEL_ROOT/qwen38"
else
  echo "already present: $QWEN38_MODEL"
fi

echo "==> Ornith-1.5-35B-A3B AD-Q5_K-Q4_K"
if [[ ! -s $ORNITH15_MODEL ]]; then
  need_hf_cli
  "$HF" download AtomicChat/Ornith-1.5-35B-A3B-GGUF \
    Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf --local-dir "$MODEL_ROOT/ornith15"
else
  echo "already present: $ORNITH15_MODEL"
fi

echo "==> Ornith-1.5 vision projector (BF16)"
if [[ ! -s $ORNITH15_MMPROJ ]]; then
  need_hf_cli
  MMPROJ_NAME=mmproj-Ornith-1.5-35B-BF16.gguf
  MMPROJ_DIR=$(dirname -- "$ORNITH15_MMPROJ")
  "$HF" download ornith-ai/Ornith-1.5-35B-A3B-GGUF \
    "$MMPROJ_NAME" --local-dir "$MMPROJ_DIR"
  DOWNLOADED_MMPROJ="$MMPROJ_DIR/$MMPROJ_NAME"
  if [[ $DOWNLOADED_MMPROJ != "$ORNITH15_MMPROJ" ]]; then
    mv -f -- "$DOWNLOADED_MMPROJ" "$ORNITH15_MMPROJ"
  fi
else
  echo "already present: $ORNITH15_MMPROJ"
fi

echo "==> Shisa fixed Ornith-1.5 MTP3 draft"
if [[ ! -s $ORNITH15_MTP_MODEL ]]; then
  CONVERT_VENV=$ROOT/venv-convert
  if [[ ! -x $CONVERT_VENV/bin/python ]]; then
    python3 -m venv "$CONVERT_VENV"
  fi
  "$CONVERT_VENV/bin/pip" install -U pip
  "$CONVERT_VENV/bin/pip" install -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt"

  BF16="$MODEL_ROOT/ornith15/mtp-shisa-ornith15-bf16.gguf"
  if [[ ! -s $BF16 ]]; then
    HF_TOKEN=${HF_TOKEN:-} "$CONVERT_VENV/bin/python" "$SRC/convert_hf_to_gguf.py" \
      --remote --mtp --outtype bf16 --outfile "$BF16" \
      shisa-ai/Ornith-1.5-35B-A3B-MTP
  fi

  # Start from Q8_0, but explicitly keep every substantial tensor in the
  # trained MTP block at BF16. The only large tensors left to quantize are the
  # duplicated token embedding and output projection. This reproduces the
  # 2.64 GiB draft used in our tests.
  "$SRC/build-qwen/bin/llama-quantize" \
    --tensor-type 'blk\..*(attn_(k|q|v|output)|ffn_(down|gate|up)_(exps|shexp)|nextn\.eh_proj)\.weight=bf16' \
    "$BF16" "$ORNITH15_MTP_MODEL" Q8_0
  if [[ ${KEEP_MTP_BF16:-0} != 1 ]]; then
    rm -f "$BF16"
  fi
else
  echo "already present: $ORNITH15_MTP_MODEL"
fi

echo
echo "Models ready:"
ls -lh "$QWEN38_MODEL" "$ORNITH15_MODEL" "$ORNITH15_MTP_MODEL" "$ORNITH15_MMPROJ"
