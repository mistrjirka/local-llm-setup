#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PREFIX=${PREFIX:-$HOME/.local/share/local-llm-setup}
BIN_LINK_DIR=${BIN_LINK_DIR:-$HOME/.local/bin}
LLAMA_CPP_REPO=${LLAMA_CPP_REPO:-https://github.com/mistrjirka/llama.cpp.git}
LLAMA_CPP_REF=${LLAMA_CPP_REF:-v100-optimized}
CUDA_ARCHS=${CUDA_ARCHS:-70;86}
JOBS=${JOBS:-$(nproc)}
WITH_MODELS=0
DENSE_MODEL=""
MOE_MODEL=""
MTP_MODEL=""

while (( $# )); do
  case "$1" in
    --models) WITH_MODELS=1; shift ;;
    --dense-model)
      [[ $# -ge 2 ]] || { echo "--dense-model requires a path" >&2; exit 2; }
      DENSE_MODEL=$2; shift 2 ;;
    --moe-model)
      [[ $# -ge 2 ]] || { echo "--moe-model requires a path" >&2; exit 2; }
      MOE_MODEL=$2; shift 2 ;;
    --mtp-model)
      [[ $# -ge 2 ]] || { echo "--mtp-model requires a path" >&2; exit 2; }
      MTP_MODEL=$2; shift 2 ;;
    --dense-model=*) DENSE_MODEL=${1#*=}; shift ;;
    --moe-model=*) MOE_MODEL=${1#*=}; shift ;;
    --mtp-model=*) MTP_MODEL=${1#*=}; shift ;;
    -h|--help)
      cat <<EOF
usage: ./install.sh [--models] [--dense-model PATH] [--moe-model PATH] [--mtp-model PATH]

Model paths:
  --dense-model PATH   existing Qwen3.8 GGUF used by the normal-MMQ build
  --moe-model PATH     existing Ornith GGUF used by the FORCE_MMQ build
  --mtp-model PATH     existing Ornith MTP draft GGUF
  --models             download/build only model artifacts whose configured paths are missing

Environment overrides:
  PREFIX=$PREFIX
  LLAMA_CPP_REF=$LLAMA_CPP_REF
  CUDA_ARCHS=$CUDA_ARCHS
  JOBS=$JOBS
EOF
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for model_path in "$DENSE_MODEL" "$MOE_MODEL" "$MTP_MODEL"; do
  if [[ -n $model_path && ! -s $model_path ]]; then
    echo "model path does not exist or is empty: $model_path" >&2
    exit 2
  fi
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }; }
for x in git cmake python3 curl tar sha256sum; do need "$x"; done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Install a CUDA toolkit before running this installer." >&2
  exit 1
fi

mkdir -p "$PREFIX" "$PREFIX/bin" "$PREFIX/config" "$PREFIX/systemd" "$BIN_LINK_DIR"

SRC=$PREFIX/llama.cpp
if [[ -d $SRC/.git ]]; then
  echo "==> Updating llama.cpp fork"
  git -C "$SRC" fetch --prune origin
  git -C "$SRC" checkout "$LLAMA_CPP_REF"
  git -C "$SRC" pull --ff-only origin "$LLAMA_CPP_REF"
else
  echo "==> Cloning llama.cpp fork"
  git clone --branch "$LLAMA_CPP_REF" --single-branch "$LLAMA_CPP_REPO" "$SRC"
fi

COMMON_CMAKE=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_CUDA=ON
  "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHS"
  -DGGML_CUDA_FA=ON
  -DGGML_CUDA_GRAPHS=ON
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
  -DGGML_SCHED_MAX_COPIES=4
)

echo "==> Building Qwen variant (normal MMQ heuristic)"
cmake -S "$SRC" -B "$SRC/build-qwen" "${COMMON_CMAKE[@]}" -DGGML_CUDA_FORCE_MMQ=OFF
cmake --build "$SRC/build-qwen" --target llama-server llama-quantize llama-fit-params -j "$JOBS"

echo "==> Building Ornith variant (FORCE_MMQ)"
cmake -S "$SRC" -B "$SRC/build-ornith-mmq" "${COMMON_CMAKE[@]}" -DGGML_CUDA_FORCE_MMQ=ON
cmake --build "$SRC/build-ornith-mmq" --target llama-server -j "$JOBS"

install -m 0755 "$REPO_DIR/bin/llama_cache_proxy.py" "$PREFIX/bin/llama_cache_proxy.py"
install -m 0755 "$REPO_DIR/bin/run-qwen38.sh" "$PREFIX/bin/run-qwen38.sh"
install -m 0755 "$REPO_DIR/bin/run-ornith15.sh" "$PREFIX/bin/run-ornith15.sh"
install -m 0755 "$REPO_DIR/bin/start.sh" "$PREFIX/bin/start.sh"
install -m 0755 "$REPO_DIR/download-models.sh" "$PREFIX/bin/download-models.sh"

echo "==> Installing llama-swap"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python3 - "$TMP/release.env" <<'PY'
import json, platform, urllib.request, sys
out=sys.argv[1]
arch={"x86_64":"amd64","amd64":"amd64","aarch64":"arm64","arm64":"arm64"}.get(platform.machine().lower())
if not arch:
    raise SystemExit(f"unsupported architecture: {platform.machine()}")
with urllib.request.urlopen("https://api.github.com/repos/mostlygeek/llama-swap/releases/latest", timeout=30) as f:
    release=json.load(f)
assets={a["name"]:a["browser_download_url"] for a in release["assets"]}
needle=f"linux_{arch}.tar.gz"
name=next((n for n in assets if n.endswith(needle)), None)
checks=next((n for n in assets if n.endswith("_checksums.txt")), None)
if not name or not checks:
    raise SystemExit("could not find llama-swap Linux release assets")
with open(out,"w") as f:
    f.write(f"TAG={release['tag_name']}\nASSET={name}\nURL={assets[name]}\nCHECKS_URL={assets[checks]}\n")
PY
# shellcheck source=/dev/null
source "$TMP/release.env"
curl -fL --retry 3 "$URL" -o "$TMP/$ASSET"
curl -fL --retry 3 "$CHECKS_URL" -o "$TMP/checksums.txt"
EXPECTED=$(awk -v f="$ASSET" '$2==f || $2=="*"f {print $1; exit}' "$TMP/checksums.txt")
if [[ -n $EXPECTED ]]; then
  ACTUAL=$(sha256sum "$TMP/$ASSET" | awk '{print $1}')
  [[ $ACTUAL == "$EXPECTED" ]] || { echo "llama-swap checksum mismatch" >&2; exit 1; }
fi
tar -xzf "$TMP/$ASSET" -C "$TMP"
SWAP_BIN=$(find "$TMP" -type f -name llama-swap -perm -u+x | head -1)
[[ -n $SWAP_BIN ]] || { echo "llama-swap binary missing from release archive" >&2; exit 1; }
install -m 0755 "$SWAP_BIN" "$PREFIX/bin/llama-swap"

python3 - "$REPO_DIR/config/llama-swap.yaml.in" "$PREFIX/config/llama-swap.yaml" "$PREFIX" <<'PY'
from pathlib import Path
import sys
src,dst,prefix=sys.argv[1:]
Path(dst).write_text(Path(src).read_text().replace("@PREFIX@", prefix))
PY

if [[ ! -f $PREFIX/config/config.env ]]; then
  cp "$REPO_DIR/config/config.env.example" "$PREFIX/config/config.env"
else
  echo "==> Keeping existing $PREFIX/config/config.env"
fi

# CLI paths override the corresponding configured model paths without replacing
# the rest of an existing config.env.
python3 - "$PREFIX/config/config.env" "$DENSE_MODEL" "$MOE_MODEL" "$MTP_MODEL" <<'PYCFG'
from pathlib import Path
import shlex, sys
path = Path(sys.argv[1])
overrides = {
    "QWEN38_MODEL": sys.argv[2],
    "ORNITH15_MODEL": sys.argv[3],
    "ORNITH15_MTP_MODEL": sys.argv[4],
}
lines = path.read_text().splitlines()
for key, value in overrides.items():
    if not value:
        continue
    value = str(Path(value).expanduser().resolve())
    replacement = f"{key}={shlex.quote(value)}"
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = replacement
            break
    else:
        lines.append(replacement)
path.write_text("\n".join(lines) + "\n")
PYCFG

python3 - "$REPO_DIR/systemd/local-llm-setup.service.in" "$PREFIX/systemd/local-llm-setup.service" "$PREFIX" <<'PY'
from pathlib import Path
import sys
src,dst,prefix=sys.argv[1:]
Path(dst).write_text(Path(src).read_text().replace("@PREFIX@", prefix))
PY

ln -sfn "$PREFIX/bin/start.sh" "$BIN_LINK_DIR/local-llm-swap"
ln -sfn "$PREFIX/bin/download-models.sh" "$BIN_LINK_DIR/local-llm-download-models"

COMMIT=$(git -C "$SRC" rev-parse HEAD)
echo "$COMMIT" > "$PREFIX/llama.cpp.commit"
echo "$TAG" > "$PREFIX/llama-swap.version"

echo
echo "Installed to: $PREFIX"
echo "llama.cpp:  $COMMIT"
echo "llama-swap: $TAG"
if (( WITH_MODELS )); then
  "$PREFIX/bin/download-models.sh"
else
  echo "Next: $PREFIX/bin/download-models.sh"
fi
echo "Start: $PREFIX/bin/start.sh"
