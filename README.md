# local-llm-setup

This is my local two-model llama.cpp setup for a V100 32 GB + RTX 3060 Ti 8 GB machine.

The main model is Qwen3.8-27B. Ornith-1.5-35B-A3B is used for subagents. llama-swap loads only the model that is needed, while a small wrapper saves llama.cpp slot state before unloading and restores it when that model comes back. This means switching to a subagent does not require rebuilding the main agent's long prompt cache from scratch.

The setup intentionally builds llama.cpp twice. Qwen3.8 is much faster on the normal Volta/cuBLAS path, while Ornith's routed MoE layers are much faster with `GGML_CUDA_FORCE_MMQ=ON`.

## Install

Requirements are a working NVIDIA/CUDA development environment plus `git`, `cmake`, a C/C++ compiler, Python 3, `curl` and `tar`.

```bash
git clone https://github.com/mistrjirka/local-llm-setup.git
cd local-llm-setup
./install.sh --models
```

If the target GGUFs already exist, point the installer at them instead of moving or downloading them again:

```bash
./install.sh \
  --dense-model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --moe-model /path/to/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf \
  --mtp-model /path/to/mtp-shisa-ornith15-bf16block-q8-embedout.gguf
```

`--dense-model` selects the model used by the normal-MMQ Qwen profile and `--moe-model` selects the model used by the FORCE_MMQ Ornith profile. These flags configure paths; the launch parameters are still tuned for Qwen3.8 and Ornith-1.5 rather than arbitrary dense/MoE architectures.

The flags can be mixed with `--models`. In that case existing paths are reused and only missing artifacts are downloaded or built. For example, if both target GGUFs already exist but the fixed MTP draft does not:

```bash
./install.sh --models \
  --dense-model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --moe-model /path/to/Ornith-1.5-35B-A3B-AD-Q5_K-Q4_K.gguf
```

Without `--models`, the installer only builds the software. Models can then be downloaded separately:

```bash
~/.local/bin/local-llm-download-models
```

The default install directory is:

```text
~/.local/share/local-llm-setup
```

Start it with:

```bash
~/.local/bin/local-llm-swap
```

The OpenAI-compatible API is then on `127.0.0.1:8080` by default. The configured model IDs are `qwen38` and `ornith15`.

## What gets installed

The installer clones my `v100-optimized` llama.cpp branch and builds two variants from the same source. This branch is the runtime integration branch: it follows current llama.cpp upstream while carrying the tested Volta optimizations and the local cache/prefill features used by this setup. Individual upstream PR work remains isolated on separate branches.

| Model | llama.cpp build | Reason |
| --- | --- | --- |
| Qwen3.8-27B | normal MMQ heuristic | keeps the fast V100 FP16/cuBLAS path for large dense matmuls |
| Ornith-1.5-35B-A3B | `GGML_CUDA_FORCE_MMQ=ON` | avoids the very slow large-batch routed-MoE fallback on Volta |

It also downloads the current Linux llama-swap release and installs the cache-preserving wrapper from this repository.

The default model files are:

- Qwen3.8-27B `UD-Q5_K_XL`
- Ornith-1.5-35B-A3B `AD-Q5_K-Q4_K`
- the fixed Shisa Ornith-1.5 MTP head, exported as a llama.cpp draft GGUF

For the MTP draft, the trained MTP block remains BF16. Only the duplicated token embedding and output projection are converted to Q8_0 to save VRAM.

## Why two builds

On the exact `100k cached + 1k new + 64 generated` workload, using our current fork on this machine:

| Qwen3.8-27B | PP | TG |
| --- | ---: | ---: |
| normal build | **452.8 tok/s** | 26.66 tok/s |
| FORCE_MMQ | 323.2 tok/s | 26.54 tok/s |

So globally forcing MMQ costs Qwen about 29% of prompt-processing performance.

For Ornith with the fixed MTP3 head:

| Ornith-1.5 AD-Q6_K | PP | TG |
| --- | ---: | ---: |
| vanilla llama.cpp | 544.1 tok/s | 67.26 tok/s |
| our fork, normal MMQ heuristic | 644.2 tok/s | 68.99 tok/s |
| **our fork + FORCE_MMQ** | **882.6 tok/s** | **70.80 tok/s** |

The generated 64-token sequence was identical in those matched tests. FORCE_MMQ is therefore useful for Ornith but should not be enabled globally for Qwen.

The reason is the routed MoE path. On Volta, a large quantized `MUL_MAT_ID` can otherwise fall back to a path that copies routing information to the CPU, synchronizes the CUDA stream, sorts tokens by expert on the CPU, copies the result back, and launches expert matmuls separately. FORCE_MMQ keeps this operation on the GPU. Qwen's large dense matmuls are different and benefit from the normal FP16 Tensor Core/cuBLAS route.

## Long-context subagent profile

The default Ornith profile is tuned for the exact V100 32 GB + 3060 Ti 8 GB setup:

```text
4 slots
250112 tokens per slot
1000448 total llama.cpp context
Q8_0 K cache
Q8_0 V cache
AD-Q5_K-Q4_K target
Shisa fixed MTP3
FORCE_MMQ
```

I tested the full configuration with four requests generating concurrently. It fits and all four slots remain available. To preserve Q8/Q8 KV at this context size, some expert tensors from late Ornith layers are intentionally left on the CPU.

This placement assumes llama.cpp sees the V100 as `CUDA0` and the 3060 Ti as `CUDA1`. Check the startup log on another system before using the same tensor overrides.

For a different GPU layout, edit:

```text
~/.local/share/local-llm-setup/config/config.env
```

`ORNITH15_EXTRA_ARGS` can replace the default tensor placement completely.

## Cache preservation

There are two cache layers in this setup.

While a model is running, llama.cpp uses its normal RAM prompt cache and `--cache-idle-slots`, so interleaved requests do not needlessly destroy idle prefixes.

When llama-swap needs to unload a model, `llama_cache_proxy.py` waits for active requests to finish and saves every explicit server slot with llama.cpp's `/slots/{id}?action=save` API. When that model is started again, all existing slot snapshots are restored before the wrapper reports itself healthy.

For Ornith this means `slot0.bin` through `slot3.bin` are kept independently. Qwen currently uses one explicit 262k slot.

By default snapshots are stored under:

```text
/dev/shm/local-llm-setup
```

That makes save/restore fast but means snapshots disappear on reboot. If reboot persistence is more important, set `LLAMA_CACHE_ROOT` in `config.env` to a directory on normal storage.

llama-swap's graceful unload timeout is set to 120 seconds so a multi-GB Qwen snapshot is not killed during save.

## Reasoning effort

Qwen3.8 receives `reasoning_effort` directly per request.

Ornith does not have Qwen3.8's native low/medium/xhigh effort levels, so the wrapper maps the same external field to `thinking_budget_tokens`. The default mapping is:

```text
none    0
low     2048
medium  8192
high    32768
xhigh   unlimited/model default
```

It can be changed with `ORNITH15_REASONING_MAP` in `config.env`.

## Configuration

The main local configuration file is:

```text
~/.local/share/local-llm-setup/config/config.env
```

Useful settings include:

```bash
LLAMA_SWAP_LISTEN="127.0.0.1:8080"
LLAMA_CACHE_ROOT="/dev/shm/local-llm-setup"
ORNITH15_PARALLEL=4
ORNITH15_CTX_PER_SLOT=250112
QWEN38_CACHE_RAM_MIB=65536
ORNITH15_CACHE_RAM_MIB=32768
```

Existing model files can be reused by changing `QWEN38_MODEL`, `ORNITH15_MODEL` and `ORNITH15_MTP_MODEL` rather than downloading another copy.

## systemd user service

The installer renders a user-service file into the install directory. To enable it:

```bash
mkdir -p ~/.config/systemd/user
cp ~/.local/share/local-llm-setup/systemd/local-llm-setup.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-llm-setup.service
```

Inspect logs with:

```bash
journalctl --user -u local-llm-setup.service -f
```

## Updating

Run the installer again. It fast-forwards the configured llama.cpp branch and rebuilds both variants. Existing `config.env` is preserved. The default branch is `v100-optimized`; set `LLAMA_CPP_REF=<branch-or-tag>` when invoking `install.sh` to test another branch without editing the installer.

```bash
cd local-llm-setup
git pull
./install.sh
```

The model downloader skips files that already exist.

## Notes

This repository is deliberately hardware-specific rather than a generic llama.cpp installer. The tested default profile assumes 40 GB of combined NVIDIA VRAM split as V100 32 GB + 3060 Ti 8 GB. The wrapper itself is model-agnostic, but tensor placement, context size and the choice to FORCE_MMQ for Ornith are tuned for this machine.
