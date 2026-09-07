# local-llm-setup

This is my local two-model llama.cpp setup for a V100 32 GB + RTX 2080 Ti 22 GB machine.

The main model is Qwen3.8-27B. Ornith-1.5-35B-A3B is used for subagents. llama-swap loads only the model that is needed, while a small wrapper saves llama.cpp slot state before unloading and restores it when that model comes back. This means switching to a subagent does not require rebuilding the main agent's long prompt cache from scratch.

The setup uses one `v100-optimized` llama.cpp CUDA build. Qwen keeps normal dense dispatch, while Ornith enables `GGML_CUDA_VOLTA_FORCE_MMQ=moe` so only Volta routed-expert matmuls are forced through MMQ.

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
  --moe-model /path/to/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf \
  --mtp-model /path/to/mtp-shisa-ornith15-all-Q5_0.gguf \
  --mmproj-model /path/to/mmproj-Ornith-1.5-35B-BF16.gguf
```

`--dense-model` selects the Qwen target, `--moe-model` selects the Ornith target, and `--mmproj-model` can reuse an existing Ornith vision projector. These flags configure paths; the launch parameters are still tuned for Qwen3.8 and Ornith-1.5 rather than arbitrary dense/MoE architectures.

The flags can be mixed with `--models`. In that case existing paths are reused and only missing artifacts are downloaded or built. For example, if both target GGUFs already exist but the fixed MTP draft does not:

```bash
./install.sh --models \
  --dense-model /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --moe-model /path/to/Ornith-1.5-35B-A3B-AD-Q6_K-Q5_K.gguf
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

The installer clones my `v100-optimized` llama.cpp branch and builds one CUDA server used by both profiles. This branch is the runtime integration branch: it follows current llama.cpp upstream while carrying the tested Volta optimizations and the local cache/prefill features used by this setup. Individual upstream PR work remains isolated on separate branches.

| Model | CUDA dispatch | Reason |
| --- | --- | --- |
| Qwen3.8-27B | normal heuristic | keeps the fast dense V100 path |
| Ornith-1.5-35B-A3B | runtime `GGML_CUDA_VOLTA_FORCE_MMQ=moe` | forces MMQ only for Volta MoE experts |

It also downloads the current Linux llama-swap release and installs the cache-preserving wrapper from this repository.

The default model files are:

- Qwen3.8-27B `UD-Q5_K_XL`
- Ornith-1.5-35B-A3B `AD-Q6_K-Q5_K`
- the Shisa 12K KL-distilled Ornith MTP head, exported as a Q5_0 draft GGUF
- the official Ornith BF16 vision projector `mmproj-Ornith-1.5-35B-BF16.gguf`

The MTP draft is fully Q5_0. In a matched local 100k+1k+256 test it used 1.36 GB instead of 2.78 GB, raised TG from 78.17 to 99.70 tok/s, and produced the same target-token SHA.

## Runtime CUDA dispatch

Both models use the same normal CUDA build. Ornith adds selective Volta MoE MMQ (`GGML_CUDA_VOLTA_FORCE_MMQ=moe`) plus the tuned GQA8 ncols2 path.

### Qwen3.8 400k YaRN profile

The default Qwen launcher uses the validated **409600-token** extended-context profile on the V100 32 GB + RTX 2080 Ti 22 GB pair. Qwen3.8 is native at 262144 tokens, so the launcher enables static YaRN only when `QWEN38_CTX_SIZE` is above that native limit.

```text
V100 32 GB + RTX 2080 Ti 22 GB
split mode: tensor
llama.cpp device order: CUDA1,CUDA0
split: 4:5 (RTX 2080 Ti : V100)
context: 409600
YaRN original context: 262144
YaRN factor: 1.5625 (= 409600 / 262144)
q8_0 target K/V cache; FP16 MTP K/V
4096 batch / 2048 ubatch
MTP n-max=3 / draft ubatch 512 with prompt deferral
1 pipeline copy
internal host-staged CUDA AllReduce
```

llama.cpp needs both the RoPE scaling and a metadata override so its server slot is not capped at the GGUF's native 262144-token metadata. The launcher therefore adds, for the default 400k profile:

```text
--ctx-size 409600
--override-kv qwen35.context_length=int:409600
--rope-scaling yarn
--rope-scale 1.5625
--yarn-orig-ctx 262144
```

This exact capacity was validated on the target machine with `UD-Q5_K_XL`: a real 400000-token q8_0 cache followed by +1001 prompt tokens measured **230.43 PP/s**, **4.47 s TTFT**, and **28.12 TG/s** over 256 generated tokens with MTP3 (189/197 drafted tokens accepted). After the request the RTX 2080 Ti used about 20.94/22.0 GiB and the V100 about 23.05/32.0 GiB. The one-time 100k→400k cache construction measured 379.09 PP/s and is not the steady agent-turn figure.

Qwen's guidance for static YaRN is to choose a factor matching the context actually needed rather than always using the 4x 1M setting; for example it recommends factor 2 for 524288. The launcher follows the same rule and computes `QWEN38_CTX_SIZE / QWEN38_NATIVE_CTX` if `QWEN38_YARN_SCALE` is unset. Static YaRN is not enabled at or below 262144 because it can hurt short-context behavior.

During tuning llama.cpp enumerated the V100 as `CUDA0` and the RTX 2080 Ti as `CUDA1`, so the launcher deliberately passes `--device CUDA1,CUDA0`. If enumeration differs on another host, set `QWEN38_DEVICE_ORDER`. The topology-specific settings remain the measured 128 KiB internal-AllReduce copy threshold, SM75 large-prompt cuBLAS crossover at batch 256, Volta Qwen kernels, and exact Qwen3.8 MTP shortlist.

To return to native context, set `QWEN38_CTX_SIZE=262144`; the launcher then omits all YaRN/metadata-override arguments. For the historical native-context profile, FP16 target KV and 4096 ubatch remain available through `QWEN38_CACHE_TYPE_K=f16`, `QWEN38_CACHE_TYPE_V=f16`, and `QWEN38_UBATCH_SIZE=4096`. Extended-context slot snapshots are namespaced by context and KV type so an old 262k/F16 state is never restored into the 400k/q8 profile.

## Long-context subagent profile

The default Ornith profile is **four persistent 400000-token slots** with target Q8 KV:

```text
4 fixed slots x 400000 tokens
YaRN: 400000 / 262144 = 1.52587890625
target KV: Q8_0 / Q8_0
target weights: AD-Q6_K-Q5_K (26.25 GB)
split mode: layer, CUDA1,CUDA0, split 4:5
Shisa 12K KL-distilled Q5_0 draft on CUDA0 (V100)
MTP n-max=2
draft KV: Q4_0 / Q4_0
2048 batch / 128 ubatch / 64 draft ubatch
BF16 vision projector on CPU
```

Target KV deliberately stays Q8_0. Draft KV is only speculative state: every accepted token is verified by the Q8 target, so draft quantization can change speed/acceptance but not final target output.

The 26.25 GB `AD-Q6_K-Q5_K` improves AtomicChat's BF16-reference mean KLD from 0.025137 (the old `AD-Q5_K-Q4_K`) to 0.015793 and top-1 agreement from 93.52% to 94.85%. The exact candidate GGUF tensor layout was allocation-tested at 4x400k with Q8 target KV, the CPU vision projector, and MTP2. MTP3 also starts only with a much tighter ~35 MiB RTX margin, so MTP2 is the safer default; in a matched current-weight test MTP2 gave 85.70 TG/s versus 91.24 for MTP3 with identical target output.

Ornith is native at 262144. The launcher enables YaRN and the `qwen35moe.context_length` override only above native context.

## Cache preservation

There are two cache layers in this setup.

While a model is running, llama.cpp uses its normal RAM prompt cache and `--cache-idle-slots`, so interleaved requests do not needlessly destroy idle prefixes.

When llama-swap needs to unload a model, `llama_cache_proxy.py` waits for active requests to finish and saves every explicit server slot with llama.cpp's `/slots/{id}?action=save` API. When that model is started again, all existing slot snapshots are restored before the wrapper reports itself healthy.

A five-slot MTP-enabled restart test saved distinct 635/754/873/992/1111-token states and restored all five exact counts before the proxy became ready. The production profile uses four slots, so all four subagent contexts survive an Ornith -> Qwen -> Ornith swap.

For Ornith this means `slot0.bin` through `slot3.bin` are kept independently. Qwen uses one 409600-token slot. Ornith snapshots are namespaced by model, context, parallel count and target KV format because its recurrent state is configuration-sensitive.

By default snapshots are stored under:

```text
/dev/shm/local-llm-setup
```

That makes save/restore fast but means snapshots disappear on reboot. If reboot persistence is more important, set `LLAMA_CACHE_ROOT` in `config.env` to a directory on normal storage.

A 100k Ornith Q8 slot snapshot measured about 1.16 GB, so budget roughly **18-19 GB** for four nearly-full 400k slots. The Ornith llama-swap unload timeout is 300 seconds so the wrapper can finish those writes even on substantially slower storage.

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
ORNITH15_CTX_PER_SLOT=400000
QWEN38_CACHE_RAM_MIB=65536
ORNITH15_CACHE_RAM_MIB=32768
ORNITH15_MMPROJ="$HOME/models/local-llm-setup/ornith15/mmproj-Ornith-1.5-35B-BF16.gguf"
ORNITH15_MMPROJ_OFFLOAD=0
```

Existing model files can be reused by changing `QWEN38_MODEL`, `ORNITH15_MODEL`, `ORNITH15_MTP_MODEL` and `ORNITH15_MMPROJ` rather than downloading another copy. The projector stays on CPU by default so the tuned four-slot GPU placement retains its VRAM headroom; set `ORNITH15_MMPROJ_OFFLOAD=1` to move it to a GPU, optionally with `ORNITH15_MMPROJ_DEVICE=CUDA0` (or another llama.cpp device).

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

Run the installer again. It fast-forwards the configured llama.cpp branch and rebuilds the shared CUDA server. Existing `config.env` is preserved. The installer migrates only the exact old stock Ornith model/MTP paths and `250112` context default; customized values are left untouched. The default branch is `v100-optimized`; set `LLAMA_CPP_REF=<branch-or-tag>` when invoking `install.sh` to test another branch without editing the installer.

```bash
cd local-llm-setup
git pull
./install.sh
```

The model downloader skips files that already exist.

## Notes

This repository is deliberately hardware-specific rather than a generic llama.cpp installer. The Qwen default targets 54 GB of combined NVIDIA VRAM from a V100 32 GB + RTX 2080 Ti 22 GB and uses the validated 409600-token YaRN profile. The wrapper itself is model-agnostic; Qwen tensor placement and CUDA dispatch settings are tuned for this machine, while Ornith uses its separately configurable four-slot 400k/Q8 profile.
