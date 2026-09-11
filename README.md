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
  --mtp-model /path/to/mtp-shisa-ornith15-all-q4.gguf \
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
- the Shisa 12K KL-distilled Ornith MTP head, exported as a Q4_0 draft GGUF
- the official Ornith BF16 vision projector `mmproj-Ornith-1.5-35B-BF16.gguf`

The production MTP draft is fully Q4_0. The four-active-slot profile uses MTP1, q4_0 draft K/V and target-head reuse; target verification still determines every emitted token.

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

The conservative fallback Ornith profile is **four persistent 350000-token fixed slots**. It uses the same target and speculative policy as the shared-400k profile, but reserves independent fixed slots instead of relying on a common prefix:

```text
4 fixed slots x 350000 tokens
YaRN: 350000 / 262144 = 1.33514404296875
target KV: Q8_0 / Q8_0
target weights: AD-Q6_K-Q5_K (26.25 GB)
split mode: layer, device order CUDA1,CUDA0, split 14:35
Shisa 12K KL-distilled Q4_0 draft on CUDA1 (RTX 2080 Ti), target LM head reused from CUDA0
MTP n-max=1
draft KV: q4_0 / q4_0
512 batch / 128 ubatch / 64 draft ubatch
1 pipeline copy
BF16 vision projector on CPU
```

The real full `AD-Q6_K-Q5_K` GGUF is validated with this geometry, including the CPU vision projector. In fixed-350k mode all four slots report `n_ctx=350000` with MTP enabled. Draft-owned tensors stay on the first draft device (`CUDA1`, RTX 2080 Ti); `CUDA0` is also exposed to the draft scheduler only so the graph can reuse the target LM head already resident on the V100.

An earlier near-limit capacity benchmark at **348744 cached + 1000 prompt + 256 generated** measured **388.91 PP/s** (2.571 s for the 1k append) and **33.11 TG/s** with the older MTP3/Q8-draft-KV policy (150/312 drafts accepted). The previous `AD-Q5_K-Q4_K` model under the same geometry measured **385.62 PP/s / 33.26 TG/s** (149/315 accepted), so moving to `AD-Q6_K-Q5_K` was **+0.85% PP / -0.45% TG** in this test while improving quantization fidelity. The 348744-token one-time Q6/Q5 cache construction averaged **776.60 PP/s**; its deferred MTP catch-up is persisted by the `.draft`/`.spec` slot companions.

The 26.25 GB `AD-Q6_K-Q5_K` improves AtomicChat's BF16-reference mean KLD from 0.025137 (the old `AD-Q5_K-Q4_K`) to 0.015793 and top-1 agreement from 93.52% to 94.85%. The current four-slot production profile uses the all-Q4 Shisa head, **MTP1**, target-head reuse, target `batch=512` / `ubatch=128`, draft `ubatch=64`, and **q4_0 draft K/V**. In the matched four-active-slot sweep q4_0 draft K/V used **19,118 MiB** on the RTX versus **19,802 MiB** for Q8 draft K/V while aggregate throughput was effectively tied (**105.27 vs 105.25 tok/s**). Target verification determines final output tokens; draft quantization only changes speculation quality and cost.

Ornith is native at 262144. The launcher enables YaRN and the `qwen35moe.context_length` override only above native context. Reducing the cap from four 400k slots to four 350k slots saves roughly **2.7-2.9 GiB of reserved Q8 KV** on this GPU pair; it does not slow inference and reduces attention work when an agent actually reaches the shorter limit.

### Default shared-prefix 400k mode

The default `ORNITH15_SHARED_400K=1` exposes **four logical 400000-token slots** while reserving a **1,400,000-token physical unified Q8 KV pool**. This mode keeps the same Q6/Q5 target weights, Q8 target KV, q4_0 draft KV, MTP1, target-head reuse, CPU vision projector and 14:35 placement. It adds llama.cpp's exact `--slot-fork-prefix` path so subagents branching from the same parent reference the same attention-KV cells instead of duplicating their common prefix.

The capacity is intentionally conditional rather than four independent 400k guarantees. For four histories of length `L_i` with a common prefix `P`, physical attention-KV occupancy is approximately `sum(L_i) - 3P`. At four full 400k histories, the 1.4M pool therefore needs at least **66,667 common-prefix tokens**. A 100k common parent leaves about 100k tokens of additional physical-pool margin at four full logical caps. Completely unrelated four-way 400k histories still need 1.6M cells and do not fit this GPU pair.

The current production geometry is validated with the real full `AD-Q6_K-Q5_K` GGUF on the V100 32 GB + RTX 2080 Ti 22 GB pair: four logical 400k slots over the 1.4M unified pool, Q8 target KV, q4_0 draft KV, Q4 Shisa MTP1, target-head reuse, and 14:35 placement. In the matched four-slot run startup used **19,118 MiB** on the RTX 2080 Ti and **31,291 MiB** on the V100, leaving about **2,883 MiB** and **1,204 MiB** respectively. The physical KV buffer is preallocated, so growing a shared agent does not progressively allocate more VRAM. The 400k shared-prefix profile is now the default. Set `ORNITH15_SHARED_400K=0` to use the conservative 350k fixed-slot profile when four completely unrelated histories must each have guaranteed capacity.

Shared-400k snapshots use a separate namespace including the unified-pool size, target/draft KV types, MTP depth and draft-model name. Do not reuse fixed-slot snapshots across the two layouts.

## Cache preservation

There are three cache levels in this setup.

1. **Live GPU slots.** Shared-400k mode keeps up to four active unified-KV slots resident, with exact common-prefix K/V cells shared between compatible agents. `--no-cache-idle-slots` avoids proactively evicting useful live branches.
2. **Parked RAM agents.** The same server has a size-bounded host prompt cache (`ORNITH15_CACHE_RAM_MIB`, 32 GiB by default). When a live slot is replaced or KV pressure requires space, its complete target state, MTP draft state and speculative carry are parked in RAM. A later request automatically restores the deepest matching agent history; if a compatible parent prefix is still live, target and draft state restore directly onto that prefix instead of duplicating it. One physical slot can therefore serve multiple inactive agent sessions over time.
3. **Process-restart persistence.** When llama-swap unloads the model, `llama_cache_proxy.py` saves both the live explicit slots and the whole parked-agent RAM bank. On the next model start the parked bank and live slots are restored before the wrapper reports healthy.

The parked bank is persisted as `prompt-cache.bin` beside the normal slot files. A Qwen end-to-end test saved two parked agents as a 625.5 MB bank, killed/recreated the server, restored that bank in 141 ms, and resumed the parked agent with the same output SHA and MTP acceptance as the live-cache control. Dense Qwen3.8 and hybrid/recurrent Ornith both passed A -> B -> C -> A RAM-restore exactness tests.

The wrapper also preserves optional `.draft` and `.spec` companions emitted by MTP-aware llama.cpp builds. Those carry the draft KV and the small per-sequence speculative state alongside the ordinary target `slotN.bin`, avoiding a long draft catch-up after a model swap. On older servers where these companions are absent, behavior is unchanged.

The stronger shared-prefix restart test used one 10k parent plus three divergent ~11.5k children. All four states were saved, the Ornith process exited, a fresh process restored all four before the proxy became ready, and all four then continued concurrently with only 51 new prompt tokens each. The restored continuation matched a clean full-prefill reference SHA exactly, including MTP behavior.

For Ornith, `slot0.bin` through `slot3.bin` preserve the currently-live execution slots while `prompt-cache.bin` preserves additional parked sessions. Qwen still has one 409600-token live slot, but that slot can now multiplex multiple parked agent histories through the same RAM cache. Ornith snapshots are namespaced by model, context, parallel count and target/draft KV/MTP geometry because its recurrent and speculative state is configuration-sensitive.

By default snapshots are stored under:

```text
/dev/shm/local-llm-setup
```

That makes save/restore fast but means snapshots disappear on reboot. If reboot persistence is more important, set `LLAMA_CACHE_ROOT` in `config.env` to a directory on normal storage.

A 100k Ornith Q8 target snapshot measured about 1.16 GB. MTP-aware live-slot saves add `.draft` and tiny `.spec` companions; budget roughly **20 GB** for four nearly-full 400k live snapshots, plus up to `ORNITH15_CACHE_RAM_MIB` for the persisted parked-agent bank in the worst case. With the default 32 GiB RAM bank, `/dev/shm` or an alternate `LLAMA_CACHE_ROOT` should therefore have substantial free space. The Ornith llama-swap unload timeout is 300 seconds so the wrapper can finish those writes even on slower storage.

Qwen uses a 64 GiB RAM-cache ceiling by default, although actual persisted size is only the states currently parked. If `LLAMA_CACHE_ROOT` points to slower disk and the bank becomes large, increase llama-swap's graceful unload timeout accordingly.

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
ORNITH15_CTX_PER_SLOT=350000
# Default: four logical 400k slots over a 1.4M shared physical Q8 KV pool.
# Requires shared agent prefixes for aggregate capacity; set 0 for fixed 350k.
ORNITH15_SHARED_400K=1
ORNITH15_SHARED_CTX_PER_SLOT=400000
ORNITH15_SHARED_KV_POOL=1400000
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

Run the installer again. It fast-forwards the configured llama.cpp branch and rebuilds the shared CUDA server. Existing `config.env` values are preserved. The installer migrates only exact historical stock Ornith defaults, leaves customized values untouched, and appends the new shared-400k option keys only when they are missing. The default branch is `v100-optimized`; set `LLAMA_CPP_REF=<branch-or-tag>` when invoking `install.sh` to test another branch without editing the installer.

```bash
cd local-llm-setup
git pull
./install.sh
```

The model downloader skips files that already exist.

## Notes

This repository is deliberately hardware-specific rather than a generic llama.cpp installer. The Qwen default targets 54 GB of combined NVIDIA VRAM from a V100 32 GB + RTX 2080 Ti 22 GB and uses the validated 409600-token YaRN profile. The wrapper itself is model-agnostic; Qwen tensor placement and CUDA dispatch settings are tuned for this machine, while Ornith defaults to four shared-prefix 400k/Q8 logical slots over a 1.4M physical pool; fixed 350k/Q8 remains available as the conservative fallback.
