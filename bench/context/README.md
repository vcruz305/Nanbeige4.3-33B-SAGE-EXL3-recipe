# Extended context experiments — Nanbeige4.3 SAGE EXL3

The model's nominal 512K RoPE/position ceiling is **not** a claim that its 4.50-bpw EXL3 pack can serve 512K on one 24GB RTX 4090. Actual single-GPU long-prompt retrieval tests are below; all used the same released model weights and pinned ExLlamaV3 through isolated TabbyAPI.

## Verified GPU tests (October 26, 2026)

| Allocated context | Actual prompt input | KV K/V | Input embedding | Peak VRAM | Retrieval |
|---:|---:|---|---|---:|---:|
| 53,248 | 52,728 | Q4/Q4 | GPU | 23,120 MiB | 3/3 |
| 65,536 | 65,004 | Q3/Q3 | GPU | 22,980 MiB | 3/3 |
| 81,920 | 81,405 | Q2/Q2 | GPU | 22,680 MiB | 3/3 |
| **131,072** | **130,542** | **Q2/Q2** | **CPU/system RAM** | **23,220 MiB** | **3/3** |

All successful attempts loaded the entire released decoder EXL3 pack without altering its weights, processed a native-template prompt almost filling the specified cache, generated a real response, and recovered synthetic markers at three different depths. The 128K request took 161.055 seconds on this specific WSL2 RTX 4090. Model input embedding is quantized in the original pack and kept in system RAM in this mode to save VRAM; this is a **measured selective embedding offload**, not full model offload.

Resource protection: exclusive GPU lock, monitoring of device VRAM, host available memory and filesystem space, **23,950 MiB hard device cutoff**, 48 GiB host available RAM and 42 GiB root disk free safety floors during the original experiments. The watchdog could terminate only the spawned server process group. The GPU returned to 214 MiB idle after each run.

## 256K attempted and rejected safely

A 262,144-token capacity request generated a 261,619-token synthetic test prompt. Its **Q2/Q2 KV cache alone needs 8,320 MiB** (52 layers, eight KV heads, 128 dimensions and per-32-element scales). Accounting for the model and a CPU-resident embedding, the preflight estimated **27,576 MiB of VRAM**, exceeding the 23,950 MiB safety cutoff. It **refused the 256K GPU launch**; there was no 256K model decode, no OOM, and no claimed 256K pass.

The original 4.50-bpw model is too large to hold itself plus a fully resident 256K Q2 cache in 24GB. An actual 256K GPU run requires further engineering, e.g. model-weight/KV host offload (not provided by these scripts), a larger GPU, multi-GPU sharding, or a substantially smaller model quantization. A positional metadata limit is *not* a solution to insufficient memory.

## Use 128K with the recipe (one suitable CUDA GPU)

After downloading the publicly released model and adopting an isolated compatible ExLlamaV3 + TabbyAPI runtime:

~~~bash
PROFILE=extended bash exllamav3-tabby/serve.sh
~~~

The `extended` profile uses `MAX_SEQ_LEN=131072`, `CACHE_SIZE=131072`, `cache_mode: 2,2` (2-bit K/V), one concurrent sequence, and `EXL3_EMBED_TABLE_CUDA=0` so the packed input embedding remains in system RAM.

### Exact run commands on the verified type of host (Linux/WSL2 + CUDA)

Prepare an isolated, compatible ExLlamaV3 + TabbyAPI checkout as documented in [installation](../../docs/installation.md). These commands intentionally do **not** install/upgrade GPU runtime code or touch unrelated processes:

~~~bash
# From this GitHub repo directory. Replace the three runtime paths
# with your separately installed CUDA-PyTorch runtime and source checkouts.
export BASE_PYTHON=/path/to/your/cuda-venv/bin/python
export EXLLAMAV3_SRC=/path/to/exllamav3
export TABBY_DIR=/path/to/tabbyAPI
bash exllamav3-tabby/setup.sh

# All 54 EXL3 safetensors are PUBLIC and downloaded without credentials.
bash exllamav3-tabby/download.sh
# Optional re-verification (reads 18.35 GiB to compute 63 file hashes):
bash exllamav3-tabby/download.sh --verify-only

# Preview the actual cache and model options without touching the GPU.
PROFILE=extended DRY_RUN=1 bash exllamav3-tabby/serve.sh

# Match the tested RTX 4090 config (128K/131072; Q2 K/V; CPU embedding).
# RAM/disk limits are from the original workstation; do not weaken safety
# without checking your host. The API binds to loopback only.
GPU_INDEX=0 PROFILE=extended \
  MAX_SEQ_LEN=131072 CACHE_SIZE=131072 \
  CACHE_MODE=Q2 EMBEDDING_PLACEMENT=cpu \
  RAM_MIN_AVAILABLE_GIB=48 ROOT_MIN_FREE_GIB=42 \
  PORT=8898 bash exllamav3-tabby/serve.sh

# In another terminal after the server is ready:
PORT=8898 bash exllamav3-tabby/smoke.sh
~~~

The short equivalent invocation (with portable default **8 GiB host RAM / 4 GiB root disk** floors) is `PROFILE=extended bash exllamav3-tabby/serve.sh`. The 128K test used **48 GiB / 42 GiB minimum** for host RAM/disk as stronger protection. The runtime also needs `nvidia-smi` and keeps **at least 614 MiB or 2.5% of the selected GPU VRAM in reserve**, whichever is larger. It refuses overlapping workers, performs a conservative capacity preflight, watches memory while running, and can terminate only its own Tabby process group. The documented 128K mode is for **one selected GPU**; it does not configure multi-GPU tensor parallelism.

**Interpretation of the measured result:** 130,542 actual prompt tokens, **161.055 seconds** for the entire request including prefill and short generation, **23,220 MiB peak GPU usage**, three synthetic markers recovered correctly. This is not a sustained 128K decoding-rate measurement or a Sixcat score.


Optional manual settings for *new unqualified experiments*: `MAX_SEQ_LEN`, `CACHE_SIZE`, `CACHE_MODE` (aliases Q2/Q3/Q4/Q6/Q8, or native 2,2 / 3,3 etc), and `EMBEDDING_PLACEMENT=cpu` / `gpu`. **The runtime refuses configurations whose estimated memory demand exceeds the selected device's guard**. An estimate never establishes fitness; real measurements and the owned-process watchdog remain mandatory.

On any other GPU, model compatibility and context/headroom are unvalidated. Q2 or Q3 KV may have meaningful quality loss compared with higher precision. The 3/3 retrieval test is a very small smoke, **not a 128K reasoning benchmark or a replacement for Sixcat**. The scored Sixcat v0.8.3 104/120 used **36,864 context and Q4 KV** and has not been rerun at 128K.

Machine-readable, sanitized receipts: [measured-results.json](measured-results.json). Raw prompts and watchdog event logs remain on the local test host and are not included in the GitHub recipe.

## Memory analysis for 200K

See [200K_MEMORY_OPTIONS.md](200K_MEMORY_OPTIONS.md) for exact Q2/Q3/Q4 cache geometry, experimental PCIe-pinned weight offload and quality/performance-oriented bitrate estimates. Keep predicted savings separate from measured working configurations.
