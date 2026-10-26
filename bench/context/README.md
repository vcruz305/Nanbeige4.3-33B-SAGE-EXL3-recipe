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

Optional manual settings for *new unqualified experiments*: `MAX_SEQ_LEN`, `CACHE_SIZE`, `CACHE_MODE` (aliases Q2/Q3/Q4/Q6/Q8, or native 2,2 / 3,3 etc), and `EMBEDDING_PLACEMENT=cpu` / `gpu`. **The runtime refuses configurations whose estimated memory demand exceeds the selected device's guard**. An estimate never establishes fitness; real measurements and the owned-process watchdog remain mandatory.

On any other GPU, model compatibility and context/headroom are unvalidated. Q2 or Q3 KV may have meaningful quality loss compared with higher precision. The 3/3 retrieval test is a very small smoke, **not a 128K reasoning benchmark or a replacement for Sixcat**. The scored Sixcat v0.8.3 104/120 used **36,864 context and Q4 KV** and has not been rerun at 128K.

Machine-readable, sanitized receipts: [measured-results.json](measured-results.json). Raw prompts and watchdog event logs remain on the local test host and are not included in the GitHub recipe.
