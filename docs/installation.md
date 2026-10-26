# Install and run

Requires Linux/WSL2, a compatible NVIDIA CUDA GPU with enough VRAM
for 18.35 GiB of weights plus inference/KV overhead, CUDA-enabled PyTorch,
ExLlamaV3 and TabbyAPI. A 24 GiB GPU is a reasonable starting point; it is
**not a promise every such card can run the tested 36K/Q4 configuration**.
This recipe can select a GPU on a multi-GPU machine with `GPU_INDEX`,
but does not configure tensor-parallel splits or CPU layer offload.
The recorded measurements were all performed on **RTX 4090 24 GB**.

## Important: locally qualified runtime versus current public checkout

The measured ExLlamaV3 source checkout was
`61cfd2cbed196cb05e512ac2201c7f1711140ada`, an **unpublished
local development revision** when this recipe was written. Do NOT
silently replace it with HEAD of a public fork and assume the same
Sixcat score or 33.82 tok/s speed. The TabbyAPI isolated snapshot
derived from `fcc1a1078e1f766dad305045c7c4d30aaefa6458`.
It had one Turing SM75 hardware capability exception, irrelevant
to RTX4090 (SM89). This recipe makes **no edits** to any checkout.

For a new host, use the [ExLlamaV3 installation guide](https://github.com/turboderp-org/exllamav3)
and [TabbyAPI installation guide](https://github.com/theroyallab/tabbyAPI)
to prepare a separate environment. Example source roots:

~~~bash
mkdir -p ~/nanbeige43-exl3/src
git clone https://github.com/vcruz305/exllamav3.git ~/nanbeige43-exl3/src/exllamav3
git clone https://github.com/vcruz305/tabbyAPI.git ~/nanbeige43-exl3/src/tabbyAPI
# Build/install them in a fresh CUDA-enabled Python env using their docs.
# This newly built environment is UNQUALIFIED until a local smoke and eval.
~~~

Record pre-installed paths without updating them:

~~~bash
export BASE_PYTHON=/path/to/cuda-venv/bin/python
export EXLLAMAV3_SRC=/path/to/exllamav3
export TABBY_DIR=/path/to/tabbyAPI
bash exllamav3-tabby/setup.sh
~~~

## Public, ungated model download and integrity validation

The model's 54 EXL3 weight shards are **publicly downloadable without
an HF account, token, license-acceptance click or access request**.
The download script explicitly uses anonymous access:

~~~bash
python3 -m pip install -U huggingface_hub
bash exllamav3-tabby/download.sh
bash exllamav3-tabby/download.sh --verify-only
~~~

Download is pinned to the public model-card revision
`e4bf35566476daf25fdf650947d10553c1af985c`.
All 63 files, including **54 safetensors**, have locked SHA-256 and
size. Default model directory is
`~/models/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw`.
Override with `MODEL_DIR` as necessary. No weights are uploaded or
published by these scripts.

## Serve

~~~bash
DRY_RUN=1 PROFILE=long bash exllamav3-tabby/serve.sh
PROFILE=long bash exllamav3-tabby/serve.sh
# New terminal:
bash exllamav3-tabby/smoke.sh
~~~

`long`: 36,864-token context Q4 KV, batch1, chunk512.
`short`: 16,384-token context Q8 KV, batch1.
`compact`: 8,192-token context Q4 KV, batch1 (**experimental**,
not a confirmed fit on a specific other GPU).
`extended`: **131,072-token (128K) context Q2 K/V, batch1**, with
**CPU-resident input embedding**. Successfully generated a 130,542-token
input response with **3/3 synthetic retrieval markers** on RTX 4090,
peak **23,220 MiB VRAM**. This is **not** a full Sixcat/quality evaluation.
Run `PROFILE=extended bash exllamav3-tabby/serve.sh` to use it.
`PORT` defaults to 8898. API **always** binds 127.0.0.1.
Resource checks refuse an occupied selected NVIDIA GPU and reserve at
least 2.5% of its VRAM or 614 MiB, whichever is greater. On the measured
RTX 4090 (24,564 MiB), this preserves a 23,949-MiB limit. Portable
default RAM/disk safety floors are 8/4 GiB free and can be raised via
`RAM_MIN_AVAILABLE_GIB` / `ROOT_MIN_FREE_GIB`. The original scoring
campaign used stricter 48/42 GiB floors. Only the owned model-server
process group may be stopped by this launcher. Quit with Ctrl-C.

Manual context experiments may specify `MAX_SEQ_LEN`, `CACHE_SIZE`,
`CACHE_MODE` (`Q2`, `Q3`, `Q4`, `Q6`, `Q8`, `FP16`, or native 2..8 bit K/V
pairs) and `EMBEDDING_PLACEMENT=cpu` or `gpu`. The recipe converts `Q2`
to native `2,2` and `Q3` to `3,3` automatically. A conservative GPU
memory budget calculation **refuses** settings predicted to breach the
selected device's reservation; this is a preflight estimate, not a
substitute for monitoring the running process. This also fixes the older
1024-file-descriptor loader issue by setting a 65,536 soft limit for the
owned server only, leaving the system-wide settings unchanged.

**256K is not validated on the RTX 4090.** The Q2 262,144-token cache
alone is 8,320 MiB, and the measured-budget preflight predicts around
27,576 MiB GPU total even with CPU embedding. The original experiment
correctly refused GPU launch under the 23,950 MiB ceiling. 256K would
require a larger device or extra dense-weight/KV offload beyond this
single-GPU recipe. See [context measurements](../bench/context/README.md).

To select a different GPU: `GPU_INDEX=1 PROFILE=compact bash
exllamav3-tabby/serve.sh` (only a configuration example, **not a
verified 8K fit** on that GPU). Larger devices can experimentally adjust
`MAX_SEQ_LEN`, `CACHE_SIZE`, or `CACHE_MODE`; record new benchmarks.
Non-NVIDIA and CPU-only inference are outside this EXL3 runtime recipe.

`DRY_RUN=1` does not touch the GPU or install any packages.
