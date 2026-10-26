# Install and run

Requires Linux/WSL2, NVIDIA RTX 4090 24 GB, compatible CUDA-enabled
PyTorch, ExLlamaV3 and TabbyAPI, and disk room for the 18.35 GiB pack.

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
mkdir -p ~/nanbeige43-exl3-4090/src
git clone https://github.com/vcruz305/exllamav3.git ~/nanbeige43-exl3-4090/src/exllamav3
git clone https://github.com/vcruz305/tabbyAPI.git ~/nanbeige43-exl3-4090/src/tabbyAPI
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

## Private model download and integrity validation

The private model requires owner access and `hf auth login`:

~~~bash
python3 -m pip install -U huggingface_hub
hf auth login
bash exllamav3-tabby/download.sh
bash exllamav3-tabby/download.sh --verify-only
~~~

Download is pinned to revision
`b273927de2b1c93e51f4d25707613cc0c28e3283`.
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
`PORT` defaults to 8898. API **always** binds 127.0.0.1.
Resource checks refuse a busy 4090 and enforce GPU <=23,950 MiB,
available host RAM >=48 GiB, root disk >=42 GiB, and a cooperative
exclusive lock. Only the child process group launched by this recipe
can be stopped by its watcher. Quit with Ctrl-C.

`DRY_RUN=1` does not touch the GPU or install any packages.
