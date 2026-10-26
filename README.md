# Nanbeige4.3-33B SAGE-EXL3 4.50 bpw — model and serving recipe

Run **Nanbeige4.3-33B**, a **33B dense** model, at **4.499880554 body bpw**
on compatible NVIDIA CUDA systems with enough VRAM, using
**ExLlamaV3 + TabbyAPI**. This is the original qualified v6 SAGE
allocation, including the exact per-tensor mixed-K configuration, private
release checksums, and reproducible serving scripts. It is **not limited
to the RTX 4090**. The following benchmark results were **measured on a
single RTX 4090**; other GPUs and operating environments need independent
memory/throughput/compatibility validation.

**Model:** [vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw](https://huggingface.co/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw)
— **PRIVATE.** Access is required; this public GitHub recipe **does not**
grant access, include weights, or change the model's visibility.

| Actual results — **RTX 4090** test machine, 2026-10-26 | Measurement |
|---|---:|
| EXL3 decoder body | **4.499880554 bpw** |
| Serialized model weights | 54 shards, **18.35 GiB** |
| Full evaluated context | **36,864 tokens, Q4 KV** |
| Peak VRAM during full Sixcat | **22,584 MiB** |
| ExLlamaV3 generated decode | **33.82 tok/s** weighted; 34.94 median |
| Sixcat v0.8.3, strict / thinking ON | **104 / 120 = 86.67%** |
| Real locally guarded Python Code | **17 / 20** |

**Qualification:** All 120 Sixcat first responses were graded, including
20 actual guarded Code executions. Two Code responses were truncated after
32,768 generated tokens while reasoning and counted as failures; this
result is **not directly comparable to a fully untruncated run**.

## Model

| | Link / value |
|---|---|
| Original checkpoint | [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) (Apache-2.0) |
| Selected private quant | [Nanbeige4.3-33B-SAGE-EXL3-4.50bpw](https://huggingface.co/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw) |
| Qualified release | Original v6 baseline, **NOT** rejected experimental full-residual QAD |
| Architecture | Llama-style dense 52 layers / 56 attention heads / 8 KV heads |
| Hidden / FFN dimensions | 7,168 / 22,016 |
| Input embeddings and output head | Quantized 8 bit each |
| EXL3 codebook | `mul1`, mixed per-tensor decoder-body bit allocation |

## Quick start

Linux/WSL2 with a compatible NVIDIA CUDA GPU, CUDA-enabled Python with
a working ExLlamaV3 installation, and an isolated TabbyAPI checkout.
The packed weights alone require **18.35 GiB VRAM**; additional memory
is needed for the model runtime and KV cache. A single GPU with at least
24 GiB VRAM is a **practical starting point, not a universal guarantee**.
A multi-GPU machine can select one GPU with `GPU_INDEX`, but **sharding,
CPU offload, CPU-only operation and non-CUDA backends are not configured
or validated by this particular script**. The original benchmarks were
measured exclusively on an RTX 4090.

~~~bash
git clone https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe.git
cd Nanbeige4.3-33B-SAGE-EXL3-recipe

# 1. Existing isolated CUDA + ExLlamaV3 + TabbyAPI runtime
export BASE_PYTHON=/path/to/your/cuda-venv/bin/python
export EXLLAMAV3_SRC=/path/to/exllamav3
export TABBY_DIR=/path/to/tabbyAPI
bash exllamav3-tabby/setup.sh    # read-only adoption; no runtime upgrades

# 2. Authenticated PRIVATE weights, pinned to a verified revision
python3 -m pip install -U huggingface_hub
hf auth login
bash exllamav3-tabby/download.sh

# 3. Verify configuration without using GPU
DRY_RUN=1 PROFILE=long bash exllamav3-tabby/serve.sh

# 4. Serve on your selected idle compatible CUDA GPU, with a safety guard
PROFILE=long bash exllamav3-tabby/serve.sh

# In a separate terminal:
bash exllamav3-tabby/smoke.sh
~~~

**Important runtime pin:** The measured ExLlamaV3 snapshot was a local
development commit `61cfd2cbed196cb05e512ac2201c7f1711140ada`,
not yet fetchable from the public fork at the time of recipe creation.
Current public checkouts are **not automatically score- or speed-qualified**.
Read [installation and provenance](docs/installation.md) before building
a fresh installation. No setup command touches a live runtime or starts GPU work.

## Profiles and hardware-independent configuration

| Profile | Context | KV cache | Batch | Status |
|---|---:|---|---:|---|
| `long` (default) | **36,864** | **Q4** | 1 | **Full Sixcat measured on RTX 4090** |
| `short` | 16,384 | Q8 | 1 | Separate **RTX 4090** smoke; not enough for long thinking-enabled Sixcat Code |
| `compact` | 8,192 | Q4 | 1 | **Untested capacity option** for other GPUs; may still fail if VRAM is insufficient |

Select profiles using `PROFILE=compact bash exllamav3-tabby/serve.sh` or
`PROFILE=long`. On hardware with different memory budgets, change
`MAX_SEQ_LEN`, `CACHE_SIZE`, and `CACHE_MODE` (`FP16`/`Q8`/`Q6`/`Q4`) for
**new unvalidated deployments** (e.g. `MAX_SEQ_LEN=65536 CACHE_MODE=Q8`).
The script requires KV cache size ≥ context and a multiple of 256 tokens.
These overrides **do not inherit** the measured 4090 speed, fit, or score.

`GPU_INDEX=1` selects the second installed NVIDIA GPU on a multi-GPU host
(the script uses exactly that device, not all GPUs). `MODEL_DIR` overrides
the private local checkpoint, `PORT` sets loopback API port (default 8898),
and `RECIPE_HOME` / `STATE_DIR` customize private runtime paths.
`GPU_LOCK_FILE` allows coordination with other cooperative workers.

The model's configuration advertises a **524,288-token positional limit**,
but **524K was NOT tested or shown to fit** on a single RTX 4090.

## Quality against BF16 source

| Separate final-logit split | Top-1 agreement | Mean KL |
|---|---:|---:|
| Held-out | **96.6016%** | **0.01360184** |
| Independent | **96.1719%** | **0.01057153** |

Teacher-relative fidelity is separate from user-task correctness:
the above numbers are not Sixcat scores.

## Complete Sixcat v0.8.3

One RTX 4090, 36,864 tokens Q4 KV, pinned ExLlamaV3 + isolated
TabbyAPI, strict first-response scoring, thinking ON, concurrency 1,
no full-suite timeout, locally executed guarded Code:

| Category | Correct | Score |
|---|---:|---:|
| Knowledge | 13/20 | 65% |
| Math | 20/20 | 100% |
| Truth | 17/20 | 85% |
| Instruction | 19/20 | 95% |
| Code | 17/20 | 85% |
| Tools | 18/20 | 90% |
| **Overall** | **104/120** | **86.67%** |

Two Code responses hit the **32,768-token output ceiling** while still in
reasoning. Both remain failures. No questions were skipped, but the result
must not be advertised as a fully untruncated comparison.

### Inference engine selection

| Local engine | Observed decode speed | Method |
|---|---:|---|
| **ExLlamaV3 + TabbyAPI** | **33.8207 tok/s** | Token-weighted native metrics over 107 longer requests |
| Experimental vLLM 0.32.0 + custom EXL3 plugin | 6.8373 tok/s | Post-first-token SSE over separate 256-token streams, INT4 KV, eager |

**ExLlamaV3 is the qualified operational choice.** These are *different
timing methods and request mixes*, not a perfectly matched kernel A/B;
do not describe the ratio as a controlled hardware speedup. A full vLLM
Sixcat score has **not** been measured.

See [VALIDATION.md](VALIDATION.md) and [benchmark methodology](bench/README.md).

## Exact quantization recipe

- [`quantization/v6-mixed-k-4.50bpw.yaml`](quantization/v6-mixed-k-4.50bpw.yaml):
  **all 364** decoder tensor bit allocations (52 × 7), actual solver output.
- [`quantization/campaign.json`](quantization/campaign.json):
  sanitized actual encoder arguments and source revision.
- [Quantization stages and requirements](quantization/README.md):
  genuine calibrated SAGE capture, measured Hessians, held-out gates and
  why the **later 52/52 locally accepted residual-QAD** experiment was
  **rejected** after global held-out KL/NLL checks.

The source campaign includes private tokenized corpora and a separate
SAGE development checkout; those artifacts are **not** redistributable
from this recipe. The published weight manifest is not an automatic
claim of a byte-identical fresh quantization.

## Repository contents

| Location | Purpose |
|---|---|
| [`exllamav3-tabby/`](exllamav3-tabby/) | Non-mutating runtime adoption, pinned private pack download and foreground owned-process serve |
| [`model.lock.json`](model.lock.json) | Exact private Hub revision + SHA-256 and size of all 63 release files |
| [`runtime.json`](runtime.json) | Qualified and not-yet-public local engine revision, Tabby provenance |
| [`bench/`](bench/README.md) | Full Sixcat summary, reproducibility flags, valid speed-comparison scope |
| [`quantization/`](quantization/README.md) | Original v6 SAGE mixed-bit allocation + reproducibility boundary |
| [`docs/`](docs/installation.md) | Installation, local-only serving, safety and troubleshooting |
| [`tests/`](tests/test_recipe.py) | CPU-only manifest/config/methodology validation |

## Safety, credit, license

Serving binds **127.0.0.1 only**, without authentication as in the
measured local benchmark. **Do not expose this API on a public interface.**
A single-worker preflight refuses an occupied selected GPU; the watchdog
reserves at least **2.5% of that GPU's VRAM or 614 MiB**, whichever is
larger, and can signal only its own child process group. On the original
24,564-MiB RTX 4090 this yields **23,949 MiB maximum**, slightly below
the measured 23,950-MiB safety cutoff. Portable default available-RAM/root-
disk safety floors are **8 GiB / 4 GiB**; tighten them with
`RAM_MIN_AVAILABLE_GIB` and `ROOT_MIN_FREE_GIB`. The **original benchmark
campaign** used 48 GiB available-RAM and 42 GiB disk safety floors.
No other CUDA devices are modified.

Original model: Nanbeige Team; EXL3: Turboderp and collaborators;
TabbyAPI: The Royal Lab; SAGE-EXL3 quant and this recipe: Victor Cruz.
Recipe code/docs **MIT**, original model **Apache-2.0**, runtimes retain
their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
No model weights, teacher checkpoint, corpus, API keys or execution logs
are stored in GitHub.

For contributors and agents: read [AGENTS.md](AGENTS.md).
