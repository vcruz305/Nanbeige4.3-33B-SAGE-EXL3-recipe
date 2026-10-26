# Nanbeige4.3-33B SAGE-EXL3 4.50 bpw — model and serving recipe

Run **Nanbeige4.3-33B**, a **33B dense** model, at **4.499880554 body bpw**
on compatible NVIDIA CUDA systems with enough VRAM, using
**ExLlamaV3 + TabbyAPI**. This is the original qualified v6 SAGE
allocation, including the exact per-tensor mixed-K configuration, public
release checksums, and reproducible serving scripts. It is **not limited
to the RTX 4090**. The following benchmark results were **measured on a
single RTX 4090**; other GPUs and operating environments need independent
memory/throughput/compatibility validation.

**Model:** [vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw](https://huggingface.co/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw)
— **PUBLIC AND UNGATED.** Anyone can download the 54 quantized weight
shards without creating an account or requesting access. The GitHub recipe
contains scripts/docs/quality evidence, while the weights reside on Hugging Face.

| Actual results — **RTX 4090** test machine, 2026-10-26 | Measurement |
|---|---:|
| EXL3 decoder body | **4.499880554 bpw** |
| Serialized model weights | 54 shards, **18.35 GiB** |
| Full **Sixcat-scored** context | **36,864 tokens, Q4 KV** |
| Largest verified **synthetic retrieval** context | **131,072 tokens (128K), Q2 KV**, CPU-resident input embedding |
| 128K tested RTX 4090 peak VRAM | **23,220 MiB** (130,542-token prompt, 3/3 markers, **161.055-second** request) |
| Peak VRAM during full 36K Sixcat | **22,584 MiB** |
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
| Public EXL3 quantization | [Nanbeige4.3-33B-SAGE-EXL3-4.50bpw](https://huggingface.co/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw) |
| Qualified release | Original v6 baseline, **NOT** rejected experimental full-residual QAD |
| Architecture | Llama-style dense 52 layers / 56 attention heads / 8 KV heads |
| Hidden / FFN dimensions | 7,168 / 22,016 |
| Input embeddings and output head | Quantized 8 bit each |
| EXL3 codebook | `mul1`, mixed per-tensor decoder-body bit allocation |

## What is Nanbeige4.3-33B?

[**Nanbeige4.3-33B**](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
is a **dense ~33.1-billion-parameter reasoning and agent model** from the
**Nanbeige Team**, built as the flagship teacher for future smaller
Nanbeige4.3 models. It targets **terminal and agentic coding, software
engineering, tool use, multi-step automation, professional cowork/office
tasks**, plus general reasoning, chat and mathematics. The original
model card lists **English and Chinese** support and the **Apache-2.0**
license. It is **not MoE**; all 52 dense transformer layers execute for
each generated token.

| Original model detail | Value |
|---|---|
| Class | `LlamaForCausalLM` / dense decoder-only Transformer |
| Layers / dimensions | 52 / hidden 7,168 / SwiGLU FFN 22,016 |
| Attention | Grouped-query: 56 query heads / 8 KV heads / head dimension 128 |
| Vocabulary | 166,912 padded token IDs |
| Positional context in original model metadata | **524,288 (512K)**, **not verified for this 4.50bpw quant** |
| Chat template | Native Nanbeige thinking and XML/JSON tool-calling template |

The original tokenizer supports `enable_thinking` (**on by default**),
`preserve_thinking` (**false** for general Q&A and **true** for multi-turn
coding/tool workflows per upstream recommendation), and tool calls with
`tool_call_format="xml"` (**recommended upstream**) or `"json"`.
The Nanbeige Team recommends **temperature 1.0, top-p 0.95** for
agentic/tool workflows and **temperature 0.6, top-p 0.95** for
reasoning/chat. Those recommendations describe the **original model**;
actual EXL3 runtime behavior depends on its chat template and sampling
implementation. The upstream model recommends much larger context/output
budgets than were validated on the 4090 in this release.

### Original Nanbeige model-card benchmarks (not quantization scores)

These are **the Nanbeige Team's reported figures for the original model**,
using their specified evaluation harnesses/highest thinking setting, and
**were not measured on this EXL3 checkpoint**. They are included to explain
the model's intended abilities, not to claim quantized benchmark parity.

| Original model benchmark | Original card figure |
|---|---:|
| Terminal-Bench 2.1 | 80.9 |
| DeepSWE | 65.5 |
| SWE-Bench Pro | 60.3 |
| Toolathlon-Verified | 70.4 |
| Automation Bench | 54.2 |
| GPQA Diamond | 89.5 |
| AIME 2026 | 95.2 |

Read the [original model card and benchmark methodology](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
and the [full public EXL3 release card](https://huggingface.co/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw)
for source attribution and limitations. The card is also mirrored in
[docs/RELEASE_MODEL_CARD.md](docs/RELEASE_MODEL_CARD.md) for reference.

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

# 2. Public, ungated weights — anonymous download, pinned/checksum-verified
python3 -m pip install -U huggingface_hub
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
| **`extended`** | **131,072** | **Q2 K/V (2,2)** | 1 | **3/3 actual long-prompt retrieval smoke on RTX 4090, host input embedding**; not Sixcat |

**Runnable 128K / 131,072-token configuration:** once the isolated
ExLlamaV3 + TabbyAPI runtime has been adopted with `setup.sh` and the
public weights have been checksum-verified with `download.sh`, launch:

~~~bash
# The values below match the local RTX 4090 128K test. Adjust available
# RAM / disk floors only to suit the resources of your own machine.
GPU_INDEX=0 PROFILE=extended \
  MAX_SEQ_LEN=131072 CACHE_SIZE=131072 \
  CACHE_MODE=Q2 EMBEDDING_PLACEMENT=cpu \
  RAM_MIN_AVAILABLE_GIB=48 ROOT_MIN_FREE_GIB=42 \
  PORT=8898 bash exllamav3-tabby/serve.sh

# In a separate shell, after the server reports ready:
PORT=8898 bash exllamav3-tabby/smoke.sh
~~~

**Shorter equivalent command with portable host-memory defaults:**
`PROFILE=extended bash exllamav3-tabby/serve.sh`.
`extended` configures a single batch/sequence, a full **131,072-token
Q2 K/V cache** and CPU/system RAM for **only the quantized input
embedding**, while the **same original 4.50-bpw decoder stays on the GPU**.
The validated test prefilled **130,542 actual native chat-template tokens**,
recovered **3 of 3 synthetic markers**, used **23,220 MiB peak RTX 4090
VRAM**, and took **161.055 seconds including prefill and response**. This
is a retrieval smoke test, not an accredited 128K reasoning benchmark.
The script checks model file SHA-256 hashes before running, refuses
other GPU workers, and monitors memory/disk during runtime.

Full setup, examples and comparisons are in
[bench/context/README.md](bench/context/README.md).

Other presets include `PROFILE=compact` and `PROFILE=long`. Override
`MAX_SEQ_LEN`, `CACHE_SIZE`, `CACHE_MODE` (`FP16`, `Q2`/`Q3`/`Q4`/`Q6`/`Q8`
or native per-key/value `2,2`/`3,3` patterns) and
`EMBEDDING_PLACEMENT=cpu` / `gpu` to experiment on other systems.
The script requires the cache to cover context and sizes it in 256-token
pages. A new conservative **VRAM capacity preflight** refuses memory
configurations that are predicted to violate the per-GPU safety ceiling;
it does not claim all settings that pass a prediction are verified.
Other GPUs or settings **do not inherit** the RTX 4090 results.

`GPU_INDEX=1` selects the second installed NVIDIA GPU on a multi-GPU host
(the script uses exactly that device, not all GPUs). `MODEL_DIR` overrides
the locally downloaded model checkpoint, `PORT` sets loopback API port (default 8898),
and `RECIPE_HOME` / `STATE_DIR` customize private runtime paths.
`GPU_LOCK_FILE` allows coordination with other cooperative workers.

The model's configuration advertises a **524,288-token positional limit**,
but **524K was NOT tested or shown to fit** on a single RTX 4090.

### Hugging Face config parsing compatibility

The released model uses **mixed per-tensor EXL3 bits** and has an
**average decoder-body bitrate of 4.499880554 bpw**. The Hugging Face
Hub's generic `quantization_config.bits` field expects an integer; putting
`4.5` there triggered a configuration parsing warning. We therefore
**omit that field only from top-level `config.json`**, while retaining
`quant_method: "exl3"`, the original `quantization_config.json` with
`bits: 4.5` and complete tensor-storage metadata, and the full
[364-tensor mixed-K allocation](quantization/v6-mixed-k-4.50bpw.yaml).

This is a **metadata/schema compatibility correction**, not a new encode.
All **54 safetensors weights remain identical** to the qualified model.
The updated Hub revision and SHA-256 are pinned in
[`model.lock.json`](model.lock.json), and an exact copy of the public
[`config.json`](evidence/config.json) is committed as a regression fixture.

## Extended context and DFlash2 experiments (RTX 4090)

The original **36,864-token Q4 KV** context was the configuration used
for the published full strict Sixcat score. Additional actual single-RTX4090
**long-prompt retrieval smoke tests** have since passed, without changing
the 4.50-bpw weights:

| Configured context | KV bits (K/V) | Actual prompt tokens | Retrieval markers | Peak VRAM |
|---|---|---:|---:|---:|
| **53,248** | **4/4** | 52,728 | **3/3** | 23,120 MiB |
| **65,536** | **3/3** | 65,004 | **3/3** | 22,980 MiB |
| **81,920** | **2/2** | 81,405 | **3/3** | 22,680 MiB |
| **131,072** | **2/2, embedding on CPU** | **130,542** | **3/3** | **23,220 MiB** |

All four used one RTX 4090, a real full-length prompt prefill followed by
a generated answer, and the original **23,950 MiB** safety cutoff. The
128K test kept the 8-bit input embedding on **system RAM** while leaving
the EXL3 decoder on the GPU. These are **synthetic retrieval smoke tests,
not full Sixcat scores**. Lower-bit KV precision may reduce reasoning
quality compared with Q4/Q8, so 128K should not be advertised as fully
quality-qualified. Other GPU capacities require independent testing.

**256K on the RTX 4090:** a 261,619-token prompt was prepared, but the
262,144-token Q2 K/V cache would require **8,320 MiB**. Even with the
input embedding offloaded to RAM, memory preflight estimated **27,576
MiB total VRAM** versus a **23,950 MiB safety cap** and correctly **refused
to start the GPU model**. **256K has NOT been run or validated.** A larger
GPU or true model/KV offloading is required; just increasing the context
number cannot make 256K fit. The model's metadata 512K limit remains
unverified for this EXL3 quantization.

See the [detailed context-test evidence and reproducible settings](bench/context/README.md).

### What would 200K cost on one RTX 4090?

**204,800-token full-context Q2 K/V needs exactly 6,500 MiB of KV memory**
with this ExLlamaV3 implementation. Relative to the measured 128K/Q2
setup (4,160 MiB KV and 23,220 MiB peak GPU), the additional 2,340 MiB
projects the existing all-GPU-decoder 4.50-bpw model at **~25,560 MiB**:
it **would not fit** below our 23,950 MiB VRAM cap without more offloading.
The model is **33.1B dense with 52 attention layers**, so the KV cache is
not small even though it already uses only eight KV heads and 2-bit values.

A separate native pinned-memory sanity test preserved **24/24 baseline
greedy output tokens** after moving the 8-bit output head and five decoder
blocks to CUDA-addressable host memory, saving roughly **2.38 GiB**. But
short-prompt decode fell from ~35 tok/s to **~0.36 tok/s**: this is a poor
daily-serving option. A **new ~3.9-bpw mixed-precision decoder** is worth
evaluating for *usable* 200K Q2 on 24 GB, but is **only an estimate** (not
encoded or scored). See the [200K memory math and candidate bitrate
comparison](bench/context/200K_MEMORY_OPTIONS.md). The original
**4.50-bpw release and its Sixcat scores remain unchanged**.

The independent [DFlash2 draft-model test](bench/dflash2/README.md)
using [ProCreations' native ExLlamaV3 DFlash2 draft](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2/tree/exllamav3)
measured **1.20×** and **1.25×** decoding speedups on short coding/agentic
prompts (Q4 target KV, 8K cache), but **greedy output sequences diverged
from the target-only baseline**. Lossless compatibility is **not proven**;
DFlash2 remains experimental/opt-in and was **not used** in the Sixcat
score or the extended-context tests.

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

The source campaign uses separate tokenized calibration corpora and a
SAGE development checkout; these research inputs are **not** part of the
public model weights or GitHub recipe. All 54 qualified inference weight
shards are freely downloadable on Hugging Face. The published weight manifest is not an automatic
claim of a byte-identical fresh quantization.

## Repository contents

| Location | Purpose |
|---|---|
| [`exllamav3-tabby/`](exllamav3-tabby/) | Non-mutating runtime adoption, pinned **public** pack download and foreground owned-process serve |
| [`model.lock.json`](model.lock.json) | Exact **public** model revision + SHA-256 and size of all 63 release files |
| [`runtime.json`](runtime.json) | Measured development engine revision, Tabby provenance and runtime limits |
| [`bench/`](bench/README.md) | Full Sixcat, experimental DFlash2 and **53K–128K context** receipts, plus 256K safety-preflight outcome |
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
