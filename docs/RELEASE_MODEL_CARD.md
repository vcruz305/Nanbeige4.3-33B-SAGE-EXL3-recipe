---
language:
  - en
  - zh
license: apache-2.0
base_model: Nanbeige/Nanbeige4.3-33B
pipeline_tag: text-generation
gated: false
tags:
  - nanbeige
  - exl3
  - exllamav3
  - sage-exl3
  - quantization
  - agentic-coding
  - tool-use
  - reasoning
---
# Nanbeige4.3-33B — SAGE EXL3 4.50 bpw

**Public, freely downloadable EXL3 quantization of [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), produced with SAGE-EXL3 by [Victor Cruz](https://huggingface.co/vcruz305).** No access request or Hugging Face login is required to download the model. The upstream model's **Apache-2.0 license** applies.

**[GitHub: full quantization and inference recipe](https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe)** — includes the exact 364-tensor mixed-bit allocation, 54-shard checksum manifest, local serving instructions and benchmark provenance.

| Release | Value |
|---|---|
| Base checkpoint | [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) |
| Architecture | **33.1B dense Llama-style** reasoning/agent model; **not MoE** |
| EXL3 quant | **4.499880554 decoder-body bpw**, `mul1` codebook, 8-bit embeddings and output head |
| Serialized model weights | **18.35 GiB** (19,700,587,306 bytes), 54 safetensors |
| Tested hardware | **One NVIDIA RTX 4090 (24 GB)**; the model is not 4090-specific |
| Largest successfully scored configuration | **36,864-token context, Q4 KV, batch 1** |
| Observed ExLlamaV3 generation | **33.82 tokens/s token-weighted** across 107 long requests |
| Strict Sixcat v0.8.3 | **104 / 120 = 86.67%**, thinking ON, 20 locally executed Code tests |

## What is Nanbeige4.3-33B?

The **Nanbeige Team** describes Nanbeige4.3-33B as the flagship **teacher agent** of the Nanbeige4.3 family, intended to distill capable agentic behaviors into smaller future models. It is **dense**, meaning all 52 transformer layers participate in each token rather than routing token computation through MoE experts.

The upstream model focuses on **agentic coding, terminal/software engineering, tool use, multi-step automation, and professional cowork/office tasks**, as well as general reasoning, conversation and mathematics. Its published languages are **English and Chinese**. See the [original model card](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) for the authors' evaluation settings, benchmark comparisons and general usage instructions.

### Architecture and model size

| Field | Original model / EXL3 release |
|---|---|
| Architecture | `LlamaForCausalLM`; model type `llama` |
| Parameters | ~33.1 billion, **dense** |
| Layers | **52** |
| Hidden / FFN dimension | **7,168 / 22,016** (SwiGLU/SiLU) |
| Attention | **56 query heads, 8 key/value heads** (GQA), head size **128** |
| Vocabulary | **166,912** padded tokens |
| Position limit advertised by original | **524,288 tokens (512K)** — **not demonstrated on this quant at 24 GB** |
| Input embedding and output head | Untied tables; both **8-bit quantized** in this EXL3 |

### Original model benchmarks — NOT EXL3 measurements

These benchmark figures were **reported by the Nanbeige Team for the original model**, with high thinking effort. They have **not** been reproduced as BF16-vs-EXL3 task comparisons and must not be presented as quantization scores.

| Benchmark | Original Nanbeige4.3-33B card |
|---|---:|
| Terminal-Bench 2.1 | 80.9 |
| DeepSWE | 65.5 |
| SWE-Bench Pro | 60.3 |
| Toolathlon-Verified | 70.4 |
| Automation Bench | 54.2 |
| GPQA Diamond | 89.5 |
| AIME 2026 | 95.2 |

The source authors report harness-specific choices, including `preserve_thinking=true` for the model evaluation. See the [original table and footnotes](https://huggingface.co/Nanbeige/Nanbeige4.3-33B#2-model-performance).

## Download and serve

The model weights are **public and ungated** under Apache-2.0. This is **EXL3**, *not* BF16 Transformers weights or GGUF. Use an inference engine that actually supports this model's EXL3 format and tokenizer.

~~~bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw \
  --local-dir ./Nanbeige4.3-33B-SAGE-EXL3-4.50bpw
~~~

For installation, precise SHA-256 verification, and single-GPU inference, follow the [public ExLlamaV3 + TabbyAPI recipe](https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe). The **18.35 GiB weight footprint excludes KV cache and inference-runtime overhead**; 24 GiB is a practical starting point, not a fit guarantee. The recipe supports selecting an individual compatible NVIDIA CUDA GPU; multi-GPU sharding, CPU-only and non-CUDA runtimes are not automatically configured. **All published speed/fit figures below were measured on a single RTX 4090, not every GPU.**

The original qualified engine used a **local development ExLlamaV3 commit `61cfd2cbed196cb05e512ac2201c7f1711140ada`** that was not confirmed publicly fetchable at release creation; fresh runtime deployments must verify correct behavior rather than assuming benchmark parity.

### Reasoning and tool-call configuration

The **original tokenizer/chat template** supports the following controls. Preserve the bundled native Nanbeige template; using generic Llama/Qwen prompt formatting can change behavior.

- `enable_thinking`: controls thinking on/off for the current response, **on by default** upstream.
- `preserve_thinking`: retains prior assistant reasoning in multi-turn context. Nanbeige recommends **false** for ordinary chat/Q&A and **true** for multi-turn agents, coding or professional tool workflows.
- Passing `tools` enables structured tool messages. Nanbeige recommends `tool_call_format="xml"` for best tool use; `json` is also supported for compatibility.
- The measured TabbyAPI setup used reasoning enabled, XML tool formatting and its `qwen3_coder` tool parser.

Original-author suggested sampling (for the **base model**, not separately optimized for this EXL3): **temperature 1.0/top_p 0.95** for agentic tool workflows; **temperature 0.6/top_p 0.95** for reasoning and chat. The original card recommends extremely large completion budgets, but these **are not validated to fit on a 24 GB GPU in this EXL3 release**.

## SAGE quantization fidelity

### Hugging Face config compatibility

The Hub's generic config parser requires `quantization_config.bits` to be
an integer. This model uses **mixed per-tensor EXL3 precision**, so the
nominal **4.5 bpw** figure is an **average**, not a uniform integer width.
Accordingly, the top-level `config.json` retains
`quantization_config.quant_method: "exl3"` **without a misleading integer**
`bits` field. The EXL3-native **`quantization_config.json`** preserves its
original `bits: 4.5` and packed tensor-storage metadata, and the exact
achieved **4.499880554 bpw** and 364 tensor allocations are documented in
the [SAGE quantization recipe](https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe).
**No model weights, layer bit assignments, or tokenizer files were changed.**

This is the original **qualified v6 baseline** at **4.499880554 decoder-body bpw**, with measured mixed-K allocation and an 8-bit embedding/output head. It is **not** the experimental full-residual QAD re-encode.

| BF16 teacher-relative final-logit split | Full-vocabulary top-1 agreement | Mean KL |
|---|---:|---:|
| Held-out | **96.6016%** | **0.01360184** |
| Independent | **96.1719%** | **0.01057153** |

These numbers describe closeness to the BF16 teacher, **not** task accuracy. The later full-residual QAD research candidate locally passed 52/52 layers but worsened held-out KL and both global NLLs; it was rejected, preserving this higher-fidelity v6 pack.

## RTX 4090 — measured runtime results (other GPUs untested)

| Tested mode | KV | Context | Peak GPU memory | Generated speed |
|---|---|---:|---:|---:|
| Short smoke | Q8 | 16,384 | 21,854 MiB | 32.188 tok/s across **only 15 token intervals** |
| Full Sixcat | **Q4** | **36,864** | **22,584 MiB** | **33.8207 tok/s token-weighted**, **34.94 median**, 107 long server requests |

The full 36K scored run served through pinned ExLlamaV3 + isolated TabbyAPI on one RTX 4090, batch/concurrency 1 and no CPU layer offload. Speed uses native TabbyAPI server metrics, **not** Sixcat wall-clock throughput.

### Complete Sixcat v0.8.3, thinking ON

All **120 first responses** were scored under **strict policy**, 20 per category, with **20 guarded local Python Code executions** and no overall deadline:

| Category | Correct | Score |
|---|---:|---:|
| Knowledge | 13/20 | 65% |
| Math | 20/20 | 100% |
| Truth | 17/20 | 85% |
| Instruction | 19/20 | 95% |
| Code | 17/20 | 85% |
| Tools | 18/20 | 90% |
| **Total** | **104/120** | **86.67%** |

**Important caveat:** Two Code completions hit the **32,768-token output cap while still reasoning**, and remain scored failures. The 120-item evaluation completed, but its score is **not directly comparable to a fully untruncated run**. There were also looping failures. No failed answers were repaired.

An isolated **vLLM 0.32.0 + custom EXL3 plugin** was able to load and decode the same pack with 36K INT4 KV, but an experimental ~6.84 tok/s SSE probe used a different workload/timing method. It does **not** establish an apples-to-apples engine speed ratio, and no full vLLM Sixcat score is claimed. **ExLlamaV3 + TabbyAPI is the qualified operational choice** for the measured setup.

## Additional RTX 4090 context tests (after the scored 36K release)

These are **long-prompt synthetic retrieval smoke tests**, not
higher-context Sixcat scores. The original SAGE-EXL3 model weights were
unchanged. Each successful run actually prefilled a nearly full native
chat-template prompt and returned the correct three synthetic markers:

| Configured context | Actual prompt | K/V cache | Input embedding | Peak VRAM | Synthetic retrieval |
|---:|---:|---|---|---:|---:|
| 53,248 | 52,728 | Q4/Q4 | GPU | 23,120 MiB | 3/3 |
| 65,536 | 65,004 | Q3/Q3 | GPU | 22,980 MiB | 3/3 |
| 81,920 | 81,405 | Q2/Q2 | GPU | 22,680 MiB | 3/3 |
| **131,072** | **130,542** | **Q2/Q2** | **CPU/system RAM** | **23,220 MiB** | **3/3** |

The **128K** configuration uses the original EXL3 weights and offloads
only the **packed 8-bit input embedding** to system RAM, while keeping the
decoder-body weights and Q2 K/V cache on the RTX 4090. The full-length
128K request took **161.055 seconds** and completed without safety events
under a **23,950 MiB** VRAM cutoff. Lower-bit KV and host embedding
placement may affect performance and accuracy; no 128K Sixcat or
full-corpus long-context fidelity result is claimed.

**256K safety outcome:** a 261,619-token prompt was prepared for a
262,144-token Q2/Q2 configuration. Its cache would require **8,320 MiB**,
and projected total VRAM with the host embedding was **27,576 MiB**,
above the safety cap. The preflight **refused GPU launch**. There was
**no 256K inference run** or verified capacity on the RTX 4090.
Reaching 256K with this model requires additional CPU/model/KV offload
support or a larger-memory GPU. See the complete
[context receipts and limitations](https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe/tree/main/bench/context).

### DFlash2 draft — experimental short-speed comparison

The native
[ProCreations Nanbeige4.3-33B DFlash2](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2/tree/exllamav3)
draft was loaded with **this exact 4.50-bpw pack** on RTX 4090 at **8K Q4
target KV**. It reached **42.81 tok/s coding** versus 35.58 plain
(**1.20×**) and **44.30 tok/s agentic** versus 35.57 plain (**1.25×**)
on two short 192-output-token samples. However, the greedy token streams
**diverged** from the no-draft baseline at tokens 113 and 66, so **lossless
output equivalence is not proven** and draft inference is not enabled by
default in the public recipe. These short tests do **not** prove DFlash2
performance or correctness at 128K. The full 120-question Sixcat score
was recorded with **no draft**. See the
[experimental DFlash2 report](https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe/tree/main/bench/dflash2).

## Limits, license and credits

- Original Nanbeige model config advertises **512K context**. The largest *completed scored* context for this quant was **36,864 tokens** on RTX 4090. Do not extrapolate to 512K on a 24 GB GPU.
- Other devices and inference backends must validate VRAM headroom, context length, chat template, KV precision and tool-parser behavior themselves.
- **Original model and upstream benchmark figures:** Nanbeige Team. **SAGE EXL3 quantization, recipe and measurements:** Victor Cruz.
- **License:** Apache-2.0 for the base model/quantized model. Inference engines and GitHub recipe code retain their own licenses.

Original source and credit: [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B). Detailed reproducibility and per-tensor allocation: [Nanbeige4.3-33B-SAGE-EXL3-recipe](https://github.com/vcruz305/Nanbeige4.3-33B-SAGE-EXL3-recipe).
