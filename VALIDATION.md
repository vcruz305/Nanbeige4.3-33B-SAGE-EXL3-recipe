# Validation: Nanbeige4.3-33B SAGE EXL3 on RTX 4090

**Selected weights:** the original qualified v6 EXL3 4.50-bpw model, 54 shards
(19,700,587,306 bytes = 18.35 GiB), NOT the later full-residual QAD experiment.

## Source fidelity against BF16 teacher

| Split | Top-1 agreement | Mean KL (nats) |
|---|---:|---:|
| Held-out | 96.6016% | 0.01360184 |
| Independent | 96.1719% | 0.01057153 |

The quantization allocation is the real per-tensor 364-projection mixed-K
YAML in [quantization](quantization/README.md), with 64 × 1024 calibration
and separate 10 × 1024 held-out/independent splits. Head and embeddings
are 8-bit, body achieved 4.499880554 bpw.

## Full Sixcat v0.8.3 strict first-response run

On one RTX 4090, pinned ExLlamaV3 at `61cfd2cb` with isolated TabbyAPI,
Q4 KV, 36,864 context, 1 concurrent sequence, thinking ON,
full 120 questions including guarded local Code:

| Knowledge | Math | Truth | Instruction | Code | Tools | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 13/20 | 20/20 | 17/20 | 19/20 | 17/20 | 18/20 | **104/120 = 86.67%** |

Verified 120 distinct first responses, 20 each category and no omitted
questions. **Two Code completions were truncated at 32,768 generated tokens
in the thinking channel, and remained failures.** This is a completed strict
score but is *not fully untruncated*.

## GPU and speed

| Run | KV | Context | Peak VRAM | Decode |
|---|---|---:|---:|---:|
| Initial 16K smoke | Q8 | 16,384 | 21,854 MiB | 32.188 tok/s, 15 post-first intervals only |
| Full Sixcat | **Q4** | **36,864** | **22,584 MiB** | **33.8207 weighted tok/s**, 34.94 median, 107 long server requests |

The alternative isolated vLLM 0.32.0 / custom EXL3 plugin used INT4-KV
with Triton eager at 36K and achieved **6.8373 tok/s mean post-first SSE**
in three separate 256-token trials. Native Tabby metrics and SSE wall timing
are **not identical benchmark methods**. ExLlamaV3 is the qualified
operational choice, not proof of a controlled speed ratio.

The model metadata supports a theoretical 524,288 positions; 524K was
**not measured on a 24 GB RTX 4090**.

## Rejected candidate

52/52 local residual-QAD layers passed their limited local checks,
but global BF16 held-out KL worsened **0.01360 → 0.01658** and independent
KL improved **0.01057 → 0.01005** while **both NLL measures worsened**.
This did not justify promotion. The selected Hub model and SHA lock are
the original qualified v6 baseline.

## Runtime caveat

The measured ExLlamaV3 commit was a **local unpushed development**
revision `61cfd2cbed196cb05e512ac2201c7f1711140ada`, not yet
fetchable from the public fork at recipe creation. Do not claim new
current-fork installations reproduce the original score or speeds.
See [runtime.json](runtime.json) and [installation](docs/installation.md).

## Subsequent RTX 4090 context smoke tests (not Sixcat)

Actual Q4 **53,248** / 52,728-input, Q3 **65,536** / 65,004-input,
Q2 **81,920** / 81,405-input, and Q2 **131,072** / 130,542-input tests
completed full-context prefill and generation with **3/3 synthetic retrieval
markers**. Peak VRAM was **23,120 / 22,980 / 22,680 / 23,220 MiB**, respectively.
The 128K mode used a **CPU-resident quantized input embedding**; the
original decoder remains GPU-resident. Original VRAM cutoff was 23,950 MiB.
These runs are **not** full Sixcat or general long-context reasoning quality.

A **262,144-token Q2** attempt prepared 261,619 native prompt tokens but
failed its capacity preflight **before launching a GPU worker**: 8,320 MiB
of KV payload and estimated 27,576 MiB VRAM against the 23,950 MiB cap.
Never claim 256K worked on the 4090 with this 4.50-bpw pack.
See [context receipts and operating instructions](bench/context/README.md).

DFlash2 short native tests: 35.58 -> 42.81 tok/s coding (1.20x) and
35.57 -> 44.30 tok/s agentic (1.25x) with a separate 1.96GB draft.
Greedy sequences diverged, so lossless equivalence remains unverified.
See [DFlash2 measured evidence](bench/dflash2/README.md).
