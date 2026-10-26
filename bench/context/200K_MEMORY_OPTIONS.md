# 200K context — single-RTX4090 memory engineering

This document distinguishes **measured 128K results** from **model-size projections** and experimental CPU weight pinning. The original model is **33B dense**, not a small MoE: it has 52 attention layers, eight KV heads and 128 dimensions per KV head. Its KV cache is large even at 2 bits.

## Native ExLlamaV3 KV geometry

The installed `CacheLayer_quant` packs 32 K/V features per quantization group and stores per-group FP16 scales. At **Q2 for both keys and values**:

- 1,024 KV features × 2 bits = 256 bytes for keys, and 256 for values, per transformer layer and token
- Two FP16 scale arrays, 32 entries each = 128 more bytes per layer/token
- 640 bytes per layer/token × 52 layers = **33,280 bytes/token (32.5 KiB/token)**

| Cache allocation | Q2 K/V | Q3 K/V | Q4 K/V |
|---:|---:|---:|---:|
| 131,072 tokens (128K) | 4,160 MiB | 5,824 MiB | 7,488 MiB |
| 204,800 tokens (200K) | **6,500 MiB** | 9,100 MiB | 11,700 MiB |
| 262,144 tokens (256K) | 8,320 MiB | 11,648 MiB | 14,976 MiB |

Q2 is the **lowest native KV precision supported** in this installed ExLlamaV3 cache. Do not assume it matches Q4 quality: only the full 36K Q4 configuration has been evaluated by Sixcat.

## Existing 4.50-bpw model at 200K, no extra offloading

The observed 128K/Q2 RTX4090 run used **23,220 MiB** at peak with the quantized input embedding in RAM. A 200K/Q2 cache needs **2,340 MiB more** than the 128K cache, predicting approximately **25,560 MiB** with everything else unchanged: **1,610 MiB above** the 23,950 MiB safety limit. This means a simple 200K context knob change is unsafe.

## New target model bitrate — *projection, not a completed encode*

The SAGE v6 decoder body contains **30,725,373,952 parameters**. Holding 8-bit input embedding/output head unchanged, and changing **only average decoder-body bpw**, gives the following first-order byte budget. Uses 128K observed peak as baseline and accounts for the 200K additional Q2 KV allocation.

| Candidate decoder-body bpw | Estimated saved weight memory | Projected RTX4090 200K/Q2 peak | Safety margin vs 23,950 MiB |
|---|---:|---:|---:|
| **4.49988** (existing) | 0 MiB | **25,560 MiB** | 1,610 MiB **over** |
| 4.10 | 1,465 MiB | 24,095 MiB | 145 MiB **over** |
| 4.00 | 1,831 MiB | 23,729 MiB | **221 MiB**, too tight |
| 3.95 | 2,014 MiB | 23,546 MiB | 404 MiB, borderline |
| **3.90** | **2,197 MiB** | **23,363 MiB** | **587 MiB**, better |

These are **estimates**, not a measured 3.90/4.0 EXL3 checkpoint, not encoded files, not validated memory use, and not established fidelity. Kernel scratch buffers, byte alignment and module allocation can alter the real peak. Reducing model body from 4.50 to 3.90 saves roughly 2.1 GiB, but fidelity must be independently checked.

## Pinned-memory experiment with the unchanged 4.50-bpw model

ExLlamaV3 on this particular WSL/RTX4090 environment supports **CUDA-addressable pinned-host aliases** for EXL3 linear trellis weights. A 1 MiB mapped-host correctness probe passed. In a separate **8K/Q4 short-prompt** test, host-pinning the 8-bit output head freed about 1.1 GiB; five decoder blocks freed another 1.26 GiB. Greedy output tokens remained identical to the baseline **over 24 tokens**.

This memory-saving path is **extremely slow**:

| 24-token native short test | Post-first-token decode |
|---|---:|
| Target GPU-resident baseline | ~35 tokens/sec |
| Output head pinned in RAM | **1.23 tokens/sec** |
| Output head + five full decoder blocks pinned | **0.36 tokens/sec** |

Do not extrapolate short-prompt throughput to 200K prefill; the pinned-host path uses PCIe and can be much slower than a GPU-resident model. A 200K context with pinned layers is an **experimental capacity proof only**, not a recommended deployment. Full-context results, if obtained, will be recorded separately.

## Recommended direction

For **practical** 200K inference on a single 24 GiB RTX 4090, keep the original **4.50-bpw quality-qualified release unchanged**, and test a separate **~3.9 decoder-body-bpw SAGE candidate** with an 8-bit output head and Q2 K/V. A 4.0-bpw candidate might fit but the projected ~221 MiB of headroom is too small to recommend without substantial additional verified memory reductions.

Require independent BF16 fidelity measurement (KL, top-1, NLL), real long-context retrieval and useful end-to-end throughput, explicit VRAM monitoring, and all 120 Sixcat tasks before claiming parity with the existing release. A 200K Q4 cache (11.7 GiB) would require a larger-memory system or dramatically different inference architecture; squeezing decoder bits further cannot guarantee acceptable quality.

The 200K predictions **do not modify** the current 4.50-bpw model weights.
