# Benchmarks and limitations

This model recipe can be used on other compatible CUDA GPUs, but
**all benchmark numbers below are from the original RTX 4090**, preserved in
[`measured-results.json`](measured-results.json). This repo deliberately
excludes raw challenge prompts, full inference transcripts and
model weights (the weights are available separately on the public,
ungated Hugging Face model repository).

## Full Sixcat v0.8.3

After the selected model is actually serving from a tested runtime,
install/tag [Sixcat](https://github.com/vcruz305/sixcat) at `v0.8.3`
and execute the strict full suite. The original run used these exact
scoring flags (port and model name adjusted for this recipe):

~~~bash
python -m sixcat \
  --base-url http://127.0.0.1:8898/v1 \
  --model Nanbeige4.3-33B-SAGE-EXL3-4.50bpw \
  --artifact-id Nanbeige43-v6-4.50bpw-new-device-36k-q4 \
  --policy strict --thinking on --limit 20 --concurrency 1 \
  --max-minutes 0 --ctx 36864 --request-timeout 3600 \
  --out ./nanbeige-sixcat-v083-full.json --no-resume
~~~

The original server identified its pack as `output-qad-4.50` at local
port 18086. Read your own `/v1/models` before invoking the new server.
Sixcat runs guarded Python Code locally: review its execution environment.
Some hard reasoning items run over 15 minutes. Do not impose an arbitrary
30-minute evaluation deadline or count partial runs as complete results.

Actual verified full score: **104/120 (86.67%)**; 20 graded per
Knowledge/Math/Truth/Instruction/Code/Tools. Two Code items exhausted
their 32,768-token reasoning budget and counted as failures; therefore
the result is **not directly comparable to fully untruncated runs**.

## Engine performance

- ExLlamaV3 + TabbyAPI: **33.8207 tokens/s weighted** across 107 long
  server-side generations; 34.94 median, Q4 KV 36,864 context.
- Experimental vLLM 0.32.0 + EXL3 plugin: **6.8373 tok/s mean SSE
  post-first-token** over three 256-token prompts (1K, 4K, 8K),
  36K INT4 KV with Triton and CUDA graphs off; 16K prompt used 16 tokens.

These have different request sets and timing methods, so the engine
choice is operational, **not a controlled like-for-like speed ratio**.
A full vLLM Sixcat evaluation was not performed.
