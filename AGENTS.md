# Agent instructions for this recipe

- This repository documents the qualified **v6 baseline** 4.50 body bpw;
  the later full-residual QAD candidate was NOT selected.
- Never commit model weights, source BF16 checkpoints, private tokenized
  corpora, credentials, user/private conversations, logs or runtime caches.
- The user authorized this EXL3 model's public release; Hugging Face
  is **public, ungated**, while this GitHub repository holds no weights.
  Do not revert visibility or change unrelated repositories.
  Never expose private calibration corpora, credentials or internal traces.
- Never claim an incomplete Sixcat score, skip Code, disable thinking,
  or repair the first response. Real verified score: 104/120 with two
  Code completions truncated at 32,768 tokens.
- Do not overlap workers on the selected CUDA GPU, alter imported code, kill
  unrelated PIDs, relax the selected device VRAM-reserve guard, disable RAM/disk
  safety checks, or rent paid compute.
- Never claim a current runtime is the original unpublished local
  ExLlamaV3 `61cfd2cb` pin without proof.
- Test changes CPU-only first: `python3 -m unittest discover -s tests -v`,
  `bash -n` on all scripts and `DRY_RUN=1 bash exllamav3-tabby/serve.sh`.
  New GPU results need their own receipts, not recycled RTX 4090 benchmarks.

- 128K/Q2 on a 24GB RTX 4090 is a verified synthetic long-prompt
  retrieval **smoke test** only, with CPU-resident input embedding and
  23,220 MiB peak VRAM. Do not claim 128K Sixcat or Q2 quality parity.
- A 256K/Q2 attempt was rejected by memory preflight at an estimated
  27,576 MiB; **no 256K decoding was performed**. Never bypass the
  GPU cap or imply that 256K worked on this 4090.
- Experimental ProCreations DFlash2 trials showed 1.20-1.25x faster
  short decoding but **divergent greedy token sequences**. Do not
  enable draft by default or assert lossless equivalence without repair.
