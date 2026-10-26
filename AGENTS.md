# Agent instructions for this recipe

- This repository documents the qualified **v6 baseline** 4.50 body bpw;
  the later full-residual QAD candidate was NOT selected.
- Never commit model weights, source BF16 checkpoints, private tokenized
  corpora, credentials, user/private conversations, logs or runtime caches.
- The Hugging Face model remains PRIVATE until separately authorized.
  Public recipe docs are not permission to publish model shards.
- Never claim an incomplete Sixcat score, skip Code, disable thinking,
  or repair the first response. Real verified score: 104/120 with two
  Code completions truncated at 32,768 tokens.
- Do not overlap RTX4090 GPU workers, alter imported code, kill
  unrelated PIDs, relax 23,950 MiB GPU / 48 GiB RAM / 42 GiB disk
  thresholds, or rent paid compute.
- Never claim a current runtime is the original unpublished local
  ExLlamaV3 `61cfd2cb` pin without proof.
- Test changes CPU-only first: `python3 -m unittest discover -s tests -v`,
  `bash -n` on all scripts and `DRY_RUN=1 bash exllamav3-tabby/serve.sh`.
  New GPU results need their own receipts, not recycled benchmarks.
