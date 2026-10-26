DFlash2 speculative decoding — experimental

Draft model: https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2/tree/exllamav3
Draft revision: 4b7c7f8509d5286b7264db096d764b4a163d74f4
Draft weight SHA256: b031ede26614f8800110e062f81fbaf3d1974dfc43bb32f346e6de941dc0efff

Tested October 26, 2026 on one RTX 4090 with the qualified SAGE 4.50-bpw target, pinned local ExLlamaV3, Q4 target KV, FP16 draft KV, 8192 cache tokens, 7 draft tokens per verification round, 192 generated tokens and greedy sampling.

| Prompt | Target only | Target + DFlash2 | Speedup |
|---|---:|---:|---:|
| Coding (430 input tokens) | 35.5754 tok/s | 42.8075 tok/s | 1.2033x |
| Agentic (573 input tokens) | 35.5697 tok/s | 44.2963 tok/s | 1.2453x |

The draft accepted 72 and 75 tokens across 120 and 117 verification rounds respectively. These are real local measurements, not the publisher's speed claims.

IMPORTANT: Generated greedy token sequences differed from the baseline at token 113 (coding) and token 66 (agentic). Lossless output equivalence has NOT been demonstrated. Do not claim lossless speculative decoding or enable it by default. This test does not validate long-context performance or Sixcat scores.

To try the draft, download the author's native ExLlamaV3 branch:

```bash
hf download ProCreations/Nanbeige4.3-33B-DFlash2 --revision exllamav3 --local-dir ./Nanbeige4.3-33B-DFlash2-exllamav3
```

See the [native quickstart](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2/blob/exllamav3/quickstart.py). Its Q8 target cache and 65,536-token settings are NOT the 24 GB tested configuration. Use a controlled 8192-token Q4 target cache for reproduction and ensure GPU headroom.

The original Sixcat 104/120 was measured without DFlash2 and remains unchanged. The complete measured summary is [measured-results.json](measured-results.json).
