# Troubleshooting

| Symptom | Verified or safe next action |
|---|---|
| HF 401/403 | The model is public and ungated. Check the exact model ID, Hub status, any proxy/cache credential interference, and retry anonymously; no access request is required |
| Shard missing or hash mismatch | Re-download the **locked public revision**, then verify all 63 SHA-256 hashes. Do not delete unrelated files |
| Sixcat Code HTTP400 “65 pages / 64 available” | 16K cannot handle long reasoning. Use measured **36,864 context Q4 KV** instead, not reduced Code token limits |
| Q8 at 36K OOM | Not qualified; Q8 **16K** or Q4 **36K** are the measured combinations |
| GPU VRAM reaches dynamic reserved ceiling | Owned safety watcher stops only the launched model server; it reserves ≥2.5% or 614 MiB, whichever is larger |
| Source model advertises 524K | That is metadata, NOT a demonstrated 24GB RTX4090 context |
| New upstream checkout behaves differently | Exact measured ExLlamaV3 `61cfd2cb` was not fetchable publicly at release. Re-test chat template, tool calls and reasoning, don't copy past score |
| Sixcat run produces partial JSONL | Only a full 120/120 validated score is a real published result; Code must execute locally |
| QAD per-layer looks better, global metrics worse | Keep original v6 pack. Experimental full-residual QAD was rejected |
| Runtime sees busy selected GPU | Choose a different `GPU_INDEX`, or do not start. Never kill unrelated jobs |
| HF "Configuration Parsing Warning: quantization_config.bits must be an integer" | Fixed in public config revision `32ec2b7`: the top-level `config.json` omits fractional `bits`, but EXL3-native `quantization_config.json` retains nominal `bits: 4.5` and per-tensor metadata. Upgrade the local `config.json` without replacing weight shards. |

If the selected GPU does not have enough VRAM for the 18.35-GiB
weights plus runtime/KV buffers, the single-device runner will refuse
to launch or fail safely. This does **not** mean the model is compatible
with every NVIDIA GPU, and other GPUs have **no inherited scores**.
