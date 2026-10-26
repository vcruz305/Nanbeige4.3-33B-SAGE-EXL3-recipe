# Original qualified SAGE EXL3 v6 quantization

The exact **364 tensor mixed-bit assignments** are in
[`v6-mixed-k-4.50bpw.yaml`](v6-mixed-k-4.50bpw.yaml).
[`campaign.json`](campaign.json) records source revision and
encoder arguments with private local absolute paths removed.

The actual BF16 master was [Nanbeige/Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B)
at `c1093c501cc6460427c0ac634db7223d0450c0ee`. Architecture:
33B dense, 52 layers, 56 Q / 8 KV attention heads, 7,168 hidden,
22,016 SwiGLU intermediate, 166,912 padded vocabulary, separate input
and output tables (both encoded at 8-bit).

## Staged process

1. Prepare four native-template, mutually disjoint token sets:
   **64×1024 calibration**, **10×1024 sensitivity**, **10×1024 held-out**
   and **10×1024 independent**. Audit all 256-token sliding windows
   for leakage; verify native thinking/tool delimiters. Calibration
   included shell error/recovery and tool-related tokens.
2. Capture genuine BF16 teacher activations and shaped Hessian/salience
   statistics on the actual 52×7 dense tensors. Measure real sensitivity,
   including boundary-weighted final-logit KL, and compile the per-tensor
   `marginal_with_pair_exchange` allocation.
3. EXL3-convert to nominal **4.50 body bpw**, actually **4.499880554**,
   `mul1` codebook, head bits **8**, embeddings **8**, shard size **384 MB**,
   Hessian regularization 0.025, calibration 64×1024, GPU 0.
4. Score untouched BF16-vs-quantized held-out and independent full-vocab
   logits, inspect cosine over reconstructed actual weights, verify
   24 GB runtime with correct K/V, then run local agent tests and
   full 120-item Sixcat evaluation.
5. Accept **only** the qualified original v6 baseline. Do not promote
   exploratory 52-layer full-residual QAD: held-out KL and both NLL
   measures worsened.

The exact experimental corpus, BF16 teacher shards, checkpoints, and
private SAGE driver **are not redistributed in this GitHub repo**.
A fresh corpus or different decoder is a new experiment; the supplied
YAML does not magically recreate the original measured fidelity.

## Owner-only source campaign

The original staged driver lives in a separate SAGE development checkout.
If you possess its dependencies and original locally licensed data:

~~~bash
export SAGE_CAMPAIGN_ROOT=/path/to/SAGE-EXL3-nanbeige43-33b-4090
bash quantization/run-stages.sh preflight
bash quantization/run-stages.sh status
# Only after source/corpora are ready and the chosen GPU is idle:
bash quantization/run-stages.sh capture
bash quantization/run-stages.sh measure-seed
bash quantization/run-stages.sh measure-pilot
bash quantization/run-stages.sh measure-finish
bash quantization/run-stages.sh compile
bash quantization/run-stages.sh reference
bash quantization/run-stages.sh encode-qad
bash quantization/run-stages.sh gate-qad
~~~

The original stage driver pins source teacher EXL3 at `02aef45c`,
encoder at `4d98d39a`, inline-QAD encoder `61cfd2cb` and refuses
a busy GPU; some of those are local research checkout revisions.
The wrapper merely invokes a named stage in that **separate checkout**;
it does not create models or download/upload data by itself.

Original structural IDs: 166100 start, 166101 end, 166102 EOT,
166103/166104 think start/end, 166105/166106 tool start/end.
The `tool_response` tag is **not** one token. Using a generic model's
delimiter IDs produces invalid calibration/gating.
