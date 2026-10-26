#!/usr/bin/env python3
"""Non-scoring loopback TabbyAPI smoke probe; does not invoke Sixcat."""
import json
import os
import sys
import urllib.error
import urllib.request

port = int(os.environ.get("PORT", "8898"))
if not 1 <= port <= 65535:
    sys.exit("Invalid PORT")
base = f"http://127.0.0.1:{port}/v1"
with urllib.request.urlopen(base + "/models", timeout=20) as response:
    models = json.load(response)["data"]
expected = os.path.basename(os.environ.get("MODEL_DIR", "Nanbeige4.3-33B-SAGE-EXL3-4.50bpw").rstrip("/"))
if expected not in [m["id"] for m in models]:
    sys.exit(f"Expected model '{expected}' missing from /v1/models; models={models}")
body = {"model": expected, "messages": [{"role": "user", "content": "Reply with one short sentence."}],
        "temperature": 0, "max_tokens": 128, "stream": False}
req = urllib.request.Request(base + "/chat/completions", method="POST",
                             data=json.dumps(body).encode("utf8"),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=240) as response:
    result = json.load(response)
assert result.get("choices"), "No completion choices"
assert result["choices"][0].get("message") is not None, "No OpenAI message"
print("PASS: locally generated OpenAI-compatible completion from", expected)
print("NOTE: this is not a scored Sixcat result and does not establish GPU speed.")
