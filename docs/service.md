# Local-only serving

The server runs in the foreground and never installs a daemon or
stops anyone else's model process. It supports machines with one or
multiple NVIDIA CUDA GPUs; `GPU_INDEX` selects **one** physical GPU.
The launcher refuses that GPU if occupied, takes a cooperative exclusive
lock, monitors memory and disk every 10 seconds, and stops only its own
child process group on a safety breach.

- VRAM ceiling: selected GPU total minus at least **2.5% or 614 MiB**;
  the original RTX 4090 cap remains **23,949 MiB** (below the tested
  23,950-MiB safety cutoff)
- Available host RAM floor: 8 GiB by portable default
- Root filesystem free floor: 4 GiB by portable default
- Set `RAM_MIN_AVAILABLE_GIB=48 ROOT_MIN_FREE_GIB=42` to retain the
  original campaign's stricter host-memory and disk headroom floors
- Default API: http://127.0.0.1:8898/v1

GPU model, VRAM total and selected index, input-embedding residency,
and a **conservative KV/model VRAM preflight estimate** are recorded in
the private local runtime receipt. The preflight blocks likely-OOM
configurations (including 256K Q2 on a 24 GB RTX 4090) before any new
GPU process is started. Actual runtime memory is still monitored every
10 seconds and only the owned process group may be terminated. The model still requires ~18.35 GiB of weights
plus runtime overhead and a KV cache on **one** selected GPU; multi-GPU
memory pooling, CPU offload and non-NVIDIA backends are not configured.

The actual Tabby experiment was unauthenticated on **loopback only**;
the recipe also fixes the bind address to 127.0.0.1. Anyone with
access to localhost can invoke it. For another host, use a secure
SSH tunnel or authenticated reverse proxy, **never expose it directly**:

~~~bash
ssh -L 8898:127.0.0.1:8898 USER@YOUR_GPU_HOST
~~~

Local runtime receipts, logs and state are kept under ignored
`.state/`. A shared `GPU_LOCK_FILE` can coordinate other
cooperating inference jobs. A separate GPU-idle check protects against
noncooperating processes. No background service is enabled.
