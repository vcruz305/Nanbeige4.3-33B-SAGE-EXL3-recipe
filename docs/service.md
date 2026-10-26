# Local-only serving

The server runs in the foreground and never installs a daemon or
stops anyone else's model process. The launcher refuses an occupied
RTX4090 before creating its own process group. It takes an exclusive
file lock, monitors GPU/RAM/disk every 10 seconds, and stops **only its
own** child process group if thresholds are violated.

- GPU VRAM ceiling: 23,950 MiB
- Available host RAM floor: 48 GiB
- Root filesystem free floor: 42 GiB
- Default API: http://127.0.0.1:8898/v1

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
