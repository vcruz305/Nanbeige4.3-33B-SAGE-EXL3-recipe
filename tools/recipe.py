#!/usr/bin/env python3
"""Safe, local Nanbeige EXL3 download, runtime and launch helpers.

No GPU is accessed for 'config', 'verify' or 'setup --adopt'.
This file never manages someone else's processes and does not upload weights.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "model.lock.json"
MODEL_ID = "vcruz305/Nanbeige4.3-33B-SAGE-EXL3-4.50bpw"
MODEL = Path(os.environ.get("MODEL_DIR", str(Path.home() / "models" / MODEL_ID.split("/")[-1]))).expanduser().absolute()
RECIPE_HOME = Path(os.environ.get("RECIPE_HOME", str(Path.home() / "nanbeige43-exl3-4090"))).expanduser().absolute()
STATE = Path(os.environ.get("STATE_DIR", str(ROOT / ".state"))).expanduser().absolute()
GPU_LIMIT_MIB = 23950
MIN_AVAILABLE_RAM = 48 * 1024**3
MIN_FREE_ROOT = 42 * 1024**3
PROFILES = {
    "long": {"context": 36864, "cache": "Q4"},
    "short": {"context": 16384, "cache": "Q8"},
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json_private(path: Path, data: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp-")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        os.replace(name, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_release_metadata(lock):
    if lock["repo_id"] != MODEL_ID or len(lock["files"]) != 63:
        raise ValueError("Unexpected release manifest")
    if sum(x["path"].endswith(".safetensors") for x in lock["files"]) != 54:
        raise ValueError("Expected 54 weight shards")
    if sum(x["size"] for x in lock["files"] if x["path"].endswith(".safetensors")) != 19700587306:
        raise ValueError("Expected weight-bytes do not match qualified release")
    names = [x["path"] for x in lock["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate manifest file")
    for item in lock["files"]:
        name = item["path"]
        if not name or Path(name).is_absolute() or ".." in Path(name).parts or "/" in name:
            raise ValueError("Unsafe manifest path")
        if item["size"] < 1 or len(item["sha256"]) != 64:
            raise ValueError("Bad hash/size entry")


def sha256_file(path: Path):
    dig = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def verify_model(model_dir: Path = MODEL, *, full_sha: bool = True, lock=None):
    lock = lock or read_json(LOCK)
    validate_release_metadata(lock)
    if not model_dir.is_dir():
        raise ValueError("Model directory missing: " + str(model_dir))
    expected = {f["path"] for f in lock["files"]}
    actual_shards = {f.name for f in model_dir.glob("*.safetensors")}
    expected_shards = {p for p in expected if p.endswith(".safetensors")}
    if expected_shards != actual_shards:
        raise ValueError("Weight shard set mismatch; missing/extraneous: " + str(sorted(expected_shards ^ actual_shards)[:5]))
    for entry in lock["files"]:
        path = model_dir / entry["path"]
        if not path.is_file() or path.is_symlink() and not path.resolve().is_file():
            raise ValueError("Missing model file: " + entry["path"])
        if path.stat().st_size != entry["size"]:
            raise ValueError("Wrong size for " + entry["path"])
        if full_sha and sha256_file(path) != entry["sha256"]:
            raise ValueError("SHA-256 mismatch for " + entry["path"])
    cfg = read_json(model_dir / "config.json")
    q = cfg.get("quantization_config", {})
    if cfg.get("architectures") != ["LlamaForCausalLM"] or cfg.get("num_hidden_layers") != 52:
        raise ValueError("Not the expected 52-layer Nanbeige model")
    if q.get("quant_method") != "exl3" or float(q.get("bits", 0)) != 4.5 or q.get("codebook") != "mul1":
        raise ValueError("Not the selected 4.50bpw EXL3 release")
    if not (model_dir / "tokenizer_config.json").is_file():
        raise ValueError("Missing native tokenizer/chat template")
    return {"files": len(expected), "shards": len(expected_shards), "shard_bytes": lock["weight_bytes"], "sha256_checked": full_sha}


def config(profile: str = "long", *, model_dir: Path = MODEL, port: int = 8898):
    if profile not in PROFILES:
        raise ValueError("Only validated 'long' (36K Q4) and 'short' (16K Q8) profiles are supported")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    v = PROFILES[profile]
    return {
        "network": {
            "host": "127.0.0.1", "port": port, "disable_auth": True,
            "disable_fetch_requests": True, "api_servers": ["oai"],
        },
        "model": {
            "backend": "exllamav3",
            "model_dir": str(model_dir.parent),
            "model_name": model_dir.name,
            "max_seq_len": v["context"],
            "cache_size": v["context"],
            "cache_mode": v["cache"],
            "max_batch_size": 1,
            "chunk_size": 512,
            "gpu_split_auto": False,
            "cpu_moe_offload_layers": 0,
            "tool_format": "qwen3_coder",
            "template_vars_default": {"tool_call_format": "xml"},
            "reasoning": True,
        },
        "memory": {"cuda_malloc_async": False},
        "logging": {
            "log_prompt": False, "log_requests": False, "log_chat_completion_requests": False,
        },
    }


def yaml_from_mapping(values):
    # Standard JSON is a valid YAML document. TabbyAPI uses PyYAML to load config.
    return json.dumps(values, indent=2, ensure_ascii=False) + "\n"


def gpu_binary():
    for executable in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
        if os.path.isfile(executable):
            return executable
    found = shutil.which("nvidia-smi")
    if not found:
        raise RuntimeError("nvidia-smi missing")
    return found


def gpu_used_mib():
    proc = subprocess.run([gpu_binary(), "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=10, check=True)
    lines = [x for x in proc.stdout.strip().splitlines() if x.strip()]
    if len(lines) != 1:
        raise RuntimeError("Require a single visible NVIDIA GPU: CUDA_VISIBLE_DEVICES=0")
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 3 or "4090" not in parts[0]:
        raise RuntimeError("Recipe verified only on RTX 4090; actual GPU: " + parts[0])
    used, total = (int(x) for x in parts[1:])
    if total < 24000:
        raise RuntimeError("Unexpected GPU VRAM amount")
    return used


def available_ram():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not available")


def safe_resources(*, before_load=False):
    if available_ram() < MIN_AVAILABLE_RAM:
        raise RuntimeError("Available RAM below 48 GiB safety floor")
    if shutil.disk_usage("/").free < MIN_FREE_ROOT:
        raise RuntimeError("Root filesystem below 42 GiB free")
    used = gpu_used_mib()
    if used > (1536 if before_load else GPU_LIMIT_MIB):
        raise RuntimeError("GPU VRAM safety cutoff crossed: " + str(used) + " MiB")
    return used


def record_runtime(args):
    if args.adopt:
        tabby = Path(os.environ.get("TABBY_DIR", "")).expanduser()
        exl = Path(os.environ.get("EXLLAMAV3_SRC", "")).expanduser()
        py = Path(os.environ.get("BASE_PYTHON", sys.executable)).expanduser()
        if not tabby.is_dir() or not (tabby / "main.py").is_file():
            raise ValueError("TABBY_DIR must be an existing TabbyAPI checkout")
        if not exl.is_dir() or not (exl / "exllamav3").is_dir():
            raise ValueError("EXLLAMAV3_SRC must be an existing source checkout")
        if not py.is_file():
            raise ValueError("BASE_PYTHON is not a working Python executable")
        def head(path):
            p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
            return p.stdout.strip() if p.returncode == 0 else "archive-no-git"
        metadata = {"python": str(py.resolve()), "tabby_src": str(tabby.resolve()),
                    "exllama_src": str(exl.resolve()), "exllama_commit": head(exl),
                    "tabby_commit": head(tabby), "mode": "adopt", "runtime_qualified_here": False}
        write_json_private(STATE / "runtime.json", metadata)
        print("ADOPTED: existing source paths recorded without editing them; serving still requires model validation.")
        if metadata["exllama_commit"] != read_json(ROOT / "runtime.json")["tested_exllamav3_commit"]:
            print("NOTICE: checkout differs or is an archive; benchmark parity not established.")
        return
    raise ValueError("Only --adopt is supported automatically. Install TabbyAPI/ExLlamaV3 in an isolated environment first; see docs/installation.md.")


def release_download(args):
    lock = read_json(LOCK)
    validate_release_metadata(lock)
    if not args.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub: python3 -m pip install -U huggingface_hub") from exc
        print("Authenticated PRIVATE HF download; never save tokens in this repo.")
        snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"],
                          local_dir=str(MODEL), allow_patterns=[e["path"] for e in lock["files"]],
                          max_workers=1)
    print("Verifying all 63 files by SHA-256; this reads 18.35 GiB locally.")
    print(json.dumps(verify_model(full_sha=True), indent=2))


def guarded_stop(proc):
    # Only the session leader created by this invocation, never a foreign GPU job.
    if proc.poll() is not None:
        return
    try:
        if os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            if os.getpgid(proc.pid) == proc.pid:
                os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def serve(args):
    # A new local install is NOT automatically the same executable as the qualified run.
    metadata = read_json(STATE / "runtime.json")
    py = Path(metadata["python"])
    tabby = Path(metadata["tabby_src"])
    exl = Path(metadata["exllama_src"])
    if not py.is_file() or not (tabby / "main.py").is_file() or not (exl / "exllamav3").is_dir():
        raise RuntimeError("Runtime paths from setup --adopt are missing")
    result = verify_model(full_sha=os.environ.get("VERIFY_SHA256", "1") != "0")
    STATE.mkdir(parents=True, exist_ok=True)
    lockpath = Path(os.environ.get("GPU_LOCK_FILE", str(RECIPE_HOME / "gpu-exclusive.lock"))).expanduser().absolute()
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("GPU lock is busy; no overlapping model worker allowed")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        initial = safe_resources(before_load=True)
        run = STATE / ("run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()))
        run.mkdir(mode=0o700, parents=True, exist_ok=False)
        cfg = config(args.profile, model_dir=MODEL, port=args.port)
        cfgfile = run / "config.yml"
        cfgfile.write_text(yaml_from_mapping(cfg), encoding="utf8")
        cfgfile.chmod(0o600)
        write_json_private(run / "provenance.json", {
            "model": MODEL_ID, "model_revision": read_json(LOCK)["revision"],
            "profile": args.profile, "runtime": metadata, "verification": result,
            "before_load_gpu_mib": initial, "qualified_runtime_exact_match": (
                metadata.get("exllama_commit") == read_json(ROOT / "runtime.json")["tested_exllamav3_commit"]),
        })
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "0", "EXL3_EMBED_TABLE_CUDA": "1",
                    "EXL3_GRAPHS": "0", "EXL3_QT_OPTIMIZED": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
                    "PYTHONPATH": str(exl) + os.pathsep + str(tabby) + os.pathsep + env.get("PYTHONPATH", "")})
        logpath = run / "tabby.log"
        print(f"Launching ONLY owned TabbyAPI process, profile={args.profile}, GPU cutoff={GPU_LIMIT_MIB} MiB.")
        print(f"OpenAI API: http://127.0.0.1:{args.port}/v1; unauthenticated LOOPBACK ONLY. Log: {logpath}", flush=True)
        with logpath.open("w", encoding="utf8") as log:
            proc = subprocess.Popen([str(py), "main.py", "--config", str(cfgfile)],
                                    cwd=str(tabby), env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            stopped = False
            def request_stop(_signum, _frame):
                nonlocal stopped
                stopped = True
                guarded_stop(proc)
            old_term = signal.signal(signal.SIGTERM, request_stop)
            old_int = signal.signal(signal.SIGINT, request_stop)
            failures_since = None
            try:
                while proc.poll() is None and not stopped:
                    try:
                        safe_resources()
                        failures_since = None
                    except Exception as exc:
                        # A genuine breach terminates only the owned Tabby process group.
                        if "safety cutoff" in str(exc) or "below" in str(exc):
                            print(f"SAFETY STOP (owned process only): {exc}", file=sys.stderr, flush=True)
                            guarded_stop(proc)
                            raise RuntimeError(str(exc))
                        failures_since = failures_since or time.monotonic()
                        if time.monotonic() - failures_since >= 60:
                            guarded_stop(proc)
                            raise RuntimeError(f"Monitoring unavailable for 60 s: {exc}")
                    time.sleep(10)
            finally:
                guarded_stop(proc)
                signal.signal(signal.SIGTERM, old_term)
                signal.signal(signal.SIGINT, old_int)
            raise SystemExit(proc.returncode or 0)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description="Nanbeige EXL3 4090 recipe controls")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("config")
    p.add_argument("--profile", choices=PROFILES, default=os.environ.get("PROFILE", "long"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8898)))
    s = sub.add_parser("setup")
    s.add_argument("--adopt", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--verify-only", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--fast", action="store_true", help="size+architecture only; weaker than SHA-256")
    server = sub.add_parser("serve")
    server.add_argument("--profile", choices=PROFILES, default=os.environ.get("PROFILE", "long"))
    server.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8898)))
    args = parser.parse_args()
    try:
        if args.command == "config":
            print(yaml_from_mapping(config(args.profile, port=args.port)), end="")
        elif args.command == "setup":
            record_runtime(args)
        elif args.command == "download":
            release_download(args)
        elif args.command == "verify":
            print(json.dumps(verify_model(full_sha=not args.fast), indent=2))
        elif args.command == "serve":
            serve(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
