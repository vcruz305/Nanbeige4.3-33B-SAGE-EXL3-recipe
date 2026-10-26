import hashlib
import importlib.util
import json
import os
import subprocess
from unittest import mock
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recipe", ROOT / "tools" / "recipe.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class ManifestTests(unittest.TestCase):
    def test_release_manifest_identity(self):
        lock = json.loads((ROOT / "model.lock.json").read_text())
        recipe.validate_release_metadata(lock)
        self.assertEqual(lock["shard_count"], 54)
        self.assertEqual(lock["weight_bytes"], 19700587306)
        self.assertTrue(lock["private_at_release"])
        self.assertEqual(lock["revision"], "b273927de2b1c93e51f4d25707613cc0c28e3283")

    def test_checksum_helper(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "fixture"
            p.write_bytes(b"test-only")
            self.assertEqual(recipe.sha256_file(p), hashlib.sha256(b"test-only").hexdigest())

    def test_profiles_are_fail_closed(self):
        cfg = recipe.config("long", port=8898, env={})
        self.assertEqual(cfg["model"]["max_seq_len"], 36864)
        self.assertEqual(cfg["model"]["cache_mode"], "Q4")
        self.assertEqual(cfg["model"]["max_batch_size"], 1)
        self.assertEqual(cfg["network"]["host"], "127.0.0.1")
        self.assertTrue(cfg["network"]["disable_fetch_requests"])
        self.assertEqual(json.loads(recipe.yaml_from_mapping(cfg)), cfg)
        short = recipe.config("short", env={})
        self.assertEqual((short["model"]["max_seq_len"], short["model"]["cache_mode"]), (16384, "Q8"))
        compact = recipe.config("compact", env={})
        self.assertEqual((compact["model"]["max_seq_len"], compact["model"]["cache_mode"]), (8192, "Q4"))
        custom = recipe.config("long", env={"MAX_SEQ_LEN": "65536", "CACHE_SIZE": "65536", "CACHE_MODE": "Q8"})
        self.assertEqual(custom["model"]["max_seq_len"], 65536)
        self.assertEqual(custom["model"]["cache_mode"], "Q8")
        with self.assertRaises(ValueError):
            recipe.config("long", env={"MAX_SEQ_LEN": "65536", "CACHE_SIZE": "32768"})
        with self.assertRaises(ValueError):
            recipe.config("long", env={"CACHE_MODE": "bad"})
        with self.assertRaises(ValueError):
            recipe.config("512k", env={})
        with self.assertRaises(ValueError):
            recipe.config("long", port=0, env={})

    def test_portable_gpu_monitoring_and_original_cap(self):
        # The original 4090 memory ceiling must not become less safe.
        self.assertLessEqual(recipe.gpu_memory_limit_mib(24564), 23950)
        self.assertEqual(recipe.gpu_memory_limit_mib(24564), 23949)
        # Other compatible GPUs are not rejected by model name or memory size.
        self.assertGreater(recipe.gpu_memory_limit_mib(81920), 79000)
        query = subprocess.CompletedProcess([], 0, "NVIDIA H100 80GB HBM3, 0, 81920\n", "")
        with mock.patch.object(recipe, "gpu_binary", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(recipe.subprocess, "run", return_value=query) as run, \
             mock.patch.dict(os.environ, {"GPU_INDEX": "2"}):
            stats = recipe.gpu_info()
        self.assertEqual(stats["index"], 2)
        self.assertEqual(stats["name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(stats["total_mib"], 81920)
        self.assertIn("2", run.call_args.args[0])

    def test_generic_gpu_guard_and_device_selector(self):
        with mock.patch.dict(os.environ, {"GPU_INDEX": "-1"}):
            with self.assertRaises(ValueError):
                recipe.gpu_index()
        with mock.patch.dict(os.environ, {"GPU_INDEX": "1"}):
            self.assertEqual(recipe.gpu_index(), 1)
        with mock.patch.object(recipe, "gpu_info", return_value={"index": 1, "name": "NVIDIA RTX A6000", "used_mib": 120, "total_mib": 49140}), \
             mock.patch.object(recipe, "available_ram", return_value=50 * 1024**3), \
             mock.patch.object(recipe.shutil, "disk_usage", return_value=mock.Mock(free=50 * 1024**3)), \
             mock.patch.dict(os.environ, {"RAM_MIN_AVAILABLE_GIB": "8", "ROOT_MIN_FREE_GIB": "4"}):
            self.assertEqual(recipe.safe_resources(before_load=True), 120)
        with mock.patch.object(recipe, "gpu_info", return_value={"index": 1, "name": "NVIDIA CUDA GPU", "used_mib": 120, "total_mib": 16384}), \
             mock.patch.object(recipe, "available_ram", return_value=50 * 1024**3), \
             mock.patch.object(recipe.shutil, "disk_usage", return_value=mock.Mock(free=50 * 1024**3)):
            with self.assertRaisesRegex(RuntimeError, "cannot fit the 18.35 GiB weights"):
                recipe.safe_resources(before_load=True)

    def test_benchmark_receipts(self):
        scores = json.loads((ROOT / "bench" / "measured-results.json").read_text())
        cat = scores["sixcat"]["by_category"]
        self.assertEqual(sum(v["passed"] for v in cat.values()), 104)
        self.assertEqual(sum(v["total"] for v in cat.values()), 120)
        self.assertEqual(scores["sixcat"]["truncated_code_generations"], 2)
        self.assertTrue(scores["speed_comparison_not_strict_apples_to_apples"])

    def test_qualified_allocation_not_residual_qad(self):
        data = (ROOT / "quantization" / "v6-mixed-k-4.50bpw.yaml").read_text()
        tensors = [line for line in data.splitlines() if line.startswith("  model.layers.")]
        self.assertEqual(len(tensors), 52 * 7)
        self.assertIn("achieved_bpw: 4.499880554228381", data)
        campaign = json.loads((ROOT / "quantization" / "campaign.json").read_text())
        self.assertFalse(campaign["full_residual_qad_selected"])


if __name__ == "__main__":
    unittest.main()
