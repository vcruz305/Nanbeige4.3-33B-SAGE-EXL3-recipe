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
        self.assertTrue(lock["initial_private_staging"])
        self.assertEqual(lock["current_visibility"], "public")
        self.assertFalse(lock["gated"])
        self.assertEqual(lock["revision"], "e4bf35566476daf25fdf650947d10553c1af985c")

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

    def test_measured_128k_q2_profile_and_256k_preflight(self):
        # Actual RTX4090 128K result: host-placed input embedding, GPU
        # resident original decoder weights and 2-bit K/V.
        cfg = recipe.config("extended", env={})
        self.assertEqual(cfg["model"]["max_seq_len"], 131072)
        self.assertEqual(cfg["model"]["cache_size"], 131072)
        self.assertEqual(cfg["model"]["cache_mode"], "2,2")
        self.assertTrue(recipe.cpu_embedding_enabled("extended", env={}))
        self.assertEqual(recipe.kv_storage_mib(cfg), 4160.0)
        safe = recipe.gpu_vram_budget(
            cfg, cpu_embedding=True, weight_bytes=19700587306,
            total_mib=24564, used_mib=214)
        self.assertGreaterEqual(safe["estimated_headroom_mib"], 300)
        self.assertEqual(safe["gpu_limit_mib"], 23949)
        # The original 23,950MiB cutoff is not relaxed to squeeze 256K.
        too_big = recipe.config("extended", env={
            "MAX_SEQ_LEN": "262144", "CACHE_SIZE": "262144", "CACHE_MODE": "Q2"})
        self.assertEqual(recipe.kv_storage_mib(too_big), 8320.0)
        with self.assertRaisesRegex(RuntimeError, "Safe context budget exceeded"):
            recipe.gpu_vram_budget(too_big, cpu_embedding=True,
                                  weight_bytes=19700587306, total_mib=24564)
        # More-capable GPUs can configure 256K; not claimed tested.
        larger = recipe.gpu_vram_budget(too_big, cpu_embedding=True,
                                       weight_bytes=19700587306, total_mib=81920)
        self.assertGreater(larger["estimated_headroom_mib"], 10000)
        q3 = recipe.config("long", env={"MAX_SEQ_LEN":"65536", "CACHE_MODE":"Q3"})
        self.assertEqual(q3["model"]["cache_mode"], "3,3")
        self.assertEqual(recipe.kv_storage_mib(q3), 2912.0)
        self.assertFalse(recipe.cpu_embedding_enabled("long", env={}))
        self.assertTrue(recipe.cpu_embedding_enabled("long", env={"EMBEDDING_PLACEMENT":"cpu"}))
        with self.assertRaises(ValueError):
            recipe.cpu_embedding_enabled("long", env={"EMBEDDING_PLACEMENT":"cloud"})

    def test_release_native_quant_metadata_accepts_hf_fix(self):
        config = json.loads((ROOT / "evidence" / "config.json").read_text())
        top = config["quantization_config"]
        self.assertNotIn("bits", top)
        native = {"quant_method":"exl3", "codebook":"mul1", "bits":4.5,
                  "head_bits":8, "tensor_storage":{"example":{"trellis":True}}}
        self.assertTrue(recipe.validate_exl3_quant_metadata(top, native))
        legacy = {**top, "bits":4.5}
        self.assertTrue(recipe.validate_exl3_quant_metadata(legacy, native))
        with self.assertRaises(ValueError):
            recipe.validate_exl3_quant_metadata({**top, "bits":4}, native)
        with self.assertRaises(ValueError):
            recipe.validate_exl3_quant_metadata(top, {**native, "bits":4})

    def test_public_context_smoke_receipts_honest(self):
        data=json.loads((ROOT / "bench/context/measured-results.json").read_text())
        tested=data["validated_cases"]
        self.assertEqual([case["context_capacity_tokens"] for case in tested],
                         [53248, 65536, 81920, 131072])
        self.assertTrue(all(case["synthetic_needle_count_passed"]==3 for case in tested))
        self.assertTrue(all(case["gpu_peak_used_mib"]<23950 for case in tested))
        self.assertFalse(any(case["full_model_sixcat_scored"] for case in tested))
        last=tested[-1]
        self.assertEqual(last["embedding_location"], "CPU / system RAM")
        self.assertEqual(last["actual_input_tokens"], 130542)
        self.assertEqual(last["gpu_peak_used_mib"], 23220)
        attempt=data["256k_attempt"]
        self.assertEqual(attempt["context_capacity_tokens"],262144)
        self.assertEqual(attempt["kv_cache_mib"],8320)
        self.assertEqual(attempt["estimated_gpu_peak_mib"],27576)
        self.assertFalse(attempt["gpu_server_launched"])
        trial=json.loads((ROOT / "bench/dflash2/measured-results.json").read_text())
        self.assertFalse(trial["all_greedy_match"])
        self.assertEqual(trial["status"],"COMPLETE")

    def test_benchmark_receipts(self):
        scores = json.loads((ROOT / "bench" / "measured-results.json").read_text())
        cat = scores["sixcat"]["by_category"]
        self.assertEqual(sum(v["passed"] for v in cat.values()), 104)
        self.assertEqual(sum(v["total"] for v in cat.values()), 120)
        self.assertEqual(scores["sixcat"]["truncated_code_generations"], 2)
        self.assertTrue(scores["speed_comparison_not_strict_apples_to_apples"])

    def test_hf_config_integer_schema_compatibility(self):
        # EXL3 is mixed-K, 4.5bpw is a nominal *average*. Do not round it
        # to 4 or 5 to satisfy the Hub's generic integer-only `bits` schema.
        config_path = ROOT / "evidence" / "config.json"
        config = json.loads(config_path.read_text())
        q = config["quantization_config"]
        self.assertEqual(q["quant_method"], "exl3")
        self.assertEqual(q["codebook"], "mul1")
        self.assertEqual(q["head_bits"], 8)
        self.assertNotIn("bits", q)
        lock = json.loads((ROOT / "model.lock.json").read_text())
        record = next(f for f in lock["files"] if f["path"] == "config.json")
        self.assertEqual(record["size"], config_path.stat().st_size)
        self.assertEqual(record["sha256"], hashlib.sha256(config_path.read_bytes()).hexdigest())
        text = (ROOT / "docs" / "RELEASE_MODEL_CARD.md").read_text()
        self.assertIn("integer", text)
        self.assertIn("quantization_config.json", text)
        self.assertIn("4.499880554", text)

    def test_model_access_is_public_and_card_explains_base_model(self):
        lock = json.loads((ROOT / "model.lock.json").read_text())
        self.assertEqual(lock["current_visibility"], "public")
        self.assertFalse(lock["gated"])
        self.assertTrue(lock["initial_private_staging"])
        card = (ROOT / "docs" / "RELEASE_MODEL_CARD.md").read_text()
        self.assertIn("gated: false", card)
        self.assertIn("Public, freely downloadable EXL3", card)
        for expected in ("52", "56 query heads", "8 key/value heads", "524,288",
                         "Toolathlon-Verified", "preserve_thinking", "Apache-2.0",
                         "104/120", "RTX 4090", "not EXL3"):
            self.assertIn(expected.lower(), card.lower())
        code = (ROOT / "tools" / "recipe.py").read_text()
        self.assertIn("token=False, max_workers=1", code)
        self.assertNotIn("Authenticated PRIVATE HF download", code)

    def test_qualified_allocation_not_residual_qad(self):
        data = (ROOT / "quantization" / "v6-mixed-k-4.50bpw.yaml").read_text()
        tensors = [line for line in data.splitlines() if line.startswith("  model.layers.")]
        self.assertEqual(len(tensors), 52 * 7)
        self.assertIn("achieved_bpw: 4.499880554228381", data)
        campaign = json.loads((ROOT / "quantization" / "campaign.json").read_text())
        self.assertFalse(campaign["full_residual_qad_selected"])


if __name__ == "__main__":
    unittest.main()
