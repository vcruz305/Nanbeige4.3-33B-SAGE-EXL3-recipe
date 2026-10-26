import hashlib
import importlib.util
import json
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
        cfg = recipe.config("long", port=8898)
        self.assertEqual(cfg["model"]["max_seq_len"], 36864)
        self.assertEqual(cfg["model"]["cache_mode"], "Q4")
        self.assertEqual(cfg["model"]["max_batch_size"], 1)
        self.assertEqual(cfg["network"]["host"], "127.0.0.1")
        self.assertTrue(cfg["network"]["disable_fetch_requests"])
        self.assertEqual(json.loads(recipe.yaml_from_mapping(cfg)), cfg)
        short = recipe.config("short")
        self.assertEqual((short["model"]["max_seq_len"], short["model"]["cache_mode"]), (16384, "Q8"))
        with self.assertRaises(ValueError):
            recipe.config("512k")
        with self.assertRaises(ValueError):
            recipe.config("long", port=0)

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
