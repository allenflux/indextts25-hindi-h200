"""Fixed TRAIN gate CPU contracts and actual tiny-official before/after forward.

The tiny fixture is synthetic; production's full dataset validation is separate.
No GPU, actual large checkpoint or quality inference is run by these tests.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hi_h200_native import pure_overfit as gate
from hi_h200_native.assets import MODEL_REPO, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None and OFFICIAL.is_dir()
if AVAILABLE:
    import torch
    from hi_h200_native.model import make_tiny_official_model, PURE_LORA_PROFILE, file_sha256


class GateMetadataTests(unittest.TestCase):
    def test_predeclared_aggregate_boundary_and_nonfinite_fail_closed(self):
        self.assertTrue(gate.aggregate([10.] * 4, [9.] * 4)["mechanism_gate_passed"])
        self.assertFalse(gate.aggregate([10.] * 4, [9.00001] * 4)["mechanism_gate_passed"])
        self.assertFalse(gate.aggregate([10.] * 4, [10.] * 4)["mechanism_gate_passed"])
        for values in ([1., 2., 3.], [1., 2., 3., float("nan")], [0., 1., 1., 1.]):
            with self.assertRaises(ValueError):
                gate.aggregate(values, [1.] * 4)

    def test_default_help_imports_no_torch(self):
        code = "import sys; from hi_h200_native.pure_overfit import main;\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code==0\nassert 'torch' not in sys.modules; assert 'indextts' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=20,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allow-cuda", result.stdout)

    def test_nontrain_and_unapproved_cuda_reject_before_adapter_or_torch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as folder:
            config = Path(folder) / "dataset.json"
            config.write_text('{"split":"test"}')
            with self.assertRaisesRegex(ValueError, "training split"):
                gate.preflight(SimpleNamespace(workspace_root=ROOT, dataset=config))
        with self.assertRaisesRegex(ValueError, "allow-cuda"):
            gate.execute(SimpleNamespace(allow_cuda=False), {})

    @unittest.skipUnless(OFFICIAL.is_dir(), "Pinned local official source needed, no weights")
    def test_metadata_is_truthfully_pending_and_has_no_private_identifiers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as folder:
            root = Path(folder)
            for name in ("model", "audio", "prepared", "features"):
                (root / name).mkdir()
            for name in ("manifest.jsonl", "provenance.json"):
                (root / "features" / name).write_text("{}")
            manifest = root / "model" / "model-manifest.json"
            manifest.write_text(json.dumps({"repository": MODEL_REPO, "snapshot_revision": MODEL_REVISION,
                                           "files": {"gpt.pth": gate.BASE_GPT_SHA256}}))
            options = {"schema_version": 1, "split": "train", "official_repo": str(OFFICIAL),
                "model_dir": str(root / "model"), "model_manifest": str(manifest),
                "audio_root": str(root / "audio"), "prepared_dir": str(root / "prepared"),
                "feature_root": str(root / "features"), "feature_manifest": str(root / "features/manifest.jsonl"),
                "provenance": str(root / "features/provenance.json")}
            dataset = root / "dataset.json"
            dataset.write_text(json.dumps(options))
            adapter = root / "adapter.pt"
            adapter.write_bytes(b"metadata fixture only, never a tensor checkpoint")
            import hashlib
            args = ["--workspace-root", str(ROOT.parent), "--dataset", str(dataset), "--adapter", str(adapter),
                    "--adapter-sha256", hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    "--output-dir", str(ROOT.parent / ("not-created-gate-" + root.name))]
            code = "import sys; from hi_h200_native.pure_overfit import main; assert main("+repr(args)+")==0; assert 'torch' not in sys.modules"
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=20,
                env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report["dataset_fully_validated"])
            self.assertFalse(report["adapter_tensor_validation"])
            self.assertFalse(report["adapter_training_dataset_binding_verified"])
            self.assertFalse(report["mechanism_gate_passed"])
            self.assertNotIn(str(root), result.stdout)
            self.assertFalse(Path(args[-1]).exists())
            for protected in (ROOT / ".git" / "pure-gate-must-not-create", ROOT / "src" / "pure-gate-must-not-create"):
                blocked = SimpleNamespace(workspace_root=ROOT.parent, dataset=dataset, adapter=adapter,
                    adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(), output_dir=protected, seed=1234)
                with self.assertRaisesRegex(ValueError, "overlap"):
                    gate.preflight(blocked)
                self.assertFalse(protected.exists())


@unittest.skipUnless(AVAILABLE, "Actual Torch and pinned official class required")
class ActualTinyGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_four_example_loss_before_after_and_frozen_state(self):
        class SyntheticFeatures:
            _ids = ["synthetic-0", "synthetic-1", "synthetic-2", "synthetic-3", "zz-not-selected"]
            identity = {"split": "train", "fixture": "synthetic_not_audio"}
            def __len__(self): return 5
            def __getitem__(self, index):
                if index >= 4: raise AssertionError("The fixed gate must not select other examples")
                return {"id": self._ids[index], "text_ids": torch.tensor([3,4,5]),
                    "codes": torch.tensor([2,3,4,5]), "spk_emb": torch.linspace(-.1,.1,192),
                    "emo_vec": torch.linspace(-.05,.05,16) + index / 100}
        dataset = SyntheticFeatures()
        def fresh():
            return make_tiny_official_model(OFFICIAL, seed=151, training_profile=PURE_LORA_PROFILE)
        trained = fresh().eval()
        optimizer = torch.optim.AdamW(trained.optimizer_groups(0), lr=.03, foreach=False)
        for index in range(80):
            optimizer.zero_grad(set_to_none=True)
            trained(dataset[index % 4])["loss"].backward()
            optimizer.step()
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as folder:
            path = Path(folder) / "adapter.pt"
            payload = {"schema_version": 1, "kind": "hi_h200_native_adapter", "step": 80,
                "base_identity": trained.native_identity, "adapter_spec": trained.adapter_spec(),
                "trainable": trained.trainable_state()}
            torch.save(payload, path)
            model = fresh()
            report = gate.compare_four(model, dataset, path, file_sha256(path), device="cpu", seed=1234,
                                       deadline=time.monotonic() + 30)
            self.assertTrue(report["mechanism_gate_passed"])
            self.assertEqual([row["index"] for row in report["cases"]], [0,1,2,3])
            self.assertLessEqual(report["mean_loss_ratio"], .9)
            self.assertTrue(report["frozen_state_unchanged"])
            self.assertEqual(report["trainable_tensors"], 4)  # Actual one-layer tiny GPT.
            serialized = json.dumps(report)
            self.assertFalse(any(uid in serialized for uid in dataset._ids))
            self.assertNotIn(str(path), serialized)
            with self.assertRaisesRegex(ValueError, "zero-delta"):
                gate.compare_four(trained, dataset, path, file_sha256(path), device="cpu", seed=1234,
                                  deadline=time.monotonic() + 30)
            dataset.identity = {"split": "val"}
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                gate.compare_four(fresh(), dataset, path, file_sha256(path), device="cpu", seed=1234,
                                  deadline=time.monotonic() + 30)


if __name__ == "__main__":
    unittest.main()
