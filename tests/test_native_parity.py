"""Actual tiny official CPU model parity; no fake model or pretrained claim."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import torch
    from hi_h200_native.model import make_tiny_official_model, import_official_model
    from hi_h200_native.export import merge_official_state
    from hi_h200_native.parity import stock_forward, compare_outputs, RTOL, ATOL


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), "Needs real Torch and pinned official source")
class NativeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_actual_native_adapter_versus_fresh_stock_after_real_updates(self):
        native = make_tiny_official_model(OFFICIAL, seed=67, lora_dropout=.1).eval()
        batch = {"text_ids":torch.tensor([3,4,5]), "codes":torch.tensor([2,3,4,5,6,7,8,9]),
                 "spk_emb":torch.linspace(-.2,.2,192), "emo_vec":torch.linspace(-.1,.1,16)}
        optimizer = torch.optim.AdamW(native.optimizer_groups(0), lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            native(batch)["loss"].backward()
            optimizer.step()
        with torch.no_grad():
            expected = native(batch)
        stock = import_official_model(OFFICIAL)(**native.native_config["gpt"]).eval()
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            checkpoint = Path(directory) / "gpt.pth"
            torch.save({"model":merge_official_state(native)}, checkpoint)
            stock.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        self.assertEqual(type(stock).__name__, "UnifiedVoice")
        self.assertFalse(any("lora_" in name or ".base." in name for name in stock.state_dict()))
        with torch.no_grad():
            actual = stock_forward(stock, batch, native.text_loss_weight)
        result = compare_outputs(expected, actual)
        self.assertTrue(result["passed"], result)
        self.assertEqual((result["rtol"],result["atol"]), (1e-4,1e-4))
        self.assertTrue(result["metrics"]["text_targets"]["exact_equal"])
        self.assertTrue(result["metrics"]["code_logits"]["argmax_equal"])
        self.assertLess(result["metrics"]["loss"]["max_abs_error"], 1e-4)
        # Preserve fixed acceptance thresholds, and show that a real delta fails.
        actual["code_logits"][0,0,0] += .1
        failed = compare_outputs(expected, actual)
        self.assertFalse(failed["passed"])
        actual["code_logits"][0,0,0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            compare_outputs(expected, actual)

    def test_pure_profile_execute_reconstructs_actual_native_and_reports_validated_spec(self):
        from hi_h200_native.model import PURE_LORA_PROFILE, file_sha256
        from hi_h200_native.export import export_merged_gpt, verify_export_bundle
        from hi_h200_native.parity import execute
        native = make_tiny_official_model(OFFICIAL, seed=71, training_profile=PURE_LORA_PROFILE).eval()
        batch = {"text_ids":torch.tensor([3,4,5]), "codes":torch.tensor([2,3,4,5,6,7,8,9]),
                 "spk_emb":torch.linspace(-.2,.2,192), "emo_vec":torch.linspace(-.1,.1,16)}
        optimizer = torch.optim.AdamW(native.optimizer_groups(0), lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            native(batch)["loss"].backward()
            optimizer.step()
        class SyntheticFeature:
            identity = {"kind":"synthetic-feature-for-actual-tiny-official-model"}
            def __len__(self): return 1
            def __getitem__(self, index):
                if index != 0: raise IndexError(index)
                return batch
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as folder:
            root = Path(folder)
            adapter, bundle, dataset_path = root / "pure.pt", root / "merged", root / "synthetic.dataset.json"
            dataset_path.write_text('{"fixture":"synthetic, no production feature provenance"}')
            payload = {"schema_version":1,"kind":"hi_h200_native_adapter","step":3,
                "base_identity":native.native_identity,"adapter_spec":native.adapter_spec(),"trainable":native.trainable_state()}
            torch.save(payload,adapter)
            adapter_sha = file_sha256(adapter)
            export_merged_gpt(native, OFFICIAL, bundle, adapter_path=adapter, adapter_sha256=adapter_sha)
            merged = verify_export_bundle(bundle)
            provenance = json.loads((bundle / "provenance.json").read_text())
            report = {"dataset_config_sha256":file_sha256(dataset_path),"merged_gpt_sha256":merged["gpt_sha256"]}
            args = SimpleNamespace(adapter=adapter,adapter_sha256=adapter_sha,dataset=dataset_path,
                merged_bundle=bundle,index=0,device="cpu",allow_cuda=False)
            def actual_tiny_constructor(*unused, **kwargs):
                self.assertEqual(kwargs["training_profile"], PURE_LORA_PROFILE)
                self.assertEqual(kwargs["hindi_row"], 17)
                self.assertEqual(kwargs["text_rows"], [])
                return make_tiny_official_model(OFFICIAL,seed=71,training_profile=kwargs["training_profile"])
            # Only unavailable full production asset/data entry points are injected.
            # Both GPTs, strict adapter load, merge container and forwards are real.
            with patch("hi_h200_native.model.load_native_model",side_effect=actual_tiny_constructor) as loader, \
                    patch("hi_h200_native.data.load_native_dataset",return_value=SyntheticFeature()):
                result = execute(args,{"official_repo":OFFICIAL,"model_dir":root,"model_manifest":root / "unused.json"},
                    native.native_config,native.native_identity,provenance,report)
            self.assertEqual(loader.call_count,1)
            self.assertTrue(result["comparison"]["passed"])
            self.assertTrue(result["adapter_profile_verified"])
            self.assertEqual(result["training_profile"], PURE_LORA_PROFILE)
            self.assertEqual(result["adapter_spec"],native.adapter_spec())
            self.assertEqual(result["adapter_spec"]["heads"],[])
            self.assertEqual(result["adapter_spec_sha256"],provenance["adapter_spec_sha256"])
            self.assertEqual(result["status"],"tiny_official_single_batch_parity_passed")
            self.assertFalse(result["full_pretrained_forward_parity_verified"])
            self.assertEqual((result["comparison"]["rtol"],result["comparison"]["atol"]),(1e-4,1e-4))

    def test_cli_help_has_no_tensor_model_import(self):
        script = "import sys; from hi_h200_native.parity import main;\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code == 0\nassert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src")), timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--execute", completed.stdout)
        self.assertEqual((RTOL, ATOL), (1e-4,1e-4))


if __name__ == "__main__":
    unittest.main()
