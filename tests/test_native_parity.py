"""Actual tiny official CPU model parity; no fake model or pretrained claim."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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

    def test_cli_help_has_no_tensor_model_import(self):
        script = "import sys; from hi_h200_native.parity import main;\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code == 0\nassert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src")), timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--execute", completed.stdout)
        self.assertEqual((RTOL, ATOL), (1e-4,1e-4))


if __name__ == "__main__":
    unittest.main()
