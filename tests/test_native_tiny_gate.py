"""Comparison guards and actual three-process CPU official-tiny-GPT gate."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from hi_h200_native.tiny_gate import TinyGateError, _validate_args, compare_states, main, run_gate

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_SOURCE", ROOT / ".runtime/external-official"))
TORCH = importlib.util.find_spec("torch") is not None
NATIVE = TORCH and importlib.util.find_spec("transformers") is not None and OFFICIAL.is_dir()


def arguments(output):
    return argparse.Namespace(official_repo=OFFICIAL, output_dir=output, device="cpu", allow_cuda=False,
        updates=6, stop_update=3, memory_limit_mib=512, min_free_mib=4096, atol=1e-5, rtol=1e-4)


class TinyGateArgumentTests(unittest.TestCase):
    def test_cuda_requires_deliberate_flag_and_one_visible_device(self):
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            args = arguments(Path("not-created"))
            args.device = "cuda:0"
            with self.assertRaisesRegex(TinyGateError, "allow-cuda"):
                _validate_args(args)
            args.allow_cuda = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            with self.assertRaisesRegex(TinyGateError, "one explicit"):
                _validate_args(args)
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_excessive_tolerance_or_allocation_is_rejected(self):
        for name, value in (("atol", .1), ("rtol", float("nan")),
                            ("memory_limit_mib", 4096), ("min_free_mib", 1024),
                            ("updates", 30), ("stop_update", 6)):
            args = arguments(Path("not-created"))
            setattr(args, name, value)
            with self.subTest(name=name), self.assertRaises(TinyGateError):
                _validate_args(args)

    def test_main_reports_blocked_not_passed_without_cuda_permission(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            code = main(["run", "--official-repo", str(OFFICIAL),
                         "--output-dir", "not-created", "--device", "cuda:0"])
        self.assertEqual(code, 2)
        record = json.loads(error.getvalue())
        self.assertEqual(record["status"], "tiny_gate_failed_or_blocked")
        self.assertFalse(record["cuda_bf16_verified"])


@unittest.skipUnless(TORCH, "Optional real Torch required")
class TinyGateComparisonTests(unittest.TestCase):
    def state(self):
        import torch
        result = {key: None for key in ("schema_version", "kind", "config", "identity", "runtime", "step",
                  "sampler", "parameter_groups", "model_identity", "adapter_spec", "scaler")}
        result.update(trainable={"weight": torch.tensor([1., 2.])},
                      optimizer={"moment": torch.tensor([.1, .2])},
                      scheduler={"last_epoch": 6}, rng={"torch_cpu": torch.tensor([1, 2], dtype=torch.uint8)})
        return result

    def test_reports_floating_tolerance_but_requires_exact_rng(self):
        import torch
        expected, actual = self.state(), self.state()
        actual["trainable"]["weight"] += 1e-6
        report = compare_states(expected, actual, atol=1e-5, rtol=1e-4)
        self.assertGreater(report["trainable"]["max_absolute_error"], 0)
        self.assertFalse(report["trainable"]["all_tensors_exact"])
        actual["rng"]["torch_cpu"] = torch.tensor([1, 3], dtype=torch.uint8)
        with self.assertRaisesRegex(TinyGateError, "Nonfloating"):
            compare_states(expected, actual, atol=1e-2, rtol=1e-2)

    def test_nonfinite_or_changed_runtime_never_passes(self):
        expected, actual = self.state(), self.state()
        actual["runtime"] = {"torch": "different"}
        with self.assertRaisesRegex(TinyGateError, "runtime"):
            compare_states(expected, actual, atol=1e-5, rtol=1e-4)
        actual = self.state()
        actual["optimizer"]["moment"][0] = float("nan")
        with self.assertRaisesRegex(TinyGateError, "Nonfinite"):
            compare_states(expected, actual, atol=1e-5, rtol=1e-4)


@unittest.skipUnless(NATIVE, "Real Torch, Transformers and pinned official source required")
class TinyGateProcessTests(unittest.TestCase):
    def test_actual_three_process_cpu_stop_and_resume(self):
        runtime = ROOT / ".runtime/native-tiny-gate-tests"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as temporary:
            output = Path(temporary) / "gate"
            report = run_gate(arguments(output))
            self.assertEqual(report["status"], "tiny_gate_passed")
            self.assertFalse(report["cuda_bf16_verified"])
            self.assertTrue(report["separate_process_stop_resume_verified"])
            self.assertEqual(report["optimizer_updates_before_stop"], 3)
            self.assertEqual(report["optimizer_updates_after_resume"], 3)
            self.assertFalse(report["orchestrator_cuda_context_created"])
            self.assertFalse(report["pretrained_model_loaded"])
            self.assertFalse(report["hindi_quality_verified"])
            self.assertTrue(report["comparison"]["trainable"]["all_tensors_exact"])
            self.assertTrue((output / "gate-result.json").is_file())
            self.assertFalse((output / "split/STOP").exists())
            self.assertEqual(report["segments"]["resume"]["actual_model_class"], "UnifiedVoice")
            self.assertEqual(report["segments"]["resume"]["actual_logit_dtypes"], ["torch.float32"])
