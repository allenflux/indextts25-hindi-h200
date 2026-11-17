"""Real CPU loss/gradient equivalence for the deterministic 2D token CE path."""
import importlib.util
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_SOURCE", ROOT / ".runtime/external-official"))
TORCH = importlib.util.find_spec("torch") is not None
NATIVE = TORCH and importlib.util.find_spec("transformers") is not None and OFFICIAL.is_dir()
if TORCH:
    import torch
    from torch.nn import functional as F
    from hi_h200_native.model import _sequence_cross_entropy, NativeModelError


@unittest.skipUnless(TORCH, "Real Torch required; no synthetic loss substitute")
class SequenceCrossEntropyTests(unittest.TestCase):
    def test_flattened_loss_and_input_gradients_match_legacy_sequence_ce(self):
        generator = torch.Generator().manual_seed(73)
        for dtype in (torch.float32, torch.float64):
            # Strided, multi-example logits expose incorrect flatten order.
            values = torch.randn(2, 7, 10, generator=generator, dtype=dtype)[..., ::2]
            targets = torch.randint(7, (2, 5), generator=generator)
            original = values.detach().requires_grad_()
            flattened = values.detach().clone().requires_grad_()
            old = F.cross_entropy(original.float(), targets)
            new = _sequence_cross_entropy(flattened, targets)
            old.backward()
            new.backward()
            torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(flattened.grad, original.grad, rtol=1e-6, atol=1e-7)

    def test_inconsistent_token_axes_are_rejected(self):
        with self.assertRaisesRegex(NativeModelError, "Token loss expects"):
            _sequence_cross_entropy(torch.zeros(1, 7, 5), torch.zeros(1, 4, dtype=torch.long))


@unittest.skipUnless(NATIVE, "Pinned official tiny GPT and real Torch required")
class OfficialModelCrossEntropyTests(unittest.TestCase):
    def test_real_official_model_loss_and_all_trainable_gradients_match(self):
        from hi_h200_native.model import make_tiny_official_model
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = make_tiny_official_model(OFFICIAL, seed=37).eval()
            batch = {"text_ids": torch.tensor([3, 4, 5]), "codes": torch.tensor([2, 3, 4, 5]),
                     "spk_emb": torch.linspace(-.1, .1, 192),
                     "emo_vec": torch.linspace(-.05, .05, 16)}
            result = model(batch)
            legacy = (F.cross_entropy(result["code_logits"].float(), result["code_targets"])
                      + model.text_loss_weight * F.cross_entropy(result["text_logits"].float(), result["text_targets"]))
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            new_gradients = torch.autograd.grad(result["loss"], parameters, retain_graph=True, allow_unused=True)
            old_gradients = torch.autograd.grad(legacy, parameters, allow_unused=True)
            torch.testing.assert_close(result["loss"], legacy, rtol=1e-6, atol=1e-7)
            nonzero = 0
            for current, old in zip(new_gradients, old_gradients):
                self.assertEqual(current is None, old is None)
                if current is not None:
                    torch.testing.assert_close(current, old, rtol=1e-5, atol=1e-7)
                    nonzero += int(torch.count_nonzero(current).item() > 0)
            self.assertGreater(nonzero, 3)
        finally:
            torch.set_num_threads(previous)
