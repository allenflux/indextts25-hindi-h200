"""Real CPU tensor tests against the pinned official UnifiedVoice, not a fake model.

Small random weights/synthetic codec IDs test math and state handling only. They
are not Hindi speech, pretrained checkpoint validation or a GPU quality claim.
"""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import torch
    from hi_h200_native.model import (Conv1DLoRA, NativeModelError, import_official_model,
                                      make_tiny_official_model, verify_official_source,
                                      load_native_adapter, file_sha256)


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), "Native tests require real Torch/Transformers and the pinned official checkout")
class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        import_official_model(OFFICIAL)

    def batch(self):
        return {"text_ids": torch.tensor([3, 4, 5]), "codes": torch.tensor([2, 3, 4, 5]),
                "spk_emb": torch.linspace(-.1, .1, 192), "emo_vec": torch.linspace(-.05, .05, 16)}

    def test_real_official_forward_backward_and_frozen_rows(self):
        model = make_tiny_official_model(OFFICIAL, seed=7).eval()
        base = model.base
        self.assertEqual(type(base).__name__, "UnifiedVoice")
        frozen = base.gpt.h[0].attn.c_attn.base.weight.detach().clone()
        language = base.lang_embedding.weight.detach().clone()
        text = base.text_embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.optimizer_groups(.3), lr=.02)
        result = model(self.batch())
        self.assertEqual(tuple(result["code_logits"].shape), (1, 34, 5))
        self.assertEqual(tuple(result["text_logits"].shape), (1, 65, 4))
        self.assertEqual(result["code_targets"].tolist(), [[2, 3, 4, 5, 33]])
        self.assertEqual(result["text_targets"].tolist(), [[3, 4, 5, 1]])
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertGreater(base.gpt.h[0].attn.c_attn.lora_B.grad.abs().sum().item(), 0)
        self.assertIsNone(base.gpt.h[0].attn.c_attn.base.weight.grad)
        self.assertIsNone(base.emo_layer.weight.grad)
        self.assertIsNone(base.spk_emb_proj.weight.grad)
        self.assertGreater(base.lang_embedding.weight.grad[17].abs().sum().item(), 0)
        others = torch.arange(base.lang_embedding.num_embeddings) != 17
        self.assertEqual(base.lang_embedding.weight.grad[others].abs().sum().item(), 0)
        optimizer.step()
        self.assertTrue(torch.equal(base.lang_embedding.weight.detach()[others], language[others]))
        self.assertFalse(torch.equal(base.lang_embedding.weight.detach()[17], language[17]))
        self.assertTrue(torch.equal(base.text_embedding.weight.detach()[20], text[20]))
        self.assertTrue(torch.equal(base.text_embedding.weight.detach()[0:2], text[0:2]))
        self.assertFalse(torch.equal(base.text_embedding.weight.detach()[3], text[3]))
        self.assertTrue(torch.equal(base.gpt.h[0].attn.c_attn.base.weight, frozen))

    def test_logits_match_official_generation_prefix_and_full_teacher_context(self):
        model = make_tiny_official_model(OFFICIAL, seed=13).eval()
        data = model.training_inputs(self.batch())
        base = model.base
        from indextts.gpt.model_v2 import GPT2InferenceModel
        inference = GPT2InferenceModel(base.gpt.config, base.gpt, base.mel_pos_embedding,
                                      base.mel_embedding, base.final_norm, base.mel_head, kv_cache=False).eval()
        inference.store_mel_emb(data["prefix"])
        fake_prefix = torch.ones((1, data["prefix"].shape[1]), dtype=torch.long)
        ids = torch.cat((fake_prefix, data["mel_input"]), dim=1)
        with torch.no_grad():
            actual = inference(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False, return_dict=True).logits[:, fake_prefix.shape[1]:]
            trained = model(self.batch())["code_logits"].transpose(1, 2)
        torch.testing.assert_close(actual, trained, rtol=1e-5, atol=1e-6)
        # Language conditioning must cover framed START/STOP as well as raw text.
        framed = torch.tensor([[0, 3, 4, 5, 1]])
        expected = base.text_embedding(framed) + base.text_pos_embedding(framed) + base.lang_embedding(torch.tensor([17]))
        torch.testing.assert_close(data["text_embeddings"], expected)
        torch.testing.assert_close(data["condition"][:, 1:], torch.zeros((1, 2, 16)))

    def test_real_optimizer_updates_reduce_small_random_model_loss(self):
        model = make_tiny_official_model(OFFICIAL, seed=11).eval()
        optimizer = torch.optim.AdamW(model.optimizer_groups(0), lr=.03)
        initial = model(self.batch())["loss"].item()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            result = model(self.batch())
            result["loss"].backward()
            optimizer.step()
        final = model(self.batch())["loss"].item()
        self.assertLess(final, initial * .8)

    def test_full_trainable_roundtrip_keeps_both_heads(self):
        model = make_tiny_official_model(OFFICIAL, seed=23).eval()
        optimizer = torch.optim.AdamW(model.optimizer_groups(0), lr=.02)
        model(self.batch())["loss"].backward()
        optimizer.step()
        state = model.trainable_state()
        self.assertIn("base.text_head.weight", state)
        self.assertIn("base.mel_head.weight", state)
        self.assertIn("base.lang_embedding.weight", state)
        restored = make_tiny_official_model(OFFICIAL, seed=23).eval()
        restored.load_trainable_state(state)
        torch.testing.assert_close(model(self.batch())["loss"], restored(self.batch())["loss"], rtol=0, atol=0)
        bad = dict(state)
        bad.pop("base.text_head.weight")
        with self.assertRaises(NativeModelError):
            restored.load_trainable_state(bad)
        bad = {name: value.clone() for name, value in state.items()}
        bad["base.text_head.weight"][0,0] = float("nan")
        with self.assertRaises(NativeModelError):
            restored.load_trainable_state(bad)

    def test_restricted_adapter_file_roundtrip_and_digest_base_spec_guards(self):
        model = make_tiny_official_model(OFFICIAL, seed=41).eval()
        optimizer = torch.optim.AdamW(model.optimizer_groups(0), lr=.01)
        model(self.batch())["loss"].backward()
        optimizer.step()
        payload = {"schema_version": 1, "kind": "hi_h200_native_adapter", "step": 1,
                   "base_identity": model.native_identity, "adapter_spec": model.adapter_spec(),
                   "trainable": model.trainable_state()}
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(payload, path)
            restored = make_tiny_official_model(OFFICIAL, seed=41).eval()
            result = load_native_adapter(restored, path, file_sha256(path))
            self.assertEqual(result["status"], "native_adapter_loaded")
            self.assertFalse(result["synthesis_verified"])
            torch.testing.assert_close(model(self.batch())["loss"], restored(self.batch())["loss"], rtol=0, atol=0)
            with self.assertRaisesRegex(NativeModelError, "SHA256"):
                load_native_adapter(restored, path, "0" * 64)
            payload["base_identity"] = {"kind": "different_base"}
            torch.save(payload, path)
            with self.assertRaisesRegex(NativeModelError, "base identity"):
                load_native_adapter(restored, path, file_sha256(path))
            payload["base_identity"] = model.native_identity
            payload["adapter_spec"] = dict(model.adapter_spec(), hindi_row=18)
            torch.save(payload, path)
            with self.assertRaisesRegex(NativeModelError, "settings/row"):
                load_native_adapter(restored, path, file_sha256(path))

    def test_invalid_adapter_tensors_do_not_partially_mutate_model(self):
        model = make_tiny_official_model(OFFICIAL, seed=43).eval()
        before = model.trainable_state()
        bad = {name: value.clone() for name, value in before.items()}
        first, last = next(iter(bad)), list(bad)[-1]
        bad[first].add_(1)
        bad[last].flatten()[0] = float("nan")
        with self.assertRaises(NativeModelError):
            model.load_trainable_state(bad)
        for name, value in model.trainable_state().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_real_hf_conv1d_lora_orientation_and_zero_initial_delta(self):
        from transformers.pytorch_utils import Conv1D
        layer = Conv1D(7, 5)
        lora = Conv1DLoRA(layer, rank=2, alpha=4, dropout=0)
        x = torch.randn(2, 3, 5)
        torch.testing.assert_close(lora(x), layer(x), rtol=0, atol=0)
        with torch.no_grad():
            lora.lora_A.fill_(.2)
            lora.lora_B.fill_(.3)
        expected = layer(x) + (x @ lora.lora_A.T @ lora.lora_B.T) * 2
        torch.testing.assert_close(lora(x), expected)

    def test_raw_reserved_tokens_bad_shapes_and_nonfinite_features_fail(self):
        model = make_tiny_official_model(OFFICIAL, seed=29)
        for key, value in (("text_ids", torch.tensor([3, 0, 4])), ("codes", torch.tensor([1, 32])),
                           ("text_ids", torch.ones(2, 3, dtype=torch.long)),
                           ("text_ids", torch.tensor([3.,4.])),
                           ("spk_emb", torch.zeros(191)), ("emo_vec", torch.full((16,), float("nan")))):
            batch = self.batch()
            batch[key] = value
            with self.subTest(key=key), self.assertRaises(NativeModelError):
                model(batch)

    def test_pinned_source_gate_rejects_noncheckout_without_executing_it(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            with self.assertRaises(NativeModelError):
                verify_official_source(directory)


if __name__ == "__main__":
    unittest.main()
