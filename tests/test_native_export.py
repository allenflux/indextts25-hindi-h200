"""Actual official GPT merge/export/reload tests on a tiny CPU model."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import torch
    from torch.nn import functional as F
    from hi_h200_native.model import NativeModelError, file_sha256, import_official_model, make_tiny_official_model
    from hi_h200_native.export import export_merged_gpt, merge_official_state, verify_export_bundle


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), "Native export tests require real Torch and pinned official source")
class NativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root = Path(self.temporary.name)
        self.model = make_tiny_official_model(OFFICIAL, seed=71).eval()
        self.batch = {"text_ids": torch.tensor([3,4,5]), "codes": torch.tensor([2,3,4,5]),
                      "spk_emb": torch.linspace(-.1,.1,192), "emo_vec": torch.linspace(-.05,.05,16)}
        self.adapter = self.root / "native_adapter.pt"
        self.output = self.root / "fresh_bundle"

    def tearDown(self):
        self.temporary.cleanup()

    def save_adapter(self):
        self.payload = {"schema_version":1,"kind":"hi_h200_native_adapter","step":3,
                        "base_identity":self.model.native_identity,"adapter_spec":self.model.adapter_spec(),
                        "trainable":self.model.trainable_state()}
        torch.save(self.payload, self.adapter)
        return file_sha256(self.adapter)

    def export(self, **kwargs):
        return export_merged_gpt(self.model, OFFICIAL, self.output, adapter_path=self.adapter,
                                 adapter_sha256=file_sha256(self.adapter), **kwargs)

    def test_real_merged_official_reload_matches_native_logits_and_loss(self):
        optimizer = torch.optim.AdamW(self.model.optimizer_groups(.1), lr=.02)
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            self.model(self.batch)["loss"].backward()
            optimizer.step()
        before = self.model(self.batch)
        original_matrix = self.model.base.gpt.h[0].attn.c_attn.base.weight.detach().clone()
        original_lora = self.model.base.gpt.h[0].attn.c_attn.lora_B.detach().clone()
        self.save_adapter()
        result = self.export()
        self.assertEqual(result["status"], "complete_native_gpt_bundle")
        self.assertTrue(torch.equal(original_matrix, self.model.base.gpt.h[0].attn.c_attn.base.weight))
        self.assertTrue(torch.equal(original_lora, self.model.base.gpt.h[0].attn.c_attn.lora_B))
        saved = torch.load(self.output / "gpt.pth", map_location="cpu", weights_only=True)
        self.assertEqual(set(saved), {"model"})
        state = saved["model"]
        self.assertFalse(any("lora_" in key or ".base." in key for key in state))
        self.assertIn("text_head.weight", state)
        self.assertIn("mel_head.weight", state)
        self.assertIn("lang_embedding.weight", state)
        fresh = import_official_model(OFFICIAL)(**self.model.native_config["gpt"]).eval()
        self.assertEqual(set(fresh.state_dict()), set(state))
        fresh.load_state_dict(state, strict=True)
        # Exercise the actual official checkpoint container loader too.
        from indextts.utils.checkpoint import load_checkpoint
        self.assertEqual(load_checkpoint(fresh, str(self.output / "gpt.pth")), {})
        speaker = fresh.spk_emb_proj(self.batch["spk_emb"].reshape(1,192)).unsqueeze(1)
        condition = torch.cat((speaker + self.batch["emo_vec"].reshape(1,1,16), torch.zeros(1,2,16)), dim=1)
        text = self.batch["text_ids"].reshape(1,-1)
        _, prefix, _ = fresh.prepare_gpt_inputs(condition, text, torch.tensor([17]))
        mel = torch.cat((torch.tensor([[32]]), self.batch["codes"].reshape(1,-1)), dim=1)
        mel_embedding = fresh.mel_embedding(mel) + fresh.mel_pos_embedding(mel)
        text_logits, code_logits = fresh.get_logits(condition, prefix[:,3:], fresh.text_head, mel_embedding, fresh.mel_head)
        text_logits = text_logits[:,:,:4]
        loss = F.cross_entropy(code_logits, torch.tensor([[2,3,4,5,33]])) + .1 * F.cross_entropy(text_logits, torch.tensor([[3,4,5,1]]))
        torch.testing.assert_close(code_logits, before["code_logits"], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(text_logits, before["text_logits"], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(loss, before["loss"], rtol=1e-5, atol=2e-6)
        provenance = json.loads((self.output / "provenance.json").read_text())
        self.assertFalse(provenance["full_pretrained_forward_parity_verified"])
        self.assertFalse(provenance["old_language_behavior_guaranteed_unchanged"])
        self.assertIn("tiny random", provenance["scope"])
        self.assertEqual((self.output / "LICENSE").read_bytes(), (OFFICIAL / "LICENSE").read_bytes())
        self.assertNotIn("optimizer", state)

    def test_existing_outputs_and_protected_roots_are_never_overwritten(self):
        self.save_adapter()
        self.output.mkdir()
        original = self.output / "keep.txt"
        original.write_text("preserve")
        with self.assertRaises(NativeModelError):
            self.export()
        self.assertEqual(original.read_text(), "preserve")
        blocked = self.root / "original_model"
        blocked.mkdir()
        self.model.native_model_dir = str(blocked)
        self.output = blocked / "must_not_write"
        with self.assertRaisesRegex(NativeModelError, "isolated"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_bad_adapter_digest_identity_spec_and_missing_head_block_export(self):
        self.save_adapter()
        with self.assertRaisesRegex(NativeModelError, "SHA256"):
            export_merged_gpt(self.model, OFFICIAL, self.output, adapter_path=self.adapter, adapter_sha256="0"*64)
        self.assertFalse(self.output.exists())
        for key, value in (("base_identity", {"official_commit":"wrong"}),
                           ("adapter_spec", dict(self.model.adapter_spec(), hindi_row=18))):
            self.save_adapter()
            self.payload[key] = value
            torch.save(self.payload, self.adapter)
            with self.assertRaises(NativeModelError):
                self.export()
            self.assertFalse(self.output.exists())
        self.save_adapter()
        self.payload["trainable"].pop("base.text_head.weight")
        torch.save(self.payload, self.adapter)
        with self.assertRaises(NativeModelError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_changed_frozen_state_and_unselected_embedding_rows_block(self):
        self.save_adapter()
        with torch.no_grad():
            self.model.base.spk_emb_proj.weight[0,0].add_(1)
        with self.assertRaisesRegex(NativeModelError, "Frozen state"):
            self.export()
        self.assertFalse(self.output.exists())
        self.model = make_tiny_official_model(OFFICIAL, seed=71).eval()
        self.save_adapter()
        with torch.no_grad():
            self.model.base.lang_embedding.weight[0,0].add_(1)
        with self.assertRaisesRegex(NativeModelError, "protected language"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_nonfinite_or_changed_lora_scaling_is_not_exported(self):
        self.save_adapter()
        self.payload["trainable"]["base.text_head.weight"][0,0] = float("nan")
        torch.save(self.payload, self.adapter)
        with self.assertRaises(NativeModelError):
            self.export()
        self.assertFalse(self.output.exists())
        self.model = make_tiny_official_model(OFFICIAL, seed=71).eval()
        self.model.base.gpt.h[0].attn.c_attn.scale *= 2
        with self.assertRaisesRegex(NativeModelError, "scaling"):
            merge_official_state(self.model)

    def test_incomplete_or_tampered_bundle_is_rejected(self):
        self.save_adapter()
        self.export()
        self.assertEqual(verify_export_bundle(self.output)["status"], "complete_native_gpt_bundle")
        marker = self.output / "COMPLETE.json"
        original = marker.read_bytes()
        marker.unlink()
        with self.assertRaisesRegex(NativeModelError, "incomplete"):
            verify_export_bundle(self.output)
        marker.write_bytes(original)
        with (self.output / "gpt.pth").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(NativeModelError, "changed"):
            verify_export_bundle(self.output)


if __name__ == "__main__":
    unittest.main()
