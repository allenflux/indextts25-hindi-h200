"""Actual official tiny GPT CPU checks for the separate pure-attention profile.

Synthetic token/code tensors test math/state only; no pretrained/Hindi quality or
GPU result is implied. Existing mixed rows/heads adapters remain a distinct route.
"""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hi_h200_native.train import read_config

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None and OFFICIAL.is_dir()
if AVAILABLE:
    import torch
    from hi_h200_native.model import (LEGACY_PROFILE, PURE_LORA_PROFILE, NativeModelError,
        make_tiny_official_model, file_sha256, load_native_adapter, preserved_state_digest,
        import_official_model)
    from hi_h200_native.engine import (LoopConfig, StatefulTrainer, TrainingError,
        cosine_scheduler, seed_everything)
    from hi_h200_native.export import export_merged_gpt, export_from_adapter
    from hi_h200_native.checkpoint_adapter import validate_checkpoint, publish_adapter


def batch():
    return {"text_ids": torch.tensor([3, 4, 5]), "codes": torch.tensor([2, 3, 4, 5]),
            "spk_emb": torch.linspace(-.1, .1, 192), "emo_vec": torch.linspace(-.05, .05, 16)}


class PureConfigTests(unittest.TestCase):
    def test_profile_is_explicit_and_old_config_stays_unmodified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as directory:
            folder = Path(directory)
            for name in ("native-train.dataset.json", "native-val.dataset.json"):
                (folder / name).write_text("{}")
            original = json.loads((ROOT / "configs/native-train.example.json").read_text())
            path = folder / "train.json"
            path.write_text(json.dumps(original))
            old = read_config(path)
            self.assertNotIn("training_profile", old["lora"])
            original["lora"]["training_profile"] = "attention_lora_only_v1"
            path.write_text(json.dumps(original))
            self.assertEqual(read_config(path)["lora"]["training_profile"], "attention_lora_only_v1")
            original["lora"]["training_profile"] = "pure_but_train_heads"
            path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "profile"):
                read_config(path)


@unittest.skipUnless(AVAILABLE, "Optional real Torch/official GPT required; no substitute model")
class PureAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def model(self, profile=PURE_LORA_PROFILE, dropout=0):
        return make_tiny_official_model(OFFICIAL, seed=151, lora_dropout=dropout, training_profile=profile)

    def trainer(self, profile=PURE_LORA_PROFILE):
        model = self.model(profile, dropout=.2)
        seed_everything(321)
        optimizer = torch.optim.AdamW(model.optimizer_groups(.02), lr=.01, foreach=False)
        scheduler = cosine_scheduler(optimizer, max_updates=4, warmup_updates=1)
        config = LoopConfig(max_updates=4, grad_accumulation=2, seed=71, save_interval=2,
                            validation_interval=2, max_validation_examples=1)
        # Actual random tiny model and real autograd; fixed synthetic features.
        return StatefulTrainer(model, optimizer, scheduler, [batch()], config,
            {"fixture": "actual-tiny-pure-profile", "adapter_spec": model.adapter_spec()},
            validation_dataset=[batch()])

    def payload(self, model, step=1):
        return {"schema_version": 1, "kind": "hi_h200_native_adapter", "step": step,
                "base_identity": model.native_identity, "adapter_spec": model.adapter_spec(),
                "trainable": model.trainable_state()}

    def test_only_attention_ab_gradients_and_all_other_bytes_remain_frozen(self):
        model = self.model().eval()
        self.assertEqual(len(model.lora_targets), 2)  # One real tiny GPT layer.
        expected = {"base." + target + suffix for target in model.lora_targets for suffix in (".lora_A", ".lora_B")}
        self.assertEqual(set(model.trainable_state()), expected)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 192)
        self.assertEqual(model.adapter_spec()["heads"], [])
        self.assertEqual(model.adapter_spec()["text_rows"], [])
        self.assertTrue(model.adapter_spec()["frozen_embeddings"])
        frozen = {name: value.detach().clone() for name, value in model.state_dict().items() if name not in expected}
        before = preserved_state_digest(model)
        optimizer = torch.optim.AdamW(model.optimizer_groups(.1), lr=.03, foreach=False)
        initial = model(batch())["loss"].item()
        a_nonzero = False
        for index in range(30):
            optimizer.zero_grad(set_to_none=True)
            model(batch())["loss"].backward()
            for name, parameter in model.named_parameters():
                if name in expected:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    if index > 0 and name.endswith(".lora_A") and parameter.grad.abs().sum() > 0:
                        a_nonzero = True
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        self.assertTrue(a_nonzero)
        self.assertLess(model(batch())["loss"].item(), initial * .98)
        self.assertEqual(preserved_state_digest(model), before)
        for name, value in frozen.items():
            # Byte equality also distinguishes signed zero, unlike float equality.
            self.assertTrue(torch.equal(model.state_dict()[name].contiguous().view(torch.uint8),
                                        value.contiguous().view(torch.uint8)), name)

    def test_legacy_spec_unchanged_and_pure_guards_protect_hi_row_and_heads(self):
        legacy = self.model(LEGACY_PROFILE)
        self.assertEqual(legacy.adapter_spec()["format"], "hi_h200_native_batch1_v1")
        self.assertNotIn("training_profile", legacy.adapter_spec())
        self.assertEqual(legacy.adapter_spec()["heads"], ["mel_head", "text_head"])
        model = self.model()
        before = preserved_state_digest(model)
        with torch.no_grad():
            model.base.lang_embedding.weight[17, 0].add_(1)
        self.assertNotEqual(preserved_state_digest(model), before)
        model = self.model()
        model.base.text_head.weight.requires_grad_(True)
        for action in (model.trainable_state, model.optimizer_groups, model.adapter_spec,
                       lambda: model(batch()), lambda: preserved_state_digest(model)):
            with self.assertRaisesRegex(NativeModelError, "exactly.*A/B"):
                action()
        with self.assertRaisesRegex(NativeModelError, "text_rows"):
            from hi_h200_native.model import NativeHindiModel
            NativeHindiModel(self.model().base, lora_rank=2, training_profile=PURE_LORA_PROFILE, text_rows=(3,))

    def test_pure_adapter_roundtrip_and_cross_profile_or_extra_tensor_rejected(self):
        model = self.model().eval()
        optimizer = torch.optim.AdamW(model.optimizer_groups(0), lr=.01)
        model(batch())["loss"].backward()
        optimizer.step()
        payload = self.payload(model)
        self.assertTrue(all(name.endswith((".lora_A", ".lora_B")) for name in payload["trainable"]))
        path = self.root / "adapter.pt"
        torch.save(payload, path)
        fresh = self.model().eval()
        load_native_adapter(fresh, path, file_sha256(path))
        torch.testing.assert_close(model(batch())["code_logits"], fresh(batch())["code_logits"], rtol=0, atol=0)
        legacy = self.model(LEGACY_PROFILE)
        payload["base_identity"] = legacy.native_identity  # Isolate strict spec rejection.
        torch.save(payload, path)
        with self.assertRaisesRegex(NativeModelError, "settings/row"):
            load_native_adapter(legacy, path, file_sha256(path))
        state = model.trainable_state()
        state["base.mel_head.weight"] = model.base.mel_head.weight.detach().clone()
        with self.assertRaisesRegex(NativeModelError, "inventory"):
            fresh.load_trainable_state(state)

    def test_real_pure_stop_resume_is_exact_and_mixed_resume_is_rejected(self):
        continuous = self.trainer()
        continuous.run(self.root / "continuous")
        interrupted = self.trainer()
        stopped = interrupted.run(self.root / "stopped", stop_after_updates=2)
        resumed = self.trainer()
        resumed.run(self.root / "stopped", resume=stopped["checkpoint"])
        self.assertEqual(resumed.step, 4)
        for name, value in continuous.model.trainable_state().items():
            torch.testing.assert_close(value, resumed.model.trainable_state()[name], rtol=0, atol=0)
        self.assertEqual(preserved_state_digest(continuous.model), preserved_state_digest(resumed.model))
        legacy = self.trainer(LEGACY_PROFILE)
        # Even copying the outer user identity cannot make incompatible state resumable.
        legacy.identity = interrupted.identity
        with self.assertRaisesRegex(TrainingError, "base or adapter"):
            legacy.load(stopped["checkpoint"])
        self.assertEqual(legacy.step, 0)
        legacy_run = self.trainer(LEGACY_PROFILE)
        old = legacy_run.run(self.root / "legacy", stop_after_updates=2)
        pure = self.trainer()
        pure.identity = legacy_run.identity
        with self.assertRaisesRegex(TrainingError, "base or adapter"):
            pure.load(old["checkpoint"])
        self.assertEqual(pure.step, 0)

    def test_real_pure_checkpoint_to_adapter_preserves_only_ab(self):
        trainer = self.trainer()
        result = trainer.run(self.root / "run", stop_after_updates=2)
        state = torch.load(result["checkpoint"], weights_only=True, map_location="cpu")
        model = self.model(dropout=.2)
        options = {"loop": asdict(trainer.config), "optimizer": {"learning_rate": .01, "weight_decay": .02}}
        payload = validate_checkpoint(state, model, trainer.identity, options, 2, 1)
        output = self.root / "selected"
        published = publish_adapter(payload, model, output, {"selected_step": 2})
        self.assertTrue(published["adapter_load_verified"])
        saved = torch.load(output / "adapter.pt", weights_only=True, map_location="cpu")
        self.assertTrue(all(name.endswith((".lora_A", ".lora_B")) for name in saved["trainable"]))
        self.assertNotIn("optimizer", saved)
        restored = self.model(dropout=.2).eval()
        load_native_adapter(restored, output / "adapter.pt", file_sha256(output / "adapter.pt"))
        trainer.model.eval()
        torch.testing.assert_close(trainer.model(batch())["loss"], restored(batch())["loss"], rtol=0, atol=0)

    def test_optional_pure_export_strict_stock_reload_has_frozen_heads_and_rows(self):
        model = self.model().eval()
        optimizer = torch.optim.AdamW(model.optimizer_groups(0), lr=.01)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(batch())["loss"].backward()
            optimizer.step()
        path = self.root / "pure.pt"
        torch.save(self.payload(model, 3), path)
        digest = file_sha256(path)
        output = self.root / "private-acceptance-merge"
        # Inject only the real tiny constructor instead of unavailable large assets.
        # The production adapter parser, strict load, merge and official schema run.
        with patch("hi_h200_native.export.load_native_model", side_effect=lambda *args, **kwargs:
                   make_tiny_official_model(OFFICIAL, seed=151, training_profile=kwargs["training_profile"])) as load:
            export_from_adapter(OFFICIAL, self.root / "unused-base", self.root / "unused-manifest", path, digest, output)
        self.assertEqual(load.call_args.kwargs["training_profile"], PURE_LORA_PROFILE)
        stock = import_official_model(OFFICIAL)(**model.native_config["gpt"]).eval()
        state = torch.load(output / "gpt.pth", weights_only=True, map_location="cpu")["model"]
        stock.load_state_dict(state, strict=True)
        for name in ("lang_embedding.weight", "text_embedding.weight", "text_head.weight", "text_head.bias", "mel_head.weight", "mel_head.bias"):
            self.assertTrue(torch.equal(state[name], model.base.state_dict()[name]))
        inputs = model.training_inputs(batch())
        text, code = stock.get_logits(inputs["condition"], inputs["text_embeddings"], stock.text_head,
                                     inputs["mel_embeddings"], stock.mel_head)
        torch.testing.assert_close(code, model(batch())["code_logits"], rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(text[:, :, :4], model(batch())["text_logits"], rtol=1e-5, atol=2e-6)
        self.assertIn("两个head均保持原始值", (output / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
