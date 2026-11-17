"""Native CLI gates plus real dataset/tokenizer/GPT/optimizer CPU integration.

The integration uses explicitly synthetic tones, handcrafted test features and
random tiny official weights. It does NOT forge production extraction provenance
or claim to load an official pretrained snapshot.
"""
from __future__ import annotations
import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import wave

from hi_h200_native.train import (launch_lease, main, protect_output, read_config,
                                  split_identity, train)

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
NATIVE = (importlib.util.find_spec("torch") is not None
          and importlib.util.find_spec("transformers") is not None
          and importlib.util.find_spec("whisper") is not None and OFFICIAL.is_dir())


class NativeTrainGateTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / ".runtime/native-train-tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.root = Path(self.temp.name)
        for name in ("train.json", "val.json"):
            (self.root / name).write_text("{}")
        self.config = {
            "schema_version": 1, "train_dataset": "train.json", "val_dataset": "val.json",
            "output_dir": "run",
            "loop": {"max_updates": 4, "grad_accumulation": 2, "seed": 31, "device": "cpu",
                     "precision": "fp32", "grad_clip": 1., "save_interval": 2,
                     "validation_interval": 2, "max_validation_examples": 2},
            "optimizer": {"learning_rate": .001, "weight_decay": .01,
                          "warmup_updates": 1, "min_lr_ratio": .1},
            "lora": {"rank": 2, "alpha": 4., "dropout": .1, "text_loss_weight": .1}}
        self.path = self.root / "native.json"
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.config))

    def test_real_config_paths_and_unknown_fields(self):
        options = read_config(self.path)
        self.assertEqual(options["train_dataset"], str(self.root / "train.json"))
        self.assertEqual(options["output_dir"], str(self.root / "run"))
        self.config["fake_trainer"] = True
        self.write()
        with self.assertRaisesRegex(ValueError, "Unknown"):
            read_config(self.path)

    def test_nested_unknown_nan_and_duplicate_config_rejected(self):
        self.config["lora"]["silently_ignored"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "Unknown"):
            read_config(self.path)
        del self.config["lora"]["silently_ignored"]
        self.config["optimizer"]["learning_rate"] = float("nan")
        self.write()
        with self.assertRaises(ValueError):
            read_config(self.path)
        self.path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_config(self.path)

    def test_cuda_permission_gate_precedes_asset_or_torch_loading(self):
        self.config["loop"].update(device="cuda:0", precision="bf16")
        self.write()
        with self.assertRaisesRegex(ValueError, "explicit --allow-cuda"):
            train(self.path)
        self.assertFalse((self.root / "run").exists())

    def test_missing_assets_fail_cleanly_without_success(self):
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            status = main(["--config", str(self.path)])
        self.assertEqual(status, 2)
        record = json.loads(out.getvalue())
        self.assertEqual(record["status"], "blocked_or_failed")
        self.assertFalse(record["training_success_asserted"])
        self.assertFalse((self.root / "run").exists())

    def test_output_cannot_overlap_original_source_model_or_data(self):
        model = self.root / "model"
        model.mkdir()
        for output in (model, model / "runs", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlaps"):
                protect_output(output, [model])
        self.assertEqual(protect_output(self.root / "independent", [model]), self.root / "independent")

    def test_launch_lease_prevents_concurrent_model_loads(self):
        output = self.root / "run"
        with launch_lease(output):
            with self.assertRaisesRegex(ValueError, "Another native launch"):
                with launch_lease(output):
                    self.fail("concurrent lease acquired")

    def test_split_names_shared_identity_and_all_leak_sets(self):
        def dataset(split, prefix):
            return SimpleNamespace(identity={"split": split, **{key: "same" for key in
                ("official_commit", "model_snapshot_revision", "model_manifest_sha256",
                 "tokenizer_sha256", "config_sha256", "frontend")}},
                **{key: frozenset({prefix + key}) for key in
                   ("speaker_ids", "clip_ids", "audio_hashes", "text_hashes", "feature_hashes")})
        first, second = dataset("train", "train-"), dataset("val", "val-")
        self.assertEqual(set(split_identity(first, second)), {"train", "val"})
        for key in ("speaker_ids", "clip_ids", "audio_hashes", "text_hashes", "feature_hashes"):
            changed = dataset("val", "val-")
            setattr(changed, key, getattr(first, key))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "leakage"):
                split_identity(first, changed)
        second.identity["tokenizer_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differ"):
            split_identity(first, second)
        second = dataset("test", "val-")
        with self.assertRaisesRegex(ValueError, "held-out val"):
            split_identity(first, second)


@unittest.skipUnless(NATIVE, "Real Torch/official GPT/tokenizer dependencies required; no mocked success")
class NativeTrainIntegrationTests(unittest.TestCase):
    def test_actual_native_dataset_official_tokenizer_gpt_and_optimizer(self):
        import numpy as np
        import torch
        from hi_h200.data import prepare, read_jsonl
        from hi_h200_native.data import (NativeFeatureDataset, PREPARED_FILES, encoded_text,
                                        file_sha256, model_limits, official_tokenizer)
        from hi_h200_native.engine import LoopConfig, StatefulTrainer, cosine_scheduler, seed_everything
        from hi_h200_native.model import NativeHindiModel, OFFICIAL_COMMIT, import_official_model
        directory = ROOT / ".runtime/native-train-tests"
        directory.mkdir(parents=True, exist_ok=True)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                root = Path(temporary)
                model_dir, audio, features = root / "tiny-tokenizer", root / "audio", root / "features"
                for folder in (model_dir, audio, features):
                    folder.mkdir()
                # Real official byte-BPE tokenizer; a tiny test vocabulary, not
                # the shipped multilingual vocabulary and never a model snapshot.
                ranks = "".join(base64.b64encode(bytes([i])).decode() + " " + str(i) + "\n" for i in range(256))
                (model_dir / "multilingual_zh_ja_yue_char_del.tiktoken").write_text(ranks)
                cls = import_official_model(OFFICIAL)
                tokenizer = official_tokenizer(OFFICIAL, model_dir)
                texts = ("नमस्ते मित्र।", "आज मौसम अच्छा है।", "कृपया यहाँ बैठिए।",
                         "यह कहानी सुंदर है।", "कल घर जाएँगे।", "फिर मिलेंगे मित्र।")
                source = []
                for index, text in enumerate(texts):
                    filename = f"tone{index}.wav"
                    with wave.open(str(audio / filename), "wb") as stream:
                        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                        stream.writeframes(struct.pack("<h", index + 1) * 24000)
                    source.append(dict(id=f"clip{index}", speaker_id=f"speaker{index//2}",
                                       audio=filename, text=text, language="hi", consent=True,
                                       source="synthetic-integration-not-aligned-speech", license="synthetic-only"))
                raw = root / "raw.jsonl"
                raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source))
                prepare(raw, audio, root / "prepared")
                prepared = list(read_jsonl(root / "prepared/train.jsonl"))
                manifest_rows, inventory = [], []
                for index, row in enumerate(prepared):
                    text_ids = encoded_text(tokenizer, row["text_norm"])
                    path = features / (row["id"] + ".npz")
                    np.savez(path, text_ids=np.asarray(text_ids, dtype="int32"),
                             codes=np.arange(8, dtype="int32"),
                             spk_emb=np.full(192, .01 * (index + 1), dtype="float32"),
                             emo_vec=np.full(16, .02 * (index + 1), dtype="float32"))
                    manifest_rows.append(dict(id=row["id"], speaker_id=row["speaker_id"],
                        text_norm=row["text_norm"], feature=path.name, text_len=len(text_ids), code_len=8))
                    inventory.append(dict(id=row["id"], feature=path.name, sha256=file_sha256(path),
                        audio_pcm_sha256=row["audio_sha256"], text_sha256=row["text_sha256"]))
                manifest = features / "manifest.jsonl"
                manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows))
                config = {"version": "2.5", "gpt": {"number_text_tokens": 512, "max_text_tokens": 128,
                    "max_mel_tokens": 24, "model_dim": 16, "number_mel_codes": 34,
                    "start_mel_token": 32, "stop_mel_token": 33, "start_text_token": 0, "stop_text_token": 1},
                    "semantic_codec": {"codebook_size": 32}}
                provenance = {"features": inventory, "prepared_sha256": {
                    name: file_sha256(root / "prepared" / name) for name in PREPARED_FILES},
                    "fixture": "handcrafted synthetic arrays; actual audio extraction NOT executed"}
                dataset = NativeFeatureDataset(prepared, features, manifest, provenance,
                    model_limits(config), tokenizer, {"official_commit": OFFICIAL_COMMIT,
                    "model_snapshot_revision": "tiny-random-test-not-a-pretrained-snapshot"})
                seed_everything(81)
                base = cls(layers=1, model_dim=16, heads=2, max_text_tokens=128, max_mel_tokens=24,
                    max_conditioning_inputs=3, number_text_tokens=512, number_mel_codes=34,
                    start_text_token=0, stop_text_token=1, start_mel_token=32, stop_mel_token=33,
                    checkpointing=False, spk_cond_mode="campplus", use_accel=False,
                    emo_condition_module={"output_size":16,"linear_units":32,"attention_heads":2,
                        "num_blocks":1,"input_layer":"linear","perceiver_mult":1})
                model = NativeHindiModel(base, lora_rank=2, lora_alpha=4., lora_dropout=.1,
                                         text_rows=dataset.text_rows)
                model.native_identity = {"kind": "explicit-random-official-GPT-CPU-integration"}
                optimizer = torch.optim.AdamW(model.optimizer_groups(), lr=.003, foreach=False)
                scheduler = cosine_scheduler(optimizer, max_updates=2)
                initial = model.trainable_state()
                trainer = StatefulTrainer(model, optimizer, scheduler, dataset,
                    LoopConfig(max_updates=2, grad_accumulation=2, save_interval=1,
                        validation_interval=1), identity={"fixture": "synthetic", "data": dataset.identity})
                result = trainer.run(root / "run")
                self.assertEqual(result["updates_this_run"], 2)
                self.assertTrue(Path(result["checkpoint"]).is_file())
                self.assertTrue(any(not torch.equal(initial[name], value)
                                    for name, value in model.trainable_state().items()))
                rows = [json.loads(line) for line in (root / "run/updates.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in rows], [1, 2])
                self.assertTrue(all(row["micro_examples"] == 2 and row["update_wall_seconds"] > 0 and row["examples_per_second"] > 0 for row in rows))
                self.assertTrue(all("cuda_peak_allocated_bytes" not in row for row in rows))
                self.assertEqual(type(model.base).__name__, "UnifiedVoice")
                self.assertIn(tokenizer.encode("<|hi|>", allowed_special={"<|hi|>"})[0], dataset.text_rows)
                self.assertFalse(dataset.identity["runtime_audio_feature_parity_verified"])
        finally:
            torch.set_num_threads(old_threads)
