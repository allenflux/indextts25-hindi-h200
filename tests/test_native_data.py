"""Original native data checks with real NumPy files and synthetic PCM only."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from hi_h200.data import prepare, read_jsonl
from hi_h200_native.data import (FRONTEND, OFFICIAL_COMMIT, PREPARED_FILES,
                                NativeDataError, file_sha256, load_native_dataset,
                                model_limits, read_npz)

AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]
TEXTS = ("नमस्ते मेरे मित्र।", "आज मौसम अच्छा है।", "कृपया यहाँ बैठिए।",
         "मुझे हिंदी बोलना पसंद है।", "कल हम घर जाएँगे।", "यह एक सुंदर कहानी है।")


class TestTokenizer:
    """Test-only deterministic token mapping, never an actual model tokenizer."""
    def encode(self, text, allowed_special):
        assert allowed_special == {"<|hi|>"}
        assert text.startswith("<|hi|>")
        return [300] + [2 + ord(c) % 250 for c in text[6:]]


@unittest.skipUnless(AVAILABLE, "Optional native NumPy dependency not installed")
class NativeDataTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="native-data-", dir=runtime)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio"
        self.audio.mkdir()
        self.prepared = self.root / "prepared"
        rows = []
        for index, text in enumerate(TEXTS):
            filename = f"clip{index}.wav"
            with wave.open(str(self.audio / filename), "wb") as out:
                out.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                out.writeframes(struct.pack("<h", index + 1) * 24000)
            rows.append(dict(id=f"clip{index}", speaker_id=f"speaker{index // 2}", audio=filename,
                             text=text, language="hi", source="synthetic-unit-test-only",
                             license="synthetic-no-speech", consent=True))
        raw = self.root / "raw.jsonl"
        raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        prepare(raw, self.audio, self.prepared)
        self.rows = list(read_jsonl(self.prepared / "train.jsonl"))
        self.features = self.root / "features"
        self.features.mkdir()
        self.manifest = self.features / "manifest.jsonl"
        self.provenance = self.features / "provenance.json"
        self.tokenizer = TestTokenizer()
        manifest, inventory = [], []
        for index, row in enumerate(self.rows):
            arrays = self.arrays(row, index)
            filename = row["id"] + ".npz"
            np.savez(self.features / filename, **arrays)
            manifest.append(dict(id=row["id"], speaker_id=row["speaker_id"], text_norm=row["text_norm"],
                                 feature=filename, text_len=len(arrays["text_ids"]), code_len=len(arrays["codes"])))
            inventory.append(dict(id=row["id"], feature=filename, sha256=file_sha256(self.features / filename),
                                  audio_pcm_sha256=row["audio_sha256"], text_sha256=row["text_sha256"]))
        self.write_manifest(manifest)
        self.identity = dict(official_commit=OFFICIAL_COMMIT, model_snapshot_revision="a" * 40,
                             model_manifest_sha256="b" * 64, tokenizer_sha256="c" * 64)
        self.config = {"version": "2.5", "gpt": {"number_text_tokens": 512, "max_text_tokens": 600,
                       "max_mel_tokens": 1815, "model_dim": 16, "number_mel_codes": 34,
                       "start_mel_token": 32, "stop_mel_token": 33,
                       "start_text_token": 0, "stop_text_token": 1}, "semantic_codec": {"codebook_size": 32}}
        self.expected_extractor = {"implementation": "synthetic-test-only-not-extracted-speech", "source_sha256": "d" * 64}
        self.prov = dict(schema_version=1, kind="hi_h200_native_features", **self.identity,
                         frontend=FRONTEND, language="hi", feature_manifest_sha256=file_sha256(self.manifest),
                         prepared_sha256={name: file_sha256(self.prepared / name) for name in PREPARED_FILES},
                         extractor=dict(self.expected_extractor),
                         features=inventory)
        self.write_provenance()
        # Existing dummy paths satisfy config path confinement; model verifier is
        # explicitly injected in these unit tests, never used by the CLI.
        self.model = self.root / "model"
        self.model.mkdir()
        self.model_manifest = self.root / "model_manifest.json"
        self.model_manifest.write_text("{}")
        self.official = self.root / "official"
        self.official.mkdir()
        self.data_config = self.root / "data.json"
        self.options = dict(schema_version=1, prepared_dir="prepared", audio_root="audio",
                            feature_manifest="features/manifest.jsonl", feature_root="features",
                            provenance="features/provenance.json", model_dir="model",
                            model_manifest="model_manifest.json", official_repo="official", split="train")
        self.write_config()

    def tearDown(self):
        self.temp.cleanup()

    def arrays(self, row, index=0):
        np = self.np
        return dict(codes=np.arange(8, dtype="int32"),
                    text_ids=np.asarray(self.tokenizer.encode("<|hi|> " + row["text_norm"], {"<|hi|>"}), dtype="int32"),
                    spk_emb=np.full(192, index + 1, dtype="float32"),
                    emo_vec=np.full(16, index + 11, dtype="float32"))

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def write_provenance(self):
        self.provenance.write_text(json.dumps(self.prov, ensure_ascii=False))

    def write_config(self):
        self.data_config.write_text(json.dumps(self.options))

    def load(self):
        return load_native_dataset(self.data_config, tokenizer=self.tokenizer,
                                   asset_verifier=lambda *args: (self.config, self.identity),
                                   extractor_identity_provider=lambda: dict(self.expected_extractor))

    @unittest.skipUnless(TORCH and importlib.util.find_spec("transformers") is not None
                         and importlib.util.find_spec("whisper") is not None
                         and Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official")).is_dir(),
                         "Actual pinned official tiny model/tokenizer dependencies required")
    def test_example_config_drives_real_tiny_assets_and_native_dataset_validation(self):
        """The shipped JSON is the entry point; missing schema must fail here.

        This uses real random tiny official weights, byte-BPE and physical PCM/
        NPZ files, not a pretrained snapshot or an audio-extraction quality test.
        Only the handcrafted features' test-only extractor identity is injected.
        """
        import torch
        from hi_h200_native.data import encoded_text, official_tokenizer
        from hi_h200_native.model import import_official_model, load_verified_base
        official = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official")).resolve()
        # An actual tiny official base can be saved and strictly reloaded; the
        # name/revision below explicitly serves as a local synthetic test fixture.
        tiny = {"layers":1, "model_dim":16, "heads":2, "max_text_tokens":128,
                "max_mel_tokens":24, "max_conditioning_inputs":3,
                "number_text_tokens":512, "number_mel_codes":34,
                "start_text_token":0, "stop_text_token":1,
                "start_mel_token":32, "stop_mel_token":33, "checkpointing":False,
                "emo_condition_module":{"output_size":16,"linear_units":32,"attention_heads":2,
                                        "num_blocks":1,"input_layer":"linear","perceiver_mult":1}}
        actual_config = {"version":"2.5", "gpt":tiny, "semantic_codec":{"codebook_size":32}}
        (self.model / "config.yaml").write_text(json.dumps(actual_config))
        ranks = "".join(base64.b64encode(bytes([i])).decode() + " " + str(i) + "\n" for i in range(256))
        (self.model / "multilingual_zh_ja_yue_char_del.tiktoken").write_text(ranks)
        (self.model / "LICENSE").write_bytes((official / "LICENSE").read_bytes())
        cls = import_official_model(official)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(321)
            base = cls(**tiny, spk_cond_mode="campplus", use_accel=False)
        torch.save(base.state_dict(), self.model / "gpt.pth")
        files = {name:file_sha256(self.model / name) for name in
                 ("config.yaml","multilingual_zh_ja_yue_char_del.tiktoken","LICENSE","gpt.pth")}
        self.model_manifest.write_text(json.dumps({"repository":"IndexTeam/IndexTTS-2.5",
                                                   "snapshot_revision":"f" * 40,"files":files}))
        _, config, identity = load_verified_base(official, self.model, self.model_manifest)
        self.assertEqual(config, actual_config)
        self.assertEqual(identity["config_sha256"], files["config.yaml"])
        tokenizer = official_tokenizer(official, self.model)
        records, inventory = [], []
        for index, row in enumerate(self.rows):
            arrays = self.arrays(row, index)
            arrays["text_ids"] = self.np.asarray(encoded_text(tokenizer, row["text_norm"]), dtype="int32")
            filename = row["id"] + ".npz"
            self.np.savez(self.features / filename, **arrays)
            records.append(dict(id=row["id"],speaker_id=row["speaker_id"],text_norm=row["text_norm"],
                                feature=filename,text_len=len(arrays["text_ids"]),code_len=len(arrays["codes"])))
            inventory.append(dict(id=row["id"],feature=filename,sha256=file_sha256(self.features / filename),
                                  audio_pcm_sha256=row["audio_sha256"],text_sha256=row["text_sha256"]))
        self.write_manifest(records)
        self.prov.update({key:identity[key] for key in
                          ("official_commit","model_snapshot_revision","model_manifest_sha256","tokenizer_sha256")})
        self.prov.update(features=inventory,feature_manifest_sha256=file_sha256(self.manifest))
        self.write_provenance()
        example = json.loads((ROOT / "configs/native-dataset.example.json").read_text())
        # Fill only keys already supplied by the example; never supply a missing
        # schema_version from the implementation or a hand-written test config.
        replacements = {**self.options, "official_repo":str(official)}
        for key in example:
            if key != "schema_version" and key in replacements:
                example[key] = replacements[key]
        self.data_config.write_text(json.dumps(example))
        dataset = load_native_dataset(self.data_config,
                    extractor_identity_provider=lambda: dict(self.expected_extractor))
        self.assertEqual(len(dataset), 2)
        batch = dataset[0]
        self.assertEqual(batch["text_ids"].tolist(), encoded_text(tokenizer, self.rows[0]["text_norm"]))
        self.assertEqual(dataset.identity["model_manifest_sha256"], file_sha256(self.model_manifest))
        self.assertNotEqual(batch["id"], batch["prompt_id"])
        self.assertEqual(tuple(batch["spk_emb"].shape), (192,))
        self.assertFalse(dataset.identity["runtime_audio_feature_parity_verified"])
        # Demonstrate the production schema boundary explicitly for the exact
        # previously broken example, rather than merely asserting a JSON value.
        del example["schema_version"]
        self.data_config.write_text(json.dumps(example))
        with self.assertRaisesRegex(NativeDataError, "schema/split"):
            load_native_dataset(self.data_config)
        from hi_h200_native.extract import ExtractionError, extract_config
        with self.assertRaisesRegex(ExtractionError, "Invalid native extraction configuration"):
            extract_config(self.data_config, device="cpu")

    def test_changed_extractor_recipe_or_source_requires_reextraction(self):
        for field, value in (("implementation", "foreign-extractor.v1"),
                             ("source_sha256", "e" * 64)):
            self.prov["extractor"] = dict(self.expected_extractor)
            self.prov["extractor"][field] = value
            self.write_provenance()
            with self.subTest(field=field), self.assertRaisesRegex(NativeDataError, "extractor recipe/source"):
                self.load()

    def test_production_default_rejects_declared_test_extractor(self):
        with self.assertRaisesRegex(NativeDataError, "extractor recipe/source"):
            load_native_dataset(self.data_config, tokenizer=self.tokenizer,
                                asset_verifier=lambda *args: (self.config, self.identity))

    def test_real_npz_dataset_identity_and_prompt_contract(self):
        dataset = self.load()
        self.assertEqual(len(dataset), 2)
        self.assertFalse(dataset.identity["runtime_audio_feature_parity_verified"])
        self.assertIn(300, dataset.text_rows)
        self.assertEqual(len(dataset.speaker_ids), 1)
        self.assertEqual(len(dataset.clip_ids), 2)
        self.assertEqual(len(dataset.audio_hashes), 2)
        self.assertEqual(len(dataset.text_hashes), 2)
        self.assertEqual(len(dataset.feature_hashes), 2)
        self.assertEqual(dataset.identity["split"], "train")
        if TORCH:
            first = dataset[0]
            self.assertNotEqual(first["id"], first["prompt_id"])
            self.assertEqual(first["text_ids"].dtype.__str__(), "torch.int64")
            self.assertTrue((first["spk_emb"] == 2).all())
            self.assertTrue((first["emo_vec"] == 11).all())

    def test_model_source_mismatch_and_stale_provenance_fail(self):
        self.prov["official_commit"] = "0" * 40
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "different source"):
            self.load()
        self.prov["official_commit"] = OFFICIAL_COMMIT
        self.prov["prepared_sha256"]["manifest.json"] = "0" * 64
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "stale"):
            self.load()

    def test_missing_clip_and_stale_tokens_fail(self):
        manifest = list(read_jsonl(self.manifest))
        self.write_manifest(manifest[:1])
        self.prov["feature_manifest_sha256"] = file_sha256(self.manifest)
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "omitted"):
            self.load()
        self.write_manifest(manifest)
        row = self.rows[0]
        arrays = self.arrays(row)
        arrays["text_ids"][1] += 1
        path = self.features / (row["id"] + ".npz")
        self.np.savez(path, **arrays)
        self.prov["features"][0]["sha256"] = file_sha256(path)
        self.prov["feature_manifest_sha256"] = file_sha256(self.manifest)
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "cached tokens"):
            self.load()

    def test_audio_tamper_and_changed_feature_fail(self):
        dataset = self.load()
        path = self.features / (self.rows[0]["id"] + ".npz")
        arrays = self.arrays(self.rows[0])
        arrays["spk_emb"][0] = 123
        self.np.savez(path, **arrays)
        with self.assertRaisesRegex(NativeDataError, "changed after"):
            dataset._arrays(self.rows[0]["id"])
        with (self.audio / self.rows[0]["audio"]).open("ab") as out:
            out.write(b"tampered")
        # PCM validation rejects malformed/trailing audio only if it affects PCM;
        # replacing a sample definitively changes the audited PCM hash.
        data = bytearray((self.audio / self.rows[0]["audio"]).read_bytes())
        data[44] ^= 1
        (self.audio / self.rows[0]["audio"]).write_bytes(data)
        with self.assertRaisesRegex(NativeDataError, "Prepared data failed"):
            self.load()

    def test_paths_duplicate_and_reserved_tokens_fail(self):
        rows = list(read_jsonl(self.manifest))
        rows[0]["feature"] = "../outside.npz"
        self.write_manifest(rows)
        self.prov["feature_manifest_sha256"] = file_sha256(self.manifest)
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "Unsafe"):
            self.load()
        arrays = self.arrays(self.rows[0])
        arrays["text_ids"][1] = 0
        path = self.features / "reserved.npz"
        self.np.savez(path, **arrays)
        with self.assertRaisesRegex(NativeDataError, "framing"):
            read_npz(path, model_limits(self.config))
        arrays["text_ids"][1] = 100
        arrays["codes"][0] = 32
        self.np.savez(path, **arrays)
        with self.assertRaisesRegex(NativeDataError, "codec"):
            read_npz(path, model_limits(self.config))

    def test_nonfinite_arrays_and_invalid_model_limits_fail(self):
        arrays = self.arrays(self.rows[0])
        arrays["emo_vec"][0] = self.np.nan
        path = self.features / "nonfinite.npz"
        self.np.savez(path, **arrays)
        with self.assertRaisesRegex(NativeDataError, "Nonfinite"):
            read_npz(path, model_limits(self.config))
        self.config["version"] = "2.0"
        with self.assertRaisesRegex(NativeDataError, "version 2.5"):
            self.load()

    def test_object_npz_and_symlink_escape_fail(self):
        arrays = self.arrays(self.rows[0])
        arrays["text_ids"] = self.np.asarray([object()], dtype=object)
        outside = self.root / "outside.npz"
        self.np.savez(outside, **arrays)
        with self.assertRaisesRegex(NativeDataError, "safely read"):
            read_npz(outside, model_limits(self.config))
        rows = list(read_jsonl(self.manifest))
        linked = self.features / "escaped.npz"
        linked.symlink_to(outside)
        rows[0]["feature"] = "escaped.npz"
        self.write_manifest(rows)
        self.prov["feature_manifest_sha256"] = file_sha256(self.manifest)
        self.prov["features"][0]["feature"] = "escaped.npz"
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "escapes"):
            self.load()

    def test_wrong_speaker_and_noninteger_lengths_fail(self):
        rows = list(read_jsonl(self.manifest))
        rows[0]["speaker_id"] = "another-speaker"
        self.write_manifest(rows)
        self.prov["feature_manifest_sha256"] = file_sha256(self.manifest)
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "speaker/text"):
            self.load()
        rows[0]["speaker_id"] = self.rows[0]["speaker_id"]
        rows[0]["text_len"] = True
        self.write_manifest(rows)
        self.prov["feature_manifest_sha256"] = file_sha256(self.manifest)
        self.write_provenance()
        with self.assertRaisesRegex(NativeDataError, "length metadata"):
            self.load()

    def test_unknown_config_and_ambiguous_json_fail(self):
        self.options["unexpected"] = True
        self.write_config()
        with self.assertRaisesRegex(NativeDataError, "schema"):
            self.load()
        self.data_config.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(NativeDataError, "Invalid"):
            self.load()


if __name__ == "__main__":
    unittest.main()
