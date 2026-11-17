"""CPU-only security and integration tests; never import the external backend."""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
import json
from pathlib import Path
import tempfile
import unittest
import wave
import sys
from unittest import mock

from scripts import check_backend as backend
from scripts import check_features as features
from scripts import monitor_gpu as monitor
from scripts import render_backend_config as renderer

try:
    import numpy as np
except ImportError:
    np = None

RUNTIME = Path(__file__).resolve().parents[1] / ".runtime" / "backend-tests"


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=RUNTIME)
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path


class BackendTests(WorkspaceTest):
    def test_static_argparse_inspection_never_executes_source(self):
        source = self.write("train.py", "raise RuntimeError('must never import')\np.add_argument('--amp')\np.add_argument('--no-train-heads',dest='train_heads')\n")
        self.assertEqual(backend.cli_destinations(source), {"amp", "train_heads"})

    def test_strict_json_rejects_ambiguous_audit_records(self):
        for value in ('{"permission_confirmed":false,"permission_confirmed":true}', '{"learning_rate":NaN}', '{"learning_rate":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend.strict_json(value)

    def test_source_inspection_disables_fsmonitor(self):
        result = mock.Mock(stdout="expected\n")
        with mock.patch.object(backend.subprocess, "run", return_value=result) as run:
            self.assertEqual(backend.git_read(self.root, "status", "--porcelain"), "expected")
        command = run.call_args.args[0]
        self.assertIn("core.fsmonitor=false", command)
        self.assertIn("core.hooksPath=/dev/null", command)
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_rights_record_needs_exact_commit_and_explicit_confirmation(self):
        record = self.write("rights.json", json.dumps({"backend_commit": "a" * 40, "permission_confirmed": False, "permission_basis": "Permission still not obtained"}))
        with self.assertRaises(backend.PreflightError):
            backend.check_rights_record(record, "a" * 40)
        record.write_text(json.dumps({"backend_commit": "b" * 40, "permission_confirmed": True, "permission_basis": "Test-only declared permission"}))
        with self.assertRaises(backend.PreflightError):
            backend.check_rights_record(record, "a" * 40)

    def test_weight_hash_mismatch_and_symlink_escape_fail(self):
        model = self.root / "model"
        model.mkdir()
        self.write("model/config.yaml", "version: 2.5\n")
        spec = {"repository": "IndexTeam/IndexTTS-2.5", "minimum_files": ["config.yaml"]}
        manifest = self.write("weights.json", json.dumps({"repository": spec["repository"], "snapshot_revision": "a" * 40, "files": {"config.yaml": "0" * 64}}))
        with self.assertRaises(backend.PreflightError):
            backend.check_weights(model, manifest, spec)
        other = self.write("outside", "not a model file")
        (model / "escaped").symlink_to(other)
        with self.assertRaises(backend.PreflightError):
            backend.contained_file(model, "escaped")

    def test_unknown_backend_fields_are_rejected(self):
        config = backend.read_json(backend.ROOT / "configs/hindi-h200.example.json")
        fields = set(config)
        config["deepspeed"] = True
        with self.assertRaisesRegex(backend.PreflightError, "Unknown backend fields"):
            renderer.validate_config(config, fields)

    def test_manifest_uses_real_backend_pair_schema_and_checks_all_rows(self):
        for name in ("first.npz", "second.npz"):
            self.write(name, "not loaded by this schema test")
        row = {"id": "target", "speaker": "speaker-a", "prompt_features": "first.npz", "target_features": "second.npz", "lang": 17, "lang_alias": "hi", "prompt_lang": 17, "cross_lingual": False, "text_len": 3, "code_len": 8}
        path = self.write("pairs.jsonl", json.dumps(row) + "\n")
        _, metadata = renderer.manifest_path(path)
        self.assertEqual(metadata["speakers"], {"speaker-a"})
        self.assertEqual(metadata["rows"], 1)
        with path.open("a") as stream:
            stream.write('{"id":"bad"}\n')
        with self.assertRaises(backend.PreflightError):
            renderer.manifest_path(path)

    def test_output_cannot_overwrite_original_weights(self):
        model = self.root / "model"
        with self.assertRaises(backend.PreflightError):
            renderer.output_path(model / "new-run", (model,))

    def test_monitor_rejects_nonfinite_telemetry_without_gpu_access(self):
        for invalid in ("inf", "NaN", "-1"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                monitor.parse_sample("0, GPU-example, 140000, " + invalid + ", 50, 300")
        self.assertEqual(monitor.parse_sample("0, GPU-example, 140000, 12000, 50, 300")[3], "12000")


@unittest.skipIf(np is None, "Optional NumPy not available; backend feature gate remains unvalidated in this interpreter")
class FeatureTests(WorkspaceTest):
    def fixture(self):
        sys.path.insert(0, str(backend.ROOT / "src"))
        from hi_h200.data import prepare, read_jsonl, build_pairs
        from scripts.make_demo import TEXTS
        audio_root = self.root / "audio"
        audio_root.mkdir()
        raw = []
        for index, text in enumerate(TEXTS):
            relative = f"clip-{index}.wav"
            with wave.open(str(audio_root / relative), "wb") as audio:
                audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                audio.writeframes((100 + index).to_bytes(2, "little", signed=True) * 24000)
            raw.append({"id": f"clip-{index}", "speaker_id": f"speaker-{index // 2}", "audio": relative, "text": text, "language": "hi", "source": "synthetic-cpu-test-only", "license": "CC0-1.0", "consent": True})
        source = self.write("raw.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw))
        prepared = self.root / "prepared"
        prepare(source, audio_root, prepared)
        expected = list(read_jsonl(prepared / "train.jsonl"))
        cache = self.root / "cache"
        cache.mkdir()
        records = []
        for row in expected:
            path = cache / (row["id"] + ".npz")
            np.savez(path, codes=np.arange(8, dtype=np.int32), text_ids=np.array([42, 10, 11], dtype=np.int32), spk_emb=np.zeros(192, dtype=np.float32), emo_vec=np.zeros(1280, dtype=np.float32))
            records.append({"id": row["id"], "features": path.name, "speaker": row["speaker_id"], "lang": 17, "lang_alias": "hi", "text": row["text_norm"], "text_len": 3, "code_len": 8})
        feature_manifest = cache / "utterances_train.jsonl"
        feature_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        pairs = cache / "pairs_train.jsonl"
        build_pairs(feature_manifest, prepared, "train", cache, pairs, audio_root,
                    max_text_tokens=600, max_code_tokens=1815, max_prompt_code_tokens=1815)
        args = argparse.Namespace(manifest=feature_manifest, pairs=pairs, feature_root=cache, prepared=prepared, audio_root=audio_root, split="train", number_text_tokens=60509, max_text_tokens=600, max_code_tokens=1815, max_prompt_code_tokens=1815, codebook_size=8192, emotion_dim=1280)
        return args, records

    def test_real_arrays_and_token_parity(self):
        args, _ = self.fixture()
        frontend = mock.Mock()
        frontend.encode.return_value = [42, 10, 11]
        report = features.audit(args, np, frontend)
        self.assertEqual(report["clips"], 2)
        self.assertTrue(report["tokenizer_parity_verified"])
        self.assertFalse(report["gpu_training_verified"])
        self.assertFalse(report["ready_for_training"])

    def test_reserved_codec_ids_fail(self):
        args, records = self.fixture()
        path = args.feature_root / records[0]["features"]
        np.savez(path, codes=np.full(8, 8192, dtype=np.int32), text_ids=np.array([42,10,11], dtype=np.int32), spk_emb=np.zeros(192, dtype=np.float32), emo_vec=np.zeros(1280, dtype=np.float32))
        with self.assertRaisesRegex(backend.PreflightError, "Reserved or out-of-range"):
            features.audit(args, np)

    def test_manifest_length_lie_fails(self):
        args, records = self.fixture()
        records[0]["text_len"] = 2
        args.manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
        with self.assertRaisesRegex(backend.PreflightError, "lengths disagree"):
            features.audit(args, np)

    def test_same_clip_prompt_fails(self):
        args, _ = self.fixture()
        pair_rows = list(features.rows(args.pairs))
        pair_rows[0]["prompt_features"] = pair_rows[0]["target_features"]
        args.pairs.write_text("".join(json.dumps(row) + "\n" for row in pair_rows))
        with self.assertRaisesRegex(backend.PreflightError, "distinct prompt"):
            features.audit(args, np)

    def test_old_frontend_cache_fails(self):
        args, _ = self.fixture()
        frontend = mock.Mock()
        frontend.encode.return_value = [43, 10, 11]
        with self.assertRaisesRegex(backend.PreflightError, "Cached token IDs differ"):
            features.audit(args, np, frontend)

    def test_malicious_npy_shape_rejected_before_array_loading(self):
        args, records = self.fixture()
        path = args.feature_root / records[0]["features"]
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<i4", "fortran_order": False, "shape": (10 ** 12,)})
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("codes.npy", header.getvalue())
            for name in ("text_ids.npy", "spk_emb.npy", "emo_vec.npy"):
                archive.writestr(name, b"not reached")
        with mock.patch.object(np, "load", side_effect=AssertionError("allocation should not occur")):
            with self.assertRaises(backend.PreflightError):
                features.inspect_npz(path, records[0], np, {"max_code_tokens":1815,"max_text_tokens":600,"emotion_dim":1280,"codebook_size":8192,"number_text_tokens":60509})

    def test_feature_limits_bound_to_actual_model_config(self):
        args, _ = self.fixture()
        cfg = argparse.Namespace(gpt=argparse.Namespace(number_text_tokens=60509, max_text_tokens=600, max_mel_tokens=1815, model_dim=1280, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193), semantic_codec=argparse.Namespace(codebook_size=8192))
        features.check_model_limits(cfg, args)
        args.number_text_tokens += 1
        with self.assertRaisesRegex(backend.PreflightError, "actual hashed model config"):
            features.check_model_limits(cfg, args)

    def test_structural_report_cannot_render_training_config(self):
        args, _ = self.fixture()
        report = features.audit(args, np)
        path = self.write("audit.json", json.dumps(report))
        with self.assertRaisesRegex(backend.PreflightError, "backend-tokenizer parity"):
            renderer.feature_report(path, args.pairs, "train", {})


if __name__ == "__main__":
    unittest.main()
