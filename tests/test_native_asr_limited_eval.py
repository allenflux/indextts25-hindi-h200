"""CPU-only schema/metric/audio/privacy tests; no weights or fake ASR execution."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from hi_h200_native import asr_limited_eval as a

ROOT = Path(__file__).resolve().parents[1]


class LimitedASRTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="limited-asr-", dir=ROOT / ".runtime")
        self.root = Path(self.temp.name)
        self.wav = self.root / "private.wav"
        self.audio()
    def tearDown(self):
        self.temp.cleanup()
    def audio(self, *, seconds=1.2, rate=24000, channels=1, width=2):
        with wave.open(str(self.wav), "wb") as out:
            out.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            out.writeframes(b"".join(struct.pack("<h", 500 if i % 2 else -500)
                                   for i in range(int(rate * seconds) * channels)) if width == 2 else b"\x80" * int(rate * seconds) * channels * width)
    def row(self, *, condition="baseline", language="hi", uid="private-id", text="क़लम यहाँ रखिए।", **extra):
        return dict(id=uid, language=language, condition=condition, text=text, audio="private.wav", **extra)
    def manifest(self, rows, name="cases.json"):
        path = self.root / name
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return path
    def load(self, rows):
        return a.load_cases(self.root, [self.manifest(rows)])[0]
    def test_decode_options_are_fixed_except_language(self):
        original = dict(a.base.SETTINGS)
        for language in a.LANGUAGES:
            options = a.decode_settings(language)
            self.assertEqual({**options, "language":"hi"}, original)
            self.assertEqual(options["language"], language)
        self.assertEqual(a.base.SETTINGS, original)
        with self.assertRaises(a.base.ASRError):
            a.decode_settings("fr")
    def test_primary_marks_joiners_and_fixed_caseless_policy(self):
        ref = "क़्\u200dष क्\u200cष ₹२०।"
        self.assertEqual(a.metric_views(ref, ref, "hi")["strict"]["cer"], 0)
        normalized = a.metric_views(ref, ref, "hi")["punctuation_only"]["reference"]
        for mark in ("\u093c", "\u094d", "\u200d", "\u200c", "₹", "२०"):
            self.assertIn(mark, normalized)
        metric = a.metric_views("ÁRBOL", "árbol", "es")
        self.assertGreater(metric["strict"]["cer"], 0)
        self.assertEqual(metric["caseless"]["cer"], 0)
        self.assertGreater(a.metric_views("ÁRBOL", "arbol", "es")["caseless"]["cer"], 0)
        self.assertNotIn("caseless", a.metric_views("नमस्ते", "नमस्ते", "hi"))
    def test_zh_ja_wer_is_explicitly_uninterpretable(self):
        for language, text in (("zh", "你好世界"), ("ja", "こんにちは世界")):
            for value in a.metric_views(text, text, language).values():
                self.assertEqual(value["cer"], 0)
                self.assertIsNone(value["wer"])
                self.assertFalse(value["wer_interpretable"])
    def test_batch_results_and_reference_array_preserve_original_targets(self):
        text = "Hello WORLD."
        row = self.row(language="en", text=text, status="generated", frontend_verified=True)
        row.pop("condition")
        row["actual_stock_text"] = "hello world."
        row["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        first = self.manifest(dict(variant="original", cases=[row]), "original.json")
        second = self.manifest(dict(variant="merged", cases=[row]), "merged.json")
        ref = self.manifest([self.row(condition="reference", uid="heldout")], "reference.json")
        cases, identities = a.load_cases(self.root, [first, second, ref])
        self.assertEqual([x["condition"] for x in cases], ["baseline", "merged", "reference"])
        self.assertEqual(cases[0]["text"], text)
        self.assertEqual(len(identities), 3)
        self.assertTrue(all(x["status"] == "preflight_ready" for x in cases))
    def test_missing_or_failed_cases_become_private_coverage_gaps(self):
        failed = self.row(condition="merged", status="failed")
        failed.pop("audio")
        cases = self.load([self.row(), failed, self.row(language="en", uid="missing-side")])
        gaps, complete, eligible = a.coverage(cases, executed=False)
        self.assertEqual(complete["hi"], 0)
        self.assertFalse(eligible)
        self.assertIn("source_failed", {x["reason"] for x in gaps})
        self.assertIn("missing_pair_condition", {x["reason"] for x in gaps})
        summary = a.public_summary(cases, executed=False)
        self.assertTrue(summary["not_full_regression_passed"])
        self.assertFalse(summary["quality_gate_passed"])
        started = self.row(condition="merged", status="attempt_started")
        started.pop("audio")
        unfinished = self.load([self.row(), started])
        self.assertEqual(unfinished[1]["gap"], "source_attempt_started")
        pending_summary = a.public_summary(unfinished, executed=False)
        self.assertEqual(pending_summary["complete_pairs_by_language"]["hi"], 0)
        self.assertEqual(pending_summary["coverage_gap_counts"]["hi"]["source_attempt_started"], 1)
        self.assertFalse(pending_summary["quality_gate_passed"])
    def test_invalid_language_duplicates_text_hash_and_pair_mismatch_reject(self):
        scenarios = ([self.row(language="fr")], [self.row(), self.row()],
                     [self.row(), self.row(condition="merged", text="अलग पाठ।")],
                     [self.row(text_sha256="0" * 64)])
        for rows in scenarios:
            with self.subTest(rows=len(rows)), self.assertRaises(a.base.ASRError):
                self.load(rows)
    def test_frontend_unverified_missing_audio_and_invalid_wave_are_gaps(self):
        row = self.row(frontend_verified=False)
        self.assertEqual(self.load([row])[0]["gap"], "source_frontend_unverified")
        batch_row = self.row()
        batch_row.pop("condition")
        path = self.manifest(dict(variant="original", cases=[batch_row]))
        self.assertEqual(a.load_cases(self.root, [path])[0][0]["gap"], "source_frontend_unverified")
        with self.assertRaises(a.base.ASRError):
            self.load([self.row(frontend_verified="false")])
        row = self.row()
        row["audio"] = "missing.wav"
        self.assertEqual(self.load([row])[0]["gap"], "audio_missing")
        for options in (dict(seconds=.5), dict(seconds=21), dict(channels=2), dict(width=1), dict(rate=16000)):
            self.audio(**options)
            with patch.object(a.base, "verify_model", side_effect=AssertionError("No model preflight during manifest check")):
                self.assertEqual(self.load([self.row()])[0]["gap"], "audio_preflight_failed")
    def test_external_paths_and_symlink_escape_reject_even_failed_case(self):
        for value in ("../outside.wav", "https://example.invalid/audio.wav"):
            row = self.row(status="failed")
            row["audio"] = value
            with self.assertRaises(a.base.ASRError):
                self.load([row])
        outside = self.root.parent / (self.root.name + "-outside.wav")
        outside.write_bytes(self.wav.read_bytes())
        try:
            (self.root / "escape.wav").symlink_to(outside)
            row = self.row()
            row["audio"] = "escape.wav"
            with self.assertRaises(a.base.ASRError):
                self.load([row])
        finally:
            outside.unlink()
    def test_global_40_case_bound_including_failed_records(self):
        rows = [self.row(uid=f"probe-{i}", status="not_executed") for i in range(40)]
        self.assertEqual(len(self.load(rows)), 40)
        left = self.manifest(rows[:21], "left.json")
        right = self.manifest(rows[21:] + [self.row(uid="extra", status="failed")], "right.json")
        with self.assertRaises(a.base.ASRError):
            a.load_cases(self.root, [left, right])
    def test_reference_calibration_is_strictly_language_specific(self):
        cases = self.load([self.row(condition="reference", uid="native-hi"), self.row(language="en"),
                           self.row(language="en", condition="merged")])
        # Only metric bookkeeping is synthesized here; no ASR engine or transcript is simulated.
        for row in cases:
            row["status"] = "asr_computed"
            row["metrics"] = a.metric_views(row["text"], row["text"], row["language"])
        summary = a.public_summary(cases, executed=True)
        self.assertTrue(summary["calibrated_reference_available"]["hi"])
        self.assertFalse(summary["calibrated_reference_available"]["en"])
        self.assertTrue(all(not summary["calibrated_reference_available"][lang] for lang in ("zh", "ja", "es", "ar")))
        public = json.dumps(summary, ensure_ascii=False)
        for private in ("private-id", "native-hi", "private.wav", "क़लम", str(self.root), "hypothesis", "reference_chars_private"):
            self.assertNotIn(private, public)
        self.assertEqual(summary["complete_pairs_by_language"]["en"], 1)
        self.assertTrue(summary["not_full_regression_passed"])
    def test_unpaired_metrics_are_not_aggregated_as_comparable(self):
        cases = self.load([self.row()])
        cases[0].update(status="asr_computed", metrics=a.metric_views(cases[0]["text"], "", "hi"))
        summary = a.public_summary(cases, executed=True)
        self.assertEqual(summary["metrics"]["hi"], {})
        self.assertEqual(summary["complete_pairs_by_language"]["hi"], 0)
    def test_metadata_mode_never_imports_torch_whisper_or_downloads(self):
        manifest = self.manifest([self.row(), self.row(condition="merged")])
        code = "from hi_h200_native.asr_limited_eval import main; import sys; r=main(sys.argv[1:]); assert 'torch' not in sys.modules and 'whisper' not in sys.modules; raise SystemExit(r)"
        result = subprocess.run([sys.executable, "-B", "-c", code, "--project-root", str(self.root), "--input", str(manifest)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertFalse(summary["asr_executed"])
        self.assertFalse(summary["download"])
        self.assertFalse(summary["calibrated_reference_available"]["hi"])
    def test_execute_rejects_unknown_weights_before_torch_import(self):
        cases, identities = a.load_cases(self.root, [self.manifest([self.row()])])
        folder = self.root / "wrong-model"
        folder.mkdir()
        (folder / "small.pt").write_bytes(b"not weights")
        args = a.argparse.Namespace(output=self.root / "new-result", device="cpu", allow_cuda=False, gpu_id=None)
        with patch.object(a.base, "package", return_value="MIT metadata already tested separately"):
            with self.assertRaises(a.base.ASRError):
                a.execute(args, self.root, folder, cases, identities)
        self.assertFalse(args.output.exists())
    def test_no_runnable_case_writes_honest_private_and_sanitized_public_reports(self):
        cases, identities = a.load_cases(self.root, [self.manifest([self.row(status="failed")])])
        args = a.argparse.Namespace(output=self.root / "new-result", device="cpu", allow_cuda=False, gpu_id=None)
        with patch.object(a.base, "package", side_effect=AssertionError("No ASR package needed")), patch.object(a.base, "verify_model", side_effect=AssertionError("No model")):
            summary = a.execute(args, self.root, self.root / "unused-model", cases, identities)
        self.assertFalse(summary["asr_executed"])
        self.assertFalse(summary["quality_gate_passed"])
        private = json.loads((args.output / "private-asr.json").read_text())
        public = json.loads((args.output / "public-summary.json").read_text())
        self.assertIn("क़लम", json.dumps(private, ensure_ascii=False))
        self.assertNotIn("क़लम", json.dumps(public, ensure_ascii=False))
        self.assertFalse(private["model_verified"])
        self.assertEqual((args.output.stat().st_mode & 0o777), 0o700)
        self.assertEqual(((args.output / "private-asr.json").stat().st_mode & 0o777), 0o600)
        with self.assertRaises(a.base.ASRError):
            a.execute(args, self.root, self.root / "unused-model", cases, identities)


if __name__ == "__main__":
    unittest.main()
