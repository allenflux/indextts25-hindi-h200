"""Pure CPU ASR preflight/metric tests; no ASR weights, network, or fake transcripts."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from hi_h200_native import asr_eval as a

ROOT = Path(__file__).resolve().parents[1]


class ASREvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native-asr-", dir=ROOT / ".runtime")
        self.root = Path(self.temp.name)
        self.source = self.root / "source.wav"
        self.write_wave(self.source)
    def tearDown(self):
        self.temp.cleanup()
    def write_wave(self, path, *, rate=24000, seconds=1, channels=1, width=2, silence=False):
        with wave.open(str(path), "wb") as out:
            out.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            frames = int(rate * seconds)
            if width == 2:
                out.writeframes(b"".join(struct.pack("<h", 0 if silence else (700 if i % 2 else -700)) for i in range(frames * channels)))
            else:
                out.writeframes(bytes([0 if silence else 100]) * frames * channels * width)
    def cases(self):
        return [dict(id="calibration", condition="reference", audio="source.wav", text="क़लम यहाँ रखिए।"),
                dict(id="paired", condition="baseline", audio="source.wav", text="नमस्ते मित्र।"),
                dict(id="paired", condition="merged", audio="source.wav", text="नमस्ते मित्र।")]
    def manifest(self, rows=None):
        path = self.root / "cases.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (self.cases() if rows is None else rows)))
        return path
    def args(self):
        return a.argparse.Namespace(input=self.manifest(), output=self.root / "result", device="cpu", allow_cuda=False, gpu_id=None)
    def test_unicode_marks_joiners_and_numbers_are_preserved(self):
        value = "क़\u094d\u200dष क्\u200cष ₹२० 20"
        normalized = a.normalized(value, True)
        for mark in ("\u093c", "\u094d", "\u200d", "\u200c", "₹", "२०", "20"):
            self.assertIn(mark, normalized)
        self.assertEqual(a.scores("क़लम", "क़लम")["cer"], 0)
        self.assertEqual(a.normalized(" हिंदी!। ", True), "हिंदी")
        with self.assertRaises(a.ASRError):
            a.normalized("हिंदी\u202e")
    def test_raw_and_punctuation_views_are_separate(self):
        self.assertGreater(a.scores("नमस्ते!", "नमस्ते")["cer"], 0)
        self.assertEqual(a.scores("नमस्ते!", "नमस्ते", True)["cer"], 0)
        self.assertEqual(a.scores("एक दो", "एक तीन")["wer"], .5)
        self.assertEqual(a.scores("एक", "")["wer"], 1)
        self.assertGreater(a.scores("एक", "एक दो तीन")["wer"], 1)
        with self.assertRaises(a.ASRError):
            a.scores("", "कुछ")
    def test_edit_distance_true_insert_delete_substitute(self):
        self.assertEqual(a.edit_distance("abc", "abc"), 0)
        self.assertEqual(a.edit_distance("abc", "ac"), 1)
        self.assertEqual(a.edit_distance("abc", "axc"), 1)
        self.assertEqual(a.edit_distance("", "abc"), 3)
    def test_local_paths_and_symlink_escape_are_blocked(self):
        with self.assertRaises(a.ASRError):
            a.contained(self.root, "../outside.wav")
        outside = self.root.parent / (self.root.name + "-outside.wav")
        try:
            self.write_wave(outside)
            (self.root / "link.wav").symlink_to(outside)
            with self.assertRaises(a.ASRError):
                a.contained(self.root, "link.wav", exists=True)
        finally:
            outside.unlink(missing_ok=True)
    def test_cases_require_calibration_labels_and_paired_text(self):
        self.assertEqual(len(a.load_cases(self.root, self.manifest())), 3)
        for rows in (self.cases()[1:], self.cases()[:2], self.cases() + self.cases()[:1]):
            with self.assertRaises(a.ASRError):
                a.load_cases(self.root, self.manifest(rows))
        rows = self.cases()
        rows[0]["text"] = "..."
        with self.assertRaises(a.ASRError):
            a.load_cases(self.root, self.manifest(rows))
        rows = self.cases()
        rows[-1]["text"] = "अलग पाठ"
        with self.assertRaises(a.ASRError):
            a.load_cases(self.root, self.manifest(rows))
    def test_bad_waveform_rejected_before_model_load(self):
        for kwargs in (dict(seconds=.5), dict(seconds=21), dict(channels=2), dict(width=1), dict(rate=16000), dict(silence=True)):
            self.write_wave(self.source, **kwargs)
            with patch.object(a, "verify_model", side_effect=AssertionError("Model preflight must not run")):
                with self.assertRaises(Exception) as error:
                    a.execute(self.args(), self.root, self.root / "models")
            self.assertNotIsInstance(error.exception, AssertionError)
    def test_truncated_wave_rejected_before_model_load(self):
        self.source.write_bytes(self.source.read_bytes()[:-4])
        with patch.object(a, "verify_model", side_effect=AssertionError("Must not load")):
            with self.assertRaises(Exception) as error:
                a.execute(self.args(), self.root, self.root / "models")
        self.assertNotIsInstance(error.exception, AssertionError)
    def test_full_22050_and_24000_to_16000_waveform(self):
        for rate in (22050, 24000):
            self.write_wave(self.source, rate=rate)
            output = a.waveform(self.source)
            self.assertEqual((output.shape, str(output.dtype)), ((16000,), "float32"))
    def test_missing_wrong_size_or_hash_model_blocks(self):
        folder = self.root / "model"
        folder.mkdir()
        with self.assertRaises(a.ASRError):
            a.verify_model(folder)
        (folder / "small.pt").write_bytes(b"synthetic-not-weights")
        with self.assertRaises(a.ASRError):
            a.verify_model(folder)
        with patch.object(a, "MODEL_BYTES", (folder / "small.pt").stat().st_size):
            with self.assertRaises(a.ASRError):
                a.verify_model(folder)
        target = self.root / "fixture-not-weights"
        target.write_bytes((folder / "small.pt").read_bytes())
        (folder / "small.pt").unlink()
        (folder / "small.pt").symlink_to(target)
        with patch.object(a, "MODEL_BYTES", target.stat().st_size), patch.object(a, "MODEL_SHA", hashlib.sha256(target.read_bytes()).hexdigest()):
            with self.assertRaises(a.ASRError):
                a.verify_model(folder)
    def test_download_is_fixed_bounded_noauth_and_sha_checked(self):
        payload = b"synthetic fixture, never loaded as ASR weights"
        calls = []
        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(payload))}
        class Opener:
            def open(self, request, **kwargs):
                calls.append((request.full_url, dict(request.header_items())))
                return Response(payload)
        with patch.object(a, "MODEL_BYTES", len(payload)), patch.object(a, "MODEL_SHA", hashlib.sha256(payload).hexdigest()), patch.object(a, "package", return_value="MIT License\nOpenAI"):
            result = a.download_model(self.root / "download", 500000000, opener=Opener())
        self.assertEqual(result["status"], "downloaded_verified")
        self.assertEqual(calls[0][0], a.MODEL_URL)
        self.assertFalse(any(key.lower() in {"authorization", "cookie", "proxy-authorization"} for key in calls[0][1]))
        with patch.object(a, "MODEL_BYTES", len(payload)), patch.object(a, "MODEL_SHA", "0" * 64), patch.object(a, "package", return_value="MIT License\nOpenAI"):
            with self.assertRaises(a.ASRError):
                a.download_model(self.root / "bad-sha-download", 500000000, opener=Opener())
        self.assertFalse((self.root / "bad-sha-download" / "small.pt").exists())
        with self.assertRaises(a.ASRError):
            a.download_model(self.root / "unused", 100)
    def test_runtime_origin_and_fresh_cuda_guards(self):
        installed = self.root / "installed"
        expected = installed / "whisper" / "__init__.py"
        expected.parent.mkdir(parents=True)
        expected.write_text("# fixture path; never imported")
        dist = SimpleNamespace(locate_file=lambda relative: installed / relative)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        whisper = SimpleNamespace(__file__=str(expected))
        with patch.object(a.importlib.metadata, "distribution", return_value=dist):
            a.runtime_guard(torch, whisper, "cpu")
            a.runtime_guard(torch, whisper, "cuda")
            with self.assertRaises(a.ASRError):
                a.runtime_guard(torch, SimpleNamespace(__file__=str(__file__)), "cpu")
            torch.cuda.is_initialized = lambda: True
            with self.assertRaises(a.ASRError):
                a.runtime_guard(torch, whisper, "cuda")

    def test_default_plan_never_loads_or_downloads(self):
        output = io.StringIO()
        with patch.object(a, "download_model", side_effect=AssertionError("No download")), patch.object(a, "execute", side_effect=AssertionError("No ASR")), contextlib.redirect_stdout(output):
            result = a.main(["--project-root", str(self.root)])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertFalse(report["asr_executed"])
        self.assertFalse(report["download"])
        self.assertEqual((report["bytes"], len(report["sha256"])), (483617219, 64))
    def test_redirects_cannot_change_destination(self):
        redirect = a.FixedRedirect()
        request = a.urllib.request.Request(a.MODEL_URL)
        for url in ("http://openaipublic.azureedge.net/model", "https://example.invalid/model", "https://user:password@openaipublic.azureedge.net/model", "https://openaipublic.azureedge.net:444/model"):
            with self.assertRaises(a.ASRError):
                redirect.redirect_request(request, None, 302, "found", {}, url)


if __name__ == "__main__":
    unittest.main()
