"""CPU-only adversarial and leakage checks; all generated audio is synthetic."""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from hi_h200.cli import main
from hi_h200.data import DataError, build_pairs, prepare, read_jsonl, validate
from hi_h200.text import normalize_hindi

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
TEXTS = ["नमस्ते मेरे मित्र।", "आज मौसम अच्छा है।", "कृपया यहाँ बैठिए।", "मुझे हिंदी बोलना पसंद है।",
         "कल हम घर जाएँगे।", "यह एक सुंदर कहानी है।", "फिर मिलेंगे मित्र।", "आप कैसे हैं मित्र।"]


class TextTests(unittest.TestCase):
    def test_nfc_and_whitespace_preserve_marks(self):
        text = "  क़\tक्\u200dष\nक्\u200cष   हिंदी  "
        normalized = normalize_hindi(text)
        self.assertEqual(normalized, "क़ क्\u200dष क्\u200cष हिंदी")
        self.assertIn("\u093c", normalized)
        self.assertIn("\u094d", normalized)
        self.assertEqual(normalize_hindi("क़लम"), normalize_hindi("क़लम"))

    def test_numbers_require_both_assertions(self):
        for text in ("कीमत ₹२० है।", "कीमत 20 है।", "समय Ⅳ है।"):
            for kwargs in ({}, {"allow_as_spoken_numbers": True}, {"numbers_as_spoken": True}):
                with self.assertRaises(DataError):
                    normalize_hindi(text, **kwargs)
            self.assertEqual(normalize_hindi(text, allow_as_spoken_numbers=True,
                                             numbers_as_spoken=True), text)

    def test_control_and_tokens_fail(self):
        for text in ("हिंदी\x00 है", "हिंदी\u202e है", "हिंदी\u2066 है", "हिंदी<|hi|>", "हिंदी\ufeff है"):
            with self.subTest(text=repr(text)), self.assertRaises(DataError):
                normalize_hindi(text)

    def test_requires_hindi_and_valid_length(self):
        for text in ("English only", "ह", 7):
            with self.assertRaises(DataError):
                normalize_hindi(text)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        RUNTIME.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="data-test-", dir=RUNTIME)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio"
        self.audio.mkdir()
        self.raw = self.root / "raw.jsonl"
        self.output = self.root / "prepared"
        self.rows = []
        for index in range(6):
            name = f"clip_{index}.wav"
            self.write_wav(self.audio / name, index + 1)
            self.rows.append({"id": f"clip_{index}", "speaker_id": f"speaker_{index // 2}", "audio": name,
                              "text": TEXTS[index], "language": "hi", "source": "synthetic-unit-test",
                              "license": "synthetic-no-speech", "consent": True})
        self.save_rows()

    def tearDown(self):
        self.temp.cleanup()

    def write_wav(self, path, value=1, rate=24000, frames=24000):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack("<h", value) * frames)

    def save_rows(self):
        self.raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")

    def prepare(self, **kwargs):
        return prepare(self.raw, self.audio, self.output, **kwargs)

    def mutate_prepared(self, mutation):
        self.prepare()
        path = self.output / "train.jsonl"
        rows = list(read_jsonl(path))
        mutation(rows)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_prepare_validate_and_backend_export(self):
        manifest = self.prepare()
        self.assertEqual(validate(self.output, self.audio)["status"], "valid")
        self.assertEqual(sum(count["clips"] for count in manifest["counts"].values()), 6)
        speakers = []
        for split in ("train", "val", "test"):
            rows = list(read_jsonl(self.output / f"{split}.jsonl"))
            self.assertEqual(len(rows), 2)
            speakers.append({row["speaker_id"] for row in rows})
            for row in rows:
                self.assertNotEqual(row["id"], row["prompt_id"])
                self.assertEqual(row["text_norm"], row["text_raw"])
            exported = list(read_jsonl(self.output / f"backend_{split}.jsonl"))
            self.assertEqual(set(exported[0]), {"id", "audio", "text", "speaker", "duration"})
            self.assertTrue(Path(exported[0]["audio"]).is_absolute())
        self.assertFalse(speakers[0] & speakers[1] or speakers[1] & speakers[2] or speakers[0] & speakers[2])

    def test_stable_splits_ignore_input_order(self):
        self.prepare()
        first = (self.output / "train.jsonl").read_text()
        self.rows.reverse()
        self.save_rows()
        second = self.root / "second"
        prepare(self.raw, self.audio, second)
        self.assertEqual(first, (second / "train.jsonl").read_text())

    def test_audio_duplicates_rejected(self):
        self.rows[1]["audio"] = self.rows[0]["audio"]
        self.save_rows()
        with self.assertRaisesRegex(DataError, "duplicate PCM"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_nfc_text_duplicates_rejected_across_speakers(self):
        self.rows[0]["text"], self.rows[2]["text"] = "क़लम", "क़लम"
        self.save_rows()
        with self.assertRaisesRegex(DataError, "duplicate normalized text"):
            self.prepare()

    def test_path_traversal_and_urls_rejected(self):
        for name in ("../outside.wav", "/tmp/audio.wav", "https://example.org/audio.wav", "a\\b.wav", "./clip_0.wav"):
            self.rows[0]["audio"] = name
            self.save_rows()
            with self.subTest(name=name), self.assertRaises(DataError):
                self.prepare()

    def test_symlink_escape_rejected(self):
        outside = self.root / "outside.wav"
        self.write_wav(outside)
        (self.audio / "escape.wav").symlink_to(outside)
        self.rows[0]["audio"] = "escape.wav"
        self.save_rows()
        with self.assertRaisesRegex(DataError, "outside audio-root"):
            self.prepare()

    def test_rate_duration_and_truncation(self):
        self.write_wav(self.audio / "clip_0.wav", rate=16000)
        with self.assertRaisesRegex(DataError, "sample rate"):
            self.prepare()
        self.write_wav(self.audio / "clip_0.wav", frames=100)
        with self.assertRaisesRegex(DataError, "duration"):
            self.prepare()
        self.write_wav(self.audio / "clip_0.wav")
        path = self.audio / "clip_0.wav"
        path.write_bytes(path.read_bytes()[:-100])
        with self.assertRaisesRegex(DataError, "truncated"):
            self.prepare()

    def test_speaker_requirements_and_consent(self):
        self.rows[0]["consent"] = False
        self.save_rows()
        with self.assertRaisesRegex(DataError, "consent"):
            self.prepare()
        self.rows[0]["consent"] = True
        for row in self.rows:
            row["speaker_id"] = "only_one"
        self.save_rows()
        with self.assertRaisesRegex(DataError, "three speakers"):
            self.prepare()

    def test_reserved_or_path_like_ids_rejected(self):
        for value in ("../evil", "abc/evil", ".dot", "abc.def", "a b", "नमस्ते"):
            self.rows[0]["id"] = value
            self.save_rows()
            with self.subTest(value=value), self.assertRaises(DataError):
                self.prepare()

    def test_numbers_metadata_is_explicit(self):
        self.rows[0]["text"] = "कीमत ₹२० है।"
        self.rows[0]["numbers_as_spoken"] = True
        self.save_rows()
        with self.assertRaisesRegex(DataError, "digits/currency"):
            self.prepare()
        self.prepare(allow_as_spoken_numbers=True)
        self.assertEqual(validate(self.output, self.audio)["status"], "valid")

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("unchanged")
        with self.assertRaisesRegex(DataError, "already exists"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "unchanged")

    def test_unsafe_json_rejected(self):
        for text in ('{"id":"a","id":"b"}\n', '{"number":NaN}\n', '\n'):
            self.raw.write_text(text)
            with self.assertRaises(DataError):
                self.prepare()

    def test_prompt_self_pair_tamper_detected(self):
        self.mutate_prepared(lambda rows: rows[0].update(prompt_id=rows[0]["id"], prompt_audio=rows[0]["audio"]))
        with self.assertRaisesRegex(DataError, "different valid clip"):
            validate(self.output, self.audio)

    def test_prepared_hash_tamper_detected(self):
        self.mutate_prepared(lambda rows: rows[0].update(audio_sha256="0" * 64))
        with self.assertRaisesRegex(DataError, "recomputed"):
            validate(self.output, self.audio)

    def test_speaker_leak_tamper_detected(self):
        self.prepare()
        train = list(read_jsonl(self.output / "train.jsonl"))
        val_path = self.output / "val.jsonl"
        val = list(read_jsonl(val_path))
        val[0]["speaker_id"] = train[0]["speaker_id"]
        val_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in val))
        with self.assertRaisesRegex(DataError, "speaker leakage"):
            validate(self.output, self.audio)

    def test_malformed_manifest_settings_fail_cleanly(self):
        self.prepare()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["settings"] = None
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(DataError, "settings keys"):
            validate(self.output, self.audio)

    def test_invalid_settings(self):
        for options in ({"min_seconds": float("nan")}, {"sample_rate": True}, {"max_seconds": 100}, {"max_chars": -1}):
            with self.subTest(options=options), self.assertRaises(DataError):
                self.prepare(**options)

    def test_cli_success_and_failure(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = main(["prepare", str(self.raw), "--audio-root", str(self.audio), "--output", str(self.output)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "prepared")
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = main(["validate", str(self.root / "missing"), "--audio-root", str(self.audio)])
        self.assertEqual(result, 2)
        self.assertIn("ERROR:", stderr.getvalue())

    def features(self):
        self.prepare()
        self.features_root = self.root / "features"
        self.features_root.mkdir()
        self.feature_manifest = self.features_root / "utterances_train.jsonl"
        self.pair_output = self.features_root / "pairs.jsonl"
        rows = []
        for source in read_jsonl(self.output / "train.jsonl"):
            name = source["id"] + ".npz"
            # Pair builder intentionally checks metadata/paths only and never deserializes arrays.
            (self.features_root / name).write_bytes(b"synthetic-path-check-placeholder")
            rows.append({"id": source["id"], "speaker": source["speaker_id"], "lang": 17, "lang_alias": "hi",
                         "features": name, "text": source["text_norm"], "text_len": 20, "code_len": 100})
        self.save_features(rows)
        return rows

    def save_features(self, rows):
        self.feature_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def pairs(self):
        return build_pairs(self.feature_manifest, self.output, "train", self.features_root, self.pair_output, self.audio,
                           max_text_tokens=120, max_code_tokens=1728, max_prompt_code_tokens=640)

    def test_feature_pairs_are_unique_and_distinct(self):
        self.features()
        result = self.pairs()
        self.assertEqual(result["pairs"], 2)
        rows = list(read_jsonl(self.pair_output))
        self.assertEqual(len({row["id"] for row in rows}), 2)
        for row in rows:
            self.assertNotEqual(row["target_features"], row["prompt_features"])
            self.assertEqual(row["prompt_lang"], 17)
            self.assertFalse(row["cross_lingual"])

    def test_feature_missing_and_wrong_language_rejected(self):
        rows = self.features()
        self.save_features(rows[:1])
        with self.assertRaisesRegex(DataError, "every prepared clip"):
            self.pairs()
        rows[0]["lang"] = 0
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "Hindi slot"):
            self.pairs()

    def test_feature_long_prompt_does_not_fallback(self):
        rows = self.features()
        for row in rows:
            row["code_len"] = 1000
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "prompt token limit"):
            self.pairs()
        self.assertFalse(self.pair_output.exists())

    def test_feature_path_escape_rejected(self):
        rows = self.features()
        outside = self.root / "outside.npz"
        outside.write_bytes(b"placeholder")
        rows[0]["features"] = "../outside.npz"
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "confined"):
            self.pairs()

    def test_backend_export_tampering_detected(self):
        self.prepare()
        path = self.output / "backend_train.jsonl"
        rows = list(read_jsonl(path))
        rows[0]["audio"] = "/tmp/evil.wav"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        with self.assertRaisesRegex(DataError, "backend export"):
            validate(self.output, self.audio)

    def test_stale_feature_text_rejected(self):
        rows = self.features()
        rows[0]["text"] = "यह पुराना पाठ है।"
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "reviewed source"):
            self.pairs()

    def test_control_path_rejected(self):
        name = "bad\nfile.wav"
        self.write_wav(self.audio / name)
        self.rows[0]["audio"] = name
        self.save_rows()
        with self.assertRaisesRegex(DataError, "control/format"):
            self.prepare()

    def test_pairs_revalidate_prepared_data(self):
        self.features()
        path = self.output / "backend_train.jsonl"
        path.write_text("{}\n")
        with self.assertRaisesRegex(DataError, "backend export"):
            self.pairs()

    def test_feature_control_path_rejected(self):
        rows = self.features()
        path = self.features_root / "bad\nfile.npz"
        path.write_bytes(b"placeholder")
        rows[0]["features"] = path.name
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "control/format"):
            self.pairs()

    def test_feature_token_limit_and_speaker_mismatch(self):
        rows = self.features()
        rows[0]["text_len"] = 121
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "token limits"):
            self.pairs()
        rows[0]["text_len"] = 20
        rows[0]["speaker"] = "other"
        self.save_features(rows)
        with self.assertRaisesRegex(DataError, "speaker"):
            self.pairs()


if __name__ == "__main__":
    unittest.main()
