"""Original bounded importer tests: mock network, real tiny Parquet and PCM audio.

No authentication, actual IV-R rows/audio, or large downloads are used here.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import wave

from hi_h200.data import read_jsonl, validate
from hi_h200_native import hf_data as hf

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pyarrow", "soundfile", "scipy"))
TEXTS = ("नमस्ते मेरे मित्र।", "आज मौसम अच्छा है।", "कृपया यहाँ बैठिए।", "मुझे हिंदी बोलना पसंद है।",
         "कल हम घर जाएँगे।", "यह एक सुंदर कहानी है।", "फिर मिलेंगे मित्र।", "आप कैसे हैं मित्र।",
         "पानी साफ़ होना चाहिए।", "कृपया दरवाज़ा खोलिए।", "बच्चे बाहर खेल रहे हैं।", "मुझे चाय पसंद है।",
         "सूरज धीरे धीरे निकल रहा है।", "पेड़ के नीचे छाया है।", "उसने एक पत्र लिखा।", "आसमान नीला दिखाई देता है।",
         "फूलों की खुशबू अच्छी है।", "हम रास्ते में रुक गए।", "यह किताब बहुत पुरानी है।", "रात को तारे चमकते हैं।",
         "वह धीरे बोल रही है।", "हमारा गाँव सुंदर है।", "मेरा घर पास में है।", "तुम्हारा धन्यवाद मित्र।")


def fake_api(*, account=hf.EXPECTED_ACCOUNT, revision=hf.REVISION, license_value="cc-by-4.0", bad_size=False):
    calls = []
    names = [name for name, _ in hf.SHARDS]
    selected_sizes = [size for _, size in hf.SHARDS]
    remaining = hf.HINDI_BYTES - sum(selected_sizes)
    siblings = [SimpleNamespace(rfilename=name, size=size + (1 if bad_size and index == 0 else 0),
                                lfs=SimpleNamespace(size=size + (1 if bad_size and index == 0 else 0), sha256="a" * 64))
                for index, (name, size) in enumerate(hf.SHARDS)]
    for index in range(hf.HINDI_FILES - len(names)):
        size = remaining if index == 0 else 0
        siblings.append(SimpleNamespace(rfilename=f"Hindi/metadata-fixture-{index}.parquet", size=size,
                                        lfs=SimpleNamespace(size=size, sha256="b" * 64)))
    content = b"TTS fixture card"
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    siblings.append(SimpleNamespace(rfilename="README.md", size=len(content), blob_id=blob, lfs=None))
    class API:
        def whoami(self, **kwargs):
            calls.append(("whoami", kwargs))
            return {"name": account, "auth": {"accessToken": "SECRET_MUST_NOT_BE_RETURNED"}}
        def auth_check(self, repo, **kwargs):
            calls.append(("auth_check", repo, kwargs))
        def dataset_info(self, repo, **kwargs):
            calls.append(("dataset_info", repo, kwargs))
            return SimpleNamespace(sha=revision, card_data={"license": license_value}, siblings=siblings)
    return API(), calls


class HfMetadataTests(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="native-hf-", dir=runtime)
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def test_fixed_plan_budget_account_and_no_token_payload(self):
        api, calls = fake_api()
        report, shards, docs = hf.metadata_plan(api)
        self.assertEqual([s.filename for s in shards], [name for name, _ in hf.SHARDS])
        self.assertEqual(sum(s.size for s in shards), 946963686)
        self.assertLessEqual(report["planned_bytes"], 1000000000)
        self.assertEqual(report["corpus_bytes_downloaded"], 0)
        self.assertNotIn("SECRET", json.dumps(report))
        self.assertFalse(report["official_test_used"])
        for call in calls:
            self.assertIs(call[-1]["token"], None)
    def test_wrong_account_revision_license_or_size_block(self):
        for kwargs in ({"account": "other"}, {"revision": "b" * 40}, {"license_value": "other"}, {"bad_size": True}):
            api, _ = fake_api(**kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(hf.ImportBlocked):
                hf.metadata_plan(api)
    def test_budget_rejected_before_download_or_output(self):
        api, _ = fake_api()
        with self.assertRaises(hf.ImportBlocked):
            hf.metadata_plan(api, max_bytes=946963685)
        for maximum in (0, True, 1000000001):
            with self.assertRaises(hf.ImportBlocked):
                hf.metadata_plan(api, max_bytes=maximum)
    def test_metadata_default_never_calls_downloader(self):
        api, _ = fake_api()
        args = hf.parser().parse_args(["--output", str(self.root / "metadata")])
        def denied(**kwargs):
            raise AssertionError("Metadata-only run tried downloading")
        result = hf.run(args, api=api, download=denied)
        self.assertEqual(result["status"], "metadata_only")
        self.assertFalse((args.output / "downloads").exists())
        with self.assertRaises(hf.ImportBlocked):
            hf.run(args, api=api, download=denied)
    def test_rights_record_is_operator_declaration_not_acceptance(self):
        path = self.root / "rights.json"
        record = dict(dataset_repo=hf.REPOSITORY, revision=hf.REVISION, terms_accepted=True,
                      license_reviewed="CC-BY-4.0", permission_basis="User personally accepted the publisher access conditions and reviewed the model card.")
        path.write_text(json.dumps(record))
        result = hf.rights_record(path)
        self.assertIn("not independent proof", result["scope"])
        for key, value in (("terms_accepted", False), ("revision", "a" * 40), ("license_reviewed", "MIT")):
            changed = {**record, key: value}
            path.write_text(json.dumps(changed))
            with self.assertRaises(hf.ImportBlocked):
                hf.rights_record(path)
        with self.assertRaises(hf.ImportBlocked):
            hf.rights_record(None)
    def test_ambient_token_overrides_stop_without_reading_values(self):
        api, calls = fake_api()
        class NamesOnly(dict):
            def __getitem__(self, key):
                raise AssertionError("No token value may be read")
            def get(self, key, *args):
                raise AssertionError("No token value may be read")
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
            with patch.object(hf.os, "environ", NamesOnly({name: "synthetic-secret"})):
                with self.assertRaisesRegex(hf.ImportBlocked, "Ambient token"):
                    hf.metadata_plan(api)
        self.assertEqual(calls, [])
        with patch.object(hf.os, "environ", {}):
            with self.assertRaises(hf.ImportBlocked):
                hf.auth_environment(require_home=True)
        with patch.object(hf.os, "environ", {"HF_HOME": str(self.root)}):
            hf.auth_environment(require_home=True)

    def test_transport_errors_do_not_echo_tokens_or_urls(self):
        class API:
            def whoami(self, **kwargs):
                raise RuntimeError("SECRET token https://signed.example?credential=SECRET")
        with self.assertRaises(hf.ImportBlocked) as caught:
            hf.metadata_plan(API())
        self.assertNotIn("SECRET", str(caught.exception))
        with patch.object(hf, "run", side_effect=RuntimeError("SECRET_URL")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(hf.main(["--output", str(self.root / "unused")]), 2)
            self.assertNotIn("SECRET", output.getvalue())
    def test_download_hash_and_root_validation(self):
        folder = self.root / "out"
        (folder / "downloads").mkdir(parents=True)
        path = folder / "downloads" / "file"
        path.write_bytes(b"own fixture")
        pin = hf.FilePin("file", path.stat().st_size, "0" * 64)
        with self.assertRaisesRegex(hf.ImportBlocked, "SHA256"):
            hf.download_pin(pin, folder, lambda **_: str(path))
        pin = hf.FilePin("file", path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        result, _ = hf.download_pin(pin, folder, lambda **_: str(path))
        self.assertEqual(result, path)
        outside = self.root / "outside"
        outside.write_bytes(path.read_bytes())
        with self.assertRaises(hf.ImportBlocked):
            hf.download_pin(pin, folder, lambda **_: str(outside))


@unittest.skipUnless(AVAILABLE, "Optional PyArrow/SoundFile/SciPy/NumPy importer dependencies missing")
class HfAudioParquetTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import soundfile as sf
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.np, self.sf, self.pa, self.pq = np, sf, pa, pq
        self.temp = tempfile.TemporaryDirectory(prefix="native-hf-audio-", dir=ROOT / ".runtime")
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def audio(self, index=0, duration=1.2, rate=48000, stereo=False):
        t = self.np.arange(int(rate * duration), dtype="float64") / rate
        samples = .1 * self.np.sin(2 * self.np.pi * (180 + index * 13) * t)
        if stereo:
            samples = self.np.stack([samples, samples * .7], axis=1)
        buffer = io.BytesIO()
        self.sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
        return {"bytes": buffer.getvalue(), "path": f"publisher_clip_{index}.wav"}
    def parquet(self, rows, name="fixture.parquet"):
        path = self.root / name
        table = self.pa.Table.from_pylist(rows)
        self.pq.write_table(table, path, row_group_size=7)
        return path
    def rows(self):
        return [dict(speaker_id=f"source-speaker-{i // 4}", duration=1.2, lang="Hindi",
                     normalized=text, text=text, verbatim=text, audio=self.audio(i)) for i, text in enumerate(TEXTS)]
    def run_with_fake_downloads(self, first_rows, second_rows, *, max_rows=hf.MAX_ROWS):
        first = self.parquet(first_rows, "lazy-first.parquet")
        second = self.parquet(second_rows, "lazy-second.parquet")
        card = self.root / "card.md"
        card.write_text("Synthetic TTS fixture card, not actual IV-R data.")
        payload = card.read_bytes()
        docs = (hf.FilePin("README.md", len(payload), None,
                           hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()),)
        pins = tuple(hf.FilePin(name, path.stat().st_size, hf.digest_file(path))
                     for (name, _), path in zip(hf.SHARDS, (first, second)))
        rights = self.root / "rights.json"
        rights.write_text(json.dumps(dict(dataset_repo=hf.REPOSITORY, revision=hf.REVISION,
                                         terms_accepted=True, license_reviewed="CC-BY-4.0",
                                         permission_basis="Synthetic operator declaration used only for a CPU unit test.")))
        output = self.root / "lazy-import"
        args = hf.parser().parse_args(["--output", str(output), "--download", "--rights-record", str(rights),
                                       "--max-examined-rows", str(max_rows)])
        sources = {"README.md": card, hf.SHARDS[0][0]: first, hf.SHARDS[1][0]: second}
        calls = []
        def download(**kwargs):
            self.assertIsNone(kwargs["token"])
            self.assertEqual(kwargs["endpoint"], hf.ENDPOINT)
            self.assertEqual(kwargs["revision"], hf.REVISION)
            name = kwargs["filename"]
            calls.append(name)
            destination = Path(kwargs["local_dir"]) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[name], destination)
            return str(destination)
        plan = {"kind": "synthetic-test-only", "planned_bytes": sum(pin.size for pin in (*pins, *docs))}
        return args, output, calls, download, (plan, pins, docs)

    def test_complete_first_shard_never_downloads_second(self):
        args, output, calls, download, plan = self.run_with_fake_downloads(self.rows(), self.rows())
        with patch.object(hf, "metadata_plan", return_value=plan):
            result = hf.run(args, api=object(), download=download)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(calls, ["README.md", hf.SHARDS[0][0]])
        self.assertFalse((output / "downloads" / hf.SHARDS[1][0]).exists())
        audit = json.loads((output / "audit" / "downloaded_files.json").read_text())
        self.assertTrue(audit["import_completed"])
        self.assertEqual(len(audit["files"]), 2)

    def test_insufficient_first_shard_downloads_exact_second_once(self):
        rows = self.rows()
        args, output, calls, download, plan = self.run_with_fake_downloads(rows[:12], rows[12:])
        with patch.object(hf, "metadata_plan", return_value=plan):
            result = hf.run(args, api=object(), download=download)
        self.assertEqual((result["speakers"], result["clips"]), (6, 24))
        self.assertEqual(calls, ["README.md", hf.SHARDS[0][0], hf.SHARDS[1][0]])
        self.assertEqual(validate(output / "prepared", output)["status"], "valid")

    def test_exhausted_row_budget_never_fetches_second(self):
        args, output, calls, download, plan = self.run_with_fake_downloads(self.rows(), self.rows(), max_rows=10)
        with patch.object(hf, "metadata_plan", return_value=plan), self.assertRaises(hf.ImportBlocked):
            hf.run(args, api=object(), download=download)
        self.assertEqual(calls, ["README.md", hf.SHARDS[0][0]])
        audit = json.loads((output / "audit" / "downloaded_files.json").read_text())
        self.assertFalse(audit["import_completed"])
        self.assertFalse((output / "import.json").exists())

    def test_actual_stereo_resampling_keeps_full_utterance(self):
        source = self.audio(stereo=True)
        wav, info = hf.decode_audio(source)
        with wave.open(io.BytesIO(wav), "rb") as handle:
            self.assertEqual((handle.getframerate(), handle.getnchannels(), handle.getsampwidth()), (24000, 1, 2))
            self.assertEqual(handle.getnframes(), 28800)
        self.assertEqual(info["source_frames"], 57600)
        self.assertIn("no target truncation", info["conversion"])
    def test_external_audio_and_long_clip_rejected(self):
        for audio in ({"path": "https://example.invalid/private.wav", "bytes": None}, self.audio(duration=7)):
            with self.assertRaises(hf.ImportBlocked):
                hf.decode_audio(audio)
    def test_real_small_parquet_and_prepared_speaker_splits(self):
        rows = self.rows()
        first = self.parquet(rows[:12], "first.parquet")
        second = self.parquet(rows[12:], "second.parquet")
        output = self.root / "import"
        output.mkdir()
        report = hf.import_parquets([(hf.SHARDS[0][0], first), (hf.SHARDS[1][0], second)], output)
        self.assertEqual((report["speakers"], report["clips"], report["examined_rows"]), (6, 24, 24))
        self.assertEqual({split: data["clips"] for split, data in report["counts"].items()}, {"train": 16, "val": 4, "test": 4})
        self.assertFalse((output / "IMPORT_INCOMPLETE").exists())
        self.assertEqual(validate(output / "prepared", output)["status"], "valid")
        raw = list(read_jsonl(output / "input.jsonl"))
        self.assertTrue(all(row["source_text_fields"]["normalized"] == row["text"] for row in raw))
        self.assertTrue(all(row["dataset_revision"] == hf.REVISION for row in raw))
        self.assertFalse(report["individual_voice_cloning_consent_independently_verified"])
    def test_only_selected_audio_is_waveform_decoded(self):
        rows = [dict(speaker_id="ignored", duration=8.0, lang="Hindi", normalized="यह लंबा नमूना है।", audio={"bytes": b"never decode", "path": "bad.wav"})] + self.rows()
        path = self.parquet(rows)
        output = self.root / "selected"
        output.mkdir()
        with patch.object(hf, "decode_audio", wraps=hf.decode_audio) as decode:
            report = hf.import_parquets([(hf.SHARDS[0][0], path)], output)
        self.assertEqual(decode.call_count, 24)
        self.assertEqual(report["skipped"]["metadata_policy"], 1)
    def test_row_bound_and_unknown_speakers_fail_without_ready(self):
        path = self.parquet(self.rows())
        output = self.root / "short"
        output.mkdir()
        with self.assertRaises(hf.ImportBlocked):
            hf.import_parquets([(hf.SHARDS[0][0], path)], output, max_rows=10)
        self.assertTrue((output / "IMPORT_INCOMPLETE").is_file())
        self.assertFalse((output / "import.json").exists())
        for value in (None, -1, "unknown", True, " ", " unknown ", " speaker "):
            with self.assertRaises(hf.ImportBlocked):
                hf._speaker(value)
    def test_ascii_framing_punctuation_not_silently_removed(self):
        rows = self.rows()
        rows[0]["normalized"] += "!"
        path = self.parquet(rows)
        output = self.root / "framing"
        output.mkdir()
        with self.assertRaises(hf.ImportBlocked):
            hf.import_parquets([(hf.SHARDS[0][0], path)], output)
        self.assertFalse((output / "prepared").exists())
    def test_official_test_shards_never_allowed(self):
        output = self.root / "test-forbidden"
        output.mkdir()
        with self.assertRaisesRegex(hf.ImportBlocked, "TRAIN"):
            hf.import_parquets([("Hindi/test-00000-of-00002.parquet", self.parquet(self.rows()))], output)


if __name__ == "__main__":
    unittest.main()
