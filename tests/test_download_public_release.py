"""Pure tiny download mocks; no real release, network, GPU, or model success."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("download_public_release", ROOT / "scripts/download_public_release.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class PublicDownloadSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="public-download-test-")
        self.work = Path(self.temp.name).resolve()
        self.old_tmp = tempfile.tempdir
    def tearDown(self):
        tempfile.tempdir = self.old_tmp
        self.temp.cleanup()
    def test_environment_isolated_and_token_values_never_read(self):
        secrets = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"}
        class NamesOnly(dict):
            def __getitem__(self, key):
                if key in secrets:
                    raise AssertionError("Never read ambient token values")
                return super().__getitem__(key)
        env = NamesOnly({name: "DO_NOT_READ" for name in secrets})
        cache = self.work / "cache"
        cache.mkdir()
        with patch.object(d.os, "environ", env):
            d.environment(cache)
        self.assertFalse(set(env) & secrets)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "XDG_CACHE_HOME", "TMPDIR", "TMP", "TEMP"):
            self.assertTrue(Path(env[key]).is_relative_to(cache))
        self.assertEqual(env["HF_HUB_OFFLINE"], "0")
        for key in ("HF_HUB_DISABLE_XET", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_HUB_DISABLE_TELEMETRY"):
            self.assertEqual(env[key], "1")
    def test_existing_directories_or_insufficient_disk_stop_before_download(self):
        def denied(**_):
            raise AssertionError("Must not download")
        with patch.object(d.shutil, "disk_usage", return_value=SimpleNamespace(free=9 * 1024**3)):
            with self.assertRaises(d.Blocked):
                d.run(self.work, downloader=denied)
        self.assertFalse((self.work / "cache").exists())
        (self.work / "public13").mkdir()
        sentinel = self.work / "public13/keep"
        sentinel.write_text("preserve")
        with self.assertRaises(d.Blocked):
            d.run(self.work, downloader=denied)
        self.assertEqual(sentinel.read_text(), "preserve")
    def test_stream_copy_hash_and_exclusive_regular_destination(self):
        source = self.work / "fixture"
        source.write_bytes(b"small fixture, not model weights")
        destination = self.work / "new"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(d.copy_verified(source, destination, source.stat().st_size, digest), digest)
        self.assertFalse(destination.is_symlink())
        with self.assertRaises(FileExistsError):
            d.copy_verified(source, destination, source.stat().st_size, digest)
        with self.assertRaises(d.Blocked):
            d.copy_verified(source, self.work / "bad", source.stat().st_size, "0" * 64)
    def tiny_metadata(self):
        # Only metadata pins are mocked; the production GPT identity stays fixed.
        bodies = {name: ("synthetic fixture: " + name).encode() for name in d.FILES - set(d.PINS)}
        payload = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in bodies.items()}
        payload["gpt.pth"] = {"bytes": d.GPT_SIZE, "sha256": d.GPT_SHA}
        bodies["manifest.json"] = json.dumps({"schema_version": 1, "kind": "public_release_whitelist", "files": payload}).encode()
        hashes = {name: entry["sha256"] for name, entry in payload.items()}
        hashes["manifest.json"] = hashlib.sha256(bodies["manifest.json"]).hexdigest()
        bodies["SHA256SUMS"] = "".join(hashes[name] + "  " + name + "\n" for name in sorted(hashes)).encode()
        pins = {name: (len(bodies[name]), hashlib.sha256(bodies[name]).hexdigest()) for name in d.PINS}
        return bodies, pins
    def test_tiny_mock_download_has_fixed_anonymous_args_but_cannot_pass_as_gpt(self):
        bodies, pins = self.tiny_metadata()
        calls = []
        def mocked(**kwargs):
            calls.append(kwargs)
            self.assertIs(kwargs["token"], False)
            self.assertEqual(kwargs["repo_id"], d.REPO)
            self.assertEqual(kwargs["revision"], d.REVISION)
            self.assertEqual(kwargs["endpoint"], "https://huggingface.co")
            self.assertTrue(Path(kwargs["cache_dir"]).is_relative_to(self.work))
            target = Path(kwargs["cache_dir"]) / kwargs["filename"]
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(bodies[kwargs["filename"]])
            return str(target)
        with patch.object(d.os, "environ", {}), patch.object(d, "PINS", pins), patch.object(d.shutil, "disk_usage", return_value=SimpleNamespace(free=11 * 1024**3)), patch("socket.socket", side_effect=AssertionError("No network")):
            with self.assertRaises(d.Blocked):
                d.run(self.work, downloader=mocked)
        self.assertEqual(len(calls), 13)
        self.assertEqual({row["filename"] for row in calls}, d.FILES)
        status = json.loads((self.work / "download-status.json").read_text())
        self.assertEqual(status["phase"], "blocked")
        self.assertFalse(status["complete"])
        self.assertFalse((self.work / "public13/gpt.pth").exists())
        self.assertEqual(len(list((self.work / "public13").iterdir())), 12)
        self.assertNotIn(str(self.work), json.dumps(status))
    def test_cache_escape_and_error_details_are_blocked(self):
        outside_cache = self.work / "explicit-outside-cache-fixture"
        outside_cache.write_bytes(b"small fixture")
        def escaped(**_):
            return str(outside_cache)
        with patch.object(d.os, "environ", {}), patch.object(d.shutil, "disk_usage", return_value=SimpleNamespace(free=11 * 1024**3)), patch("socket.socket", side_effect=AssertionError("No network")):
            with self.assertRaises(d.Blocked):
                d.run(self.work, downloader=escaped)
        self.assertFalse((self.work / "public13/manifest.json").exists())
        output = io.StringIO()
        with patch.object(d, "run", side_effect=RuntimeError("SECRET_DETAIL")), patch("sys.stdout", output):
            self.assertEqual(d.main(["--work-root", str(self.work)]), 2)
        self.assertNotIn("SECRET_DETAIL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
