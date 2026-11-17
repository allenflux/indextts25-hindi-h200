"""Bounded public asset helper tests; no network or actual model downloads."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from hi_h200_native import assets

ROOT = Path(__file__).resolve().parents[1]


def metadata_api(selected, *, gated=False, revision=None, corrupt=False):
    calls = []
    class API:
        def model_info(self, repository, **kwargs):
            calls.append((repository, kwargs))
            if kwargs.get("token") is not False or kwargs.get("files_metadata") is not True:
                raise AssertionError("Metadata request must explicitly avoid authentication")
            records = []
            for asset in selected:
                if asset.repository != repository:
                    continue
                record = dict(rfilename=asset.filename, size=asset.size, blob_id=asset.digest, lfs=None)
                if asset.hash_kind == "sha256":
                    record["lfs"] = SimpleNamespace(sha256="0" * 64 if corrupt else asset.digest, size=asset.size)
                records.append(SimpleNamespace(**record))
            return SimpleNamespace(sha=revision or kwargs["revision"], gated=gated, siblings=records)
    return API(), calls


class NativeAssetTests(unittest.TestCase):
    def setUp(self):
        runtime = ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="native-assets-", dir=runtime)
        self.root = Path(self.temp.name)
        self.destination = self.root / "new-model"
        self.cache = self.root / "cache"
        self.contents = {
            "config.yaml": b"version: 2.5\ngpt: {}\nsemantic_codec: {}\n",
            "gpt.pth": b"synthetic fixture; never loaded as weights",
            "multilingual_zh_ja_yue_char_del.tiktoken": b"synthetic tokenizer",
            "LICENSE": b"synthetic license fixture",
            "README.md": b"synthetic card fixture",
        }
        selected = []
        for name, payload in self.contents.items():
            kind = "sha256" if name == "gpt.pth" else "git_blob_sha1"
            digest = hashlib.sha256(payload).hexdigest() if kind == "sha256" else hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\x00" + payload).hexdigest()
            selected.append(assets.Asset(assets.MODEL_REPO, assets.MODEL_REVISION, name, name, len(payload), kind, digest))
        self.selected = tuple(selected)
        self.downloads = []

    def tearDown(self):
        self.temp.cleanup()

    def plan(self):
        api, _ = metadata_api(self.selected)
        return assets.plan_assets("train", api=api)

    def download(self, **kwargs):
        self.downloads.append(kwargs)
        self.assertIs(kwargs["token"], False)
        self.assertEqual(kwargs["endpoint"], "https://huggingface.co")
        self.assertEqual(kwargs["revision"], assets.MODEL_REVISION)
        base = Path(kwargs["cache_dir"]) / "public-model"
        base.mkdir(parents=True, exist_ok=True)
        path = base / kwargs["filename"]
        path.write_bytes(self.contents[kwargs["filename"]])
        return str(path)

    def stage(self, plan=None, **kwargs):
        return assets.stage_assets(plan or self.plan(), self.destination, self.cache,
                                   max_bytes=10_000, reserve_bytes=0,
                                   downloader=kwargs.pop("downloader", self.download), **kwargs)

    def test_exact_real_selection_and_public_metadata_only(self):
        api, calls = metadata_api(assets.TRAIN + assets.EXTRACT)
        plan = assets.plan_assets("train", api=api)
        self.assertEqual(len(plan["files"]), 5)
        self.assertEqual(plan["network_payload_bytes"], 3260524913)
        self.assertFalse(plan["downloaded"])
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.cache.exists())
        api, calls = metadata_api(assets.TRAIN + assets.EXTRACT)
        full = assets.plan_assets("extract", api=api)
        self.assertEqual(len(full["files"]), 13)
        self.assertEqual(len(calls), 3)
        self.assertNotIn("s2mel.pth", [row["filename"] for row in full["files"]])
        self.assertIn("hf_cache/w2v-bert-2.0/model.safetensors", [row["relative_path"] for row in full["files"]])

    def test_synth_is_exact_public_pinned_and_excludes_unused_models(self):
        api, calls = metadata_api(assets.TRAIN + assets.EXTRACT + assets.SYNTH)
        plan = assets.plan_assets("synth", api=api)
        self.assertEqual(len(plan["files"]), 20)
        self.assertEqual(plan["network_payload_bytes"], 7082513550)
        self.assertEqual(len(calls), 4)
        self.assertFalse(plan["downloaded"])
        names = {row["relative_path"] for row in plan["files"]}
        self.assertTrue({"s2mel.pth", "feat1.pt", "feat2.pt", "hf_cache/bigvgan/config.json",
                         "hf_cache/bigvgan/bigvgan_generator.pt", "hf_cache/bigvgan/LICENSE",
                         "hf_cache/bigvgan/README.md"}.issubset(names))
        self.assertFalse(any("qwen" in name.lower() or "maskgct" in name.lower() for name in names))
        big = [row for row in plan["files"] if row["repository"] == assets.BIGVGAN_REPO]
        self.assertTrue(all(row["revision"] == "633ff708ed5b74903e86ff1298cf4a98e921c513" for row in big))
        self.assertIn(assets.BIGVGAN_REPO, plan["licenses"])
        self.assertFalse(self.cache.exists())

    def test_synth_stage_preserves_separate_bigvgan_source_manifest(self):
        payload = b"tiny synthetic vocoder, never loaded"
        vocoder = assets.Asset(assets.BIGVGAN_REPO, assets.BIGVGAN_REVISION,
                               "bigvgan_generator.pt", "hf_cache/bigvgan/bigvgan_generator.pt",
                               len(payload), "sha256", hashlib.sha256(payload).hexdigest())
        self.contents["bigvgan_generator.pt"] = payload
        selected = self.selected + (vocoder,)
        api, _ = metadata_api(selected)
        def download(**kwargs):
            self.assertIs(kwargs["token"], False)
            self.assertEqual(kwargs["endpoint"], "https://huggingface.co")
            path = Path(kwargs["cache_dir"]) / kwargs["repo_id"].replace("/", "--") / kwargs["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.contents[kwargs["filename"]])
            return str(path)
        with patch.object(assets, "TRAIN", self.selected), patch.object(assets, "EXTRACT", ()), patch.object(assets, "SYNTH", (vocoder,)):
            plan = assets.plan_assets("synth", api=api)
            result = assets.stage_assets(plan, self.destination, self.cache, max_bytes=10000,
                                         reserve_bytes=0, downloader=download)
        self.assertEqual(result["stage"], "synth")
        source = json.loads((self.destination / "bigvgan.manifest.json").read_text())
        self.assertEqual(source["repository"], assets.BIGVGAN_REPO)
        self.assertEqual(source["snapshot_revision"], assets.BIGVGAN_REVISION)
        self.assertEqual(source["files"][vocoder.relative_path], vocoder.digest)
        root_manifest = json.loads((self.destination / "model-manifest.json").read_text())
        self.assertIn("bigvgan.manifest.json", root_manifest["files"])

    def test_gated_changed_revision_or_hash_fails(self):
        for changes in ({"gated": "auto"}, {"revision": "0" * 40}, {"corrupt": True}):
            api, _ = metadata_api(assets.TRAIN + assets.EXTRACT, **changes)
            with self.subTest(changes=changes), self.assertRaises(assets.AssetError):
                assets.plan_assets("train", api=api)

    def test_budget_and_disk_reserve_precede_download(self):
        with patch.object(assets, "TRAIN", self.selected):
            plan = self.plan()
            with self.assertRaisesRegex(assets.AssetError, "max-bytes"):
                assets.check_budget(plan, self.destination, self.cache, 1, 0)
            with patch.object(assets.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(assets.AssetError, "free space"):
                    assets.check_budget(plan, self.destination, self.cache, 10000, 0)
        self.assertFalse(self.cache.exists())
        self.assertEqual(self.downloads, [])

    def test_existing_destination_and_path_escape_fail(self):
        self.destination.mkdir()
        with patch.object(assets, "TRAIN", self.selected):
            with self.assertRaisesRegex(assets.AssetError, "never overwritten"):
                self.stage()
        for name in ("../escape", "/absolute", "https://example/a", "a\\b", "a\x00b", "./file"):
            with self.subTest(name=name), self.assertRaises(assets.AssetError):
                assets.safe_relative(name)
        self.assertEqual(self.downloads, [])

    def test_real_copy_checks_hashes_and_publishes_manifest_last(self):
        with patch.object(assets, "TRAIN", self.selected):
            result = self.stage()
        self.assertEqual(result["status"], "assets_verified_not_loaded")
        self.assertFalse(result["model_loaded"])
        self.assertEqual(len(self.downloads), 5)
        manifest = json.loads((self.destination / "model-manifest.json").read_text())
        self.assertEqual(set(manifest), {"repository", "snapshot_revision", "files"})
        self.assertEqual(manifest["snapshot_revision"], assets.MODEL_REVISION)
        for name, expected in manifest["files"].items():
            self.assertEqual(hashlib.sha256((self.destination / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(list(self.root.glob(".new-model.tmp-*")), [])

    def test_corrupt_payload_does_not_publish_snapshot(self):
        self.contents["gpt.pth"] = b"corrupt"
        with patch.object(assets, "TRAIN", self.selected):
            with self.assertRaisesRegex(assets.AssetError, "size/hash"):
                self.stage()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".new-model.tmp-*")), [])
        self.assertTrue(self.cache.exists())  # isolated partial cache is reusable

    def test_foreign_cache_file_and_changed_plan_fail(self):
        foreign = self.root / "foreign"
        foreign.write_bytes(b"unrelated")
        with patch.object(assets, "TRAIN", self.selected):
            with self.assertRaisesRegex(assets.AssetError, "outside the isolated"):
                self.stage(downloader=lambda **kwargs: str(foreign))
            plan = self.plan()
            plan["files"][0]["relative_path"] = "../escape"
            with self.assertRaisesRegex(assets.AssetError, "changed after"):
                self.stage(plan)
        self.assertFalse(self.destination.exists())

    def test_atomic_publication_will_not_replace_even_empty_directory(self):
        source = self.root / "stage"
        source.mkdir()
        (source / "data").write_text("new")
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            assets.publish_no_replace(source, self.destination)
        self.assertTrue(source.exists())
        self.assertEqual(list(self.destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
