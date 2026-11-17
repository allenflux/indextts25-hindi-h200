"""Tiny CPU source/identity guards only; no mock generation or GPU acceptance."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("t8_compat_test_entry", ROOT / "scripts/check_t8_compat.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class T8CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="t8-compat-", dir=ROOT / ".runtime")
        self.work = Path(self.tmp.name)
        self.source = self.work / "source"
        self.source.mkdir()
        self.destination = self.work / "copy"
        self.revision = "b" * 40
        self.base = {"modelRepository": "fixture/base", "modelRevision": "a" * 40,
                     "files": {"gpt.pth": {"size": 4, "sha256": hashlib.sha256(b"base").hexdigest()},
                               "config.yaml": {"size": 3, "sha256": hashlib.sha256(b"cfg").hexdigest()}}}
        self.contents = {
            "nodes_v3.py": b'''class OtherNode:
    def define_schema(cls):
        return schema(node_id="Other", options=["ZH", "EN", "JA", "ES", "AR"])
class T8IndexTTS25ModelLoader:
    def execute(cls, model_dir):
        manifest = load_manifest()
        return manifest["modelRevision"]
class T8IndexTTS25Generate:
    def define_schema(cls):
        return schema(node_id="T8_IndexTTS25_Generate", inputs=[combo("language", options=["ZH", "EN", "JA", "ES", "AR"])])
''',
            "runtime/inference_adapter.py": b'''def run_inference(handle, speaker_audio, text, language, duration_factor, seed):
    if language.upper() not in {"ZH", "EN", "JA", "ES", "AR"}:
        raise ValueError("unsupported:" + language)
    return handle
''',
            "runtime/text_planner.py": b'''LANGUAGE_AUTO_LIMITS = {"ZH": 120, "EN": 60, "JA": 100, "ES": 60, "AR": 80}
def auto_segment_limit(language):
    return LANGUAGE_AUTO_LIMITS[language]
''',
            "services/model_store.py": b'''from pathlib import Path
import json
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_ROOT / "manifests" / "model_2_5.json"
def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
def validate_model_dir(model_dir, *, verify_hashes=False):
    manifest = load_manifest()
    return verify_hashes
def model_fingerprint(model_dir):
    manifest = load_manifest()
    return manifest["modelRevision"]
''',
            p.BASE_MANIFEST: json.dumps(self.base).encode(),
            "LICENSE": b"Tiny license fixture; not the real T8 license.\n",
            "LICENSE_ZH.txt": b"Tiny Chinese-license interface fixture.\n",
        }
        for name, data in self.contents.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "tiny fixture only")
        revision = self.git("rev-parse", "HEAD").decode().strip()
        self.pins = {name: p._blob(data) for name, data in self.contents.items()}
        self.patch_context = patch.multiple(p, PINNED_BLOBS=self.pins, NODE_REVISION=revision,
                                           GPT_BYTES=9, GPT_SHA256=hashlib.sha256(b"candidate").hexdigest())
        self.patch_context.start()
        self.patched = p.patch_sources(self.contents)
        self.sha_context = patch.object(p, "PATCHED_SHA256", {name: hashlib.sha256(data).hexdigest()
                                                             for name, data in self.patched.items()})
        self.sha_context.start()

    def tearDown(self):
        self.sha_context.stop()
        self.patch_context.stop()
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def prepare(self):
        return p.prepare(self.source, self.destination, self.revision)

    def test_ast_transforms_preserve_node_id_and_other_ui_and_guard_semantics(self):
        tree = ast.parse(self.patched["nodes_v3.py"])
        classes = {n.name: n for n in tree.body}
        self.assertIn('node_id=\'T8_IndexTTS25_Generate\'', ast.unparse(classes["T8IndexTTS25Generate"]))
        self.assertIn("'HI'", ast.unparse(classes["T8IndexTTS25Generate"]))
        self.assertNotIn("'HI'", ast.unparse(classes["OtherNode"]))
        original, modified = {}, {}
        exec(self.contents["runtime/inference_adapter.py"], original)
        exec(self.patched["runtime/inference_adapter.py"], modified)
        with self.assertRaisesRegex(ValueError, "unsupported:HI"):
            original["run_inference"](None, {}, "नमस्ते", "HI", 1.0, 1234)
        # A sentinel return only proves guard routing in this tiny fixture.
        self.assertEqual(modified["run_inference"]("guard-only", {}, "नमस्ते", "HI", 1.0, 1234), "guard-only")
        with self.assertRaisesRegex(ValueError, "unsupported:XX"):
            modified["run_inference"](None, {}, "x", "XX", 1.0, 1234)

    def test_planner_hi_limit_and_forced_hash_policy_are_actual_python(self):
        namespace = {}
        exec(self.patched["runtime/text_planner.py"], namespace)
        self.assertEqual(namespace["auto_segment_limit"]("HI"), 60)
        self.assertEqual(namespace["auto_segment_limit"]("ZH"), 120)
        namespace = {"__file__": str(self.source / "services/model_store.py")}
        exec(self.patched["services/model_store.py"], namespace)
        self.assertTrue(namespace["validate_model_dir"]("unused", verify_hashes=False))
        self.assertEqual(namespace["MANIFEST_PATH"].name, "model_2_5.json")

    def test_inspect_is_read_only_and_source_drift_fails_before_destination(self):
        before = sorted(str(x.relative_to(self.work)) for x in self.work.rglob("*"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.main(["inspect", "--node-root", str(self.source)]), 0)
        self.assertEqual(before, sorted(str(x.relative_to(self.work)) for x in self.work.rglob("*")))
        (self.source / "nodes_v3.py").write_bytes(self.contents["nodes_v3.py"] + b"# drift\n")
        with self.assertRaisesRegex(p.CompatibilityError, "source mismatch"):
            self.prepare()
        self.assertFalse(self.destination.exists())

    def test_git_archive_excludes_private_untracked_data_and_keeps_real_base_identity(self):
        (self.source / "private-untracked-token.txt").write_bytes(b"not a real credential; fixture only")
        result = self.prepare()
        self.assertEqual(result["status"], "verified_isolated_code_copy")
        self.assertFalse(result["model_assets_verified"])
        self.assertFalse(result["gpu_tested"])
        self.assertFalse((self.destination / ".git").exists())
        self.assertFalse((self.destination / "private-untracked-token.txt").exists())
        for name in (p.BASE_MANIFEST, "LICENSE", "LICENSE_ZH.txt"):
            self.assertEqual((self.destination / name).read_bytes(), self.contents[name])
        manifest = result["manifest_data"]
        self.assertEqual(manifest["modelRepository"], p.CANDIDATE_REPOSITORY)
        self.assertEqual(manifest["modelRevision"], self.revision)
        self.assertEqual(manifest["files"]["gpt.pth"]["sha256"], p.GPT_SHA256)
        self.assertEqual(manifest["experimentalDerivative"]["baseGPT"], self.base["files"]["gpt.pth"])
        self.assertEqual(manifest["files"]["config.yaml"], {
            **self.base["files"]["config.yaml"], "sourceRepository": self.base["modelRepository"],
            "sourceRevision": self.base["modelRevision"]})
        overridden = copy.deepcopy(self.base)
        overridden["files"]["config.yaml"].update(sourceRepository="fixture/override", sourceRevision="d" * 40)
        derived = p.candidate_manifest(overridden, self.revision)
        self.assertEqual(derived["files"]["config.yaml"]["sourceRepository"], "fixture/override")
        self.assertEqual(derived["files"]["config.yaml"]["sourceRevision"], "d" * 40)
        for name, metadata in manifest["files"].items():
            if name != "gpt.pth":
                original = self.base["files"][name]
                self.assertEqual(metadata["sourceRepository"], original.get("sourceRepository", self.base["modelRepository"]))
                self.assertEqual(metadata["sourceRevision"], original.get("sourceRevision", self.base["modelRevision"]))
        self.assertEqual((self.source / "nodes_v3.py").read_bytes(), self.contents["nodes_v3.py"])

    def test_local_git_replace_cannot_substitute_an_unpinned_execution_file(self):
        original = self.git("rev-parse", "HEAD").decode().strip()
        (self.source / "extra_execution.py").write_text("raise RuntimeError('must not be archived')\n")
        self.git("add", "extra_execution.py")
        self.git("-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "replacement fixture")
        replacement = self.git("rev-parse", "HEAD").decode().strip()
        self.git("reset", "--hard", original)
        self.git("replace", original, replacement)
        # Actual git replacement is active for an unguarded local archive.
        unsafe = self.git("archive", "--format=tar", original)
        with tarfile.open(fileobj=io.BytesIO(unsafe), mode="r:") as archive:
            self.assertIn("extra_execution.py", archive.getnames())
        self.prepare()
        self.assertFalse((self.destination / "extra_execution.py").exists())
        self.assertEqual((self.destination / "nodes_v3.py").read_bytes(), self.patched["nodes_v3.py"])

    def test_destination_existing_overlap_and_symlink_are_refused(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(p.CompatibilityError, "fresh"):
            self.prepare()
        with self.assertRaises(p.CompatibilityError):
            p.prepare(self.source, self.source / "nested", self.revision)
        link = self.work / "symlink"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(p.CompatibilityError):
            p.prepare(link, self.work / "other", self.revision)

    def test_prepared_source_manifest_and_audit_tamper_fail_closed(self):
        self.prepare()
        path = self.destination / "runtime/inference_adapter.py"
        actual = path.read_bytes()
        path.write_bytes(actual + b"# changed\n")
        with self.assertRaisesRegex(p.CompatibilityError, "Patched source drift"):
            p.verify_prepared(self.destination, self.revision)
        path.write_bytes(actual)
        manifest_path = self.destination / p.CANDIDATE_MANIFEST
        actual = manifest_path.read_bytes()
        manifest = json.loads(actual)
        manifest["files"]["gpt.pth"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(p.CompatibilityError, "Derivative manifest"):
            p.verify_prepared(self.destination, self.revision)
        manifest_path.write_bytes(actual)
        with self.assertRaisesRegex(p.CompatibilityError, "audit identity"):
            p.verify_prepared(self.destination, "c" * 40)
        audit_path = self.destination / p.AUDIT
        audit = json.loads(audit_path.read_bytes())
        audit["surprise"] = "forbidden"
        audit_path.write_text(json.dumps(audit))
        with self.assertRaisesRegex(p.CompatibilityError, "audit identity"):
            p.verify_prepared(self.destination)

    def test_asset_verifier_hashes_all_tiny_files_and_rejects_tamper_or_symlinks(self):
        model = self.work / "model"
        model.mkdir()
        (model / "gpt.pth").write_bytes(b"base")
        (model / "config.yaml").write_bytes(b"cfg")
        self.assertEqual(p.verify_assets(model, self.base)["files"], 2)
        (model / "gpt.pth").write_bytes(b"evil")  # Same size must still fail.
        with self.assertRaisesRegex(p.CompatibilityError, "SHA256 mismatch"):
            p.verify_assets(model, self.base)
        (model / "gpt.pth").unlink()
        (model / "gpt.pth").symlink_to(self.source / "LICENSE")
        with self.assertRaisesRegex(p.CompatibilityError, "non-symlink"):
            p.verify_assets(model, self.base)
        malicious = copy.deepcopy(self.base)
        malicious["files"]["../escape"] = malicious["files"]["config.yaml"]
        with self.assertRaises(p.CompatibilityError):
            p.verify_assets(model, malicious)

    def test_dual_manifest_real_size_routing_and_actual_hashes_reject_tamper(self):
        self.prepare()
        base_dir, hindi_dir, unknown_dir = [self.work / name for name in ("base-assets", "hindi-assets", "unknown-assets")]
        for directory, gpt in ((base_dir, b"base"), (hindi_dir, b"candidate"), (unknown_dir, b"other")):
            directory.mkdir()
            (directory / "gpt.pth").write_bytes(gpt)
            (directory / "config.yaml").write_bytes(b"cfg")
        # Execute the same transformed load_manifest/fingerprint used in the
        # isolated plugin. This checks real files/sizes, never model generation.
        module = {"__file__": str(self.destination / "services/model_store.py")}
        exec((self.destination / "services/model_store.py").read_bytes(), module)
        self.assertEqual(module["MANIFEST_PATH"], self.destination / p.BASE_MANIFEST)
        self.assertEqual(module["load_manifest"](), self.base)
        for directory, variant, revision in ((base_dir, "base", self.base["modelRevision"]),
                                              (hindi_dir, "hindi", self.revision),
                                              (unknown_dir, "base", self.base["modelRevision"])):
            with self.subTest(variant=variant, directory=directory.name):
                actual = module["load_manifest"](directory)
                selected = p.selected_manifest(self.destination, directory, self.revision, variant=variant)
                self.assertEqual(actual, selected["manifest_data"])
                self.assertEqual(selected["variant"], variant)
                self.assertFalse(selected["assets_verified"])
                self.assertEqual(module["model_fingerprint"](directory), revision)
                if directory == unknown_dir:
                    with self.assertRaisesRegex(p.CompatibilityError, "size mismatch"):
                        p.verify_assets(directory, actual)
                else:
                    self.assertEqual(p.verify_assets(directory, actual)["files"], 2)
        with self.assertRaisesRegex(p.CompatibilityError, "requested variant"):
            p.selected_manifest(self.destination, base_dir, self.revision, variant="hindi")
        for directory, replacement in ((base_dir, b"evil"), (hindi_dir, b"tampered!")):
            (directory / "gpt.pth").write_bytes(replacement)  # Same size as selected GPT.
            manifest = module["load_manifest"](directory)
            with self.assertRaisesRegex(p.CompatibilityError, "SHA256 mismatch"):
                p.verify_assets(directory, manifest)

    def test_ast_drift_and_bad_revisions_are_rejected(self):
        for name, before, after in [("nodes_v3.py", b'"T8_IndexTTS25_Generate"', b'"OtherID"'),
                                    ("runtime/text_planner.py", b'"AR": 80', b'"AR": 81')]:
            contents = dict(self.contents)
            contents[name] = contents[name].replace(before, after)
            with self.assertRaises(p.CompatibilityError):
                p.patch_sources(contents)
        for revision in ("main", "a" * 39, "A" * 40, "a" * 40 + "\n"):
            with self.assertRaises(p.CompatibilityError):
                p.candidate_manifest(self.base, revision)

    def test_exact_oversized_entry_streams_fully_and_rejects_size_hash_or_unknown(self):
        name = next(iter(p.LARGE_SOURCE_ENTRIES))
        data = b"tiny oversized source fixture"
        info = tarfile.TarInfo(name)
        info.size = len(data)
        pin = {name: {"size": len(data), "blob": p._blob(data)}}
        with patch.object(p, "MAX_SOURCE_FILE", 8), patch.object(p, "LARGE_SOURCE_ENTRIES", pin):
            output = io.BytesIO()
            p._copy_source_entry(io.BytesIO(data), output, info)
            self.assertEqual(output.getvalue(), data)  # No ordinary-bound truncation.
            for damaged, message in ((data[:-1], "Truncated"),
                                     (b"X" + data[1:], "Git blob mismatch"),
                                     (data + b"X", "length mismatch")):
                with self.assertRaisesRegex(p.CompatibilityError, message):
                    p._copy_source_entry(io.BytesIO(damaged), io.BytesIO(), info)
            wrong_size = copy.copy(info)
            wrong_size.size -= 1
            with self.assertRaisesRegex(p.CompatibilityError, "size/type mismatch"):
                p._source_entry_pin(wrong_size)
            unknown = copy.copy(info)
            unknown.name = "unreviewed-large-file.bin"
            with self.assertRaisesRegex(p.CompatibilityError, "too large"):
                p._source_entry_pin(unknown)

    def test_archive_traversal_and_symlinks_rejected_before_new_directory(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            container = io.BytesIO()
            with tarfile.open(fileobj=container, mode="w:") as stream:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "../../outside" if kind == tarfile.SYMTYPE else ""
                member.size = 0
                stream.addfile(member, io.BytesIO())
            real_git = p._git
            def git(root, *args):
                return container.getvalue() if args[0] == "archive" else real_git(root, *args)
            with patch.object(p, "_git", side_effect=git):
                with self.assertRaises(p.CompatibilityError):
                    self.prepare()
            self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
