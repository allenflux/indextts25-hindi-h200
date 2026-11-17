"""CPU metadata/safety tests. No mocked synthesis or model-generation claims."""
import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave

from hi_h200_native.assets import Asset, MODEL_REPO, MODEL_REVISION
from hi_h200_native import synth

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))


class NativeSynthesisMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_module_and_dependency_only_cli_never_import_torch_or_model(self):
        script = "import sys; from hi_h200_native import synth; synth.main(['--check-deps-only']); assert 'torch' not in sys.modules; assert 'indextts.infer_v2_5' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, cwd=ROOT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src")), timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["cuda_accessed"])
        self.assertFalse(report["models_loaded"])
        self.assertFalse(report["synthesis_executed"])
        self.assertFalse(report["dependencies"]["runtime_import_verified"])
        help_result = subprocess.run([sys.executable, "-m", "hi_h200_native.synth", "--help"], cwd=ROOT,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src")),
                                     text=True, capture_output=True, timeout=20)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--allow-cuda", help_result.stdout)
        self.assertIn("--no-kv-cache", help_result.stdout)

    @unittest.skipUnless(OFFICIAL.is_dir(), "Requires the pinned official source, no model weights")
    def test_exact_source_frontend_and_devanagari_policy(self):
        self.assertEqual(synth.verify_source(OFFICIAL), OFFICIAL.resolve())
        text = "क़िला और शक्ति की भाषा है।"
        self.assertEqual(synth.reviewed_hindi_text(OFFICIAL, text), text)
        for unsafe in ("हिन्दी: भाषा", "हिन्दी (भाषा)", "हिन्दी <शब्द|उच्चारण>", "हिन्दी १२"):
            with self.assertRaises(ValueError):
                synth.reviewed_hindi_text(OFFICIAL, unsafe)
        deps = synth.dependency_report(official_repo=OFFICIAL)
        self.assertIn("transformers==4.52.1", deps["official_direct_constraints"])
        self.assertEqual(deps["official_python_constraint"], ">=3.10,<3.12")
        self.assertFalse(deps["stock_frontend_constructor_verified"])

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "Optional PyYAML required for config metadata")
    def test_real_file_hash_and_pinned_metadata_algorithm_rejects_changes(self):
        # Tiny FILE fixtures test hash/manifest logic only, never a stand-in model.
        import yaml
        original = self.root / "original"
        original.mkdir()
        cfg = {"version":"2.5", "gpt_checkpoint":"gpt.pth", "w2v_stat":"wav2vec2bert_stats.pt",
               "s2mel_checkpoint":"s2mel.pth", "emo_matrix":"feat2.pt", "spk_matrix":"feat1.pt"}
        contents = {"config.yaml":yaml.safe_dump(cfg).encode(), "gpt.pth":b"metadata-fixture-only"}
        expected, files = [], {}
        for name, content in contents.items():
            (original / name).write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            expected.append(Asset(MODEL_REPO, MODEL_REVISION, name, name, len(content), "sha256", digest))
            files[name] = digest
        manifest = self.root / "manifest.json"
        self.write_json(manifest, {"repository":MODEL_REPO,"snapshot_revision":MODEL_REVISION,"files":files})
        _, identity = synth.verify_synth_assets(original, manifest, assets=expected)
        self.assertEqual(identity["pinned_payload_files"], 2)
        (original / "gpt.pth").write_bytes(b"changed-same-size-bytes")
        with self.assertRaises(ValueError):
            synth.verify_synth_assets(original, manifest, assets=expected)
        (original / "gpt.pth").write_bytes(contents["gpt.pth"])
        files["../escape"] = "0" * 64
        self.write_json(manifest, {"repository":MODEL_REPO,"snapshot_revision":MODEL_REVISION,"files":files})
        with self.assertRaises(ValueError):
            synth.verify_synth_assets(original, manifest, assets=expected)

    def test_merged_inventory_identity_tampering_and_alias_rejected(self):
        original, merged = self.root / "original", self.root / "merged"
        original.mkdir()
        merged.mkdir()
        (original / "gpt.pth").write_bytes(b"original metadata fixture")
        core = {name: hashlib.sha256(name.encode()).hexdigest() for name in synth.CORE}
        base = {"model_repository":MODEL_REPO, "model_snapshot_revision":MODEL_REVISION, "model_files":core}
        (merged / "gpt.pth").write_bytes(b"merged metadata fixture")
        provenance = {"kind":"hi_h200_native_merged_official_gpt", "official_commit":synth.OFFICIAL_COMMIT,
                      "base_identity":base, "gpt_sha256":synth.sha256(merged / "gpt.pth"), "optimizer_updates":2}
        self.write_json(merged / "provenance.json", provenance)
        (merged / "LICENSE").write_text("license fixture")
        (merged / "LICENSE_ZH.txt").write_text("license fixture zh")
        (merged / "README.md").write_text("readme fixture")
        (merged / "NON_ENDORSEMENT.txt").write_text("notice fixture")
        inventory = {p.name:synth.sha256(p) for p in merged.iterdir()}
        self.write_json(merged / "COMPLETE.json", {"schema_version":1,"files":inventory})
        identity = {"files":core}
        self.assertEqual(synth.verify_merged_bundle(merged, original, identity)["optimizer_updates"], 2)
        partial = dict(inventory)
        partial.pop("LICENSE_ZH.txt")
        self.write_json(merged / "COMPLETE.json", {"schema_version":1,"files":partial})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            synth.verify_merged_bundle(merged, original, identity)
        self.write_json(merged / "COMPLETE.json", {"schema_version":1,"files":inventory})
        (merged / "gpt.pth").write_bytes(b"tamper")
        with self.assertRaisesRegex(ValueError, "hash"):
            synth.verify_merged_bundle(merged, original, identity)
        with self.assertRaisesRegex(ValueError, "independent"):
            synth.verify_merged_bundle(original, original, identity)

    def write_wav(self, name, *, frames=16000, value=1000):
        path = self.root / name
        with wave.open(str(path), "wb") as output:
            output.setparams((1,2,16000,0,"NONE","not compressed"))
            output.writeframes(array.array("h", [value] * frames).tobytes())
        return path

    def test_import_probe_requires_explicit_source_workspace_new_output(self):
        result = subprocess.run([sys.executable, "-m", "hi_h200_native.synth", "--probe-runtime-imports"],
                                cwd=ROOT, text=True, capture_output=True, timeout=20,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src")))
        self.assertEqual(result.returncode, 2)
        self.assertIn("workspace/source/new output", result.stderr)

    def test_real_pcm_wav_metadata_rejects_silence_short_prompt_and_truncation(self):
        path = self.write_wav("prompt.wav")
        info = synth.wav_info(path, prompt=True)
        self.assertEqual(info["seconds"], 1)
        self.assertEqual(info["peak_pcm16"], 1000)
        self.assertEqual(info["file_sha256"], synth.sha256(path))
        for name, count, value in (("silent.wav",16000,0), ("short.wav",8000,1000)):
            with self.assertRaises(ValueError):
                synth.wav_info(self.write_wav(name, frames=count, value=value), prompt=True)
        path.write_bytes(path.read_bytes()[:-20])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            synth.wav_info(path, prompt=True)

    def test_resource_telemetry_and_nonfinite_admission_fail_before_cuda(self):
        sample = synth.parse_gpu_sample("0, GPU-unit-test, H200, 141000, 1000, 2\n")
        self.assertEqual(sample["free_mib"], 140000)
        for row in ("0,GPU-x,H200,nan,1,2", "0,GPU-x,H200,10,11,2", "0,GPU-x,H200,10,1,101", "0,GPU-x,H200,10,1,2\n1,GPU-y,H200,10,1,2"):
            with self.assertRaises(ValueError):
                synth.parse_gpu_sample(row)
        args = SimpleNamespace(allow_cuda=True, min_free_gpu_mib=float("nan"), min_free_ram_mib=1, max_utilization=10, gpu_id="0")
        with self.assertRaisesRegex(ValueError, "threshold"):
            synth.resource_admission(args)
        args.allow_cuda = False
        with self.assertRaisesRegex(ValueError, "allow-cuda"):
            synth.resource_admission(args)

    def test_network_guard_blocks_real_socket_without_network_and_restores(self):
        previous = socket.socket
        with synth.deny_network():
            with self.assertRaisesRegex(ValueError, "DNS"):
                socket.getaddrinfo("example.invalid", 443)
            # Reject construction before connect/sendto/listen can reach the OS.
            for family in (socket.AF_INET, socket.AF_INET6):
                for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
                    with self.assertRaisesRegex(ValueError, "network socket"):
                        socket.socket(family, kind)
            with socket.socket(socket.AF_UNIX) as connection:
                self.assertEqual(connection.family, socket.AF_UNIX)
        self.assertIs(socket.socket, previous)

    def test_output_no_overwrite_overlap_or_workspace_escape(self):
        protected = self.root / "original"
        protected.mkdir()
        for path in (protected, protected / "bad", self.root.parent / "escape"):
            with self.assertRaises(ValueError):
                synth._independent_output(path, self.root, (protected,))
        target = self.root / "new-output"
        self.assertEqual(synth._independent_output(target, self.root, (protected,)), target)
        self.assertFalse(target.exists())

    def test_cache_adapter_only_passes_real_normalizer_contract(self):
        # Argument injection verifies isolation only; no FST/model execution claim.
        calls = []
        class Base:
            def __init__(self):
                self.zh_normalizer = None
                self.en_normalizer = None
                self.punctuation_policy = "base-unchanged"
        def factory(cache_dir, **kwargs):
            calls.append((Path(cache_dir), kwargs))
            return object()
        cache = self.root / "run" / "cache" / "tn"
        normalizer = synth.cache_normalizer_class(Base, cache, factories=(factory,factory), system="Linux")()
        normalizer.load()
        normalizer.load()
        self.assertEqual(len(calls), 2)
        self.assertEqual([item[0] for item in calls], [cache / "zh", cache / "en"])
        self.assertEqual(calls[0][1], {"remove_interjections":False,"remove_erhua":False,"overwrite_cache":False})
        self.assertEqual(calls[1][1], {"overwrite_cache":False})
        self.assertEqual(normalizer.punctuation_policy, "base-unchanged")
        def unsupported(**kwargs):
            raise AssertionError("Must not execute an unsupported constructor")
        with self.assertRaisesRegex(ValueError, "cache_dir"):
            synth.cache_normalizer_class(Base, self.root / "other", factories=(unsupported, factory), system="Linux")

    def test_runtime_cache_and_environment_are_isolated_then_restored(self):
        output = self.root / "output"
        output.mkdir()
        previous = os.environ.get("HF_HOME")
        previous_config = os.environ.get("XDG_CONFIG_HOME")
        previous_mpl = os.environ.get("MPLCONFIGDIR")
        before = Path.cwd()
        previous_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = False
        with synth.isolated_runtime(output, self.root, None) as environment:
            self.assertTrue(Path(environment["HF_HOME"]).is_relative_to(output))
            self.assertTrue(Path(environment["XDG_CONFIG_HOME"]).is_relative_to(output))
            self.assertTrue(Path(environment["MPLCONFIGDIR"]).is_relative_to(output))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"], "1")
            self.assertEqual(Path.cwd(), output / "runtime")
            self.assertTrue(sys.dont_write_bytecode)
        self.assertFalse(sys.dont_write_bytecode)
        sys.dont_write_bytecode = previous_bytecode
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(os.environ.get("HF_HOME"), previous)
        self.assertEqual(os.environ.get("XDG_CONFIG_HOME"), previous_config)
        self.assertEqual(os.environ.get("MPLCONFIGDIR"), previous_mpl)


if __name__ == "__main__":
    unittest.main()
