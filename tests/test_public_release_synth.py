"""CPU guards/routing tests only; no fake model or audio-generation pass."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_release_synth_entry", ROOT / "scripts/synth_public_release.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class PublicReleaseSynthesisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="public-synth-", dir=ROOT / ".runtime")
        self.work = Path(self.temp.name)
        self.original, self.public, self.source = (self.work / x for x in ("original", "public", "official"))
        for path in (self.original, self.public, self.source):
            path.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def routing_fixture(self):
        # Tiny negative files do NOT satisfy the upstream real-release verifier.
        # Stub that verifier only in explicit adapter-plumbing tests below.
        identity = {"repository":p.release.MODEL_REPO,"snapshot_revision":p.release.MODEL_REVISION,"files":{}}
        contents = {"config.yaml":b"config interface fixture",p.release.TOKENIZER:b"tokenizer interface fixture","LICENSE":b"license interface fixture"}
        for name, value in contents.items():
            (self.original / name).write_bytes(value)
            published = "MODEL_LICENSE" if name == "LICENSE" else name
            (self.public / published).write_bytes(value)
            identity["files"][name] = hashlib.sha256(value).hexdigest()
        (self.original / "gpt.pth").write_bytes(b"original fixture, not real weights")
        (self.public / "gpt.pth").write_bytes(b"public fixture, not real weights")
        identity["files"]["gpt.pth"] = p.synth.sha256(self.original / "gpt.pth")
        record = dict(base_model=p.release.MODEL_REPO,base_revision=p.release.MODEL_REVISION,
                      official_source_commit=p.synth.OFFICIAL_COMMIT,base_gpt_sha256=identity["files"]["gpt.pth"],
                      gpt_sha256=p.release.GPT_SHA,checkpoint_step=p.release.STEP)
        (self.public / "release.json").write_text(json.dumps(record))
        (self.public / "manifest.json").write_text(json.dumps({"unit_plumbing_fixture_only":True}))
        verified = dict(status="verified_public_release",files=13,gpt_sha256=p.release.GPT_SHA,
                        checkpoint_step=100,quality_gate_passed=False,uploaded=False)
        return identity, record, verified

    def args(self):
        manifest = self.original / "model-manifest.json"
        manifest.write_text("{}")
        return SimpleNamespace(workspace_root=self.work,official_repo=self.source,model_dir=self.original,
                               model_manifest=manifest,merged_bundle=self.public,prompt=None,
                               output_dir=self.work / "unused-output",text=None,variant="merged",seed=1234,no_kv_cache=False)

    def argv(self):
        args = self.args()
        result = []
        for key in ("workspace_root","official_repo","model_dir","model_manifest","merged_bundle","output_dir"):
            result += ["--"+key.replace("_","-"),str(getattr(args,key))]
        return result

    def test_real_verifier_rejects_small_fixture_without_private_native_files(self):
        identity, _, _ = self.routing_fixture()
        with self.assertRaises(p.synth.SynthesisError):
            p.verify_public_bundle(self.public,self.original,identity)
        self.assertFalse((self.public / "COMPLETE.json").exists())
        self.assertFalse((self.public / "provenance.json").exists())
        self.assertFalse((self.work / "unused-output").exists())

    def test_overlap_and_wrong_base_stop_before_public_verification(self):
        identity, _, _ = self.routing_fixture()
        with patch.object(p.release,"verify_release",side_effect=AssertionError("must stop before reading public bytes")):
            for destination in (self.original,self.original / "nested"):
                destination.mkdir(exist_ok=True)
                with self.assertRaisesRegex(p.synth.SynthesisError,"independent"):
                    p.verify_public_bundle(destination,self.original,identity)
            for change in ({"repository":"other/model"},{"snapshot_revision":"0"*40},{"files":None}):
                with self.assertRaisesRegex(p.synth.SynthesisError,"fixed verified original"):
                    p.verify_public_bundle(self.public,self.original,{**identity,**change})
            with self.assertRaisesRegex(p.synth.SynthesisError,"fixed verified original"):
                p.verify_public_bundle(self.public,self.original,None)

    def test_verified_metadata_adapter_binds_actual_base_and_public_hashes(self):
        identity, _, verified = self.routing_fixture()
        before = {x.name:x.read_bytes() for x in self.public.iterdir()}
        with patch.object(p.release,"verify_release",return_value=verified) as validation:
            result = p.verify_public_bundle(self.public,self.original,identity)
        validation.assert_called_once_with(self.public)
        self.assertEqual(result,{"format":"reviewed_public_release_v1","gpt_sha256":p.release.GPT_SHA,
            "manifest_sha256":p.synth.sha256(self.public / "manifest.json"),
            "release_sha256":p.synth.sha256(self.public / "release.json"),"optimizer_updates":100,"verified_public_files":13})
        self.assertEqual(before,{x.name:x.read_bytes() for x in self.public.iterdir()})
        self.assertNotIn("complete_sha256",result)
        self.assertNotIn("provenance_sha256",result)

    def test_public_base_binding_and_alias_fail_even_after_metadata_verifier(self):
        identity, record, verified = self.routing_fixture()
        with patch.object(p.release,"verify_release",return_value=verified):
            for key,value in (("base_gpt_sha256","0"*64),("base_revision","0"*40),("official_source_commit","0"*40),
                              ("gpt_sha256","0"*64),("checkpoint_step",500)):
                (self.public / "release.json").write_text(json.dumps({**record,key:value}))
                with self.assertRaisesRegex(p.synth.SynthesisError,"actual original GPT/source"):
                    p.verify_public_bundle(self.public,self.original,identity)
            (self.public / "release.json").write_text(json.dumps(record))
            for field in ("config.yaml",p.release.TOKENIZER,"MODEL_LICENSE"):
                old=(self.public / field).read_bytes()
                (self.public / field).write_bytes(old+b"tamper")
                with self.assertRaisesRegex(p.synth.SynthesisError,"config/tokenizer/model license"):
                    p.verify_public_bundle(self.public,self.original,identity)
                (self.public / field).write_bytes(old)
            (self.public / "gpt.pth").unlink()
            os.link(self.original / "gpt.pth",self.public / "gpt.pth")
            with self.assertRaisesRegex(p.synth.SynthesisError,"alias"):
                p.verify_public_bundle(self.public,self.original,identity)

    def test_release_verification_has_network_guard_and_safe_error(self):
        identity, _, _ = self.routing_fixture()
        previous=socket.socket
        def no_model_fixture(root):
            with self.assertRaisesRegex(p.synth.SynthesisError,"network socket"):
                socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            raise RuntimeError("Do not echo a private path or payload from a dependency")
        with patch.object(p.release,"verify_release",side_effect=no_model_fixture):
            with self.assertRaisesRegex(p.synth.SynthesisError,"Cannot verify the bounded local public") as failure:
                p.verify_public_bundle(self.public,self.original,identity)
        self.assertNotIn("private path",str(failure.exception))
        self.assertIs(socket.socket,previous)

    def test_optional_injection_preserves_native_default_and_guards(self):
        args=self.args()
        identity={"files":{}}
        with patch.object(p.synth,"verify_source",return_value=self.source),patch.object(p.synth,"verify_synth_assets",return_value=({},identity)),patch.object(p.synth,"verify_merged_bundle",side_effect=p.synth.SynthesisError("native verifier sentinel")) as native:
            with self.assertRaisesRegex(p.synth.SynthesisError,"native verifier sentinel"):
                p.synth.metadata_preflight(args)
            native.assert_called_once_with(self.public,self.original,identity)
            native.reset_mock()
            def injected(root,original,assets):
                self.assertEqual((root,original),(self.public,self.original)); self.assertIs(assets,identity)
                raise p.synth.SynthesisError("public routing sentinel")
            with self.assertRaisesRegex(p.synth.SynthesisError,"public routing sentinel"):
                p.synth.metadata_preflight(args,bundle_verifier=injected)
            native.assert_not_called()
        args.output_dir=self.original / "bad"
        with patch.object(p.synth,"verify_source",return_value=self.source),patch.object(p.synth,"verify_synth_assets",side_effect=AssertionError("output guard must run first")):
            with self.assertRaisesRegex(p.synth.SynthesisError,"overlap"):
                p.synth.metadata_preflight(args,bundle_verifier=lambda *x:None)

    def test_entry_forwards_fixed_verifier_and_keeps_execute_opt_in(self):
        report={"status":"unit metadata routing fixture, not a verified real model", "dependencies":{"missing":[]},
                "models_loaded":False,"cuda_accessed":False,"synthesis_executed":False}
        with patch.object(p.synth,"metadata_preflight",return_value=(report,{})) as metadata,patch.object(p.synth,"execute_once",side_effect=AssertionError("no default execution")) as execute,contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(p.main(self.argv()),0)
            self.assertIs(metadata.call_args.kwargs["bundle_verifier"],p.verify_public_bundle)
            self.assertFalse(json.loads(stdout.getvalue())["synthesis_executed"])
            execute.assert_not_called()
        with patch.object(p.synth,"metadata_preflight",side_effect=AssertionError("execution opt-in must fail before metadata")),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as exit:
            p.main(self.argv()+["--execute"])
        self.assertEqual(exit.exception.code,2)

    def test_help_and_dependency_mode_import_no_model_or_hub(self):
        script = "import runpy,sys;sys.argv=['synth_public_release.py','--help'];\ntry:runpy.run_path('scripts/synth_public_release.py',run_name='__main__')\nexcept SystemExit as e:assert e.code==0\nassert not {'torch','whisper','huggingface_hub','indextts.infer_v2_5'} & set(sys.modules)"
        result=subprocess.run([sys.executable,"-B","-c",script],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(ROOT / "src")),capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("13-file public best100",result.stdout)
        self.assertIn("--allow-cuda",result.stdout)
        self.assertIn("--merged-bundle",result.stdout)


if __name__ == "__main__":
    unittest.main()
