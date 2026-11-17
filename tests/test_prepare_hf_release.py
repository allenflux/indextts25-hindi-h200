"""CPU safety tests only: small fixtures never pass real model identity validation."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_hf_release", ROOT / "scripts/prepare_hf_release.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class ReleasePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="release-prep-", dir=ROOT / ".runtime")
        self.work = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def summary(self):
        # Pure arithmetic fixtures, not transcripts emitted by an ASR engine.
        rows = []
        for language in r.asr.LANGUAGES:
            for condition in ("baseline", "merged"):
                rows.append(dict(id="fixture-only", language=language, condition=condition, status="asr_computed",
                                 metrics=r.asr.metric_views("one", "one", language)))
        rows.append(dict(id="reference-fixture", language="hi", condition="reference", status="asr_computed",
                         metrics=r.asr.metric_views("one", "one", "hi")))
        return r.asr.public_summary(rows, executed=True)
    def args(self):
        for name in ("official", "assets", "bundle"):
            (self.work / name).mkdir()
        files = {asset.relative_path:"a" * 64 for asset in r.selected_assets("synth")}
        manifest = self.work / "assets/model-manifest.json"
        manifest.write_text(json.dumps(dict(repository=r.MODEL_REPO, snapshot_revision=r.MODEL_REVISION, files=files)))
        (self.work / "bundle/gpt.pth").write_bytes(b"tiny negative fixture, never a trained model")
        (self.work / "bundle/COMPLETE.json").write_text(json.dumps(dict(schema_version=1, files={"gpt.pth":r.GPT_SHA})))
        card = self.work / "reviewed-card.md"
        card.write_text("---\nlicense: other\n---\n# Experimental model\n")
        summary = self.work / "public-summary.json"
        summary.write_text(json.dumps(self.summary()))
        return r.argparse.Namespace(work_root=self.work, official_repo=self.work / "official", official_assets=self.work / "assets",
                                   asset_manifest=manifest, merged_bundle=self.work / "bundle", model_card=card,
                                   public_asr_summary=summary, output=self.work / "release")
    def test_sanitizer_rebuilds_numeric_only_aggregate(self):
        public = r.sanitize_evaluation(self.summary())
        self.assertFalse(public["quality_gate_passed"])
        self.assertFalse(public["calibrated_reference_available"]["en"])
        self.assertIsNone(public["metrics"]["ja"]["merged"]["strict"]["wer"])
        text = json.dumps(public)
        for private in ("fixture-only", "reference-fixture", '"text"', '"hypothesis"', str(self.work)):
            self.assertNotIn(private, text)
    def test_private_or_unknown_fields_and_misleading_metrics_block(self):
        mutations = [lambda x:x.update(private_transcript="must not publish"),
                     lambda x:x["metrics"]["hi"]["merged"]["strict"].update(reference="private"),
                     lambda x:x["coverage_gap_counts"]["en"].update({"publisher-id":1}),
                     lambda x:x.update(quality_gate_passed=True),
                     lambda x:x.update(asr_executed=False),
                     lambda x:x["limitations"].append("hf_" + "a" * 32),
                     lambda x:x["metrics"]["hi"]["baseline"]["strict"].update(cer=.3),
                     lambda x:x["metrics"]["ja"]["baseline"]["strict"].update(wer=1),
                     lambda x:x["calibrated_reference_available"].update(en=True),
                     lambda x:x["counts"]["hi"].update(reference=True)]
        for mutate in mutations:
            value = copy.deepcopy(self.summary())
            mutate(value)
            with self.subTest(mutation=mutations.index(mutate)), self.assertRaises(r.ReleaseError):
                r.sanitize_evaluation(value)
    def test_paths_cannot_escape_or_enter_auth_git_and_output_symlinks(self):
        outside = self.work.parent / (self.work.name + "-outside")
        outside.mkdir()
        try:
            (self.work / "escape").symlink_to(outside)
            for value in (outside, self.work / "escape", self.work / ".git/new", self.work / "hf-home/token"):
                with self.assertRaises((r.ReleaseError, FileNotFoundError)):
                    r.local(self.work, value, exists=False)
            (self.work / "hf-home").mkdir()
            (self.work / "alias").symlink_to(self.work / "hf-home")
            with self.assertRaises(r.ReleaseError):
                r.local(self.work, self.work / "alias/new", exists=False)
            (self.work / "dangling").symlink_to(self.work / "future")
            with self.assertRaises(r.ReleaseError):
                r.local(self.work, self.work / "dangling", exists=False)
        finally:
            outside.rmdir()
        self.assertEqual(r.local(self.work, ROOT / "docs/MODEL_CARD.zh.md", card=True), ROOT / "docs/MODEL_CARD.zh.md")
    def test_candidate_byte_pin_blocks_tiny_fixture_even_when_metadata_is_mocked(self):
        args = self.args()
        identity = dict(files={"config.yaml":"a" * 64,r.TOKENIZER:"a" * 64,"LICENSE":"a" * 64,"gpt.pth":"a" * 64})
        merged = dict(gpt_sha256=r.GPT_SHA, optimizer_updates=100, complete_sha256="b" * 64)
        with patch.object(r.synth, "verify_source", return_value=args.official_repo), patch.object(r.synth, "verify_synth_assets", return_value=({},identity)), patch.object(r.synth,"verify_merged_bundle",return_value=merged):
            with self.assertRaisesRegex(r.ReleaseError, "completion marker"):
                r.inspect(args)
        self.assertFalse(args.output.exists())
    def test_wrong_step_or_candidate_hash_blocks_before_any_prepare(self):
        args = self.args()
        for merged in (dict(gpt_sha256="0" * 64,optimizer_updates=100),dict(gpt_sha256=r.GPT_SHA,optimizer_updates=500),dict(gpt_sha256=r.GPT_SHA,optimizer_updates=True)):
            with patch.object(r.synth,"verify_source"), patch.object(r.synth,"verify_synth_assets",return_value=({},{})), patch.object(r.synth,"verify_merged_bundle",return_value=merged):
                with self.assertRaisesRegex(r.ReleaseError,"frozen best100"):
                    r.inspect(args)
            self.assertFalse(args.output.exists())
    def test_unknown_asset_manifest_path_blocks_before_reading_other_files(self):
        args = self.args()
        value = json.loads(args.asset_manifest.read_text())
        value["files"]["hf-home/token"] = "0" * 64
        args.asset_manifest.write_text(json.dumps(value))
        with patch.object(r.synth,"verify_source",side_effect=AssertionError("Must not reach source inspection")):
            with self.assertRaisesRegex(r.ReleaseError,"non-asset/private"):
                r.inspect(args)
        self.assertFalse(args.output.exists())
    def test_existing_or_protected_output_stops_without_mutation(self):
        args = self.args()
        args.output.mkdir()
        keep = args.output / "keep"
        keep.write_text("preserve")
        with self.assertRaises(r.ReleaseError):
            r.inspect(args)
        self.assertEqual(keep.read_text(),"preserve")
        args.output = args.merged_bundle / "bad-output"
        with self.assertRaisesRegex(r.ReleaseError,"isolated"):
            r.inspect(args)
        self.assertFalse(args.output.exists())
    def test_copy_verifies_actual_bytes_and_refuses_overwrite(self):
        source = self.work / "small-public-fixture"
        source.write_bytes(b"not a model; bounded copy safety fixture")
        destination = self.work / "new-file"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        r.copy_verified(source,destination,digest,source.stat().st_size)
        self.assertEqual(source.read_bytes(),destination.read_bytes())
        with self.assertRaises(FileExistsError):
            r.copy_verified(source,destination,digest,source.stat().st_size)
        with self.assertRaises(r.ReleaseError):
            r.copy_verified(source,self.work / "bad-sha","0" * 64,source.stat().st_size)
    def test_whitelist_staging_never_copies_private_neighbors_or_passes_as_model(self):
        inputs = self.work / "inputs"
        inputs.mkdir()
        operations = {}
        generated = {"DATA_ATTRIBUTION.md":r.ATTRIBUTION.encode(),"release.json":b"{}\n","evaluation.json":b"{}\n"}
        for name in r.PAYLOADS-set(generated):
            path=inputs/name
            path.write_bytes(b"synthetic safety fixture, not official bytes")
            operations[name]=(path,hashlib.sha256(path.read_bytes()).hexdigest())
        (inputs/"provenance.json").write_text("PRIVATE_SENTINEL /private/project/path")
        (inputs/"audio.wav").write_bytes(b"not public")
        output=self.work/"synthetic-staging"
        r.materialize(output,operations,generated)
        self.assertEqual({p.name for p in output.iterdir()},r.FILES)
        self.assertFalse((output/"provenance.json").exists())
        self.assertNotIn("PRIVATE_SENTINEL",b"".join(p.read_bytes() for p in output.iterdir()).decode(errors="ignore"))
        with self.assertRaisesRegex(r.ReleaseError,"Fixed official payload|not frozen best100|fixed/bounded"):
            r.verify_release(output)
        with self.assertRaises(FileExistsError):
            r.materialize(output,operations,generated)
    def test_public_text_rejects_credential_patterns_and_private_paths(self):
        path=self.work/"card.md"
        for value in ("hf_"+"a"*32,"ghp_"+"b"*32,"gho_"+"b"*32,"ASIA"+"A"*16,"sk-"+"a"*32,"Bearer "+"a"*32,"https://user:password@example.invalid/model","/Users/person/private/model","/app/private/model", "-----BEGIN RSA PRIVATE KEY-----"):
            path.write_text(value)
            with self.assertRaises(r.ReleaseError):r.public_text(path)
    def test_fixed_payload_algorithm_rejects_self_consistent_tampering(self):
        path=self.work/"fixed-small-public-file"
        original=b"small fixture; not a model"
        path.write_bytes(original)
        digest=hashlib.sha1(b"blob "+str(len(original)).encode()+b"\0"+original).hexdigest()
        r.pinned_bytes(path,len(original),"git_blob_sha1",digest)
        path.write_bytes(b"X"+original[1:])
        with self.assertRaisesRegex(r.ReleaseError,"Fixed official payload hash"):
            r.pinned_bytes(path,len(original),"git_blob_sha1",digest)
        # A freshly recomputed self-consistent inventory still cannot alter the fixed expected pin.
        self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(),hashlib.sha256(original).hexdigest())
    def test_code_license_pins_match_actual_reviewed_official_git_files(self):
        source=ROOT/".runtime/external-official"
        if not source.is_dir():self.skipTest("Existing reviewed official checkout not available")
        for name,(size,digest) in r.CODE_LICENSES.items():
            r.pinned_bytes(source/name,size,"sha256",digest)
    def test_model_card_license_metadata_is_checked_in_frontmatter(self):
        path=self.work/"readme.md"
        for text in ("# no frontmatter\nlicense: other\n", "---\nlicense: mit\n---\nlicense: other\n", "---\nlicense: mit\nlicense: other\n---\n", "---\nlicense: other\nlicense: other\n---\n"):
            path.write_text(text)
            with self.assertRaises(r.ReleaseError):r.reviewed_card(path)
        path.write_text("---\nlicense: other\n---\n# Reviewed fixture\n")
        self.assertEqual(r.reviewed_card(path),path.read_bytes())

    def test_oversized_release_files_block_before_hashing_or_reading(self):
        from types import SimpleNamespace
        root=self.work/"resource-fixture"
        root.mkdir()
        fixed,maximum=r.release_size_limits()
        for name in r.FILES:
            (root/name).write_bytes(b"x" * fixed.get(name, 2))
        original=Path.stat
        for name in ("gpt.pth","evaluation.json","README.md","SHA256SUMS","DATA_ATTRIBUTION.md","NON_ENDORSEMENT.txt"):
            limit=fixed.get(name,maximum.get(name))
            def enlarged(path,*args,**kwargs):
                actual=original(path,*args,**kwargs)
                return SimpleNamespace(st_mode=actual.st_mode,st_size=limit+1) if path==root/name else actual
            with self.subTest(name=name), patch.object(Path,"stat",enlarged), patch.object(r.synth,"sha256",side_effect=AssertionError("Must reject before hashing")), patch.object(r,"bounded_json",side_effect=AssertionError("Must reject before JSON read")), patch.object(r,"verify_fixed_payloads",side_effect=AssertionError("Must reject before payload read")):
                with self.assertRaisesRegex(r.ReleaseError,"fixed/bounded"):
                    r.verify_release(root)

    def test_default_cli_failure_never_materializes_or_echoes_private_exception(self):
        output=io.StringIO()
        arguments=["--work-root",str(self.work)]
        for key in ("official-repo","official-assets","asset-manifest","merged-bundle","model-card","public-asr-summary","output"):
            arguments.extend(["--"+key,str(self.work/key)])
        with patch.object(r,"inspect",side_effect=RuntimeError("PRIVATE_SENTINEL signed-url credential")), patch.object(r,"materialize",side_effect=AssertionError("Default must never write")) as mutate, contextlib.redirect_stdout(output):
            self.assertEqual(r.main(arguments),2)
        self.assertFalse(mutate.called)
        self.assertNotIn("PRIVATE_SENTINEL",output.getvalue())
        self.assertFalse(json.loads(output.getvalue())["uploaded"])

    def test_import_and_help_never_load_torch_whisper_or_hub(self):
        code="import runpy,sys; sys.argv=['prepare_hf_release.py','--help'];\ntry: runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit as e: assert e.code==0\nassert not {'torch','whisper','huggingface_hub'} & set(sys.modules)"
        # Explicit file path; no manifest, model bytes, credentials or API access.
        code=code.replace("sys.argv=['prepare_hf_release.py'","sys.argv=["+repr(str(ROOT/'scripts/prepare_hf_release.py')))
        result=subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("--prepare",result.stdout)
        self.assertIn("--verify-release",result.stdout)


if __name__ == "__main__":unittest.main()
