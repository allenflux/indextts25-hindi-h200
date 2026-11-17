"""CPU-only batch metadata/control tests; no fake audio generation success."""
import array
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from hi_h200_native import batch_synth as batch
from hi_h200_native import synth
ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO",ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(OFFICIAL.is_dir(),"Pinned local source required; no downloaded assets")
class BatchMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root = Path(self.tmp.name)
        self.prompt = self.wav("reference.wav",1)
        self.case = {"id":"fixed-hi","language":"hi","text":"क़िला और शक्ति की भाषा है।",
                     "prompt":"reference.wav","max_mel_tokens":256}

    def tearDown(self):
        self.tmp.cleanup()

    def wav(self,name,seconds,value=1000):
        path=self.root / name
        with wave.open(str(path),"wb") as stream:
            stream.setparams((1,2,16000,0,"NONE","not compressed"))
            stream.writeframes(array.array("h",[value]*int(16000*seconds)).tobytes())
        return path

    def manifest(self,rows):
        path=self.root / "private-cases.json"
        path.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
        return path

    def test_manifest_hash_fixed_seed_settings_and_canonical_marks(self):
        path=self.manifest([self.case])
        rows,digest=batch.load_cases(path,self.root,OFFICIAL)
        self.assertEqual(digest,synth.sha256(path))
        self.assertEqual(rows[0]["text"],self.case["text"])
        self.assertEqual(rows[0]["text_sha256"],batch.text_hash(self.case["text"]))
        self.assertEqual(rows[0]["seed"],1234)
        self.assertEqual(rows[0]["settings"],{"do_sample":False,"num_beams":1,"repetition_penalty":1.,"max_mel_tokens":256})
        self.assertFalse(rows[0]["generation_attempted"])
        for language,text in (("en","This is a test."),("zh","这是测试"),("ja","これはテストです"),("es","Hola mundo"),("ar","هذه تجربة")):
            self.assertEqual(batch.canonical_text(OFFICIAL,language,text),text)
        for bad in ("हिन्दी: भाषा","हिन्दी १२","हिन्दी  भाषा","हिन्दी <आवाज़|x>","हिन्दी\u202e भाषा"):
            with self.assertRaises(ValueError):
                batch.canonical_text(OFFICIAL,"hi",bad)

    def test_manifest_rejects_unsafe_cases_and_freeze_mismatch(self):
        for update in ({"id":"../escape"},{"id":"runtime"},{"language":"de"},
                       {"text_sha256":"0"*64},{"max_mel_tokens":True},{"prompt":"../outside.wav"}):
            path=self.manifest([{**self.case,**update}])
            with self.assertRaises((ValueError,OSError)):
                batch.load_cases(path,self.root,OFFICIAL)
        for rows in ([],[self.case]*2,[{**self.case,"id":"case"+str(i)} for i in range(17)]):
            with self.assertRaises(ValueError):
                batch.load_cases(self.manifest(rows),self.root,OFFICIAL)

    def test_all_case_ledger_keeps_safe_failure_fatal_and_unexecuted(self):
        rows=[dict(self.case,id="case"+str(i),status="not_executed") for i in range(4)]
        saves=[]
        def save():
            saves.append(copy.deepcopy(rows))
            batch.save_ledger(self.root,{"cases":rows})
        attempts=[]
        def failures_only(row):
            attempts.append(row["id"])
            if len(attempts)==1:
                raise synth.SynthesisError("max code / invalid PCM unit failure")
            raise batch.FatalBatchError("resource unit failure")
        batch.run_cases(rows,failures_only,save)
        self.assertEqual(attempts,["case0","case1"])
        self.assertEqual([row["status"] for row in rows],["failed","failed","not_executed","not_executed"])
        self.assertEqual(saves[0][0]["status"],"attempt_started")
        self.assertEqual([row["id"] for row in json.loads((self.root / "batch-results.json").read_text())["cases"]],["case0","case1","case2","case3"])
        self.assertFalse(any(row["status"]=="generated" for row in rows))

    def test_resource_fixture_ignores_own_utilization_but_checks_uuid_ram_memory(self):
        args=SimpleNamespace(min_free_gpu_mib=8000.,min_free_ram_mib=1024.)
        with patch.object(batch.subprocess,"run",return_value=SimpleNamespace(stdout="0,GPU-fixture,H200,16000,1000,100\n")),patch.object(synth,"available_ram_bytes",return_value=4*1024**3):
            result=batch.case_resources(args,"GPU-fixture")
            self.assertFalse(result["utilization_gate_applied"])
            with self.assertRaises(batch.FatalBatchError):
                batch.case_resources(args,"GPU-another")
        with patch.object(batch.subprocess,"run",side_effect=FileNotFoundError("unit fixture")):
            with self.assertRaises(batch.FatalBatchError):
                batch.case_resources(args,"GPU-fixture")

    def test_pcm_bounds_are_checked_without_truncation(self):
        for seconds in (1,20):
            path=self.wav("ok"+str(seconds)+".wav",seconds)
            before=path.read_bytes()
            self.assertEqual(batch.output_audio(path)["seconds"],seconds)
            self.assertEqual(path.read_bytes(),before)
        for seconds,value in ((.5,1000),(20.01,1000),(1,0)):
            with self.assertRaises(ValueError):
                batch.output_audio(self.wav("bad.wav",seconds,value))

    def test_default_preflight_metadata_fixture_imports_no_torch(self):
        # Only asset inventory fixtures are injected; never a model or synthesis.
        original,merged=self.root / "original",self.root / "merged"
        original.mkdir();merged.mkdir()
        (original / "model-manifest.json").write_text("{}")
        authored={**self.case,"prompt":str(self.prompt.relative_to(ROOT.parent))}
        manifest=self.manifest([authored])
        output=ROOT.parent / ("batch-preflight-"+self.root.name)
        args=["--workspace-root",str(ROOT.parent),"--official-repo",str(OFFICIAL),"--model-dir",str(original),
              "--model-manifest",str(original / "model-manifest.json"),"--merged-bundle",str(merged),
              "--cases",str(manifest.relative_to(ROOT)),"--output-dir",str(output),"--variant","original"]
        script="""import sys
from hi_h200_native import batch_synth as b
b.synth.verify_synth_assets=lambda *a: ({}, {'files':{'gpt.pth':'a'*64}})
b.synth.verify_merged_bundle=lambda *a: {'gpt_sha256':'b'*64}
b.synth.dependency_report=lambda *a: {'missing':[]}
original_preflight=b.preflight
def checked_preflight(args):
    import os
    plan,report=original_preflight(args)
    assert plan['cases_path'].is_absolute()
    before=os.getcwd()
    try:
        os.chdir(plan['original'])
        assert b.synth.sha256(plan['cases_path'])==report['cases_manifest_sha256']
    finally:
        os.chdir(before)
    return plan,report
b.preflight=checked_preflight
assert b.main(ARGS)==0
assert 'torch' not in sys.modules
assert 'indextts.infer_v2_5' not in sys.modules
""".replace("ARGS",repr(args))
        result=subprocess.run([sys.executable,"-c",script],cwd=ROOT,text=True,capture_output=True,timeout=30,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(ROOT / "src")))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)["models_loaded"])
        self.assertFalse(output.exists())

    def test_actual_source_api_contract_and_one_constructor_call(self):
        tree=ast.parse((OFFICIAL / "indextts/infer_v2_5.py").read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="IndexTTS2")
        infer=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="infer")
        self.assertTrue({"spk_audio_prompt","text","output_path","lang","text_normalization","stream_return"} <= {arg.arg for arg in infer.args.args})
        own=ast.parse(Path(batch.__file__).read_text())
        self.assertEqual(sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="constructor" for node in ast.walk(own)),1)
        self.assertIn("Chinese,\nEnglish, Japanese, Spanish and Arabic",(OFFICIAL / "README.md").read_text())


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(),"Actual tiny official Torch model required")
class ActualBatchGPTContractTests(unittest.TestCase):
    def test_real_official_gpt_input_audit_delegates_generate_and_rejects_mismatch(self):
        import torch
        from hi_h200_native.model import make_tiny_official_model,import_official_model
        torch.set_num_threads(1)
        native=make_tiny_official_model(OFFICIAL,seed=12)
        from indextts.utils.tokenizer import lang_to_token
        stock=import_official_model(OFFICIAL)(**native.native_config["gpt"]).eval()
        del native
        stock.post_init_gpt2_config(use_deepspeed=False,kv_cache=False,half=False)
        tts=SimpleNamespace(gpt=stock)
        tokens=[3,4,5]
        def actual_call():
            return stock.inference_speech(torch.zeros(1,4,1024),torch.tensor([tokens+[1]]),torch.tensor([17]),
                emo_vec=torch.zeros(1,16),campplus_embedding=torch.zeros(1,192),max_generate_length=3,
                min_new_tokens=3,do_sample=False,num_beams=1)
        with torch.inference_mode(),batch.audit_gpt_inputs(tts,OFFICIAL,tokens,17) as observed:
            codes,_=actual_call()
        self.assertEqual(observed["calls"],1)
        self.assertEqual(tuple(codes.shape),(1,3))
        with batch.audit_gpt_inputs(tts,OFFICIAL,[3,4,6],17):
            with self.assertRaises(ValueError):
                actual_call()
        self.assertEqual({lang:lang_to_token(lang) for lang in batch.LANGUAGES},batch.LANGUAGES)
        batch.reset_seed(cuda=False)
        first=(random.random(),torch.rand(3))
        batch.reset_seed(cuda=False)
        second=(random.random(),torch.rand(3))
        self.assertEqual(first[0],second[0])
        self.assertTrue(torch.equal(first[1],second[1]))


if __name__=="__main__":
    unittest.main()
