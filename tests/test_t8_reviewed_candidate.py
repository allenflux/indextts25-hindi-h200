"""Real tiny Git/files/asset hashes for explicit complete-GPT registration; no model/GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reviewed_t8_test_entry',ROOT/'scripts/check_t8_compat.py')
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)
STORE=b'''from pathlib import Path
import hashlib,json
PLUGIN_ROOT=Path(__file__).resolve().parents[1]
MANIFEST_PATH=PLUGIN_ROOT / "manifests" / "model_2_5.json"
class ValidationReport:
    def __init__(self,model_dir,missing=(),mismatched=(),hashes_verified=False):
        self.model_dir,self.missing,self.mismatched,self.hashes_verified=model_dir,tuple(missing),tuple(mismatched),hashes_verified
    @property
    def valid(self): return not self.missing and not self.mismatched
    def require_valid(self):
        if not self.valid:
            raise FileNotFoundError("original fixture hint")
def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())
def validate_model_dir(model_dir,*,verify_hashes=False,include_auxiliary=True):
    manifest=load_manifest()
    model_dir=Path(model_dir)
    missing,mismatched=[],[]
    for name,meta in manifest["files"].items():
        if not include_auxiliary and meta.get("group")=="auxiliary": continue
        path=model_dir/name
        if not path.is_file(): missing.append(name); continue
        data=path.read_bytes()
        if len(data)!=meta["size"] or verify_hashes and hashlib.sha256(data).hexdigest()!=meta["sha256"]: mismatched.append(name)
    return ValidationReport(model_dir,missing,mismatched,verify_hashes)
def model_fingerprint(model_dir):
    manifest=load_manifest()
    return manifest["modelRevision"]
'''
NODE=b'''class T8IndexTTS25ModelLoader:
    @classmethod
    def validate_inputs(cls,download_missing=False):
        sentinel("no requested downloads")
        return True
    @classmethod
    def execute(cls,model_dir,download_missing=False):
        manifest=load_manifest()
        return manifest["modelRevision"]
class T8IndexTTS25Generate:
    @classmethod
    def define_schema(cls):
        return schema(node_id="T8_IndexTTS25_Generate",inputs=[combo("language",options=["ZH","EN","JA","ES","AR"])])
'''
RUNTIME=b'''def run_inference(handle,speaker_audio,text,language,duration_factor,seed):
    if language.upper() not in {"ZH","EN","JA","ES","AR"}:
        raise ValueError("unsupported")
    return handle
'''
PLANNER=b'''LANGUAGE_AUTO_LIMITS={"ZH":120,"EN":60,"JA":100,"ES":60,"AR":80}
'''
DOWNLOADER='''def ensure_model_bundle(*args,**kwargs):
    sentinel("bundle writes/network")
def download_main_model(*args,**kwargs):
    sentinel("main download")
def download_auxiliary_models(*args,**kwargs):
    sentinel("auxiliary download")
def _download_hf_file_cancellable(*args,**kwargs):
    sentinel("HF download")
def build_parser():
    return parser(description="下载并校验 IndexTTS 2.5 正式模型")
def main():
    print(">> IndexTTS 2.5 正式模型 SHA-256 校验通过。")
    print(">> 正式模型 SHA-256 校验通过。")
'''.encode()


class ReviewedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='indextts25-hindi-reviewed-',dir=ROOT/'.runtime');self.work=Path(self.tmp.name)
        self.source=self.work/'source';self.source.mkdir();self.dest=self.work/'copy'
        self.base_bytes=b'base-gpt!';self.new_bytes=b'candidate';self.old_bytes=b'oldweight'
        self.assertEqual(len(self.base_bytes),len(self.new_bytes));self.assertEqual(len(self.old_bytes),len(self.new_bytes))
        self.assets={'gpt.pth':self.base_bytes,**{f'asset-{i:02}.bin':bytes([i]) for i in range(25)}}
        self.base={'modelRepository':'fixture/base','modelRevision':'a'*40,'files':{name:{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in self.assets.items()}}
        self.base['files']['asset-24.bin']['group']='auxiliary'
        self.contents={'nodes_v3.py':NODE,'runtime/inference_adapter.py':RUNTIME,'runtime/text_planner.py':PLANNER,
            'services/model_store.py':STORE,p.BASE_MANIFEST:json.dumps(self.base).encode(),'LICENSE':b'authored license fixture','LICENSE_ZH.txt':b'authored license interface'}
        for name,data in {**self.contents,p.DOWNLOADER_PATH:DOWNLOADER}.items():
            path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        self.git('init','-q');self.git('add','.');self.git('-c','user.name=CPU Fixture','-c','user.email=fixture@example.invalid','commit','-qm','authored tiny source')
        node_revision=self.git('rev-parse','HEAD').strip()
        self.constants=patch.multiple(p,PINNED_BLOBS={name:p._blob(data) for name,data in self.contents.items()},
            NODE_REVISION=node_revision,DOWNLOADER_BLOB=p._blob(DOWNLOADER),
            ORIGINAL_GPT_SHA256=hashlib.sha256(self.base_bytes).hexdigest(),ORIGINAL_GPT_BYTES=len(self.base_bytes))
        self.constants.start()
        self.reg={'format':p.REVIEWED_FORMAT,'candidate_id':'pure-test','repository':None,'revision':None,
            'training_profile':'attention_lora_only_v1','adapter_sha256':'1'*64,'gpt_sha256':hashlib.sha256(self.new_bytes).hexdigest(),
            'gpt_bytes':len(self.new_bytes),'export_complete_sha256':'2'*64,'export_provenance_sha256':'3'*64,'quality_gate_passed':False}
        self.registration=self.work/'candidate.json';self.raw=p._encoded(self.reg);self.registration.write_bytes(self.raw)
        self.digest=hashlib.sha256(self.raw).hexdigest()
        self.kw={'candidate_manifest_path':self.registration,'candidate_manifest_sha256':self.digest}
        self.base_dir=self.work/'base';self.candidate_dir=self.work/'hindi'
        for folder,gpt in ((self.base_dir,self.base_bytes),(self.candidate_dir,self.new_bytes)):
            folder.mkdir()
            for name,data in self.assets.items(): (folder/name).write_bytes(gpt if name=='gpt.pth' else data)

    def tearDown(self):self.constants.stop();self.tmp.cleanup()
    def git(self,*args):
        return subprocess.run(['git','-C',str(self.source),*args],check=True,text=True,capture_output=True).stdout
    def prepare(self):return p.prepare(self.source,self.dest,None,**self.kw)
    def bind(self):return p.bind_candidate(self.dest,self.candidate_dir,self.registration,self.digest,self.work)
    def runtime_store(self):
        values={'__file__':str(self.dest/'services/model_store.py')};exec((self.dest/'services/model_store.py').read_bytes(),values);return values

    def test_registration_hash_schema_profile_and_base_lock_fail_before_copy(self):
        with self.assertRaisesRegex(p.CompatibilityError,'SHA256 mismatch'):
            p.prepare(self.source,self.dest,None,**dict(self.kw,candidate_manifest_sha256='0'*64))
        self.assertFalse(self.dest.exists())
        for name,value in (('quality_gate_passed',0),('training_profile','legacy_rows_heads_v1'),('repository','not-paired'),('gpt_bytes',True)):
            reg=dict(self.reg);reg[name]=value;raw=p._encoded(reg);self.registration.write_bytes(raw)
            with self.subTest(name=name),self.assertRaises(p.CompatibilityError):p.read_registration(self.registration,hashlib.sha256(raw).hexdigest())
        self.registration.write_bytes(self.raw)
        bad=copy.deepcopy(self.base);bad['files']['gpt.pth']['sha256']='0'*64
        with self.assertRaisesRegex(p.CompatibilityError,'base identity'):p.reviewed_candidate_manifest(bad,self.reg)

    def test_real_git_prepare_preserves_base_licenses_and_has_truthful_local_identity(self):
        result=self.prepare();self.assertEqual(result['format'],'t8-reviewed-complete-gpt-code-v1')
        self.assertIsNone(result['candidate_repository']);self.assertIsNone(result['candidate_revision'])
        self.assertFalse(result['model_assets_verified']);self.assertFalse(result['tensor_audit_verified'])
        self.assertEqual(result['manifest_data']['modelRepository'],'local-reviewed/pure-test')
        self.assertEqual(result['manifest_data']['modelRevision'],'sha256:'+self.reg['gpt_sha256'])
        for name in (p.BASE_MANIFEST,'LICENSE','LICENSE_ZH.txt'):
            self.assertEqual((self.dest/name).read_bytes(),self.contents[name])
        for name,meta in result['manifest_data']['files'].items():
            if name!='gpt.pth':
                self.assertEqual(meta['sha256'],self.base['files'][name]['sha256'])
                self.assertEqual(meta['sourceRepository'],self.base['modelRepository'])
        with self.assertRaises(p.CompatibilityError):p.verify_prepared(self.dest)
        self.assertEqual(p.verify_prepared(self.dest,None,**self.kw)['candidate_manifest_sha256'],self.digest)

    def test_same_size_base_new_old_never_select_or_authorize_by_size(self):
        self.prepare();self.bind()
        base=p.selected_manifest(self.dest,self.base_dir,None,variant='base',**self.kw)
        candidate=p.selected_manifest(self.dest,self.candidate_dir,None,variant='hindi',**self.kw)
        self.assertEqual(base['variant'],'base');self.assertEqual(candidate['variant'],'hindi')
        self.assertNotIn('size',candidate['selector'])
        self.assertEqual(p.verify_assets(self.base_dir,base['manifest_data'])['files'],26)
        self.assertEqual(p.verify_assets(self.candidate_dir,candidate['manifest_data'])['files'],26)
        (self.candidate_dir/'gpt.pth').write_bytes(self.old_bytes)
        with self.assertRaisesRegex(p.CompatibilityError,'SHA256'):p.verify_assets(self.candidate_dir,candidate['manifest_data'])
        with self.assertRaisesRegex(p.CompatibilityError,'requested variant'):
            p.selected_manifest(self.dest,self.base_dir,None,variant='hindi',**self.kw)

    def test_runtime_marker_selects_pinned_manifest_and_forces_all_asset_hashes(self):
        self.prepare();store=self.runtime_store()
        self.assertFalse(store['validate_model_dir'](self.candidate_dir).valid) # No selector => base; same size still fails.
        self.bind()
        report=store['validate_model_dir'](self.candidate_dir,verify_hashes=False,include_auxiliary=False)
        self.assertTrue(report.valid);self.assertTrue(report.hashes_verified)
        self.assertEqual(store['model_fingerprint'](self.candidate_dir),'sha256:'+self.reg['gpt_sha256'])
        (self.candidate_dir/'asset-24.bin').unlink()
        report=store['validate_model_dir'](self.candidate_dir,verify_hashes=False,include_auxiliary=False)
        self.assertIn('asset-24.bin',report.missing)
        (self.candidate_dir/p.REVIEWED_MARKER).write_bytes(self.raw+b' ')
        with self.assertRaisesRegex(ValueError,'selector'):store['load_manifest'](self.candidate_dir)

    def test_marker_bind_is_fresh_and_only_after_complete_hash_validation(self):
        self.prepare();(self.candidate_dir/'asset-00.bin').write_bytes(b'bad')
        with self.assertRaises(p.CompatibilityError):self.bind()
        self.assertFalse((self.candidate_dir/p.REVIEWED_MARKER).exists())
        (self.candidate_dir/'asset-00.bin').write_bytes(self.assets['asset-00.bin']);self.bind()
        with self.assertRaises(FileExistsError):self.bind()
        self.assertEqual((self.candidate_dir/p.REVIEWED_MARKER).read_bytes(),self.raw)
        with self.assertRaisesRegex(p.CompatibilityError,'dedicated'):
            p.bind_candidate(self.dest,self.work,self.registration,self.digest,self.work)

    def test_bind_rejects_filesystem_broad_or_symlink_work_before_asset_reads(self):
        self.prepare()
        link=self.work.parent/(self.work.name+'-link');link.symlink_to(self.work,target_is_directory=True)
        try:
            for workspace in (Path(self.work.anchor),self.work.parent,link):
                with self.subTest(workspace=workspace),patch.object(p,'verify_assets',side_effect=AssertionError('must not read assets')), \
                        self.assertRaisesRegex(p.CompatibilityError,'dedicated canonical'):
                    p.bind_candidate(self.dest,self.candidate_dir,self.registration,self.digest,workspace)
                self.assertFalse((self.candidate_dir/p.REVIEWED_MARKER).exists())
        finally:
            link.unlink()

    def test_source_registration_manifest_and_audit_drift_fail_closed(self):
        self.prepare()
        paths=[self.dest/'services/model_store.py',self.dest/p.REVIEWED_MANIFEST,self.dest/p.REVIEWED_REGISTRATION,
               self.dest/'.reviewed-source'/'runtime/text_planner.py']
        for path in paths:
            before=path.read_bytes();path.write_bytes(before+b' ')
            with self.subTest(path=path),self.assertRaises(p.CompatibilityError):p.verify_prepared(self.dest,None,**self.kw)
            path.write_bytes(before)
        audit=json.loads((self.dest/p.AUDIT).read_text());audit['production_modified']=0
        (self.dest/p.AUDIT).write_bytes(p._encoded(audit))
        with self.assertRaisesRegex(p.CompatibilityError,'audit differs'):p.verify_prepared(self.dest,None,**self.kw)

    def test_downloads_are_rejected_before_any_sentinel_and_legacy_revision_cannot_mix(self):
        self.prepare();store=self.runtime_store();calls=[]
        def sentinel(*args):calls.append(True);raise AssertionError('side effect')
        ns={'sentinel':sentinel,'load_manifest':store['load_manifest']};exec((self.dest/'nodes_v3.py').read_bytes(),ns)
        loader=ns['T8IndexTTS25ModelLoader']
        self.assertIn('禁止自动下载',loader.validate_inputs(download_missing=True))
        with self.assertRaises(ValueError):loader.execute(self.candidate_dir,download_missing=True)
        dl={'sentinel':sentinel};exec((self.dest/p.DOWNLOADER_PATH).read_bytes(),dl)
        for name in ('ensure_model_bundle','download_main_model','download_auxiliary_models','_download_hf_file_cancellable'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):dl[name]()
        self.assertEqual(calls,[])
        with self.assertRaisesRegex(p.CompatibilityError,'mutually exclusive'):p.verify_prepared(self.dest,'a'*40,**self.kw)


if __name__=='__main__':unittest.main()
