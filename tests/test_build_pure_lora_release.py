"""Actual tiny official CPU adapter -> public whitelist; no production artifact.

Fixture pins explicitly describe synthetic local metadata and random tiny GPT.
The public CLI has no fixture/pin override. No downloads, GPU or uploads occur.
"""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tested_pure_release', ROOT / 'scripts/build_pure_lora_release.py')
g = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = g; SPEC.loader.exec_module(g)
OFFICIAL = Path(os.environ.get('HI_H200_OFFICIAL_REPO', ROOT / '.runtime/external-official'))
AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None and OFFICIAL.is_dir()
if AVAILABLE:
    import torch
    import yaml
    from hi_h200_native.model import make_tiny_official_model, PURE_LORA_PROFILE
    from hi_h200_native.assets import Asset, W2V_REPO, CAMP_REPO, MODEL_REPO, MODEL_REVISION, LICENSES


def sha(data): return hashlib.sha256(data).hexdigest()


class DefaultTests(unittest.TestCase):
    def test_help_and_import_no_torch_or_cli_pin_override(self):
        code = "import runpy,sys;sys.argv=['builder','--help'];\ntry:runpy.run_path("+repr(str(ROOT/'scripts/build_pure_lora_release.py'))+",run_name='__main__')\nexcept SystemExit as e:assert e.code==0\nassert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True,timeout=20,
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--execute',result.stdout)
        self.assertNotIn('fixture',result.stdout.lower())
        self.assertEqual(g.PRODUCTION.adapter_sha256,'15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd')
        self.assertEqual(g.PRODUCTION.adapter_bytes,5931907)


@unittest.skipUnless(AVAILABLE,'Real Torch and pinned official tiny model required')
class ActualTinyReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = make_tiny_official_model(OFFICIAL,seed=411,training_profile=PURE_LORA_PROFILE)
        batch={'text_ids':torch.tensor([3,4,5]),'codes':torch.tensor([2,3,4,5]),
            'spk_emb':torch.linspace(-.1,.1,192),'emo_vec':torch.linspace(-.05,.05,16)}
        optimizer=torch.optim.AdamW(cls.model.optimizer_groups(0),lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);cls.model(batch)['loss'].backward();optimizer.step()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.runtime')
        self.work=Path(self.tmp.name).resolve()
        self.model_dir=self.work/'original-metadata';self.model_dir.mkdir()
        self.source=self.work/'source';self.source.mkdir()
        # Copy only exact public licenses; no source code, weights or directories.
        for name in ('LICENSE','LICENSE_ZH.txt'):
            (self.source/name).write_bytes((OFFICIAL/name).read_bytes())
        self.assets=[]
        original={item.relative_path:item for item in g.PRODUCTION.assets}
        for name,item in original.items():
            if name=='config.yaml': raw=yaml.safe_dump(self.model.native_config).encode()
            elif name==g.TOKENIZER: raw=b'fixture tokenizer metadata, not the official vocabulary\n'
            elif name=='LICENSE': raw=(OFFICIAL/'LICENSE').read_bytes()
            elif name.endswith('.json'): raw=b'{"fixture":"synthetic public metadata"}\n'
            else: raw=b'# Explicit synthetic metadata fixture, no pretrained weights\n'
            fixture=replace(item,size=len(raw),hash_kind='sha256',digest=sha(raw))
            self.assets.append(fixture)
            if name in g.METADATA:
                path=self.model_dir/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        hashes={item.relative_path:item.digest for item in self.assets}
        for repository,name in ((W2V_REPO,g.EXTRAS[1]),(CAMP_REPO,g.EXTRAS[2])):
            items=[item for item in self.assets if item.repository==repository]
            value={'repository':repository,'snapshot_revision':items[0].revision,'license':LICENSES[repository],
                'files':{item.relative_path:hashes[item.relative_path] for item in items}}
            raw=g.encode(value);(self.model_dir/name).write_bytes(raw);hashes[name]=sha(raw)
        plan={'schema_version':1,'stage':'extract','metadata_verified':True,'downloaded':False,
            'network_payload_bytes':sum(item.size for item in self.assets),'files':[asdict(item) for item in self.assets],
            'licenses':{},'permission_scope':'fixture only','endpoint':'https://huggingface.co','auth':'fixture only','budget_scope':'fixture only'}
        raw=g.encode(plan);(self.model_dir/'asset-plan.json').write_bytes(raw);hashes['asset-plan.json']=sha(raw)
        manifest={'repository':MODEL_REPO,'snapshot_revision':MODEL_REVISION,'files':hashes}
        self.manifest=self.model_dir/'model-manifest.json';self.manifest.write_bytes(g.encode(manifest))
        self.identity={'official_commit':g.synth.OFFICIAL_COMMIT,'model_repository':MODEL_REPO,'model_snapshot_revision':MODEL_REVISION,
            'model_manifest_sha256':g.synth.sha256(self.manifest),'model_files':hashes,'config_sha256':hashes['config.yaml'],'tokenizer_sha256':hashes[g.TOKENIZER]}
        self.payload={'schema_version':1,'kind':'hi_h200_native_adapter','step':2,'base_identity':self.identity,
            'adapter_spec':self.model.adapter_spec(),'trainable':self.model.trainable_state()}
        self.adapter=self.work/'adapter.pt';torch.save(self.payload,self.adapter)
        self.card=self.work/'reviewed-card.md'
        self.card.write_text('---\nlicense: other\nlanguage:\n  - hi\n---\n# Experimental native pure attention adapter\nQuality is unverified. No production deployment.\n')
        self.args=SimpleNamespace(workspace_root=self.work,official_repo=self.source,model_dir=self.model_dir,
            base_manifest=self.manifest,base_manifest_sha256=g.synth.sha256(self.manifest),adapter=self.adapter,
            adapter_sha256=g.synth.sha256(self.adapter),model_card=self.card,model_card_sha256=g.synth.sha256(self.card),
            output_dir=self.work/'public',zip_output=None)
        self.pins=g.Pins(self.args.adapter_sha256,self.adapter.stat().st_size,self.args.base_manifest_sha256,2,1,16,tuple(self.assets),
                         scope='tiny_cpu_fixture_not_pretrained')
        self.source_patch=patch.object(g.synth,'verify_source',return_value=self.source);self.source_patch.start()
        self.root_patch=patch.object(g,'ROOT',self.work/'tool-code');self.root_patch.start()

    def tearDown(self): self.root_patch.stop();self.source_patch.stop();self.tmp.cleanup()

    def prepared(self): return g.inspect(self.args,_pins=self.pins)

    def resave(self):
        torch.save(self.payload,self.adapter)
        self.args.adapter_sha256=g.synth.sha256(self.adapter)
        self.pins=replace(self.pins,adapter_sha256=self.args.adapter_sha256,adapter_bytes=self.adapter.stat().st_size)

    def test_actual_tiny_cpu_package_and_optional_zip_have_only_exact_whitelist(self):
        self.args.zip_output=self.work/'public.zip'
        plan=self.prepared()
        self.assertFalse(self.args.output_dir.exists())
        result=g.build(plan)
        self.assertTrue(result['adapter_tensor_validation']);self.assertFalse(result['uploaded']);self.assertFalse(result['quality_verified'])
        self.assertEqual((self.args.output_dir/'adapter.pt').read_bytes(),self.adapter.read_bytes())
        inventory=json.loads((self.args.output_dir/'manifest.json').read_text())['files']
        self.assertFalse(any(name.endswith(('.pth','.bin','.safetensors')) for name in inventory))
        self.assertNotIn('gpt.pth',inventory)
        self.assertEqual((self.args.output_dir/'base-metadata/model-manifest.json').read_bytes(),self.manifest.read_bytes())
        summary=json.loads((self.args.output_dir/'public-summary.json').read_text())
        self.assertTrue(summary['adapter_tensor_validation']);self.assertEqual(summary['trainable_tensors'],4)
        self.assertEqual(summary['package_scope'],'tiny_cpu_fixture_not_pretrained')
        self.assertFalse(summary['base_weights_opened_or_copied'])
        self.assertFalse(summary['gpu_validation_performed_by_builder'])
        self.assertNotIn(str(self.work),json.dumps(summary))
        with zipfile.ZipFile(self.args.zip_output) as archive:
            self.assertEqual(set(archive.namelist()),set(inventory)|{'manifest.json','SHA256SUMS','COMPLETE.json'})
            self.assertTrue(all(item.compress_type==zipfile.ZIP_STORED for item in archive.infolist()))
        (self.args.output_dir/'private-log.txt').write_text('must reject')
        with self.assertRaisesRegex(ValueError,'Unexpected'):g.verify_package(self.args.output_dir,set(inventory))

    def test_bad_sha_base_identity_composite_extra_tensor_and_nonfinite_rejected_before_output(self):
        self.args.adapter_sha256='0'*64
        with self.assertRaisesRegex(ValueError,'frozen'):self.prepared()
        self.args.adapter_sha256=self.pins.adapter_sha256
        original=self.payload
        import copy
        for mode in ('identity','composite','extra','missing','nonfinite','wrong_dtype','wrong_step'):
            self.payload=copy.deepcopy(original)
            if mode=='identity':self.payload['base_identity']['model_manifest_sha256']='0'*64
            elif mode=='composite':self.payload['adapter_spec']['heads']=['text_head','mel_head']
            elif mode=='extra':self.payload['trainable']['base.text_head.weight']=torch.ones(2,2)
            elif mode=='missing':self.payload['trainable'].pop(next(iter(self.payload['trainable'])))
            elif mode=='nonfinite':next(iter(self.payload['trainable'].values())).flatten()[0]=float('nan')
            elif mode=='wrong_dtype':name=next(iter(self.payload['trainable']));self.payload['trainable'][name]=self.payload['trainable'][name].bfloat16()
            else:self.payload['step']=1
            self.resave()
            with self.subTest(mode=mode),self.assertRaises(ValueError):g.build(self.prepared())
            self.assertFalse(self.args.output_dir.exists())

    def test_missing_license_and_path_overlap_or_symlink_are_rejected(self):
        (self.source/'LICENSE_ZH.txt').unlink()
        with self.assertRaises((ValueError,OSError)):self.prepared()
        (self.source/'LICENSE_ZH.txt').write_bytes((OFFICIAL/'LICENSE_ZH.txt').read_bytes())
        for output in (self.model_dir/'public',self.source/'public',self.args.adapter,self.work/'tool-code'/'public'):
            self.args.output_dir=output
            if output.parent.name=='tool-code':output.parent.mkdir()
            with self.subTest(output=output.name),self.assertRaises(ValueError):self.prepared()
        self.args.output_dir=self.work/'public'
        link=self.work/'linked-adapter.pt';link.symlink_to(self.adapter);self.args.adapter=link
        with self.assertRaisesRegex(ValueError,'canonical'):self.prepared()

    def test_sensitive_or_draft_cards_and_generated_metadata_are_not_packaged(self):
        for text in ('A private path /home/person/secret','A token hf_'+'A'*32,'模型卡草稿：待填'):
            self.card.write_text('---\nlicense: other\n---\n'+text)
            self.args.model_card_sha256=g.synth.sha256(self.card)
            with self.subTest(text=text[:10]),self.assertRaises(ValueError):self.prepared()
            self.assertFalse(self.args.output_dir.exists())
        self.card.write_text('---\nlicense: other\n---\nExperimental; quality unverified.\n')
        self.args.model_card_sha256=g.synth.sha256(self.card)
        (self.model_dir/'asset-plan.json').write_text('{"private":"/home/person/secret"}')
        with self.assertRaises(ValueError):self.prepared()


if __name__=='__main__':unittest.main()
