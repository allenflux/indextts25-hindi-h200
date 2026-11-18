"""CPU-only 300-plan/terminal-schema contracts, not GPU training evidence.

Checkpoint fixtures contain real finite FP32 tensors and own SHA sidecars; their
metadata is explicitly synthetic. No model, nvidia-smi, HTTP or child is launched.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('tested_new_pure_training_guard',ROOT/'scripts/run_pure_lora_guarded.py')
g=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(g)
AVAILABLE=importlib.util.find_spec('torch') is not None
if AVAILABLE:
    import torch
    from hi_h200_native.engine import atomic_torch_save, _runtime, rng_state


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.runtime');self.work=Path(self.tmp.name).resolve()
        self.code=self.work/'code'
        for name in ('model.py','data.py','engine.py','train.py'):
            path=self.code/'src/hi_h200_native'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# CPU source identity fixture\n')
        self.root_patch=patch.object(g,'ROOT',self.code);self.root_patch.start()
        self.cfgdir=self.work/'configs';self.cfgdir.mkdir()
        self.model=self.work/'base';self.model.mkdir()
        self.manifest=self.model/'manifest.json';self.manifest.write_text(json.dumps({'repository':'IndexTeam/IndexTTS-2.5',
            'snapshot_revision':'c39ce5ba981572cb187443877ff559dfb246ce63',
            'files':{'gpt.pth':'43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85'}}))
        for split,count in (('train',48),('val',8)):
            folder=self.work/split;folder.mkdir()
            for name in ('audio','prepared','features','official'):(folder/name).mkdir()
            feature_manifest=folder/'features/manifest.jsonl';feature_manifest.write_text('{}\n'*count)
            provenance=folder/'provenance.json';provenance.write_text('{}')
            data={'schema_version':1,'split':split,'official_repo':str(self.work/'official'),
                'model_dir':str(self.model),'model_manifest':str(self.manifest),'audio_root':str(folder/'audio'),
                'prepared_dir':str(folder/'prepared'),'feature_root':str(folder/'features'),
                'feature_manifest':str(feature_manifest),'provenance':str(provenance)}
            (self.cfgdir/(split+'.json')).write_text(json.dumps(data))
        (self.work/'official').mkdir()
        self.options=json.loads((ROOT/'configs/native-lora-pilot48-300.example.json').read_text())
        self.options.update(train_dataset=str(self.cfgdir/'train.json'),val_dataset=str(self.cfgdir/'val.json'),output_dir=str(self.work/'run300'))
        self.config=self.cfgdir/'new300.json';self.write()
        self.args=SimpleNamespace(plan=g.PLAN300,stage='fresh2',workspace_root=self.work,config=self.config,config_sha256=g.digest(self.config))
        self.network=patch('socket.socket',side_effect=AssertionError('No network in CPU tests'));self.network.start()
        self.spawn=patch.object(g.subprocess,'Popen',side_effect=AssertionError('No trainer process in CPU tests'));self.spawn.start()

    def tearDown(self):self.spawn.stop();self.network.stop();self.root_patch.stop();self.tmp.cleanup()
    def write(self):self.config.write_text(json.dumps(self.options))
    def plan(self):return g.plan300(self.args,self.config,self.options)

    def test_explicit300_freezes_root_configuration_fold_and_six_validation_points(self):
        plan=self.plan()
        self.assertEqual(plan['record']['validation_steps'],[50,100,150,200,250,300])
        self.assertFalse(plan['record']['test_used_for_selection'])
        self.assertEqual(plan['record']['selection'],'minimum_mean_val8_then_earliest_step')
        self.assertFalse(plan['path'].exists())
        g.verify_plan_state(plan,'fresh2');g.write_new(plan['path'],plan['record'])
        g.verify_plan_state(plan,'resume300')
        with self.assertRaises(ValueError):g.verify_plan_state(plan,'fresh2')
        (self.work/'train/provenance.json').write_text('{"changed":true}')
        changed=self.plan()
        with self.assertRaisesRegex(ValueError,'changed'):g.verify_plan_state(changed,'resume300')

    def test_old100_is_not_widened_and_stage_profiles_cannot_mix(self):
        old=copy.deepcopy(self.options);old['loop']['max_updates']=100
        old_args=SimpleNamespace(plan='legacy100',stage='fresh2')
        self.assertIsNone(g.plan300(old_args,self.config,old))
        with self.assertRaisesRegex(ValueError,'100'):g.plan300(old_args,self.config,self.options)
        old_args.stage='resume300'
        with self.assertRaises(ValueError):g.plan300(old_args,self.config,old)
        self.args.stage='resume100'
        with self.assertRaisesRegex(ValueError,'resume300'):self.plan()

    def test_wrong_sha_schedule_root_or_fold_reject_before_any_child_or_gpu(self):
        correct=self.args.config_sha256;self.args.config_sha256='0'*64
        with self.assertRaisesRegex(ValueError,'SHA'):self.plan()
        self.args.config_sha256=correct
        for field,value in (('max_updates',301),('grad_accumulation',1),('validation_interval',25)):
            saved=self.options['loop'][field];self.options['loop'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'fixed'):self.plan()
            self.options['loop'][field]=saved
        self.options['output_dir']=str(self.code/'run')
        with self.assertRaisesRegex(ValueError,'overlaps'):self.plan()
        self.options['output_dir']=str(self.work/'run300')
        (self.work/'val/features/manifest.jsonl').write_text('{}\n'*7)
        with self.assertRaisesRegex(ValueError,'TRAIN48/VAL8'):self.plan()

    def test_resume_rejects_replacement_step2_even_if_current_tensor_audit_passes(self):
        plan=self.plan();g.write_new(plan['path'],plan['record'])
        run=Path(self.options['output_dir']);(run/'checkpoints').mkdir(parents=True)
        checkpoint=run/'checkpoints/step00000002.pt'
        checkpoint.write_bytes(b'CPU protocol fixture; no real training checkpoint')
        checkpoint.with_suffix('.pt.sha256.json').write_text('{}')
        original={'step':2,'checkpoint_sha256':'a'*64,'adapter_sha256':'b'*64,
                  'restricted_cpu_tensor_audit':True,'pure_ab_tensors':96,'training_dataset_examples':48,'validation_dataset_examples':8}
        g.write_new(self.cfgdir/('controller-'+g.PLAN300+'-fresh2.json'),
            {'verified_step':2,'trainer_exit_code':0,'own_trainer_exited':True,'stop_reason':None,'terminal_tensor_audit':original})
        argv=['--config',str(self.config),'--stage','resume300','--plan',g.PLAN300,
              '--workspace-root',str(self.work),'--config-sha256',self.args.config_sha256]
        for changed in ('checkpoint_sha256','adapter_sha256'):
            replacement=dict(original);replacement[changed]='c'*64
            with self.subTest(changed=changed), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','HF_HUB_OFFLINE':'1',
                    'TRANSFORMERS_OFFLINE':'1','TORCH_FORCE_WEIGHTS_ONLY_LOAD':'1'}), \
                    patch.object(g,'queue_idle',return_value=True),patch.object(g,'resource_admission',return_value=[{'uuid':'CPU-FIXTURE'}]), \
                    patch.object(g,'verify_native_terminal',return_value=replacement) as audit,patch.object(g.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'successful fresh2 audit'):g.main(argv)
                audit.assert_called_once();spawn.assert_not_called()
        self.assertFalse((self.cfgdir/('controller-'+g.PLAN300+'-resume300.json')).exists())

    @unittest.skipUnless(AVAILABLE,'Real Torch CPU tensor serialization required')
    def test_restricted_actual_terminal_files_bind96_ab_checkpoint_and_adapter(self):
        torch.set_num_threads(1)
        plan=self.plan();record=plan['record'];spec=record['adapter_spec'];step=2
        values={}
        for target in spec['lora_targets']:
            values['base.'+target+'.lora_A']=torch.zeros(8,1280)
            values['base.'+target+'.lora_B']=torch.zeros(3840 if target.endswith('.c_attn') else 1280,8)
        identity={'config_sha256':record['config_sha256'],'resolved_config':self.options,
            'native_code_sha256':record['native_code_sha256'],'adapter_spec':spec,
            'datasets':{split:{'split':split,'feature_files':{str(i):'a'*64 for i in range(count)}} for split,count in (('train',48),('val',8))}}
        base={'model_manifest_sha256':record['base_manifest_sha256']}
        optimizer={'state':{i:{'step':torch.tensor(float(step)),'exp_avg':torch.zeros_like(value),'exp_avg_sq':torch.zeros_like(value)} for i,value in enumerate(values.values())},
                   'param_groups':[{'params':list(range(96))},{'params':[]}]}
        state={'schema_version':1,'kind':'hi_h200_native_training','reason':'explicit_CPU_contract_fixture_not_training',
            'config':self.options['loop'],'identity':identity,'runtime':_runtime(),'step':step,
            'sampler':{'dataset_length':48,'micro_steps':8,'epoch':0,'offset':8},'parameter_groups':[list(values),[]],
            'trainable':values,'optimizer':optimizer,'scheduler':{'last_epoch':step},'scaler':None,'rng':rng_state(),
            'model_identity':base,'adapter_spec':spec}
        payload={'schema_version':1,'kind':'hi_h200_native_adapter','step':step,'base_identity':base,'adapter_spec':spec,'trainable':values}
        run=Path(self.options['output_dir']);run.mkdir()
        checkpoint=run/'checkpoints/step00000002.pt';adapter=run/'adapters/step00000002.pt'
        atomic_torch_save(checkpoint,state);atomic_torch_save(adapter,payload)
        (run.parent/(run.name+'.native-run.json')).write_text(json.dumps({'schema_version':1,'kind':'hi_h200_native_run','identity':identity,'limitations':['CPU contract fixture, not training']}))
        report=g.verify_native_terminal(run,2,self.options,plan)
        self.assertEqual(report['pure_ab_tensors'],96);self.assertTrue(report['restricted_cpu_tensor_audit'])
        self.assertFalse(torch.cuda.is_initialized())
        for mode in ('partial','step','identity','adapter_changed','nan','optimizer_partial'):
            broken=copy.deepcopy(state);other=copy.deepcopy(payload)
            if mode=='partial':broken['trainable'].pop(next(iter(values)))
            elif mode=='step':broken['step']=3
            elif mode=='identity':broken['identity']['config_sha256']='0'*64
            elif mode=='adapter_changed':next(iter(other['trainable'].values())).flatten()[0]=1
            elif mode=='nan':next(iter(broken['trainable'].values())).flatten()[0]=float('nan')
            else:broken['optimizer']['state'].pop(0)
            atomic_torch_save(checkpoint,broken);atomic_torch_save(adapter,other)
            with self.subTest(mode=mode),self.assertRaises(ValueError):g.verify_native_terminal(run,2,self.options,plan)


if __name__=='__main__':unittest.main()
