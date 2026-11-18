"""Pure supervisor CPU contracts; no actual Popen, GPU, network or evaluation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tested_pure_eval_guard', ROOT / 'scripts/run_pure_eval_guarded.py')
g = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(g)


class Child:
    pid = 12345
    def __init__(self, stubborn=False):
        self.code = None; self.terminated = 0; self.killed = 0; self.waits = []; self.stubborn = stubborn
    def poll(self): return self.code
    def terminate(self):
        self.terminated += 1
        if not self.stubborn: self.code = -15
    def kill(self): self.killed += 1; self.code = -9
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.code is None: raise subprocess.TimeoutExpired('only our CPU fixture',timeout)
        return self.code


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.runtime')
        self.work = Path(self.tmp.name).resolve()
        self.source = self.work / 'eval-source'; self.source.mkdir()
        self.root_patch = patch.object(g,'ROOT',self.source); self.root_patch.start()
        self.adapter = self.work / 'pure.pt'; self.adapter.write_bytes(b'no real tensor fixture')
        self.dataset = self.work / 'train.dataset.json'
        paths = {}
        for key in ('official_repo','model_dir','model_manifest','audio_root','prepared_dir','feature_root','feature_manifest','provenance'):
            path = self.work / key
            if key in ('model_manifest','feature_manifest','provenance'): path.write_text('{}')
            else: path.mkdir()
            paths[key] = str(path)
        self.dataset.write_text(json.dumps(paths))
        self.args = SimpleNamespace(module='pure_overfit',workspace_root=self.work,report_dir=self.work/'controller',
            protect_root=[],readonly_dependency_root=[],python_dependency_root=[],gpu_id='0',min_free_gpu_mib=48000.,min_free_ram_mib=64000.,max_utilization=5.,execute=False,
            module_args=['--','--workspace-root',str(self.work),'--dataset',str(self.dataset),'--adapter',str(self.adapter),
                         '--adapter-sha256','a'*64,'--output-dir',str(self.work/'module-result')])

    def tearDown(self): self.root_patch.stop(); self.tmp.cleanup()

    def run_fixture(self, child, *, exit_code=0, queue=True, stdout='0, GPU-fixture, H200, 140000, 500, 0', ram=100000*1024**2, sleep=None, monitor_error=None):
        prepared = g.plan(self.args)
        handlers = {}
        def signal_fixture(number, handler):
            old = handlers.get(number,signal.SIG_DFL); handlers[number]=handler; return old
        def finish(_): child.code=exit_code
        with patch.object(g.shared,'queue_idle',side_effect=queue if isinstance(queue,list) else None,return_value=queue), \
             patch.object(g.shared,'resource_admission',return_value=[{'uuid':'GPU-fixture'}]), \
             patch.object(g.subprocess,'Popen',return_value=child) as popen, \
             patch.object(g.subprocess,'run',side_effect=monitor_error,return_value=SimpleNamespace(stdout=stdout)), \
             patch.object(g.shared,'available_ram_bytes',return_value=ram), \
             patch.object(g.time,'sleep',side_effect=sleep or finish), \
             patch.object(g.signal,'signal',side_effect=signal_fixture):
            result=g.execute(self.args,prepared)
        return result,popen,handlers

    def test_default_plan_has_no_process_or_resource_activity_and_uses_only_fixed_module(self):
        with patch.object(g.subprocess,'Popen',side_effect=AssertionError('no process')), \
             patch.object(g.shared,'queue_idle',side_effect=AssertionError('no queue')), \
             patch.object(g.shared,'resource_admission',side_effect=AssertionError('no GPU')):
            result=g.plan(self.args)
        self.assertFalse(result['report_dir'].exists())
        self.assertFalse(result['module_output'].exists())
        self.assertEqual(result['command'][:4],[sys.executable,'-B','-m','hi_h200_native.pure_overfit'])
        self.assertIn('--execute',result['command']); self.assertIn('--allow-cuda',result['command'])
        self.assertEqual(result['command'].count('--gpu-id'),1)
        self.assertFalse(result['public']['module_report_verified'])

    def test_unknown_options_duplicate_paths_shell_or_other_module_are_rejected(self):
        for extra in (['--dataset',str(self.dataset)],['--execute'],['-c','anything'],['--output-dir=/tmp/x'],['--gpu-id','1']):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                g.module_options(self.args.module,self.args.module_args[1:]+extra)
        self.args.module='os'
        with self.assertRaises(ValueError):g.plan(self.args)

    def test_whole_source_models_dependencies_and_both_outputs_are_protected(self):
        for target in (self.source/'logs',self.work/'model_dir'/'logs',self.work/'feature_root'/'logs',self.work/'module-result'):
            self.args.report_dir=target
            with self.subTest(target=target),self.assertRaises(ValueError):g.plan(self.args)
        self.args.report_dir=self.work/'controller'; self.args.report_dir.mkdir()
        with self.assertRaises(ValueError):g.plan(self.args)
        self.args.report_dir=self.work/'fresh'
        link=self.work/'link'; link.symlink_to(self.work/'model_dir',target_is_directory=True)
        self.args.report_dir=link/'report'
        with self.assertRaises(ValueError):g.plan(self.args)

    def test_readonly_share_permits_only_original_model_dependencies_and_never_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as external:
            root=Path(external).resolve(); model=root/'assets'; model.mkdir()
            manifest=model/'model-manifest.json';manifest.write_text('{}')
            data=json.loads(self.dataset.read_text()); data.update(model_dir=str(model),model_manifest=str(manifest))
            self.dataset.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'original-model'):
                g.plan(self.args)
            self.args.readonly_dependency_root=[root]
            self.assertEqual(g.plan(self.args)['module_output'],self.work/'module-result')
            for name in ('official_repo','audio_root','prepared_dir','feature_root'):
                other=dict(data);other[name]=str(model);self.dataset.write_text(json.dumps(other))
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'original-model'):
                    g.plan(self.args)
            self.dataset.write_text(json.dumps(data))
            old=list(self.args.module_args)
            self.args.module_args[self.args.module_args.index('--adapter')+1]=str(manifest)
            with self.assertRaisesRegex(ValueError,'original-model'):g.plan(self.args)
            self.args.module_args=old;self.args.report_dir=root/'controller'
            with self.assertRaises(ValueError):g.plan(self.args)
            self.args.report_dir=self.work/'controller'
            self.args.module_args[self.args.module_args.index('--output-dir')+1]=str(root/'result')
            with self.assertRaises(ValueError):g.plan(self.args)

    def test_synthesis_external_model_share_is_rejected_to_match_child_scope(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as external:
            root=Path(external).resolve(); model=root/'assets';model.mkdir()
            manifest=model/'manifest.json';manifest.write_text('{}')
            prompt=self.work/'prompt.wav';prompt.write_bytes(b'path-contract only')
            self.args.module='adapter_synth';self.args.readonly_dependency_root=[root]
            self.args.module_args=['--workspace-root',str(self.work),'--official-repo',str(self.source),
                '--model-dir',str(model),'--model-manifest',str(manifest),'--training-base-manifest',str(manifest),
                '--adapter',str(self.adapter),'--adapter-sha256','a'*64,'--prompt',str(prompt),
                '--output-dir',str(self.work/'module-result'),'--text','नमस्ते।']
            with self.assertRaisesRegex(ValueError,'original-model'):g.plan(self.args)
            local_model=self.work/'model_dir';local_manifest=self.work/'model_manifest'
            for name,value in (('--model-dir',local_model),('--model-manifest',local_manifest),('--training-base-manifest',local_manifest)):
                self.args.module_args[self.args.module_args.index(name)+1]=str(value)
            self.assertEqual(g.plan(self.args)['command'][3],'hi_h200_native.adapter_synth')
            self.args.module_args[self.args.module_args.index('--official-repo')+1]=str(model)
            with self.assertRaisesRegex(ValueError,'original-model'):g.plan(self.args)

    def test_exit_zero_is_only_process_success_and_does_not_verify_module_or_quality(self):
        child=Child(); (report,code),popen,handlers=self.run_fixture(child)
        self.assertEqual(code,0); self.assertEqual(child.waits,[30])
        self.assertEqual(child.terminated,0); self.assertTrue(report['own_child_exited'])
        self.assertEqual(report['status'],'child_exit0_module_report_unverified')
        self.assertFalse(report['module_report_verified']); self.assertFalse(report['quality_gate_passed'])
        self.assertNotIn('shell',popen.call_args.kwargs)
        self.assertEqual(popen.call_args.kwargs['cwd'],self.args.report_dir)
        self.assertEqual(handlers[signal.SIGTERM],signal.SIG_DFL)
        self.assertTrue((self.args.report_dir/'stdout.log').is_file())
        self.assertTrue((self.args.report_dir/'controller.json').is_file())

    def test_overfit_exit_three_stays_distinct_from_operational_exit_two(self):
        for status in (3,2):
            self.args.report_dir=self.work/('controller-'+str(status))
            (report,code),_,_=self.run_fixture(Child(),exit_code=status)
            self.assertEqual(code,status)
            self.assertEqual(report['child_exit_code'],status)
            self.assertEqual('mechanism_gate_failed' in report['status'],status==3)
            self.assertFalse(report['module_report_verified'])

    def test_queue_arrival_resource_identity_and_monitor_failure_stop_only_own_child(self):
        conditions=[{'queue':[True,True,False]}, {'stdout':'0, GPU-other, H200, 140000, 500, 0'},
            {'stdout':'0, GPU-fixture, H200, 140000, 120000, 0'}, {'ram':None}, {'monitor_error':OSError('fixture')}]
        for index,kwargs in enumerate(conditions):
            with self.subTest(index=index):
                self.args.report_dir=self.work/('stopped-'+str(index))
                child=Child();(report,code),_,_=self.run_fixture(child,**kwargs)
                self.assertEqual(code,2);self.assertEqual(child.terminated,1);self.assertEqual(child.killed,0)
                self.assertIsNotNone(report['stop_reason']);self.assertTrue(report['own_child_exited'])

    def test_term_then_kill_waits_are_bounded_and_original_handlers_restored(self):
        child=Child(stubborn=True)
        (report,code),_,handlers=self.run_fixture(child,queue=[True,True,False])
        self.assertEqual(code,2);self.assertEqual(child.terminated,1);self.assertEqual(child.killed,1)
        self.assertEqual(child.waits,[30,10]);self.assertEqual(report['child_exit_code'],-9)
        self.assertEqual(handlers[signal.SIGINT],signal.SIG_DFL)

    def test_signal_and_exception_cleanup_have_no_orphan(self):
        prepared=g.plan(self.args);child=Child();handlers={}
        def register(number,handler): old=handlers.get(number,signal.SIG_DFL);handlers[number]=handler;return old
        def signal_request(_):handlers[signal.SIGTERM](signal.SIGTERM,None)
        with patch.object(g.shared,'queue_idle',return_value=True),patch.object(g.shared,'resource_admission',return_value=[{'uuid':'GPU-fixture'}]), \
             patch.object(g.subprocess,'Popen',return_value=child),patch.object(g.subprocess,'run',return_value=SimpleNamespace(stdout='0, GPU-fixture, H200, 140000, 500, 0')), \
             patch.object(g.shared,'available_ram_bytes',return_value=100000*1024**2),patch.object(g.time,'sleep',side_effect=signal_request), \
             patch.object(g.signal,'signal',side_effect=register):
            report,code=g.execute(self.args,prepared)
        self.assertEqual(code,2);self.assertEqual(report['stop_reason'],'controller_signal');self.assertIsNotNone(child.poll())
        self.args.report_dir=self.work/'exception';prepared=g.plan(self.args);child=Child()
        with patch.object(g.shared,'queue_idle',return_value=True),patch.object(g.shared,'resource_admission',return_value=[{'uuid':'GPU-fixture'}]), \
             patch.object(g.subprocess,'Popen',return_value=child),patch.object(g.subprocess,'run',return_value=SimpleNamespace(stdout='0, GPU-fixture, H200, 140000, 500, 0')), \
             patch.object(g.shared,'available_ram_bytes',return_value=100000*1024**2),patch.object(g.time,'sleep',side_effect=RuntimeError('fixture')), \
             patch.object(g.signal,'signal',return_value=signal.SIG_DFL),self.assertRaises(RuntimeError):
            g.execute(self.args,prepared)
        self.assertIsNotNone(child.poll());self.assertEqual(child.terminated,1)
        self.assertEqual(json.loads((self.args.report_dir/'controller.json').read_text())['stop_reason'],'controller_exception')

    def test_deadline_stops_before_another_telemetry_call(self):
        child=Child()
        with patch.object(g.time,'monotonic',side_effect=[0,601,602]):
            (report,code),_,_=self.run_fixture(child)
        self.assertEqual(code,2);self.assertEqual(report['stop_reason'],'600_second_deadline')
        self.assertEqual(child.terminated,1)

    def test_child_environment_binds_source_offline_and_never_forwards_tokens_or_proxy(self):
        with patch.dict(os.environ,{'HF_TOKEN':'test-secret','HTTPS_PROXY':'test-proxy','PYTHONPATH':'untrusted'},clear=False):
            env=g.child_environment('0')
        self.assertNotIn('HF_TOKEN',env);self.assertNotIn('HTTPS_PROXY',env)
        self.assertEqual(env['PYTHONPATH'],str(self.source/'src'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0');self.assertEqual(env['HF_HUB_OFFLINE'],'1')

    def test_only_explicit_bounded_dependency_overlays_follow_the_source(self):
        overlay=self.work/'synth-overlay';overlay.mkdir()
        self.args.python_dependency_root=[overlay]
        prepared=g.plan(self.args)
        with patch.dict(os.environ,{'PYTHONPATH':'untrusted-ambient'}):
            env=g.child_environment('0',prepared['python_roots'])
        self.assertEqual(env['PYTHONPATH'],os.pathsep.join([str(self.source/'src'),str(overlay)]))
        self.assertNotIn('untrusted-ambient',env['PYTHONPATH'])
        self.args.report_dir=overlay/'logs'
        with self.assertRaises(ValueError):g.plan(self.args)
        self.args.report_dir=self.work/'controller'
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as external:
            root=Path(external).resolve();shared_overlay=root/'overlay';shared_overlay.mkdir()
            self.args.python_dependency_root=[shared_overlay]
            with self.assertRaisesRegex(ValueError,'Python dependency'):g.plan(self.args)
            self.args.readonly_dependency_root=[root]
            self.assertEqual(g.plan(self.args)['python_roots'],[shared_overlay])
            link=self.work/'overlay-link';link.symlink_to(shared_overlay,target_is_directory=True)
            self.args.python_dependency_root=[link]
            with self.assertRaisesRegex(ValueError,'canonical'):g.plan(self.args)

    def test_parent_import_uses_repo_source_before_ambient_installed_package(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as external:
            fake=Path(external)/'hi_h200_native';fake.mkdir()
            (fake/'__init__.py').write_text("raise AssertionError('ambient package must not be imported')")
            script=("import importlib.util,sys,pathlib,inspect;"
                "s=importlib.util.spec_from_file_location('guard',"+repr(str(ROOT/'scripts/run_pure_eval_guarded.py'))+");"
                "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
                "assert pathlib.Path(sys.path[0])==g.ROOT/'src';"
                "assert pathlib.Path(inspect.getfile(g.shared.resource_admission)).resolve()==g.ROOT/'src/hi_h200_native/synth.py';"
                "assert 'torch' not in sys.modules")
            result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,text=True,capture_output=True,timeout=20,
                env=dict(os.environ,PYTHONPATH=external,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(result.returncode,0,result.stderr)

    def test_help_imports_no_torch_and_does_not_start_evaluation(self):
        script="import importlib.util,sys; s=importlib.util.spec_from_file_location('guard',"+repr(str(ROOT/'scripts/run_pure_eval_guarded.py'))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);\ntry:m.main(['--help'])\nexcept SystemExit as e:assert e.code==0\nassert 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,text=True,capture_output=True,timeout=20,
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--report-dir',result.stdout)


if __name__=='__main__':unittest.main()
