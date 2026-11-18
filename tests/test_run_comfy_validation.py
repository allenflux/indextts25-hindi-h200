"""CPU contracts only: no server, socket, subprocess, GPU, or model execution."""
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_comfy_validation as c


class ControllerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='comfy-controller-', dir=ROOT / '.runtime')
        self.home = Path(self.temp.name).resolve()
        self.model = self.home / 'base'
        self.model.mkdir()
        self.graph = c.build_graph(0, self.model, 18889)
        self.prompt_id = 'fixture-prompt-00000001'
        self.network = patch('socket.socket', side_effect=AssertionError('CPU tests never create a socket'))
        self.spawn = patch.object(c.subprocess, 'Popen', side_effect=AssertionError('CPU tests never start a service'))
        self.network.start(); self.spawn.start()
    def tearDown(self):
        self.spawn.stop(); self.network.stop(); self.temp.cleanup()
    def history(self):
        descriptor = {'filename':'validation_base_before_00001.wav', 'subfolder':'audio', 'type':'output'}
        return {self.prompt_id:{'prompt':[1,self.prompt_id,copy.deepcopy(self.graph),{},['5']],
            'status':{'completed':True, 'status_str':'success',
                      'messages':[['execution_start',{'prompt_id':self.prompt_id}],
                                  ['execution_success',{'prompt_id':self.prompt_id}]]},
            'outputs':{'5':{'audio':[descriptor]}}}}
    def telemetry(self):
        return {'overflow':False,
            'events':[{'event':'executing','prompt_id':self.prompt_id,'node':node} for node in sorted(c.ACTIVE)],
            'cache_entries':[{'model_dir':str(self.model), 'gpt_path':str(self.model/'gpt.pth'),
                              'users':0, 'model_id':123456, 'device':'cuda:0', 'use_bf16':True,
                              'use_fp16':False, 'completed_runs':1}]}
    def process(self, code=None):
        return SimpleNamespace(pid=12345, poll=lambda:code)
    def test_fixed_eight_nodes_five_links_and_original_loader_contract(self):
        self.assertEqual(set(self.graph), set('12345678'))
        self.assertEqual({key:node['class_type'] for key,node in self.graph.items()}, {
            '1':'T8_IndexTTS25_ModelLoader','2':'LoadAudioFromURLCached_ZHO',
            '3':'T8_IndexTTS25_SamplingConfig','4':'T8_IndexTTS25_Generate','5':'AudioEncrypt',
            '6':'EmptyLatentImage','7':'LoadAudioFromURLCached_ZHO','8':'T8_IndexTTS25_EmotionControl'})
        links = sorted((node,key,value[0],value[1]) for node,row in self.graph.items()
                       for key,value in row['inputs'].items() if isinstance(value,list))
        self.assertEqual(links,[('4','emotion','8',0),('4','model','1',0),('4','sampling','3',0),
                               ('4','speaker_audio','2',0),('5','audio','4',0)])
        loader = self.graph['1']['inputs']
        self.assertEqual((loader['device'],loader['precision']),('auto','auto'))
        self.assertIs(loader['persistent_reference_cache'],True)
        self.assertIs(loader['verify_hashes'],False)
        self.assertFalse(self.graph['5']['inputs']['encrypt'])
        self.assertEqual(self.graph['5']['inputs']['key'],'0'*32)
        self.assertNotIn('extra_pnginfo',json.dumps(self.graph))
        self.assertEqual(self.graph['4']['inputs']['quality_retry_count'],0)
    def test_first_and_third_graphs_only_change_output_prefix(self):
        first = copy.deepcopy(c.build_graph(0,self.model,18889))
        third = copy.deepcopy(c.build_graph(2,self.model,18889))
        self.assertNotEqual(first['5']['inputs']['filename_prefix'],third['5']['inputs']['filename_prefix'])
        first['5']['inputs'].pop('filename_prefix'); third['5']['inputs'].pop('filename_prefix')
        self.assertEqual(first,third)
        hindi = c.build_graph(1,self.home/'hindi',18889)
        self.assertEqual(hindi['4']['inputs']['language'],'HI')
        self.assertEqual(hindi['1']['inputs']['custom_model_path'],str(self.home/'hindi'))
    def test_graph_rejects_changed_node_link_encryption_and_retries(self):
        edits = [lambda g:g.pop('7'), lambda g:g['4'].update(class_type='WrongNode'),
                 lambda g:g['4']['inputs'].update(model=['2',0]),
                 lambda g:g['5']['inputs'].update(encrypt=True),
                 lambda g:g['5']['inputs'].update(key='1'*32),
                 lambda g:g['4']['inputs'].update(quality_retry_count=1)]
        for edit in edits:
            graph = copy.deepcopy(self.graph); edit(graph)
            with self.assertRaises(ValueError): c.validate_graph(graph)
    def test_production_port_and_nonowned_listener_are_never_contacted(self):
        for port in (8188,80,0,65536,True):
            with self.assertRaises(ValueError): c.OwnedHTTP(self.process(),port,'nonce-fixture')
            with self.assertRaises(ValueError): c.build_graph(0,self.model,port)
        for process,proof in ((self.process(),lambda *_:False),(self.process(0),lambda *_:True)):
            client = c.OwnedHTTP(process,18889,'nonce-fixture',proof=proof)
            with patch.object(c.http.client,'HTTPConnection') as connection:
                with self.assertRaises(ValueError): client.request('GET','/queue')
                connection.assert_not_called()
    def test_uncertain_post_counts_once_and_never_automatically_retries(self):
        client = c.OwnedHTTP(self.process(),18889,'nonce-fixture',proof=lambda *_:True)
        client.request = Mock(side_effect=TimeoutError('CPU timeout fixture'))
        with self.assertRaises(TimeoutError): client.submit_once(self.graph,'controller-fixture')
        self.assertEqual(client.submissions,1)
        self.assertEqual(client.request.call_count,1)
        method,path,payload = client.request.call_args.args
        self.assertEqual((method,path),('POST','/prompt'))
        self.assertEqual(set(payload),{'prompt','client_id'})
        self.assertNotIn('extra_pnginfo',json.dumps(payload))
        client.submissions = 3
        with self.assertRaises(ValueError): client.submit_once(self.graph,'controller-fixture')
        self.assertEqual(client.request.call_count,1)
    def test_invalid_prompt_acknowledgment_is_not_success(self):
        for ack in ({}, {'prompt_id':'bad'}, {'prompt_id':self.prompt_id,'node_errors':{'4':'error'}}):
            client = c.OwnedHTTP(self.process(),18889,'nonce-fixture',proof=lambda *_:True)
            client.request = Mock(return_value=ack)
            with self.assertRaises(ValueError): client.submit_once(self.graph,'controller-fixture')
            self.assertEqual(client.submissions,1)
            self.assertEqual(client.request.call_count,1)
    def test_history_fixture_contract_requires_same_job_and_final_node_five(self):
        # This checks the controller's declared schema, not a real Comfy response.
        descriptor = c.completed_history(self.history(),self.prompt_id,self.graph)
        self.assertEqual(descriptor['type'],'output')
        self.assertEqual(c.completed_history({},self.prompt_id,self.graph),None)
        bad = self.history(); bad['other-prompt'] = bad.pop(self.prompt_id)
        with self.assertRaises(ValueError): c.completed_history(bad,self.prompt_id,self.graph)
        edits = [lambda h:h[self.prompt_id]['status'].update(completed=False),
                 lambda h:h[self.prompt_id]['status'].update(status_str='error'),
                 lambda h:h[self.prompt_id]['status'].update(messages=[]),
                 lambda h:h[self.prompt_id]['status']['messages'].append(['execution_error',{}]),
                 lambda h:h[self.prompt_id]['status']['messages'].append(['execution_interrupted',{}]),
                 lambda h:h[self.prompt_id]['outputs'].update({'4':h[self.prompt_id]['outputs'].pop('5')})]
        for edit in edits:
            history = self.history(); edit(history)
            with self.assertRaises(ValueError): c.completed_history(history,self.prompt_id,self.graph)
    def test_http_200_history_is_not_semantic_success_and_redirect_is_refused(self):
        history = self.history(); history['wrong-id'] = history.pop(self.prompt_id)
        response = SimpleNamespace(status=200, read=Mock(return_value=json.dumps(history).encode()))
        connection = Mock(); connection.getresponse.return_value = response
        client = c.OwnedHTTP(self.process(),18889,'nonce-fixture',proof=lambda *_:True)
        with patch.object(c.http.client,'HTTPConnection',return_value=connection) as factory:
            value = client.request('GET','/history/'+self.prompt_id)
            with self.assertRaises(ValueError): c.completed_history(value,self.prompt_id,self.graph)
            factory.assert_called_once_with('127.0.0.1',18889,timeout=2)
        response.read.assert_called_once_with(c.MAX_JSON+1)
        connection.close.assert_called_once()
        response.status = 302
        response.read.return_value = b'SECRET_RESPONSE_BODY'
        with patch.object(c.http.client,'HTTPConnection',return_value=connection):
            with self.assertRaises(ValueError) as error: client.request('GET','/history/'+self.prompt_id)
        self.assertNotIn('SECRET_RESPONSE_BODY',str(error.exception))

    def test_history_rejects_graph_drift_and_wrong_success_id(self):
        history = self.history(); history[self.prompt_id]['prompt'][2]['4']['inputs']['text'] = 'changed'
        with self.assertRaises(ValueError): c.completed_history(history,self.prompt_id,self.graph)
        history = self.history(); history[self.prompt_id]['status']['messages'][-1][1]['prompt_id'] = 'other'
        with self.assertRaises(ValueError): c.completed_history(history,self.prompt_id,self.graph)
        history = self.history(); history[self.prompt_id]['prompt'][1] = 'other'
        with self.assertRaises(ValueError): c.completed_history(history,self.prompt_id,self.graph)
    def test_foreign_or_malformed_queue_entries_fail_without_clearing(self):
        c.check_queue({'queue_running':[], 'queue_pending':[]},set())
        c.check_queue({'queue_running':[[0,self.prompt_id]], 'queue_pending':[]},{self.prompt_id})
        for queue in ({'queue_running':[[0,'foreign']], 'queue_pending':[]},
                      {'queue_running':[], 'queue_pending':[[0,'foreign']]},
                      {'queue_running':[], 'queue_pending':{}}, {}):
            with self.assertRaises(ValueError): c.check_queue(queue,{self.prompt_id})
    def test_audio_descriptor_rejects_traversal_wrong_prefix_and_symlinks(self):
        output = self.home/'output'; (output/'audio').mkdir(parents=True)
        descriptor = c.completed_history(self.history(),self.prompt_id,self.graph)
        path = output/'audio'/descriptor['filename']; path.write_bytes(b'CPU path fixture, not generated WAV')
        self.assertEqual(c.audio_path(descriptor,output,0),path)
        for changed in ({'filename':'../'+descriptor['filename']}, {'filename':'/absolute.wav'},
                        {'filename':'validation_hindi_00001.wav'}, {'subfolder':'../audio'},
                        {'type':'input'}, {'url':'https://example.invalid/audio'}):
            value = dict(descriptor,**changed)
            with self.assertRaises(ValueError): c.audio_path(value,output,0)
        path.unlink(); outside = self.home/'outside-fixture'; outside.write_bytes(b'fixture')
        path.symlink_to(outside)
        with self.assertRaises(ValueError): c.audio_path(descriptor,output,0)
    def test_six_actual_execution_events_and_cuda_bf16_cache_are_required(self):
        state = self.telemetry()
        self.assertEqual(c.evidence(state,self.prompt_id,self.model)['model_id'],123456)
        edits = [lambda s:s.update(overflow=True), lambda s:s['events'].pop(),
                 lambda s:s['events'].append({'event':'execution_cached','prompt_id':self.prompt_id,'nodes':['4']}),
                 lambda s:s['cache_entries'][0].update(device='cpu'),
                 lambda s:s['cache_entries'][0].update(use_bf16=False),
                 lambda s:s['cache_entries'][0].update(use_fp16=True),
                 lambda s:s['cache_entries'][0].update(users=1),
                 lambda s:s['cache_entries'][0].update(gpt_path='wrong'),
                 lambda s:s['cache_entries'].append(copy.deepcopy(s['cache_entries'][0]))]
        for edit in edits:
            state = self.telemetry(); edit(state)
            with self.assertRaises(ValueError): c.evidence(state,self.prompt_id,self.model)
    def test_cached_third_graph_cannot_count_as_model_switchback(self):
        state = self.telemetry()
        state['events'] = [{'event':'executing','prompt_id':self.prompt_id,'node':'5'},
                           {'event':'execution_cached','prompt_id':self.prompt_id,'nodes':['1','2','3','4','8']}]
        with self.assertRaises(ValueError): c.evidence(state,self.prompt_id,self.model)
        state = self.telemetry()
        for event in state['events']: event['prompt_id'] = 'previous-run'
        with self.assertRaises(ValueError): c.evidence(state,self.prompt_id,self.model)
    def test_second_submit_timeout_aborts_cycle_and_cleans_only_owned_state(self):
        self._second_submit_timeout_fixture()
    def test_reviewed_manifest_pins_reach_bootstrap_and_second_stage_verification(self):
        self._second_submit_timeout_fixture(reviewed=True)
    def _second_submit_timeout_fixture(self, reviewed=False):
        # State-machine fixture only: this is not an observed Comfy/GPU result.
        import stage_comfy_validation as staging
        output = self.home/'controller-report'
        (self.home/'stage').mkdir()
        models = {'base':self.model, 'hindi':self.home/'hindi'}
        stage = {'port':18889, 'paths':{'python':'CPU_PYTHON_FIXTURE', 'runtime':str(self.home),
                                      'output':str(self.home/'server-output')}}
        graphs = [c.build_graph(i,models[case[1]],stage['port']) for i,case in enumerate(c.CASES)]
        report = {'status':'metadata_fixture', 'quality_verified':False,
            'models':{'base':{'revision':'base-fixture'}, 'hindi':{'revision':'hindi-fixture'}},
            'runs':[{'case':case[0], 'variant':case[1], 'language':case[2], 'status':'not_executed'} for case in c.CASES]}
        args = SimpleNamespace(execute=True,run_timeout=2,startup_timeout=2,max_used_gpu_mib=24576,
                               prompt_sha256='a'*64)
        registration = self.home/'reviewed.json';registration.write_text('{}')
        registration_sha = hashlib.sha256(registration.read_bytes()).hexdigest()
        if reviewed:
            args.candidate_revision=None;args.candidate_manifest=registration;args.candidate_manifest_sha256=registration_sha
        configs=[]
        def spawn_fixture(*unused,**kwargs):
            configs.append(json.loads((output/'bootstrap.json').read_text()))
            return process
        alive = {'code':None}
        process = SimpleNamespace(pid=56789,poll=lambda:alive['code'])
        cache = {'model_dir':str(self.model),'model_id':123456,'model_revision':'base-fixture',
                 'model_fingerprint':'fixture-fingerprint'}
        descriptor = {'filename':'validation_base_before_00001.wav','subfolder':'audio','type':'output'}
        audio_bytes = b'CPU orchestration bytes; not generated audio'
        audio = {'seconds':1.2,'file_sha256':hashlib.sha256(audio_bytes).hexdigest(),'pcm_sha256':'b'*64}
        client = Mock()
        client.submit_once.side_effect = [self.prompt_id,TimeoutError('CPU second-submit timeout fixture')]
        state_calls = []
        def request(method,path,*unused,**options):
            if path=='/validation/state':
                state_calls.append(path)
                return {'controller_id':report['controller_id'],'server_pid':process.pid,'ready':True,
                        'cache_entries':[] if len(state_calls)==1 else [cache]}
            if path.startswith('/object_info/'):
                return {path.rsplit('/',1)[1]:{}}
            if path=='/queue':return {'queue_running':[],'queue_pending':[]}
            if path=='/history/'+self.prompt_id:return {'CPU_HISTORY_FIXTURE_ONLY':True}
            if path.startswith('/view?'):return audio_bytes
            raise AssertionError('Unexpected state-machine request')
        client.request.side_effect = request
        timers = []
        def timer_factory(*unused):
            timer = SimpleNamespace(start=Mock(),cancel=Mock(),daemon=False)
            timers.append(timer);return timer
        def cleanup(owned):
            self.assertIs(owned,process);alive['code']=-15
        with ExitStack() as stack:
            stack.enter_context(patch.object(c,'preflight',return_value=(report,(stage,self.home/'stage',models,self.home/'prompt-fixture',output,graphs))))
            stack.enter_context(patch.object(c.sys,'platform','linux'))
            stack.enter_context(patch.object(c.synth,'resource_admission',return_value=[{'uuid':'CPU_GPU_UUID_FIXTURE'}]))
            stack.enter_context(patch.object(c,'runtime_gate',return_value={'fixture_only':True}))
            stack.enter_context(patch.object(c.probe,'queue_clear'))
            stack.enter_context(patch.object(c.socket,'socket',return_value=MagicMock()))
            verify_stage = stack.enter_context(patch.object(staging,'verify_stage',return_value={}))
            spawn = stack.enter_context(patch.object(c.subprocess,'Popen',side_effect=spawn_fixture))
            stack.enter_context(patch.object(c,'child_environment',return_value={}))
            owned_http = stack.enter_context(patch.object(c,'OwnedHTTP',return_value=client))
            stack.enter_context(patch('threading.Timer',side_effect=timer_factory))
            stack.enter_context(patch.object(c,'completed_history',return_value=descriptor))
            stack.enter_context(patch.object(c,'evidence',return_value=cache))
            stack.enter_context(patch.object(c,'audio_path',return_value=self.home/'audio-fixture'))
            stack.enter_context(patch.object(c.synth,'wav_info',return_value=audio))
            stop = stack.enter_context(patch.object(c,'stop_owned',side_effect=cleanup))
            stack.enter_context(patch.object(c.time,'sleep',side_effect=AssertionError('Fixture must not poll/retry')))
            with self.assertRaises(TimeoutError):c.run(args)
        self.assertEqual(client.submit_once.call_count,2)
        self.assertEqual([call.args[0] for call in client.submit_once.call_args_list],graphs[:2])
        self.assertEqual(report['runs'][0]['status'],'completed_quality_unverified')
        self.assertEqual(report['runs'][1]['status'],'failed_or_submission_uncertain')
        self.assertEqual(report['runs'][2]['status'],'not_executed')
        self.assertNotIn('submitted',report['runs'][2])
        stop.assert_called_once_with(process)
        self.assertEqual(owned_http.call_count,1)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertEqual(len(timers),2)
        for timer in timers:
            timer.start.assert_called_once();timer.cancel.assert_called_once()
        self.assertFalse((output/'bootstrap.json').exists())
        saved = json.loads((output/'result.json').read_text())
        self.assertEqual(saved['status'],'cycle_failed_not_verified')
        self.assertEqual(saved['error_type'],'TimeoutError')
        self.assertTrue(saved['partial_outputs_are_not_success'])
        self.assertTrue(saved['owned_process_cleanup_verified'])
        self.assertEqual(saved['owned_process_returncode'],-15)
        self.assertNotIn('nonce',json.dumps(saved))
        self.assertFalse(saved['quality_verified'])
        if reviewed:
            self.assertEqual(configs[0]['candidate_manifest_sha256'],registration_sha)
            self.assertEqual(configs[0]['candidate_manifest'],str(registration))
            verify_stage.assert_called_once_with(self.home/'stage',candidate_manifest_path=registration,candidate_manifest_sha256=registration_sha)
        else:
            self.assertNotIn('candidate_manifest',configs[0])
            verify_stage.assert_called_once_with(self.home/'stage')

    def test_shutdown_never_signals_unowned_group(self):
        process = self.process()
        with patch.object(c.os,'getpgid',return_value=process.pid+1), patch.object(c.os,'killpg') as kill:
            with self.assertRaises(ValueError): c.stop_owned(process)
            kill.assert_not_called()
        owned = SimpleNamespace(pid=12345,poll=lambda:None,wait=Mock(return_value=0))
        with patch.object(c.os,'getpgid',return_value=owned.pid), patch.object(c.os,'killpg') as kill:
            c.stop_owned(owned)
            kill.assert_called_once_with(owned.pid,signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
