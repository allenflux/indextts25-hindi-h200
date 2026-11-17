#!/usr/bin/env python3
"""Finite, owned-loopback ComfyUI base -> Hindi -> base acceptance; no production POST."""
from __future__ import annotations
import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlencode
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
import probe_t8_nodes as probe
from hi_h200_native import synth

ACTIVE = {'1', '2', '3', '4', '5', '8'}
ZERO_KEY = '0' * 32  # Public nonsecret placeholder only; encryption is disabled.
CASES = (('base_before', 'base', 'ZH', '你好，这是原始模型的切换测试。'),
         ('hindi', 'hindi', 'HI', 'नमस्ते। यह हिन्दी भाषा का एक छोटा परीक्षण है।'),
         ('base_after', 'base', 'ZH', '你好，这是原始模型的切换测试。'))
CLASS_TYPES = {'1':'T8_IndexTTS25_ModelLoader', '2':'LoadAudioFromURLCached_ZHO',
    '3':'T8_IndexTTS25_SamplingConfig', '4':'T8_IndexTTS25_Generate', '5':'AudioEncrypt',
    '6':'EmptyLatentImage', '7':'LoadAudioFromURLCached_ZHO', '8':'T8_IndexTTS25_EmotionControl'}
LINKS = [('4','emotion','8',0), ('4','model','1',0), ('4','sampling','3',0),
         ('4','speaker_audio','2',0), ('5','audio','4',0)]
MAX_JSON = 2 * 1024**2
PROMPT_SHA256 = 'fd23ddac68f2db3710481c7cbfc1d500117d9c38bc5c3ee51c33444f0d9b019d'
_STOP_LOCK = threading.RLock()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(raw):
    if len(raw) > MAX_JSON:
        raise ValueError('Bounded JSON exceeded')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def build_graph(index, model_dir, port):
    if index not in range(3) or type(port) is not int or not 1024 <= port <= 65535 or port == 8188:
        raise ValueError('Only the fixed three runs and a nonproduction high loopback port are allowed')
    label, variant, language, text = CASES[index]
    values = probe.node_parameters(model_dir, text, language)
    original_loader = dict(values['loader'], device='auto', precision='auto', verify_hashes=False, persistent_reference_cache=True)
    inputs = {'1':original_loader, '2':{'url':f'http://127.0.0.1:{port}/validation/prompt.wav', 'filename_prefix':'aud', 'force_format':'wav'},
        '3':values['sampling'], '4':dict(values['generate'], model=['1',0], speaker_audio=['2',0], sampling=['3',0], emotion=['8',0]),
        '5':{'encrypt':False, 'key':ZERO_KEY, 'format':'audio/wav', 'algorithm':'AES-GCM', 'bitrate':'192k',
             'filename_prefix':'validation_' + label, 'subfolder':'audio', 'audio':['4',0]},
        '6':{'width':64, 'height':64, 'batch_size':1}, '7':{'url':'', 'filename_prefix':'aud', 'force_format':'wav'},
        '8':{'mode':'speaker'}}
    graph = {key:{'class_type':CLASS_TYPES[key], 'inputs':inputs[key]} for key in CLASS_TYPES}
    validate_graph(graph)
    return graph


def validate_graph(graph):
    if not isinstance(graph, dict) or set(graph) != set(CLASS_TYPES):
        raise ValueError('Exact original eight node IDs are required')
    links = []
    for key, node in graph.items():
        if set(node) != {'class_type','inputs'} or node['class_type'] != CLASS_TYPES[key] or not isinstance(node['inputs'], dict):
            raise ValueError('Node type/schema changed')
        for name, value in node['inputs'].items():
            if isinstance(value, list):
                if len(value) != 2 or value[0] not in graph or type(value[1]) is not int:
                    raise ValueError('Invalid original graph link')
                links.append((key,name,value[0],value[1]))
    if sorted(links) != LINKS or graph['5']['inputs'].get('encrypt') is not False or graph['5']['inputs'].get('key') != ZERO_KEY:
        raise ValueError('Original links or nonsecret encryption-off policy changed')
    if graph['4']['inputs'].get('quality_retry_count') != 0:
        raise ValueError('This fixed acceptance does not enable ASR/candidate retries')


def owned_listener(pid, port):
    """Linux kernel socket ownership, not an HTTP assertion from an arbitrary service."""
    try:
        inodes = set()
        for path in (Path('/proc') / str(pid) / 'fd').iterdir():
            try:
                target=os.readlink(path)
            except OSError:
                continue
            if target.startswith('socket:['):
                inodes.add(target[8:-1])
        for line in (Path('/proc') / str(pid) / 'net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if fields[1] == f'0100007F:{port:04X}' and fields[3] == '0A' and fields[9] in inodes:
                return True
    except (OSError, IndexError):
        pass
    return False


class OwnedHTTP:
    def __init__(self, process, port, nonce, *, proof=owned_listener):
        if type(port) is not int or not 1024 <= port <= 65535 or port == 8188:
            raise ValueError('Production/default port is forbidden')
        self.process, self.port, self.nonce, self.proof = process, port, nonce, proof
        self.submissions = 0

    def request(self, method, path, payload=None, *, binary=False, timeout=2):
        allowed = path in {'/validation/state','/queue','/object_info'} or path.startswith('/history/') or path.startswith('/view?') or path.removeprefix('/object_info/') in CLASS_TYPES.values()
        if method == 'POST':
            allowed = path == '/prompt'
        if method not in {'GET','POST'} or not allowed or self.process.poll() is not None or not self.proof(self.process.pid, self.port):
            raise ValueError('Request is not bound to the controller-owned loopback listener')
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        raw = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        if raw is not None and len(raw) > 64 * 1024:
            raise ValueError('Prompt request exceeded its fixed bound')
        try:
            connection.request(method, path, body=raw, headers={'X-Validation-Nonce':self.nonce, 'Content-Type':'application/json'})
            response = connection.getresponse()
            limit = 8 * 1024**2 if binary else MAX_JSON
            body = response.read(limit + 1)
            if response.status != 200 or len(body) > limit:
                raise ValueError('Owned service HTTP failed; response body is not logged')
            return body if binary else read_json(body)
        finally:
            connection.close()

    def submit_once(self, graph, controller_id):
        if self.submissions >= 3:
            raise ValueError('Exactly three submissions maximum; never retry an uncertain POST')
        validate_graph(graph)
        self.submissions += 1  # Set before network I/O: a timeout must not cause a retry.
        value = self.request('POST', '/prompt', {'prompt':graph, 'client_id':controller_id})
        if not isinstance(value, dict) or set(value)!={'prompt_id','number','node_errors'} or not isinstance(value.get('node_errors'),dict) or value['node_errors'] or not re.fullmatch('[A-Za-z0-9-]{16,80}', str(value.get('prompt_id',''))):
            raise ValueError('Prompt acknowledgment is missing/invalid; it is not success')
        return value['prompt_id']


def check_queue(value, own_ids):
    if not isinstance(value, dict):
        raise ValueError('Invalid owned queue')
    for name in ('queue_running','queue_pending'):
        if not isinstance(value.get(name), list):
            raise ValueError('Missing owned queue state')
        for item in value[name]:
            if not isinstance(item, list) or len(item) < 2 or item[1] not in own_ids:
                raise ValueError('Unowned prompt appeared in isolated queue; no queue clearing is allowed')


def completed_history(value, prompt_id, graph):
    if value == {}:
        return None
    if not isinstance(value, dict) or set(value) != {prompt_id}:
        raise ValueError('History does not belong to the submitted prompt')
    item = value[prompt_id]; status = item.get('status')
    if not isinstance(status, dict) or status.get('completed') is not True or status.get('status_str') != 'success':
        raise ValueError('Prompt did not complete successfully')
    messages = status.get('messages')
    if not isinstance(messages, list) or not any(isinstance(m,list) and len(m)==2 and m[0]=='execution_success' and isinstance(m[1],dict) and m[1].get('prompt_id')==prompt_id for m in messages):
        raise ValueError('No matching execution_success evidence')
    if any(isinstance(m,list) and m and m[0] in {'execution_error','execution_interrupted'} for m in messages):
        raise ValueError('Execution error/interruption is not success')
    original = item.get('prompt')
    if not isinstance(original, list) or len(original)<3 or original[1]!=prompt_id or digest_json(original[2])!=digest_json(graph):
        raise ValueError('History graph identity mismatch')
    output = item.get('outputs',{}).get('5',{}).get('audio')
    if not isinstance(output,list) or len(output)!=1 or not isinstance(output[0],dict):
        raise ValueError('Missing final node5 AudioEncrypt WAV descriptor')
    return output[0]


def audio_path(descriptor, output_root, index):
    if set(descriptor) != {'filename','subfolder','type'} or descriptor['type'] != 'output':
        raise ValueError('Final audio must be a local output descriptor')
    filename, subfolder = descriptor['filename'], descriptor['subfolder']
    if not isinstance(filename,str) or not re.fullmatch(r'validation_' + CASES[index][0] + r'_[0-9]+\.wav',filename) or subfolder!='audio':
        raise ValueError('Unexpected output prefix/subfolder/format')
    path = output_root / subfolder / filename
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True)!=path or not path.is_relative_to(output_root):
        raise ValueError('Final audio escapes owned output or is not a regular file')
    return path


def evidence(state, prompt_id, model_dir):
    if state.get('overflow') or not isinstance(state.get('events'),list):
        raise ValueError('Bounded execution telemetry is missing/incomplete')
    events = [e for e in state['events'] if e.get('prompt_id')==prompt_id]
    actual = {e.get('node') for e in events if e.get('event')=='executing' and e.get('node') is not None}
    cached = {n for e in events if e.get('event')=='execution_cached' for n in e.get('nodes',[])}
    if actual != ACTIVE or cached & ACTIVE:
        raise ValueError('Not every active original node executed, or graph output was cached')
    if not isinstance(state.get('cache_entries'),list) or len(state['cache_entries'])>2:
        raise ValueError('Unexpected unbounded or additional model cache entries')
    matches = [e for e in state['cache_entries'] if e.get('model_dir')==str(model_dir)]
    if len(matches)!=1 or matches[0].get('gpt_path')!=str(model_dir/'gpt.pth') or matches[0].get('users')!=0 or matches[0].get('model_id') is None:
        raise ValueError('Actual lazy-loader/cache identity is not verified')
    if matches[0].get('device') not in {'cuda','cuda:0'} or matches[0].get('use_bf16') is not True or matches[0].get('use_fp16') is not False:
        raise ValueError('Original auto settings did not select the admitted CUDA/BF16 runtime')
    return matches[0]


def stop_owned(process, *, force=False):
    with _STOP_LOCK:
        if process is None or process.poll() is not None:
            return
        try:
            if os.getpgid(process.pid) != process.pid:
                raise ValueError('Refusing to signal a non-owned process group')
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        except ProcessLookupError:
            process.wait(timeout=5)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def runtime_gate(args, expected_uuid):
    result = subprocess.run(['nvidia-smi','--id',args.gpu_id,
        '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True,timeout=3)
    sample = synth.parse_gpu_sample(result.stdout)
    ram = synth.available_ram_bytes()
    if sample['uuid']!=expected_uuid or sample['used_mib']>args.max_used_gpu_mib or ram is None or ram<args.min_free_ram_mib*1024**2:
        raise ValueError('Runtime GPU UUID/total-used cap/RAM floor failed')
    probe.queue_clear('http://127.0.0.1:8188')  # Read only; production POST is never used.
    return dict(sample,available_ram_bytes=ram)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root','stage-root','base-model','hindi-model','prompt','report-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--readonly-dependency-root',type=Path,action='append',default=[])
    parser.add_argument('--candidate-revision',required=True)
    parser.add_argument('--prompt-sha256',required=True)
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--allow-cuda',action='store_true')
    parser.add_argument('--gpu-id',default='0')
    for name in ('min-free-gpu-mib','min-free-ram-mib','max-utilization','max-used-gpu-mib'):
        parser.add_argument('--'+name,type=float)
    parser.add_argument('--run-timeout',type=int,default=600)
    parser.add_argument('--startup-timeout',type=int,default=120)
    args=parser.parse_args(argv)
    try:
        result=run(args)
        print(json.dumps({'status':result['status'],'quality_verified':False,'runs':[{'case':r['case'],'status':r['status']} for r in result['runs']]},sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'blocked_or_failed','error_type':type(exc).__name__,'reason':str(exc)[:1000]}),file=sys.stderr)
        return 2


def preflight(args):
    import stage_comfy_validation as staging
    import check_t8_compat as compat
    workspace=staging.real_directory(args.workspace_root)
    stage_root=staging.real_directory(args.stage_root)
    stage=staging.verify_stage(stage_root)
    if stage['workspace_root']!=str(workspace) or not stage_root.is_relative_to(workspace) or stage['candidate_revision']!=args.candidate_revision:
        raise ValueError('Staged source/workspace/candidate identity mismatch')
    roots=[workspace]+[staging.real_directory(p) for p in args.readonly_dependency_root]
    def dependency(value):
        path=Path(value).absolute()
        if path.resolve(strict=True)!=path or not any(path.is_relative_to(root) for root in roots):
            raise ValueError('Dependency must be canonical and inside WORK or an explicit reviewed read-only dependency root')
        return path
    models={key:dependency(value) for key,value in [('base',args.base_model),('hindi',args.hindi_model)]}
    if models['base']==models['hindi'] or models['base'].is_relative_to(models['hindi']) or models['hindi'].is_relative_to(models['base']):
        raise ValueError('Original and Hindi model directories must be independent')
    identities={}
    for variant,model in models.items():
        selection=compat.selected_manifest(Path(stage['paths']['t8']),model,args.candidate_revision,variant=variant)
        verified=compat.verify_assets(model,selection['manifest_data'])
        manifest=selection['manifest_data']
        identities[variant]={'verified':verified,'repository':manifest['modelRepository'],
            'revision':manifest['modelRevision'],'gpt_sha256':manifest['files']['gpt.pth']['sha256']}
    if args.prompt_sha256!=PROMPT_SHA256:
        raise ValueError('Only the previously audited public PCM reference is served by this controller')
    prompt=dependency(args.prompt)
    prompt_info=synth.wav_info(prompt,prompt=True)
    if prompt_info['file_sha256']!=args.prompt_sha256:
        raise ValueError('Reviewed PCM prompt SHA256 mismatch')
    output=synth._independent_output(args.report_dir,workspace,(ROOT,stage_root,*models.values(),prompt))
    if output is None:
        raise ValueError('A fresh report directory is required')
    graphs=[build_graph(i,models[case[1]],stage['port']) for i,case in enumerate(CASES)]
    report={'status':'metadata_verified_no_server_started','quality_verified':False,
        'speech_completeness_verified':False,'server_launched':False,'production_post_used':False,
        'models':identities,'prompt':prompt_info,'framework_revision':stage['framework_revision'],
        'preserve_reviewed_ops':stage['preserve_reviewed_ops'],'framework_runtime_equivalence_verified':False,
        'original_node_ids_and_types_preserved':True,'original_links_preserved':True,
        'explicit_case_value_changes':['node1.custom_model_path selects independent verified base/Hindi directory',
            'node2.url serves the fixed audited public PCM prompt on this owned loopback service',
            'node4.text/language use the fixed public functional test sentences',
            'node5.filename_prefix separates fresh outputs; encrypt=false and public zero-key placeholder',
            'extra_pnginfo is omitted; the original user AES key is never read or stored'],
        'graph_cache':'none; T8 model/reference caches remain active',
        'infrastructure_differences':['isolated loopback service/dirs/database','original user loader auto settings retained',
            'derivative compatibility forces full asset hashes despite verify_hashes=false',
            'cache-only real-normalizer adapter with per-instance cache dirs','offline loopback-only Python networking',
            'bounded read-only send_sync and actual registered-loader cache telemetry','strict determinism and TF32 off'],
        'runs':[{'case':c[0],'variant':c[1],'language':c[2],'status':'not_executed',
                 'graph_sha256':digest_json(graphs[i])} for i,c in enumerate(CASES)]}
    return report,(stage,stage_root,models,prompt,output,graphs)


def child_environment(stage,gpu_uuid):
    # Do not inherit auth/proxy/netrc-related environment; do not print values.
    env={k:os.environ[k] for k in ('PATH','LD_LIBRARY_PATH','LANG','LC_ALL') if k in os.environ}
    env.update(stage['cache_environment'])
    env.update(PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'),CUDA_VISIBLE_DEVICES=gpu_uuid,
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',USE_MODELSCOPE='false',
        TOKENIZERS_PARALLELISM='false',TORCH_COMPILE_DISABLE='1',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return env


def run(args):
    report,state=preflight(args)
    if not args.execute:
        return report
    if sys.platform!='linux' or not 1<=args.run_timeout<=600 or not 1<=args.startup_timeout<=180:
        raise ValueError('Execution requires Linux and bounded startup/run deadlines')
    if args.max_used_gpu_mib is None or not 0<args.max_used_gpu_mib<=24576:
        raise ValueError('An explicit GPU total-used cap at most 24 GiB is required')
    stage,stage_root,models,prompt,output,graphs=state
    if min(shutil.disk_usage(stage_root).free,shutil.disk_usage(output.parent).free)<3*1024**3:
        raise ValueError('At least 3 GiB free owned runtime/output storage is required')
    admission=synth.resource_admission(args)
    gpu_uuid=admission[0]['uuid']
    probe.queue_clear('http://127.0.0.1:8188')
    with socket.socket() as availability:
        availability.bind(('127.0.0.1',stage['port']))
    output.mkdir(mode=0o700)
    report.update(status='controller_started',controller_id=str(uuid.uuid4()),admission=admission)
    atomic_json(output/'result.json',report)
    nonce=os.urandom(32).hex()
    config={'format':'comfy-owned-validation-bootstrap-v1','stage_root':str(stage_root),
        'controller_pid':os.getpid(),'controller_id':report['controller_id'],'nonce':nonce,
        'prompt':str(prompt),'prompt_sha256':args.prompt_sha256,'graph_hashes':[digest_json(g) for g in graphs]}
    config_path=output/'bootstrap.json';atomic_json(config_path,config);config_path.chmod(0o600)
    process=None;owned_ids=set();started=time.monotonic()
    try:
        import stage_comfy_validation as staging
        staging.verify_stage(stage_root)
        with (output/'server.log').open('xb') as log:
            process=subprocess.Popen([stage['paths']['python'],str(ROOT/'scripts/comfy_validation_server.py'),str(config_path)],
                cwd=stage['paths']['runtime'],env=child_environment(stage,gpu_uuid),stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
            client=OwnedHTTP(process,stage['port'],nonce)
            report.update(server_launched=True,server_pid=process.pid)
            deadline=time.monotonic()+args.startup_timeout
            while True:
                if time.monotonic()>=deadline or process.poll() is not None:
                    raise ValueError('Owned server startup failed/timed out; inspect private log')
                runtime_gate(args,gpu_uuid)
                try:
                    status=client.request('GET','/validation/state')
                    if status.get('controller_id')==report['controller_id'] and status.get('server_pid')==process.pid and status.get('ready') is True:
                        break
                except (OSError,ValueError):
                    pass  # Startup GET only; no prompt has been submitted.
                time.sleep(.5)
            if status.get('cache_entries'):
                raise ValueError('Fresh owned service unexpectedly already contains a model')
            for class_type in set(CLASS_TYPES.values()):
                object_info=client.request('GET','/object_info/'+class_type)
                if class_type not in object_info:
                    raise ValueError('Original graph node type is missing from the actual isolated server')
            for index,row in enumerate(report['runs']):
                runtime_gate(args,gpu_uuid)
                check_queue(client.request('GET','/queue'),owned_ids)
                row.update(status='submission_started',submitted=True)
                atomic_json(output/'result.json',report)
                deadline=time.monotonic()+args.run_timeout
                # Kill only the owned service even if a polling syscall stalls.
                import threading
                watchdog=threading.Timer(args.run_timeout,lambda: stop_owned(process,force=True))
                watchdog.daemon=True;watchdog.start()
                try:
                    prompt_id=client.submit_once(graphs[index],report['controller_id'])
                    if prompt_id in owned_ids:
                        raise ValueError('Server reused a prompt identity')
                    owned_ids.add(prompt_id);row.update(prompt_id=prompt_id,status='queued')
                    atomic_json(output/'result.json',report)
                    while True:
                        if time.monotonic()>=deadline or process.poll() is not None:
                            raise ValueError('Owned prompt deadline/process failure')
                        sample=runtime_gate(args,gpu_uuid)
                        check_queue(client.request('GET','/queue'),owned_ids)
                        descriptor=completed_history(client.request('GET','/history/'+prompt_id),prompt_id,graphs[index])
                        if descriptor is not None:
                            break
                        time.sleep(.5)
                    observed=client.request('GET','/validation/state')
                    if observed.get('controller_id')!=report['controller_id'] or observed.get('server_pid')!=process.pid:
                        raise ValueError('Owned process identity changed')
                    cached=evidence(observed,prompt_id,models[row['variant']])
                    if cached.get('model_revision')!=report['models'][row['variant']]['revision'] or not cached.get('model_fingerprint'):
                        raise ValueError('Actual registered cache revision/fingerprint differs from the verified model identity')
                    path=audio_path(descriptor,Path(stage['paths']['output']),index)
                    audio=synth.wav_info(path)
                    if not 1<=audio['seconds']<=20:
                        raise ValueError('Audio is outside the fixed short-case bound; never truncate it')
                    viewed=client.request('GET','/view?'+urlencode(descriptor),binary=True)
                    if hashlib.sha256(viewed).hexdigest()!=audio['file_sha256']:
                        raise ValueError('Final node5 view bytes differ from its fresh local output')
                    row.update(status='completed_quality_unverified',audio=audio,descriptor=descriptor,
                        actual_cache=cached,cache_snapshot=observed['cache_entries'],runtime_sample=sample,server_pid=process.pid,
                        active_nodes_executed=sorted(ACTIVE),view_sha256=hashlib.sha256(viewed).hexdigest())
                    atomic_json(output/'result.json',report)
                finally:
                    watchdog.cancel()
            first,last=report['runs'][0],report['runs'][2]
            retained=[entry for entry in report['runs'][1]['cache_snapshot'] if entry.get('model_dir')==str(models['base'])]
            report.update(status='three_run_same_process_cycle_completed_quality_unverified',
                original_cache_object_observed_in_middle=len(retained)==1 and retained[0].get('model_id')==first['actual_cache']['model_id'],
                original_cache_object_reused=len(retained)==1 and retained[0].get('model_id')==first['actual_cache']['model_id']==last['actual_cache']['model_id'],
                base_completed_runs_observed=[first['actual_cache'].get('completed_runs'),last['actual_cache'].get('completed_runs')],
                base_repeat_pcm_identical=first['audio']['pcm_sha256']==last['audio']['pcm_sha256'],
                same_server_pid_all_runs=all(r['server_pid']==process.pid for r in report['runs']),
                elapsed_seconds=time.monotonic()-started)
            atomic_json(output/'result.json',report)
        return report
    except BaseException as exc:
        report.update(status='cycle_failed_not_verified',error_type=type(exc).__name__,
            reason=str(exc)[:1000],partial_outputs_are_not_success=True)
        for row in report['runs']:
            if row['status'] not in {'completed_quality_unverified','not_executed'}:
                row['status']='failed_or_submission_uncertain'
        atomic_json(output/'result.json',report)
        raise
    finally:
        try:
            stop_owned(process)
            report['owned_process_cleanup_verified']=process is None or process.poll() is not None
            report['owned_process_returncode']=None if process is None else process.poll()
            atomic_json(output/'result.json',report)
        finally:
            # Remove only our ephemeral HTTP auth nonce. No original AES key is ever read.
            if config_path.exists():
                config_path.unlink()


if __name__=='__main__':
    raise SystemExit(main())
