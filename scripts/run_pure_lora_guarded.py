"""Private bounded launcher: no model/data downloads or ComfyUI mutations."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import signal
import sys
import time
import urllib.request
from types import SimpleNamespace

from hi_h200_native.synth import resource_admission, parse_gpu_sample, available_ram_bytes
from hi_h200_native.train import read_config, _dataset_paths, protect_output

ROOT = Path(__file__).resolve().parents[1]
PLAN300 = "pilot48-300-v1"
VAL_STEPS300 = [50, 100, 150, 200, 250, 300]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Loopback queue redirects are forbidden')


def queue_idle():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open('http://127.0.0.1:8188/queue', timeout=3) as stream:
        payload = stream.read(1024*1024+1)
    if len(payload) > 1024*1024:
        raise ValueError('Oversized queue response')
    queue = json.loads(payload)
    if not isinstance(queue, dict) or any(key not in queue or not isinstance(queue[key], list)
            for key in ('queue_running', 'queue_pending')):
        raise ValueError('Invalid queue schema; idle is unproven')
    return not queue['queue_running'] and not queue['queue_pending']


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def signal_child(child, force=False):
    try:
        child.kill() if force else child.terminate()
    except ProcessLookupError:
        pass  # A naturally completed child is still reaped with wait below.


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def bounded_json(path, maximum=16 * 1024**2):
    from hi_h200.data import load_json
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError('Missing, linked or oversized guarded metadata')
    value = load_json(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Guarded metadata must be an object')
    return value


def plan300(args, config, options):
    """Explicit new experiment only; legacy100 is deliberately not widened."""
    if args.plan != PLAN300:
        if args.stage == 'resume300' or options['loop']['max_updates'] != 100:
            raise ValueError('Legacy acceptance still permits exactly 100 planned updates')
        return None
    if args.stage not in ('fresh2', 'resume300'):
        raise ValueError('The 300-update plan requires fresh2 or resume300')
    if args.workspace_root is None or args.config_sha256 is None:
        raise ValueError('The new plan requires explicit frozen WORK and config SHA256')
    work = args.workspace_root.absolute()
    if work.resolve(strict=True) != work or not work.is_dir() or work == Path(work.anchor):
        raise ValueError('Use a canonical dedicated WORK')
    if config != args.config.absolute() or not config.is_relative_to(work) or not re.fullmatch('[0-9a-f]{64}', args.config_sha256) or digest(config) != args.config_sha256:
        raise ValueError('New plan config must be canonical, confined and match its independently frozen SHA')
    loop = {'max_updates':300, 'grad_accumulation':4, 'seed':260618, 'device':'cuda:0', 'precision':'bf16',
            'grad_clip':1.0, 'save_interval':50, 'validation_interval':50, 'max_validation_examples':8}
    optimizer = {'learning_rate':.0001, 'weight_decay':.01, 'warmup_updates':10, 'min_lr_ratio':.1}
    lora = {'training_profile':'attention_lora_only_v1', 'rank':8, 'alpha':16.0, 'dropout':0.0, 'text_loss_weight':.2}
    if options['loop'] != loop or options['optimizer'] != optimizer or options['lora'] != lora:
        raise ValueError('Config differs from the explicitly fixed pilot48-300 plan')
    dataset_configs = [Path(options[name]) for name in ('train_dataset','val_dataset')]
    paths = [_dataset_paths(path) for path in dataset_configs]
    for split, path, values in zip(('train','val'), dataset_configs, paths):
        if path.resolve(strict=True) != path or not path.is_relative_to(work) or bounded_json(path).get('split') != split:
            raise ValueError('TRAIN/VAL configs must be canonical frozen split inputs in WORK')
        if any(not value.is_relative_to(work) for name, value in values.items() if name not in ('model_dir','model_manifest')):
            raise ValueError('Source and data must stay in WORK; only original model dependencies may use the authorized share')
    if paths[0]['official_repo'] != paths[1]['official_repo'] or paths[0]['model_manifest'] != paths[1]['model_manifest']:
        raise ValueError('TRAIN/VAL original source and base must match')
    manifest = bounded_json(paths[0]['model_manifest'])
    if manifest.get('repository') != 'IndexTeam/IndexTTS-2.5' or manifest.get('snapshot_revision') != 'c39ce5ba981572cb187443877ff559dfb246ce63' or manifest.get('files',{}).get('gpt.pth') != '43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85':
        raise ValueError('The optimization must start from the fixed original base')
    run = Path(options['output_dir'])
    if not run.is_relative_to(work) or run == work:
        raise ValueError('New run must be a dedicated WORK descendant')
    protect_output(run, [ROOT, config, *dataset_configs, *[value for values in paths for value in values.values()]])
    spec = {'format':'hi_h200_native_attention_lora_only_v1', 'hindi_row':17, 'text_rows':[], 'text_loss_weight':.2,
        'lora':{'rank':8,'alpha':16.0,'dropout':0.0}, 'lora_targets':[f'gpt.h.{i}.attn.{name}' for i in range(24) for name in ('c_attn','c_proj')],
        'heads':[], 'model_dim':1280, 'training_profile':'attention_lora_only_v1', 'frozen_embeddings':True}
    dataset_inputs = {}
    for split, values, count in zip(('train','val'),paths,(48,8)):
        if any(not values[name].is_file() or not 0 < values[name].stat().st_size <= 16 * 1024**2 for name in ('feature_manifest','provenance')):
            raise ValueError('Dataset identity metadata exceeds its bound')
        with values['feature_manifest'].open('rb') as stream:
            if sum(bool(line.strip()) for line in stream) != count:
                raise ValueError('The bounded optimization requires exactly TRAIN48/VAL8 features')
        dataset_inputs[split] = {name:digest(values[name]) for name in ('feature_manifest','provenance')}
    record = {'format':'guarded-pilot48-300-v1', 'workspace_root':str(work), 'code_root':str(ROOT.resolve()), 'config_sha256':args.config_sha256,
        'resolved_config':options, 'dataset_config_sha256':{split:digest(path) for split,path in zip(('train','val'),dataset_configs)},
        'dataset_inputs_sha256':dataset_inputs, 'base_manifest_sha256':digest(paths[0]['model_manifest']),
        'native_code_sha256':{name:digest(ROOT/'src/hi_h200_native'/name) for name in ('model.py','data.py','engine.py','train.py')},
        'controller_sha256':digest(Path(__file__)), 'adapter_spec':spec, 'max_updates':300, 'validation_steps':VAL_STEPS300,
        'selection':'minimum_mean_val8_then_earliest_step', 'test_used_for_selection':False,
        'original_step100_candidate_unchanged':True, 'quality_gate_passed':False}
    return {'path':config.parent/('controller-'+PLAN300+'.plan.json'), 'record':record}


def verify_plan_state(plan, stage):
    path = plan['path']
    if stage == 'fresh2':
        if path.exists() or path.is_symlink():
            raise ValueError('Fresh optimization plan must be new; never rewrite previous history')
    elif bounded_json(path) != plan['record']:
        raise ValueError('Frozen optimization root/config/source/plan changed')


def verify_native_terminal(run, expected_step, options, plan):
    """Read this own job's complete checkpoint/adapter on CPU; never restore/train."""
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('Terminal audit must not reuse a CUDA context')
    torch.set_num_threads(1)
    from hi_h200_native.checkpoint_adapter import STATE_KEYS
    from hi_h200_native.engine import _finite_tree, _runtime
    checkpoint = run / ('checkpoints/step' + str(expected_step).zfill(8) + '.pt')
    adapter = run / ('adapters/step' + str(expected_step).zfill(8) + '.pt')
    for path in (checkpoint, adapter):
        if path.is_symlink() or path.resolve(strict=True) != path or not 0 < path.stat().st_size <= 64 * 1024**2:
            raise ValueError('Expected bounded canonical own native artifacts')
    before = {path:digest(path) for path in (checkpoint,adapter)}
    for path in (checkpoint,adapter):
        sidecar=bounded_json(path.with_suffix(path.suffix+'.sha256.json'),4096)
        if sidecar != {'schema_version':1,'step':expected_step,'sha256':before[path]}:
            raise ValueError('Actual checkpoint/adapter SHA sidecar differs')
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    payload = torch.load(adapter,map_location='cpu',weights_only=True)
    if any(digest(path) != value for path,value in before.items()):
        raise ValueError('Native terminal artifacts changed during restricted CPU audit')
    _finite_tree(state);_finite_tree(payload)
    record = plan['record']
    identity = state.get('identity',{}) if isinstance(state,dict) else {}
    run_manifest = bounded_json(run.parent/(run.name+'.native-run.json'))
    if (not isinstance(state,dict) or set(state) != STATE_KEYS or state['schema_version'] != 1 or state['kind'] != 'hi_h200_native_training'
            or state['step'] != expected_step or state['config'] != options['loop']
            or identity.get('config_sha256') != record['config_sha256'] or identity.get('resolved_config') != options
            or identity.get('native_code_sha256') != record['native_code_sha256'] or run_manifest.get('identity') != identity
            or run_manifest.get('schema_version') != 1 or run_manifest.get('kind') != 'hi_h200_native_run'
            or state['adapter_spec'] != record['adapter_spec'] or identity.get('adapter_spec') != record['adapter_spec']):
        raise ValueError('Actual checkpoint step/config/source/plan identity differs')
    if (not isinstance(payload,dict) or set(payload) != {'schema_version','kind','step','base_identity','adapter_spec','trainable'}
            or payload['schema_version'] != 1 or payload['kind'] != 'hi_h200_native_adapter' or payload['step'] != expected_step
            or payload['base_identity'] != state['model_identity'] or payload['adapter_spec'] != record['adapter_spec']
            or state['model_identity'].get('model_manifest_sha256') != record['base_manifest_sha256']):
        raise ValueError('Actual pure adapter differs from its checkpoint/base identity')
    datasets = identity.get('datasets',{})
    if (len(datasets.get('train',{}).get('feature_files',{})) != 48 or len(datasets.get('val',{}).get('feature_files',{})) != 8
            or datasets['train'].get('split') != 'train' or datasets['val'].get('split') != 'val'):
        raise ValueError('Optimization requires exactly the fixed TRAIN48/VAL8 fold')
    expected = {}
    for target in record['adapter_spec']['lora_targets']:
        expected['base.'+target+'.lora_A']=(8,1280)
        expected['base.'+target+'.lora_B']=(3840 if target.endswith('.c_attn') else 1280,8)
    for values in (state['trainable'],payload['trainable']):
        if not isinstance(values,dict) or set(values) != set(expected):
            raise ValueError('Terminal pure state must contain exactly 96 attention A/B tensors')
        for name,shape in expected.items():
            value=values[name]
            if not torch.is_tensor(value) or value.dtype != torch.float32 or value.layout != torch.strided or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError('Invalid finite FP32 terminal A/B tensor')
    if any(not torch.equal(state['trainable'][name].contiguous().view(torch.uint8),payload['trainable'][name].contiguous().view(torch.uint8)) for name in expected):
        raise ValueError('Actual checkpoint and standalone adapter tensors differ')
    if not isinstance(state['runtime'],dict) or set(state['runtime']) != set(_runtime()) or not isinstance(state['rng'],dict) or set(state['rng']) != {'python','numpy','torch_cpu','torch_cuda'}:
        raise ValueError('Checkpoint is missing complete runtime/RNG schema')
    groups=state['parameter_groups'];optimizer=state['optimizer']
    if groups != [list(expected),[]] or not isinstance(optimizer,dict) or set(optimizer) != {'state','param_groups'} or set(optimizer['state']) != set(range(96)) or len(optimizer['param_groups']) != 2 or optimizer['param_groups'][0]['params'] != list(range(96)) or optimizer['param_groups'][1]['params'] != []:
        raise ValueError('Checkpoint optimizer inventory/order differs')
    for index,name in enumerate(expected):
        entry=optimizer['state'][index]
        if set(entry) != {'step','exp_avg','exp_avg_sq'} or not torch.is_tensor(entry['step']) or entry['step'].numel() != 1 or entry['step'].item() != expected_step:
            raise ValueError('Missing actual per-parameter optimizer step')
        for field in ('exp_avg','exp_avg_sq'):
            if not torch.is_tensor(entry[field]) or entry[field].shape != state['trainable'][name].shape or entry[field].dtype != torch.float32:
                raise ValueError('Incomplete actual optimizer moment state')
    sampler=state['sampler']
    if sampler.get('dataset_length') != 48 or sampler.get('micro_steps') != expected_step*4 or sampler.get('epoch',-1)*48+sampler.get('offset',-1) != expected_step*4 or state['scheduler'].get('last_epoch') != expected_step or state['scaler'] is not None:
        raise ValueError('Incomplete optimizer-boundary sampler/scheduler state')
    return {'checkpoint_sha256':before[checkpoint], 'adapter_sha256':before[adapter], 'step':expected_step,
            'restricted_cpu_tensor_audit':True, 'pure_ab_tensors':96, 'training_dataset_examples':48, 'validation_dataset_examples':8}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['fresh2', 'resume100', 'resume300'], required=True)
    parser.add_argument('--plan', choices=['legacy100', PLAN300], default='legacy100')
    parser.add_argument('--workspace-root', type=Path)
    parser.add_argument('--config-sha256')
    args = parser.parse_args(argv)
    config = args.config.resolve(strict=True)
    options = read_config(config)
    run = Path(options['output_dir']).resolve()
    if options['lora'].get('training_profile') != 'attention_lora_only_v1':
        raise ValueError('Only a new pure attention profile is allowed')
    optimization = plan300(args,config,options)
    if options['loop']['device'] != 'cuda:0' or options['loop']['precision'] != 'bf16':
        raise ValueError('Finite acceptance requires explicit BF16 CUDA 0')
    if optimization is not None:
        verify_plan_state(optimization,args.stage)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Exactly GPU 0 must be explicitly masked')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'):
        if os.environ.get(key) != '1':
            raise ValueError('Required offline/restricted-load environment missing')
    if not queue_idle():
        raise ValueError('ComfyUI is busy; no launch')
    suffix = (PLAN300+'-' if optimization is not None else '') + args.stage
    report_path = config.parent / ('controller-' + suffix + '.json')
    log_path = config.parent / ('trainer-' + suffix + '.log')
    if report_path.exists() or log_path.exists():
        raise ValueError('An existing stage must be inspected, never overwritten')
    admission = resource_admission(SimpleNamespace(allow_cuda=True, gpu_id='0', min_free_gpu_mib=48000., min_free_ram_mib=64000., max_utilization=5.))
    boot = "import runpy,sys,torch;torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;sys.argv=['hi_h200_native.train',*sys.argv[1:]];runpy.run_module('hi_h200_native.train',run_name='__main__')"
    command = [sys.executable, '-B', '-c', boot, '--config', str(config), '--allow-cuda']
    if args.stage == 'fresh2':
        if run.exists():
            raise ValueError('Fresh run must be new')
        command += ['--stop-after-updates', '2']
    else:
        checkpoint = run / 'checkpoints/step00000002.pt'
        if not checkpoint.is_file() or not checkpoint.with_suffix('.pt.sha256.json').is_file():
            raise ValueError('Actual durable step2 checkpoint and SHA sidecar required')
        if optimization is not None:
            previous = bounded_json(config.parent/('controller-'+PLAN300+'-fresh2.json'))
            if previous.get('verified_step') != 2 or previous.get('trainer_exit_code') != 0 or previous.get('own_trainer_exited') is not True or previous.get('stop_reason') is not None:
                raise ValueError('The new plan requires its actual successful fresh2 stage')
            current_audit = verify_native_terminal(run,2,options,optimization)
            if current_audit != previous.get('terminal_tensor_audit'):
                raise ValueError('Current step2 bytes differ from the actual successful fresh2 audit')
        command += ['--resume', str(checkpoint)]
    if optimization is not None and args.stage == 'fresh2':
        write_new(optimization['path'],optimization['record'])
    started = time.monotonic()
    samples, reason = [], None
    child = None
    requested = []
    signal.signal(signal.SIGTERM, lambda number, frame: requested.append('controller_sigterm'))
    try:
        with log_path.open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            print('OWN_TRAINER_PID', child.pid, flush=True)
            while child.poll() is None:
                try:
                    output = subprocess.run(['nvidia-smi', '--id', '0', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True, timeout=5)
                    gpu = parse_gpu_sample(output.stdout)
                    ram = available_ram_bytes()
                    elapsed = time.monotonic() - started
                    samples.append({'elapsed_seconds':elapsed, 'derived_free_gpu_mib':gpu['free_mib'], 'available_ram_bytes':ram})
                    if requested:
                        reason = requested[0]
                    elif gpu['uuid'] != admission[0]['uuid'] or gpu['free_mib'] < 32000 or ram is None or ram < 64000*1024**2:
                        reason = 'resource_headroom_or_identity_changed'
                    elif elapsed > 600:
                        reason = '600_second_deadline'
                    elif not queue_idle():
                        reason = 'comfyui_work_arrived'
                except Exception:
                    reason = 'monitor_unavailable'
                if reason:
                    if run.is_dir():
                        (run / 'STOP').touch(exist_ok=True)
                    signal_child(child)  # This exact child only; never ComfyUI.
                    break
                time.sleep(2)
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                reason = (reason or 'shutdown') + ':own_child_forced_exit'
                signal_child(child, force=True)
                code = child.wait(timeout=10)
    finally:
        if child is not None and child.poll() is None:
            if run.is_dir():
                (run / 'STOP').touch(exist_ok=True)
            signal_child(child)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                signal_child(child, force=True)
                child.wait(timeout=10)
    expected_step = 2 if args.stage == 'fresh2' else (300 if optimization is not None else 100)
    tensor_audit = None
    final = None
    if code == 0 and reason is None:
        try:
            final = json.loads(log_path.read_text().splitlines()[-1])
            checkpoint = run / ('checkpoints/step' + str(expected_step).zfill(8) + '.pt')
            sidecar = json.loads(checkpoint.with_suffix('.pt.sha256.json').read_text())
            adapter = run / ('adapters/step' + str(expected_step).zfill(8) + '.pt')
            if (final['step'] != expected_step or final['native_training'] is not True or
                Path(final['checkpoint']).resolve() != checkpoint or
                sidecar['step'] != expected_step or sidecar['sha256'] != digest(checkpoint) or
                Path(final['adapter']).resolve() != adapter or final['adapter_sha256'] != digest(adapter)):
                raise ValueError('Incomplete final state')
            if optimization is not None:
                current_plan = plan300(args,config,read_config(config))
                if current_plan['record'] != optimization['record']:
                    raise ValueError('Frozen optimization inputs changed while training')
                verify_plan_state(optimization,'resume300')
                tensor_audit = verify_native_terminal(run,expected_step,options,optimization)
        except Exception:
            reason = 'requested_step_or_artifact_unverified'
    result = dict(stage=args.stage, trainer_exit_code=code, own_trainer_exited=child.poll() is not None,
                  stop_reason=reason, elapsed_seconds=time.monotonic()-started, samples=samples,
                  verified_step=expected_step if reason is None and code == 0 else None,
                  quality_gate_passed=False, comfyui_service_modified=False)
    if optimization is not None:
        result.update(plan=PLAN300, plan_sha256=digest(optimization['path']), config_sha256=digest(config),
                      validation_steps=VAL_STEPS300, selection='minimum_mean_val8_then_earliest_step',
                      test_used_for_selection=False, terminal_tensor_audit=tensor_audit)
    write_new(report_path, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
    return 0 if code == 0 and reason is None else 2


if __name__ == '__main__':
    sys.exit(main())
