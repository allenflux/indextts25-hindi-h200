"""Private bounded launcher: no model/data downloads or ComfyUI mutations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import urllib.request
from types import SimpleNamespace

from hi_h200_native.synth import resource_admission, parse_gpu_sample, available_ram_bytes
from hi_h200_native.train import read_config


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['fresh2', 'resume100'], required=True)
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    options = read_config(config)
    run = Path(options['output_dir']).resolve()
    if options['lora'].get('training_profile') != 'attention_lora_only_v1':
        raise ValueError('Only a new pure attention profile is allowed')
    if options['loop']['device'] != 'cuda:0' or options['loop']['precision'] != 'bf16' or options['loop']['max_updates'] != 100:
        raise ValueError('This finite acceptance permits exactly 100 planned BF16 updates')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Exactly GPU 0 must be explicitly masked')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TORCH_FORCE_WEIGHTS_ONLY_LOAD'):
        if os.environ.get(key) != '1':
            raise ValueError('Required offline/restricted-load environment missing')
    if not queue_idle():
        raise ValueError('ComfyUI is busy; no launch')
    report_path = config.parent / ('controller-' + args.stage + '.json')
    log_path = config.parent / ('trainer-' + args.stage + '.log')
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
        command += ['--resume', str(checkpoint)]
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
    expected_step = 2 if args.stage == 'fresh2' else 100
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
        except Exception:
            reason = 'requested_step_or_artifact_unverified'
    result = dict(stage=args.stage, trainer_exit_code=code, own_trainer_exited=child.poll() is not None,
                  stop_reason=reason, elapsed_seconds=time.monotonic()-started, samples=samples,
                  verified_step=expected_step if reason is None and code == 0 else None,
                  quality_gate_passed=False, comfyui_service_modified=False)
    write_new(report_path, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
    return 0 if code == 0 and reason is None else 2


if __name__ == '__main__':
    sys.exit(main())
