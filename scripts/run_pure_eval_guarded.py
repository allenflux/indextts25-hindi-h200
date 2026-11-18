#!/usr/bin/env python3
"""Bounded private pure-LoRA evaluation supervisor, never a quality verifier.

Default plans only. --execute permits exactly one of the two named modules,
without a shell. Queue/resource monitoring may stop only our own child. Read the
module's actual result/failure separately; exit zero alone is not acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))  # Bind parent imports before loading the shared helper.
_spec = importlib.util.spec_from_file_location('_reviewed_pure_training_guard', Path(__file__).with_name('run_pure_lora_guarded.py'))
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
MODULES = {'pure_overfit': 'hi_h200_native.pure_overfit', 'adapter_synth': 'hi_h200_native.adapter_synth'}
COMMON = {'workspace-root', 'adapter', 'adapter-sha256', 'output-dir', 'seed'}
OPTIONS = {'pure_overfit': COMMON | {'dataset', 'max-seconds'},
    'adapter_synth': COMMON | {'official-repo', 'model-dir', 'model-manifest', 'training-base-manifest',
        'prompt', 'language', 'text', 'max-text-tokens', 'max-mel-tokens'}}
REQUIRED = {'pure_overfit': COMMON - {'seed'} | {'dataset'},
    'adapter_synth': COMMON - {'seed'} | {'official-repo', 'model-dir', 'model-manifest', 'training-base-manifest', 'prompt', 'text'}}
PATH_OPTIONS = {'adapter', 'dataset', 'official-repo', 'model-dir', 'model-manifest', 'training-base-manifest', 'prompt'}
ADMISSION = ('gpu-id', 'min-free-gpu-mib', 'min-free-ram-mib', 'max-utilization')
DEADLINE = 600


def require(value, message):
    if not value:
        raise ValueError(message)


def existing(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path, 'Use canonical paths without symlink components')
    return path


def fresh(path, workspace, protected):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), 'Output must be fresh; inspect previous results instead of overwriting')
    require(path.parent.resolve(strict=True) == path.parent and path.is_relative_to(workspace), 'Fresh output parent must be canonical inside WORK')
    require(not any(path == root or path.is_relative_to(root) or root.is_relative_to(path) for root in protected),
            'Output overlaps protected source, model, inputs or the other output')
    return path


def module_options(module, argv):
    """No abbreviations, duplicates, equals syntax, arbitrary options or positionals."""
    values = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        require(token.startswith('--') and token[2:] in OPTIONS[module], 'Unknown or unsafe module argument')
        name = token[2:]
        require(name not in values and index + 1 < len(argv), 'Duplicate or missing module argument')
        value = argv[index + 1]
        require(not value.startswith('--') and '\x00' not in value, 'Invalid module argument value')
        values[name] = value
        index += 2
    require(REQUIRED[module].issubset(values), 'Missing required module arguments')
    require(re.fullmatch(r'[0-9a-f]{64}', values['adapter-sha256']), 'Explicit adapter SHA256 required')
    if module == 'adapter_synth':
        require(values.get('language', 'hi') == 'hi', 'Only the isolated Hindi synthesis route is permitted')
    return values


def plan(args):
    require(args.module in MODULES, 'Only pure_overfit or adapter_synth may be launched')
    require(re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)', args.gpu_id), 'One explicit GPU index or UUID is required')
    require(all(math.isfinite(value) for value in (args.min_free_gpu_mib, args.min_free_ram_mib, args.max_utilization))
            and args.min_free_gpu_mib >= 32000 and args.min_free_ram_mib >= 64000 and 0 <= args.max_utilization <= 100,
            'Admission requires at least 32000 MiB GPU free and 64000 MiB RAM, with finite utilization bound')
    work = existing(args.workspace_root)
    require(work.is_dir(), 'WORK must be an existing directory')
    argv = list(args.module_args)
    if argv[:1] == ['--']: argv.pop(0)
    values = module_options(args.module, argv)
    require(existing(values['workspace-root']) == work, 'Child and parent WORK must match')
    readonly = [existing(path) for path in args.readonly_dependency_root]
    require(all(path.is_dir() for path in readonly), 'Read-only dependency roots must be existing canonical directories')
    python_roots = [existing(path) for path in args.python_dependency_root]
    require(all(path.is_dir() and (path.is_relative_to(work) or any(path.is_relative_to(root) for root in readonly))
                for path in python_roots), 'Python dependency roots must be canonical directories inside WORK or an explicit read-only root')
    protected = [ROOT.resolve(), *readonly, *python_roots, *(existing(path) for path in args.protect_root)]
    for name in PATH_OPTIONS & values.keys():
        path = existing(values[name])
        shared_model = args.module == 'pure_overfit' and name in {'model-dir', 'model-manifest', 'training-base-manifest'}
        require(path.is_relative_to(work) or shared_model and any(path.is_relative_to(root) for root in readonly),
                'Only explicit original-model dependencies may use a declared read-only root')
        protected.append(path)
        values[name] = str(path)
    if args.module == 'pure_overfit':
        from hi_h200_native.train import _dataset_paths
        for name, path in _dataset_paths(Path(values['dataset'])).items():
            path = existing(path)
            shared_model = name in {'model_dir', 'model_manifest'}
            require(path.is_relative_to(work) or shared_model and any(path.is_relative_to(root) for root in readonly),
                    'Only dataset original-model dependencies may use a declared read-only root')
            protected.append(path)
    output = fresh(values['output-dir'], work, protected)
    report_dir = fresh(args.report_dir, work, [*protected, output])
    fresh(output, work, [*protected, report_dir])
    values.update({'workspace-root': str(work), 'output-dir': str(output)})
    forwarded = [item for name, value in values.items() for item in ('--' + name, value)]
    forwarded += ['--execute', '--allow-cuda']
    for name in ADMISSION:
        forwarded += ['--' + name, str(getattr(args, name.replace('-', '_')))]
    command = [sys.executable, '-B', '-m', MODULES[args.module], *forwarded]
    return dict(workspace=work, report_dir=report_dir, module_output=output, command=command, python_roots=python_roots,
        public={'status': 'planned_no_child_started', 'module': MODULES[args.module],
            'module_argv_sha256': hashlib.sha256(json.dumps(forwarded, separators=(',', ':')).encode()).hexdigest(),
            'deadline_seconds': DEADLINE, 'monitor_min_free_gpu_mib': 32000, 'monitor_min_free_ram_mib': 64000,
            'python_dependency_roots_sha256': hashlib.sha256(json.dumps([str(path) for path in python_roots], separators=(',', ':')).encode()).hexdigest(),
            'module_report_verified': False, 'quality_gate_passed': False, 'child_started': False})


def child_environment(gpu_id, python_roots=()):
    environment = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONINSPECT', 'PYTHONUSERBASE', 'PYTHONBREAKPOINT',
                 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN', 'MODELSCOPE_API_TOKEN',
                 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(name, None)
    environment.update(PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), *(str(path) for path in python_roots)]), PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
        TORCH_FORCE_WEIGHTS_ONLY_LOAD='1', CUDA_VISIBLE_DEVICES=gpu_id, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return environment


def stop_and_wait(child):
    if child.poll() is None:
        shared.signal_child(child)
    try:
        return child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        shared.signal_child(child, force=True)
        return child.wait(timeout=10)


def execute(args, prepared):
    require(shared.queue_idle(), 'ComfyUI queue is not idle; no launch')
    admission = shared.resource_admission(SimpleNamespace(allow_cuda=True, **{name.replace('-', '_'):getattr(args, name.replace('-', '_')) for name in ADMISSION}))
    require(shared.queue_idle(), 'ComfyUI work arrived during admission; no launch')
    target = prepared['report_dir']
    target.mkdir(mode=0o700, exist_ok=False)
    child, code, reason = None, None, None
    requested, samples = [], []
    previous = {}
    started = time.monotonic()
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, lambda number, frame: requested.append('controller_signal'))
        with (target / 'stdout.log').open('x') as log:
            child = subprocess.Popen(prepared['command'], stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=target, env=child_environment(args.gpu_id, prepared['python_roots']))
            while child.poll() is None:
                try:
                    elapsed = time.monotonic() - started
                    if requested:
                        reason = 'controller_signal'
                    elif elapsed >= DEADLINE:
                        reason = '600_second_deadline'
                    else:
                        result = subprocess.run(['nvidia-smi', '--id', args.gpu_id,
                            '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                            check=True, text=True, capture_output=True, timeout=5)
                        gpu, ram = shared.parse_gpu_sample(result.stdout), shared.available_ram_bytes()
                        samples.append({'elapsed_seconds': elapsed, 'derived_free_gpu_mib': gpu['free_mib'], 'available_ram_bytes': ram})
                        if gpu['uuid'] != admission[0]['uuid'] or gpu['free_mib'] < 32000 or ram is None or ram < 64000*1024**2:
                            reason = 'resource_headroom_or_identity_changed'
                        elif not shared.queue_idle():
                            reason = 'comfyui_work_arrived'
                except Exception:
                    reason = 'monitor_unavailable'
                if reason:
                    break
                time.sleep(2)
            code = stop_and_wait(child)
    except BaseException:
        reason = reason or 'controller_exception'
        raise
    finally:
        try:
            if child is not None and child.poll() is None:
                code = stop_and_wait(child)
        finally:
            for number, handler in previous.items(): signal.signal(number, handler)
            if child is not None:
                if reason is not None:
                    status = 'stopped_module_result_unverified'
                elif code == 0:
                    status = 'child_exit0_module_report_unverified'
                elif code == 3 and args.module == 'pure_overfit':
                    status = 'child_exit3_mechanism_gate_failed_report_unverified'
                else:
                    status = 'child_failed_module_report_unverified'
                report = dict(prepared['public'], status=status, child_started=True, child_pid=child.pid,
                    child_exit_code=code, own_child_exited=child.poll() is not None, stop_reason=reason,
                    elapsed_seconds=time.monotonic()-started, samples=samples, comfyui_service_modified=False)
                shared.write_new(target / 'controller.json', report)
    if code == 0 and reason is None: return report, 0
    if code == 3 and args.module == 'pure_overfit' and reason is None: return report, 3
    return report, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--module', choices=tuple(MODULES), required=True)
    for name in ('workspace-root', 'report-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protect-root', type=Path, action='append', default=[])
    parser.add_argument('--readonly-dependency-root', type=Path, action='append', default=[],
                        help='Authorized pure_overfit original model roots and overlay scope; adapter_synth still requires all assets in WORK')
    parser.add_argument('--python-dependency-root', type=Path, action='append', default=[],
                        help='Explicit existing trusted overlay under WORK or declared read-only root; source remains first')
    parser.add_argument('--gpu-id', required=True)
    for name in ('min-free-gpu-mib', 'min-free-ram-mib', 'max-utilization'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('module_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        prepared = plan(args)
        report, code = execute(args, prepared) if args.execute else (prepared['public'], 0)
        print(json.dumps({key:value for key,value in report.items() if key != 'samples'}, sort_keys=True, allow_nan=False))
        return code
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status':'supervisor_blocked_not_verified', 'failure_type':type(exc).__name__,
                          'quality_gate_passed':False, 'module_report_verified':False}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
