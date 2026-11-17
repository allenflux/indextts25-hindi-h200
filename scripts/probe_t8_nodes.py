#!/usr/bin/env python3
"""Opt-in isolated four-node T8 probe; never queues or executes a whole workflow."""
from __future__ import annotations

import argparse
import array
from contextlib import ExitStack
import hashlib
import http.client
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import unicodedata
from urllib.parse import urlsplit
import warnings
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from hi_h200_native import synth

SAMPLING = dict(do_sample=True, temperature=.8, top_p=.8, top_k=30, num_beams=3,
    repetition_penalty=10., length_penalty=0., max_mel_tokens=1504, diffusion_steps=25,
    inference_cfg_rate=.7, cfm_temperature=1., segmentation_mode='auto',
    max_text_tokens_per_segment=120, segment_silence_ms=200, pause_preset='off',
    comma_pause_ms=100, sentence_pause_ms=300, paragraph_pause_ms=600, text_normalization=True)
GENERATE = dict(duration_factor=1., target_duration_mode='off', target_duration_seconds=0.,
    postprocess_preset='off', postprocess_strength=1., seed=1051829406, quality_retry_count=0,
    quality_asr_backend='auto', quality_asr_model='base', quality_asr_device='auto', quality_threshold=.82)
LOADER = dict(model_name='IndexTTS-2.5', device='cuda:0', precision='bfloat16',
    acceleration_mode='off', use_cuda_kernel=False, release_after_run=False,
    recycle_after_runs=0, verify_hashes=True, reference_device='auto',
    reuse_spk_cond_for_emo=False, persistent_reference_cache=False,
    download_missing=False, accept_model_license=False)
DIFFERENCES = {'device': ['auto', 'cuda:0 (one explicitly selected physical GPU)'],
    'precision': ['auto', 'bfloat16'], 'custom_model_path': ['', 'isolated selected model directory'],
    'verify_hashes': [False, True], 'persistent_reference_cache': [True, False],
    'frontend_cache': 'cache-only real-normalizer adapter, explicit run-local cache_dir',
    'execution': 'four real node execute calls in a child process; no ComfyUI queue or whole workflow',
    'determinism': 'fixed original seed; strict algorithms and TF32 disabled for this process'}
CLASSES = ('T8IndexTTS25ModelLoader', 'T8IndexTTS25EmotionControl',
           'T8IndexTTS25SamplingConfig', 'T8IndexTTS25Generate')


def canonical_text(value):
    if not isinstance(value, str) or not value or len(value) > 160 or any(c in value for c in '<>|'):
        raise ValueError('Use one bounded literal text without special/annotation tokens')
    if unicodedata.normalize('NFC', value) != value or ' '.join(value.split()) != value:
        raise ValueError('Text must already use NFC and canonical whitespace')
    if any(unicodedata.category(c) in {'Cc', 'Cs'} or (unicodedata.category(c) == 'Cf' and c not in '\u200c\u200d') for c in value):
        raise ValueError('Control/surrogate/format characters except ZWJ/ZWNJ are forbidden')
    return value


def queue_clear(url):
    """Only a bounded read of the local server queue, without proxy/auth configuration."""
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', '::1'} or parsed.username or parsed.password or parsed.path not in {'', '/'} or parsed.query or parsed.fragment:
        raise ValueError('Queue observation requires a literal loopback HTTP origin')
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 8188, timeout=5)
    try:
        connection.request('GET', '/queue')
        response = connection.getresponse()
        data = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(data) > 1024 * 1024:
            raise ValueError('Queue observation failed or exceeded its bound')
        value = json.loads(data)
        if not isinstance(value, dict) or any(not isinstance(value.get(k), list) or value[k] for k in ('queue_running', 'queue_pending')):
            raise ValueError('ComfyUI queue is not currently empty')
        return {'running': 0, 'pending': 0, 'observed_at_unix': time.time(), 'reservation': False}
    finally:
        connection.close()


def node_parameters(model_dir, text, language):
    if language not in {'HI', 'ZH', 'EN', 'JA', 'ES', 'AR'}:
        raise ValueError('Unsupported explicit probe language')
    return {'loader': dict(LOADER, custom_model_path=str(model_dir)),
            'emotion': {'mode': {'mode': 'speaker'}}, 'sampling': dict(SAMPLING),
            'generate': dict(GENERATE, text=canonical_text(text), language=language)}


def execute_nodes(nodes, parameters, audio, *, state=None):
    """The real node API path; injected test objects check the call contract only."""
    for name in CLASSES:
        if not hasattr(nodes, name):
            raise ValueError('Missing real T8 node class: ' + name)
    if state is not None:
        state['four_node_calls_started'] = True
    handle = nodes.T8IndexTTS25ModelLoader.execute(**parameters['loader'])[0]
    if not handle.use_bf16 or handle.use_fp16 or any(getattr(handle, k) for k in ('use_cuda_kernel', 'use_torch_compile', 'use_accel', 'use_deepspeed')) or handle.reference_cache_dir is not None or handle.model_dir.resolve() != Path(parameters['loader']['custom_model_path']).resolve():
        raise ValueError('Node loader did not retain the required isolated controls')
    emotion = nodes.T8IndexTTS25EmotionControl.execute(**parameters['emotion'])[0]
    sampling = nodes.T8IndexTTS25SamplingConfig.execute(**parameters['sampling'])[0]
    if state is not None:
        state['generate_node_call_attempted'] = True
    result = nodes.T8IndexTTS25Generate.execute(model=handle, speaker_audio=audio,
        emotion=emotion, sampling=sampling, **parameters['generate'])
    if len(result[2]) != 1:
        raise ValueError('Unexpected additional candidate generation')
    return result[0], result[1]


def write_audio(audio, path):
    import torch
    if not isinstance(audio, dict) or set(audio) != {'waveform', 'sample_rate'}:
        raise ValueError('Expected a real Comfy AUDIO mapping')
    waveform, rate = audio['waveform'], audio['sample_rate']
    if not isinstance(waveform, torch.Tensor) or not waveform.is_floating_point() or waveform.ndim != 3 or tuple(waveform.shape[:2]) != (1, 1) or type(rate) is not int or not 8000 <= rate <= 96000:
        raise ValueError('Expected one mono floating-point AUDIO tensor')
    value = waveform.detach().cpu()
    seconds = value.shape[-1] / rate
    if not 1 <= seconds <= 20 or not torch.isfinite(value).all().item() or value.abs().max().item() == 0:
        raise ValueError('Empty/nonfinite or out-of-bound audio is not a successful probe')
    pcm = array.array('h', (value.clamp(-1, 1).reshape(-1) * 32767).round().to(torch.int16).tolist())
    if sys.byteorder != 'little':
        pcm.byteswap()
    with Path(path).open('xb') as stream:
        with wave.open(stream, 'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(rate)
            output.writeframes(pcm.tobytes())
        stream.flush(); os.fsync(stream.fileno())
    return synth.wav_info(path)


def _probe_output(path, workspace, comfy, source, model, prompt):
    """Permit only a dedicated canonical Comfy output scratch, never its code/model trees."""
    scratch = False
    if workspace == comfy or workspace.is_relative_to(comfy):
        output_root = comfy / 'output'
        if (not output_root.is_dir() or output_root.is_symlink()
                or output_root.resolve(strict=True) != output_root
                or workspace == output_root or not workspace.is_relative_to(output_root)):
            raise ValueError('WORK inside ComfyUI must be a dedicated canonical output scratch directory')
        scratch = True
    protected = (ROOT, source, model, prompt) + (() if scratch else (comfy,))
    output = synth._independent_output(path, workspace, protected)
    return output, {'comfy_output_scratch_exception': scratch,
        'fresh_output_within_work_required': True, 'repo_node_models_and_prompt_protected': True}


def preflight(args):
    if args.model_variant == "base" and args.language == "HI":
        raise ValueError("This probe does not test untrained base Hindi; choose a declared base language")
    import check_t8_compat as compat
    workspace = args.workspace_root.resolve(strict=True)
    if args.workspace_root.absolute() != workspace:
        raise ValueError("WORK must have no symlink components")
    paths = (args.node_root, args.model_dir, args.comfy_root)
    if any(p.absolute() != p.resolve(strict=True) for p in paths):
        raise ValueError("Use real directories without symlink components")
    source, model, comfy = (p.resolve(strict=True) for p in paths)
    if not source.is_relative_to(workspace) or not model.is_relative_to(workspace):
        raise ValueError('The isolated node and candidate directory must be inside WORK')
    audit = compat.verify_prepared(source, args.candidate_revision)
    selection = compat.selected_manifest(source, model, args.candidate_revision, variant=args.model_variant)
    manifest = selection["manifest_data"]
    assets = compat.verify_assets(model, manifest)
    if not (comfy / 'folder_paths.py').is_file() or not (comfy / 'comfy_api').is_dir():
        raise ValueError('Read-only ComfyUI root with the real V3 API is required')
    prompt = args.prompt.resolve(strict=True)
    if not prompt.is_relative_to(workspace) or synth.sha256(prompt) != args.prompt_sha256:
        raise ValueError('Prompt must be a hash-bound local WORK input')
    prompt_info = synth.wav_info(prompt, prompt=True)
    parameters = node_parameters(model, args.text, args.language)
    output, output_policy = _probe_output(args.output_dir, workspace, comfy, source, model, prompt)
    if output is None:
        raise ValueError('A fresh output directory is required even for metadata planning')
    return {'status': 'metadata_verified_no_model_loaded', 'compatibility': {k:v for k,v in audit.items() if k != 'manifest_data'},
        'model_variant': args.model_variant, 'output_path_policy': output_policy,
        'selected_manifest': {k:v for k,v in selection.items() if k != 'manifest_data'},
        'selected_model_identity': {'repository':manifest['modelRepository'], 'revision':manifest['modelRevision'],
            'gpt_sha256':manifest['files']['gpt.pth']['sha256'], 'gpt_bytes':manifest['files']['gpt.pth']['size']},
        'asset_identity': assets, 'prompt': prompt_info, 'parameters': parameters,
        'parameters_sha256': hashlib.sha256(json.dumps(parameters, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'differences_from_original_four_node_parameters': DIFFERENCES,
        'whole_workflow_executed': False, 'comfy_queue_submitted': False,
        'quality_verified': False, 'gpu_executed': False, 'cuda_access_attempted': False,
        'four_node_calls_started': False, 'generate_node_call_attempted': False,
        'model_load_is_lazy_in_generate': True}, (source, model, comfy, prompt, output)


def execute(args, report, state):
    source, model, comfy, prompt, output = state
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise ValueError('Use a fresh process without an existing CUDA context')
    if any(name == 'indextts' or name.startswith('indextts.') for name in sys.modules):
        raise ValueError('Use a fresh process without any imported IndexTTS tree')
    queue = queue_clear(args.comfy_url)
    admission = synth.resource_admission(args)
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise ValueError("At least 3 GiB free output/cache storage is required")
    output.mkdir(mode=0o700)
    stage, previous_argv = 'runtime_import', list(sys.argv)
    try:
        with synth.isolated_runtime(output, source, admission[0]['uuid']) as env, ExitStack() as stack, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            sys.argv = [previous_argv[0]]
            sys.path.insert(0, str(comfy))
            stack.callback(sys.path.remove, str(comfy))
            report.update(cuda_access_attempted=True, gpu_executed=None)
            import folder_paths
            setter = getattr(folder_paths, 'set_temp_directory', None)
            if not callable(setter):
                raise ValueError('Real folder_paths.set_temp_directory is required for isolated reference caching')
            setter(str(output / 'runtime/tmp'))
            stack.enter_context(synth._runtime_import(source, env))
            import torch
            import numpy as np
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                raise ValueError('One explicitly selected native-BF16 CUDA GPU is required')
            seed = GENERATE['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True; torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True)
            package_name = '_hi_h200_t8_isolated_probe'
            spec = importlib.util.spec_from_file_location(package_name, source / '__init__.py', submodule_search_locations=[str(source)])
            package = importlib.util.module_from_spec(spec); sys.modules[package_name] = package
            spec.loader.exec_module(package)
            nodes = importlib.import_module(package_name + '.nodes_v3')
            from comfy_api.latest import io
            for name in CLASSES:
                cls = getattr(nodes, name)
                if not issubclass(cls, io.ComfyNode) or Path(inspect.getfile(cls)).resolve() != source / 'nodes_v3.py':
                    raise ValueError('Runtime node class did not come from the verified isolated copy')
            current = synth.resource_admission(args)
            if current[0]['uuid'] != admission[0]['uuid']:
                raise ValueError('Physical GPU identity changed before generation')
            admission.extend(current)
            import check_t8_compat as compat
            current_selection = compat.selected_manifest(source, model, args.candidate_revision, variant=args.model_variant)
            if {k:v for k,v in current_selection.items() if k != "manifest_data"} != report["selected_manifest"]:
                raise ValueError("Selected model manifest changed after metadata verification")
            compat.verify_assets(model, current_selection["manifest_data"])
            if synth.sha256(prompt) != report["prompt"]["file_sha256"]:
                raise ValueError("Prompt changed after metadata verification")
            with wave.open(str(prompt), 'rb') as stream:
                pcm = bytearray(stream.readframes(stream.getnframes())); rate = stream.getframerate()
            waveform = torch.frombuffer(pcm, dtype=torch.int16).clone().float().reshape(1, 1, -1) / 32768
            stage = 'four_real_node_execute_calls'
            synth._write_json(output / 'attempt_started.json', dict(report, status='attempt_started', admission=admission, queue=queue))
            torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
            audio, info = execute_nodes(nodes, report['parameters'], {'waveform': waveform, 'sample_rate': rate}, state=report)
            torch.cuda.synchronize()
            if any('max_mel_tokens' in str(w.message) and 'exceed' in str(w.message).lower() for w in captured):
                raise ValueError('Generation reached its mel limit; partial audio is not a successful probe')
            stage = 'validate_audio'
            audio_info = write_audio(audio, output / 'audio.wav')
            report.update(status='four_real_nodes_generated_audio_quality_unverified', gpu_executed=True,
                elapsed_seconds=time.monotonic()-started, output=audio_info, generation_info=info,
                admission=admission, queue_observation=queue, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                cache_only_normalizer_adapter=True, frontend_full_parity_verified=False,
                warnings=[str(w.message)[:1000] for w in captured])
            synth._write_json(output / 'result.json', report)
            return report
    except BaseException as exc:
        synth._write_json(output / 'failure.json', dict(report, status='probe_failed_not_verified',
            failure_stage=stage, error_type=type(exc).__name__, reason=str(exc)[:4000], partial_outputs_not_success=True))
        raise
    finally:
        sys.argv = previous_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace-root', 'node-root', 'model-dir', 'comfy-root', 'prompt', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--prompt-sha256', required=True)
    parser.add_argument('--candidate-revision', required=True, help='Previously verified immutable HF 40-character commit; no network lookup here')
    parser.add_argument('--text', required=True)
    parser.add_argument('--model-variant', choices=('base', 'hindi'), default='hindi')
    parser.add_argument('--language', choices=('HI', 'ZH', 'EN', 'JA', 'ES', 'AR'), default='HI')
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--allow-cuda', action='store_true')
    parser.add_argument('--gpu-id', default='0')
    for name in ('min-free-gpu-mib', 'min-free-ram-mib', 'max-utilization'):
        parser.add_argument('--' + name, type=float)
    args = parser.parse_args(argv)
    try:
        report, state = preflight(args)
        if args.execute:
            report = execute(args, report, state)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'blocked_or_failed', 'error_type':type(exc).__name__, 'reason':str(exc)[:1000]}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
