#!/usr/bin/env python3
"""Independent CPU audit of the frozen pilot48 pure-LoRA merge; never publish.

No model construction, CUDA, unsafe pickle fallback or speech-quality inference.
Default verifies bounded bundle metadata only; --execute-cpu verifies full hashes,
restricted mmap tensors, frozen bytes, and an independent exact FP32 merge.
Use an external process timeout as well as the cooperative internal deadline.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = 'd9e41aac89fd00b3d71497fddb287b7f24613712'
REPOSITORY = 'IndexTeam/IndexTTS-2.5'
REVISION = 'c39ce5ba981572cb187443877ff559dfb246ce63'
TOKENIZER = 'multilingual_zh_ja_yue_char_del.tiktoken'
MAX_WEIGHT_BYTES = 4 * 1024**3
MAX_METADATA_BYTES = 128 * 1024
BUNDLE_FILES = frozenset(('gpt.pth', 'provenance.json', 'README.md', 'LICENSE', 'LICENSE_ZH.txt', 'MODEL_LICENSE', 'NON_ENDORSEMENT.txt'))
PROVENANCE_KEYS = frozenset(('schema_version', 'kind', 'scope', 'official_commit', 'base_identity', 'adapter_sha256',
    'native_implementation_files', 'adapter_spec_sha256', 'optimizer_updates', 'gpt_sha256', 'preserved_base_sha256',
    'merge_semantics', 'official_tensor_count', 'checkpoint_container', 'official_key_shape_check',
    'full_pretrained_forward_parity_verified', 'synthesis_verified', 'old_language_behavior_guaranteed_unchanged'))
LICENSE_PINS = {
    'LICENSE': (10554, '53875532237e0a97b17a41721556ff57e93193fc566e48c3a0febe6e35bb5343'),
    'LICENSE_ZH.txt': (7336, '366175ecf329274358801b8365922677c52f25e643dcf00f0bc15857b5b2777f'),
    'MODEL_LICENSE': (10553, 'cc7da9ea0f8a97ef15ab3bf0389e636ce79ffca1aef5489520796ac87d87a87b'),
    'NON_ENDORSEMENT.txt': (251, 'ff80f462446489711a2ae9c828b1d933d6224261225b154df8265c9b86ae48d6')}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


@dataclass(frozen=True)
class Pins:
    base_bytes: int
    base_sha256: str
    adapter_bytes: int
    adapter_sha256: str
    manifest_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    layers: int = 24
    model_dim: int = 1280
    rank: int = 8
    alpha: float = 16.0
    text_loss_weight: float = 0.2
    step: int = 100
    tensor_count: int = 456
    model_file_count: int = 16
    scope: str = 'fixed_published_pilot48'


PRODUCTION = Pins(3259599833, '43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85',
    5931907, '15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd',
    '11dae1733635a0d46e7edbc7c5444900fed2bbb4450767aa4d2820b2221f0e6f',
    '18adf417be3e8f5e2e48e30f7420c719170a6870619436250f360d626877870e',
    '747979631e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def hex_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def check_deadline(deadline):
    require(time.monotonic() < deadline, 'Cooperative CPU audit deadline exceeded')


def regular(value, bound):
    path = Path(value).absolute()
    require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode), 'Use canonical regular files without symlink components')
    require(0 < path.stat().st_size <= bound, 'Input file exceeds its fixed byte bound')
    return path


def stamp(path):
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def digest(path, deadline):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            check_deadline(deadline)
            block = stream.read(8 * 1024**2)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def read_json(path):
    path = regular(path, MAX_METADATA_BYTES)
    raw = path.read_bytes()
    require(len(raw) <= MAX_METADATA_BYTES, 'Metadata grew beyond its bound')
    def unique(items):
        value = {}
        for name, item in items:
            require(name not in value, 'Duplicate metadata JSON key')
            value[name] = item
        return value
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(AuditError('Nonfinite metadata JSON')))
    require(isinstance(value, dict), 'Expected a metadata JSON object')
    return value


def modules(pins):
    return [f'gpt.h.{index}.attn.{name}' for index in range(pins.layers) for name in ('c_attn', 'c_proj')]


def expected_spec(pins):
    return {'format': 'hi_h200_native_attention_lora_only_v1', 'training_profile': 'attention_lora_only_v1',
        'hindi_row': 17, 'text_rows': [], 'heads': [], 'frozen_embeddings': True,
        'text_loss_weight': pins.text_loss_weight, 'lora': {'rank': pins.rank, 'alpha': pins.alpha, 'dropout': 0.0},
        'lora_targets': modules(pins), 'model_dim': pins.model_dim}


def identity_valid(identity, pins):
    keys = {'official_commit', 'model_repository', 'model_snapshot_revision', 'model_manifest_sha256',
        'model_files', 'config_sha256', 'tokenizer_sha256'}
    require(isinstance(identity, dict) and set(identity) == keys and identity['official_commit'] == OFFICIAL
        and identity['model_repository'] == REPOSITORY and identity['model_snapshot_revision'] == REVISION
        and identity['model_manifest_sha256'] == pins.manifest_sha256
        and identity['config_sha256'] == pins.config_sha256 and identity['tokenizer_sha256'] == pins.tokenizer_sha256,
        'Frozen official base identity differs')
    files = identity['model_files']
    require(isinstance(files, dict) and len(files) == pins.model_file_count and
        all(isinstance(name, str) and not name.startswith('/') and chr(92) not in name and '..' not in name.split('/')
            and hex_digest(value) for name, value in files.items()) and files.get('gpt.pth') == pins.base_sha256
        and files.get('config.yaml') == pins.config_sha256 and files.get(TOKENIZER) == pins.tokenizer_sha256,
        'Original base file references differ')


def inspect(args, *, _pins=PRODUCTION):
    """Fixture pins are private test injection; the CLI never exposes overrides."""
    for value in (args.candidate_sha256, args.complete_sha256, args.provenance_sha256):
        require(hex_digest(value), 'Independently recorded candidate/COMPLETE/provenance SHA256 values are required')
    require(type(args.candidate_bytes) is int and 0 < args.candidate_bytes <= MAX_WEIGHT_BYTES, 'Candidate byte bound is invalid')
    work = Path(args.workspace_root).absolute()
    require(work.resolve(strict=True) == work and work.is_dir() and work != Path(work.anchor), 'WORK must be a dedicated canonical existing directory')
    base = regular(args.base, MAX_WEIGHT_BYTES)
    adapter = regular(args.adapter, 32 * 1024**2)
    require(base.stat().st_size == _pins.base_bytes and adapter.stat().st_size == _pins.adapter_bytes,
        'Frozen official base or published adapter byte count differs')
    bundle = Path(args.merged_bundle).absolute()
    require(bundle.resolve(strict=True) == bundle and bundle.is_dir() and not bundle.is_symlink(), 'Use a canonical export bundle')
    require({p.name for p in bundle.iterdir()} == BUNDLE_FILES | {'COMPLETE.json'}, 'Export bundle inventory differs')
    files = {name: regular(bundle / name, MAX_WEIGHT_BYTES if name == 'gpt.pth' else MAX_METADATA_BYTES)
             for name in BUNDLE_FILES | {'COMPLETE.json'}}
    candidate = files['gpt.pth']
    require(len({(p.stat().st_dev, p.stat().st_ino) for p in (base, adapter, candidate)}) == 3, 'Base, adapter and candidate must be independent files')
    require(candidate.stat().st_size == args.candidate_bytes, 'Candidate recorded byte count differs')
    complete, provenance = read_json(files['COMPLETE.json']), read_json(files['provenance.json'])
    require(set(complete) == {'schema_version', 'files'} and type(complete['schema_version']) is int and complete['schema_version'] == 1
        and isinstance(complete['files'], dict) and set(complete['files']) == BUNDLE_FILES
        and all(hex_digest(value) for value in complete['files'].values()), 'Export COMPLETE schema differs')
    require(complete['files']['gpt.pth'] == args.candidate_sha256 and complete['files']['provenance.json'] == args.provenance_sha256,
        'Candidate/provenance SHA disagrees with COMPLETE')
    pins = {base: _pins.base_sha256, adapter: _pins.adapter_sha256,
        **{files[name]: value for name, value in complete['files'].items()}, files['COMPLETE.json']: args.complete_sha256}
    for name, (size, value) in LICENSE_PINS.items():
        require(files[name].stat().st_size == size and complete['files'][name] == value, 'Complete fixed licenses/non-endorsement are required')
    for path, value in pins.items():
        if path not in (base, adapter, candidate):
            require(digest(path, args._deadline) == value, 'Bounded export metadata SHA256 mismatch')
    require(set(provenance) == PROVENANCE_KEYS and type(provenance['schema_version']) is int and provenance['schema_version'] == 1
        and provenance['kind'] == 'hi_h200_native_merged_official_gpt' and provenance['official_commit'] == OFFICIAL
        and provenance['scope'] == 'Hindi-intended merged official GPT replacement; acceptance still required'
        and provenance['gpt_sha256'] == args.candidate_sha256 and provenance['adapter_sha256'] == _pins.adapter_sha256
        and type(provenance['optimizer_updates']) is int and provenance['optimizer_updates'] == _pins.step
        and type(provenance['official_tensor_count']) is int and provenance['official_tensor_count'] == _pins.tensor_count
        and provenance['checkpoint_container'] == 'model'
        and provenance['official_key_shape_check'] == 'actual official meta constructor, strict=True'
        and provenance['merge_semantics'] == 'evaluation/inference with native LoRA dropout disabled'
        and all(provenance[key] is False for key in ('full_pretrained_forward_parity_verified', 'synthesis_verified', 'old_language_behavior_guaranteed_unchanged')),
        'Pure export provenance schema or frozen artifact binding differs')
    require(isinstance(provenance['native_implementation_files'], dict) and set(provenance['native_implementation_files']) == {'model.py', 'export.py'}
        and all(hex_digest(value) for value in provenance['native_implementation_files'].values())
        and hex_digest(provenance['preserved_base_sha256']), 'Invalid recorded export source/preserved-state hashes')
    identity_valid(provenance['base_identity'], _pins)
    require(provenance['adapter_spec_sha256'] == hashlib.sha256(canonical(expected_spec(_pins))).hexdigest(), 'Pure adapter specification hash differs')
    output = Path(args.output).absolute()
    partial = output.with_suffix(output.suffix + '.partial')
    protected = [ROOT, base.parent, adapter.parent, bundle, *[Path(value).resolve(strict=True) for value in args.protect_root]]
    require(output.parent.resolve(strict=True) == output.parent and output.is_relative_to(work) and not output.exists()
        and not output.is_symlink() and not partial.exists() and not partial.is_symlink(), 'Output must be a fresh canonical file inside WORK')
    require(not any(output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in protected), 'Output overlaps source, model, adapter or export bundle')
    return {'base': base, 'adapter': adapter, 'candidate': candidate, 'bundle': bundle, 'output': output,
        'hash_pins': pins, 'snapshots': {path: stamp(path) for path in pins}, 'provenance': provenance, 'pins': _pins,
        'report': {'schema_version': 1, 'kind': 'independent_pure_attention_merge_audit',
            'status': 'metadata_verified_cpu_tensor_audit_pending', 'audit_passed': False,
            'scope': _pins.scope, 'base_sha256': _pins.base_sha256, 'adapter_sha256': _pins.adapter_sha256,
            'candidate_sha256': args.candidate_sha256, 'candidate_bytes': args.candidate_bytes,
            'complete_sha256': args.complete_sha256, 'provenance_sha256': args.provenance_sha256,
            'base_manifest_sha256': _pins.manifest_sha256, 'quality_verified': False,
            'forward_parity_verified': False, 'synthesis_verified': False, 'old_language_behavior_guaranteed_unchanged': False,
            'export_source_hashes': 'Recorded provenance declarations; not an independent source checkout verification',
            'deadline_policy': 'Cooperative chunk checks; use an external hard process timeout',
            'cpu_tensor_audit_executed': False, 'all_input_hashes_verified': False, 'cuda_initialized': False}}


def unchanged(prepared, deadline):
    require({p.name for p in prepared['bundle'].iterdir()} == BUNDLE_FILES | {'COMPLETE.json'}, 'Export inventory changed during inspection')
    for path, value in prepared['hash_pins'].items():
        require(stamp(path) == prepared['snapshots'][path] and digest(path, deadline) == value
            and stamp(path) == prepared['snapshots'][path], 'An input changed or failed its independently bound SHA256')


def supported(tensor, torch):
    return (torch.is_tensor(tensor) and tensor.device.type == 'cpu' and tensor.layout == torch.strided
        and not tensor.is_quantized and tensor.is_contiguous() and tensor.dtype in
        (torch.float32, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool))


def chunks(tensor, chunk_bytes):
    flat = tensor.detach().reshape(-1)
    step = max(1, chunk_bytes // tensor.element_size())
    for start in range(0, flat.numel(), step):
        yield flat[start:start + step]


def finite(tensor, torch, chunk_bytes, deadline):
    require(supported(tensor, torch), 'Unsupported/noncontiguous CPU state tensor')
    if tensor.is_floating_point():
        for value in chunks(tensor, chunk_bytes):
            check_deadline(deadline)
            require(bool(torch.isfinite(value).all()), 'Nonfinite CPU state tensor')


def byte_equal(left, right, torch, chunk_bytes, deadline):
    a = left.detach().reshape(-1).view(torch.uint8)
    b = right.detach().reshape(-1).view(torch.uint8)
    same = True
    for start in range(0, a.numel(), chunk_bytes):
        check_deadline(deadline)
        same = bool(torch.equal(a[start:start + chunk_bytes], b[start:start + chunk_bytes])) and same
    return same


def preserved_digest(base, pins, torch, chunk_bytes, deadline):
    rename = {target + '.' + kind: target + '.base.' + kind for target in modules(pins) for kind in ('weight', 'bias')}
    value = hashlib.sha256()
    for name, tensor in sorted(((rename.get(name, name), tensor) for name, tensor in base.items()), key=lambda item: item[0]):
        value.update(canonical([name, list(tensor.shape), str(tensor.dtype)]))
        for block in chunks(tensor, chunk_bytes):
            check_deadline(deadline)
            value.update(block.view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def available_ram():
    values = []
    path = Path('/proc/meminfo')
    if path.is_file():
        match = re.search(r'(?m)^MemAvailable:\s+(\d+)\s+kB', path.read_text())
        if match: values.append(int(match.group(1)) * 1024)
    for maximum, current in (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                             ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        a, b = Path(maximum), Path(current)
        if a.is_file() and b.is_file():
            limit = a.read_text().strip()
            if limit != 'max': values.append(max(0, int(limit) - int(b.read_text().strip())))
    return min(values) if values else None


def compare(prepared, base, candidate, adapter, torch, chunk_bytes, deadline):
    pins = prepared['pins']
    require(isinstance(base, dict) and isinstance(candidate, dict) and base and candidate
        and all(isinstance(name, str) for name in (*base, *candidate)), 'Expected complete tensor state dictionaries')
    require(len(base) == pins.tensor_count and set(base) == set(candidate), 'Official/candidate tensor key inventory differs')
    require(isinstance(adapter, dict) and set(adapter) == {'schema_version', 'kind', 'step', 'base_identity', 'adapter_spec', 'trainable'}
        and type(adapter['schema_version']) is int and adapter['schema_version'] == 1 and adapter['kind'] == 'hi_h200_native_adapter'
        and type(adapter['step']) is int and adapter['step'] == pins.step, 'Published pure adapter schema/step differs')
    identity_valid(adapter['base_identity'], pins)
    require(canonical(adapter['base_identity']) == canonical(prepared['provenance']['base_identity'])
        and canonical(adapter['adapter_spec']) == canonical(expected_spec(pins)), 'Pure adapter/base/provenance specification differs')
    state = adapter['trainable']; shapes = {}
    targets = {name + '.weight' for name in modules(pins)}
    for name in modules(pins):
        shapes['base.' + name + '.lora_A'] = (pins.rank, pins.model_dim)
        shapes['base.' + name + '.lora_B'] = (pins.model_dim * (3 if name.endswith('.c_attn') else 1), pins.rank)
    require(isinstance(state, dict) and set(state) == set(shapes), 'Adapter must contain exactly the complete pure attention A/B inventory')
    for name, shape in shapes.items():
        tensor = state[name]
        finite(tensor, torch, chunk_bytes, deadline)
        require(tensor.dtype == torch.float32 and tuple(tensor.shape) == shape, 'Adapter A/B FP32 shape differs')
    frozen_same = True; formula_same = True; changed_targets = 0; checked_parameters = 0
    for name, left in base.items():
        check_deadline(deadline); right = candidate[name]
        finite(left, torch, chunk_bytes, deadline); finite(right, torch, chunk_bytes, deadline)
        require(left.shape == right.shape and left.dtype == right.dtype, 'Candidate tensor shape/dtype differs from fixed official base')
        checked_parameters += left.numel()
        if name not in targets:
            frozen_same = byte_equal(left, right, torch, chunk_bytes, deadline) and frozen_same
    require(targets.issubset(base), 'Official attention projection inventory differs')
    for target in modules(pins):
        name = target + '.weight'; left, right = base[name], candidate[name]
        width = pins.model_dim * (3 if target.endswith('.c_attn') else 1)
        require(left.dtype == torch.float32 and tuple(left.shape) == (pins.model_dim, width), 'Official attention weight layout differs')
        # Deliberately independent of the exporter. One exact-shape FP32 target
        # at a time (<20 MiB at the fixed architecture); never clone the model.
        a, b = state['base.' + target + '.lora_A'], state['base.' + target + '.lora_B']
        check_deadline(deadline)
        expected = left + (a.transpose(0, 1) @ b.transpose(0, 1)) * (pins.alpha / pins.rank)
        finite(expected, torch, chunk_bytes, deadline)
        formula_same = byte_equal(expected, right, torch, chunk_bytes, deadline) and formula_same
        changed_targets += int(not byte_equal(left, right, torch, chunk_bytes, deadline))
        del expected
    frozen_digest = preserved_digest(base, pins, torch, chunk_bytes, deadline)
    frozen_digest_match = frozen_digest == prepared['provenance']['preserved_base_sha256']
    passed = frozen_same and formula_same and changed_targets > 0 and frozen_digest_match
    return {'audit_passed': passed, 'tensor_count': len(base), 'checked_tensor_elements': checked_parameters,
        'allowed_target_tensors': len(targets), 'changed_target_tensors': changed_targets,
        'non_target_tensors': len(base) - len(targets), 'non_target_all_bitwise_equal': frozen_same,
        'all_target_independent_fp32_formula_bitwise_equal': formula_same,
        'target_formula': 'W_base + (A.transpose(0,1) @ B.transpose(0,1)) * (alpha/r); exact FP32 bytes',
        'preserved_state_digest': frozen_digest, 'provenance_preserved_digest_matches': frozen_digest_match,
        'adapter_ab_tensors': len(state), 'adapter_ab_parameters': sum(value.numel() for value in state.values())}


def audit(prepared, *, deadline, chunk_bytes, minimum_ram_bytes=16 * 1024**3):
    unchanged(prepared, deadline)  # Full base/adapter/candidate hashes before Torch.
    ram = available_ram()
    require(minimum_ram_bytes == 0 or ram is not None and ram >= minimum_ram_bytes, 'Verified available host/cgroup RAM is below the required floor')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    require(not torch.cuda.is_initialized(), 'Use a fresh CPU-only process without a CUDA context')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        objects = [torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                   for path in (prepared['base'], prepared['candidate'], prepared['adapter'])]
        raw_base, raw_candidate, adapter = objects
        if isinstance(raw_base, dict) and set(raw_base) == {'model'}:
            raw_base = raw_base['model']
        require(isinstance(raw_candidate, dict) and set(raw_candidate) == {'model'}, 'Merged checkpoint must contain only the complete official model state')
        result = compare(prepared, raw_base, raw_candidate['model'], adapter, torch, chunk_bytes, deadline)
        unchanged(prepared, deadline)  # Detect metadata, bytes or inode changes after mmap use.
        require(not torch.cuda.is_initialized(), 'Unexpected CUDA context; audit rejected')
        return result
    finally:
        torch.set_num_threads(previous)


def publish(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('workspace-root', 'base', 'adapter', 'merged-bundle', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('candidate-sha256', 'complete-sha256', 'provenance-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate-bytes', type=int, required=True)
    parser.add_argument('--execute-cpu', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=600)
    parser.add_argument('--chunk-mib', type=int, default=8)
    parser.add_argument('--min-free-ram-mib', type=int, default=16384)
    parser.add_argument('--protect-root', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    try:
        require(1 <= args.deadline_seconds <= 1800 and 1 <= args.chunk_mib <= 32 and args.min_free_ram_mib >= 16384,
            'Use bounded chunks/deadline and at least 16384 MiB available RAM')
        args._deadline = time.monotonic() + args.deadline_seconds
        prepared = inspect(args)
        report = dict(prepared['report'])
        if args.execute_cpu:
            started = time.monotonic()
            result = audit(prepared, deadline=args._deadline, chunk_bytes=args.chunk_mib * 1024**2,
                           minimum_ram_bytes=args.min_free_ram_mib * 1024**2)
            report.update(result, status='pure_merge_audit_passed' if result['audit_passed'] else 'pure_merge_audit_failed',
                cpu_tensor_audit_executed=True, all_input_hashes_verified=True, cpu_threads=1, elapsed_seconds=time.monotonic() - started)
            publish(prepared['output'], report)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 3 if args.execute_cpu and not report['audit_passed'] else 0
    except Exception as exc:
        value = {'status': 'pure_merge_audit_blocked', 'failure_type': type(exc).__name__, 'audit_passed': False, 'quality_verified': False}
        if isinstance(exc, AuditError): value['reason'] = str(exc)
        print(json.dumps(value), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
