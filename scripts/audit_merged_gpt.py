#!/usr/bin/env python3
"""Read-only, fixed-artifact GPT byte audit. Parameter equality is not speech-quality evidence."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HINDI_ROW = 17
MAX_CHECKPOINT = 4 * 1024**3


class AuditError(ValueError):
    pass


@dataclass(frozen=True)
class WeightPin:
    size: int
    sha256: str


BASE = WeightPin(3259599833, '43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85')
CANDIDATE = WeightPin(3259538991, '423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021')


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise AuditError('Cooperative CPU audit deadline exceeded')


def stamp(path):
    value = path.lstat()
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def source_path(value):
    path = Path(value).absolute()
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
        raise AuditError('Expected a regular canonical input file')
    if not 0 < path.stat().st_size <= MAX_CHECKPOINT:
        raise AuditError('Checkpoint exceeds the fixed input bound')
    return path


def sha256(path, deadline):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            check_deadline(deadline); digest.update(chunk)
    return digest.hexdigest()


def group(name):
    if re.fullmatch(r'gpt\.h\.[0-9]+\.attn\.(?:c_attn|c_proj)\.weight', name):
        return 'shared_attention_projection_weights'
    if re.fullmatch(r'gpt\.h\.[0-9]+\.attn\.(?:c_attn|c_proj)\.bias', name):
        return 'attention_projection_biases'
    if name.startswith('mel_head.'):
        return 'shared_semantic_code_head'
    if name.startswith('text_head.'):
        return 'shared_text_auxiliary_head'
    if name == 'lang_embedding.weight':
        return 'language_embedding'
    if name == 'text_embedding.weight':
        return 'text_embedding'
    if re.match(r'gpt\.h\.[0-9]+\.mlp\.', name):
        return 'transformer_mlp'
    if name.startswith('mel_embedding.'):
        return 'mel_embedding'
    if 'pos_embedding' in name:
        return 'positional_embeddings'
    if 'norm' in name or re.match(r'gpt\.h\.[0-9]+\.ln_', name):
        return 'normalization'
    return 'conditioning_and_remaining_state'


def _supported(tensor, torch):
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64,
               torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool}
    return (torch.is_tensor(tensor) and tensor.device.type == 'cpu' and tensor.layout == torch.strided
            and not tensor.is_quantized and tensor.is_contiguous() and tensor.dtype in allowed)


def _bytes(tensor, torch):
    return tensor.detach().reshape(-1).view(torch.uint8)


def _identical(left, right, torch, chunk_bytes, deadline):
    a, b = _bytes(left, torch), _bytes(right, torch)
    for offset in range(0, a.numel(), chunk_bytes):
        check_deadline(deadline)
        if not torch.equal(a[offset:offset+chunk_bytes], b[offset:offset+chunk_bytes]):
            return False
    return True


def _row_stats(left, right, torch, chunk_bytes, deadline, *, language=False):
    if left.ndim != 2 or (language and left.shape[0] <= HINDI_ROW):
        raise AuditError('Expected a two-dimensional embedding with the fixed Hindi row')
    rows, width = left.shape
    row_bytes = width * left.element_size()
    if row_bytes <= 0 or row_bytes > chunk_bytes:
        raise AuditError('Embedding row exceeds comparison chunk bound')
    a, b = _bytes(left, torch).reshape(rows, row_bytes), _bytes(right, torch).reshape(rows, row_bytes)
    step = max(1, chunk_bytes // row_bytes); changed = 0; hi_changed = False
    for start in range(0, rows, step):
        check_deadline(deadline)
        unequal = ~(a[start:start+step] == b[start:start+step]).all(dim=1)
        changed += int(unequal.sum().item())
        if language and start <= HINDI_ROW < min(rows, start+step):
            hi_changed = bool(unequal[HINDI_ROW-start].item())
    result = {'total_rows': rows, 'changed_rows': changed, 'unchanged_rows': rows-changed}
    if language:
        result.update(hindi_row=HINDI_ROW, hindi_row_changed=hi_changed,
            non_hindi_changed_rows=changed-int(hi_changed), non_hindi_all_bitwise_equal=changed==int(hi_changed))
    return result


def _compare_states(base, candidate, torch, chunk_bytes, deadline):
    if not isinstance(base, dict) or not isinstance(candidate, dict) or not base or not candidate or any(not isinstance(k,str) for k in [*base,*candidate]):
        raise AuditError('Expected nonempty tensor state dictionaries')
    common = sorted(base.keys() & candidate.keys())
    result = {'base_tensor_count':len(base), 'candidate_tensor_count':len(candidate),
        'missing_keys':sorted(base.keys()-candidate.keys()), 'unexpected_keys':sorted(candidate.keys()-base.keys()),
        'shape_mismatches':[], 'dtype_mismatches':[], 'bitwise_identical_tensors':0, 'changed_tensors':0,
        'groups':{}, 'embedding_rows':{}}
    for name in common:
        check_deadline(deadline); left, right = base[name], candidate[name]
        if not _supported(left,torch) or not _supported(right,torch):
            raise AuditError('Unsupported tensor layout/device/dtype; no implicit materialization is allowed')
        shape_equal, dtype_equal = left.shape == right.shape, left.dtype == right.dtype
        if not shape_equal:
            result['shape_mismatches'].append(name)
        if not dtype_equal:
            result['dtype_mismatches'].append(name)
        if not shape_equal or not dtype_equal:
            continue
        same = _identical(left,right,torch,chunk_bytes,deadline)
        key = 'bitwise_identical_tensors' if same else 'changed_tensors'
        result[key] += 1
        counts = result['groups'].setdefault(group(name), {'compared':0,'bitwise_identical':0,'changed':0})
        counts['compared'] += 1; counts['bitwise_identical' if same else 'changed'] += 1
        if name in {'lang_embedding.weight','text_embedding.weight'}:
            result['embedding_rows'][name] = _row_stats(left,right,torch,chunk_bytes,deadline,language=name.startswith('lang_'))
    result['key_shape_dtype_equal'] = not any(result[k] for k in ('missing_keys','unexpected_keys','shape_mismatches','dtype_mismatches'))
    return result


def _execute(base_path, candidate_path, deadline, chunk_bytes, *, _pins=(BASE,CANDIDATE)):
    paths = [source_path(base_path), source_path(candidate_path)]
    if paths[0] == paths[1]:
        raise AuditError('Base and candidate must be different files')
    before = [stamp(p) for p in paths]
    for path,pin in zip(paths,_pins):
        if path.stat().st_size != pin.size or sha256(path,deadline) != pin.sha256:
            raise AuditError('Pinned checkpoint byte count/SHA256 mismatch')
    import torch
    if torch.cuda.is_initialized():
        raise AuditError('Use a fresh CPU process without an existing CUDA context')
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        states=[]; containers={}
        for label,path in zip(('base','candidate'),paths):
            check_deadline(deadline)
            value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if not isinstance(value,dict) or not value:
                raise AuditError('Expected the reviewed official model checkpoint container')
            if isinstance(value.get('model'),dict):
                states.append(value['model']); containers[label]='model'
            elif all(torch.is_tensor(tensor) for tensor in value.values()):
                states.append(value); containers[label]='plain_tensor_state'
            else:
                raise AuditError('Expected official model or plain tensor state container')
        report=_compare_states(*states,torch,chunk_bytes,deadline)
        report['checkpoint_containers']=containers
        for path,pin,original in zip(paths,_pins,before):
            if stamp(path)!=original or sha256(path,deadline)!=pin.sha256 or stamp(path)!=original:
                raise AuditError('Checkpoint changed during the read-only audit')
        if torch.cuda.is_initialized():
            raise AuditError('Unexpected CUDA context; audit is not accepted')
        return report
    finally:
        torch.set_num_threads(previous)


def destination(value, workspace, inputs):
    output=Path(value).absolute(); parent=output.parent.resolve(strict=True)
    if parent!=output.parent or output.exists() or output.is_symlink() or output.with_suffix(output.suffix+'.partial').exists():
        raise AuditError('Audit output must be a fresh canonical file')
    if not output.is_relative_to(workspace) or any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in [ROOT,*[p.parent for p in inputs]]):
        raise AuditError('Audit output must be inside WORK and outside source/model directories')
    return output


def publish(path, report):
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(partial,path);partial.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True,type=Path);parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--workspace-root',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--execute-cpu',action='store_true')
    parser.add_argument('--deadline-seconds',type=int,default=600);parser.add_argument('--chunk-mib',type=int,default=8)
    args=parser.parse_args(argv)
    try:
        if not 1<=args.deadline_seconds<=1800 or not 1<=args.chunk_mib<=32:
            raise AuditError('Bound deadline to 1..1800 seconds and chunks to 1..32 MiB')
        workspace=args.workspace_root.absolute()
        if workspace.resolve(strict=True)!=workspace or not workspace.is_dir():
            raise AuditError('WORK must be a canonical existing directory')
        inputs=[source_path(args.base),source_path(args.candidate)]
        for path,pin in zip(inputs,(BASE,CANDIDATE)):
            if path.stat().st_size!=pin.size:
                raise AuditError('Pinned checkpoint byte count mismatch')
        output=destination(args.output,workspace,inputs)
        report={'schema_version':1,'kind':'fixed_merged_gpt_parameter_audit',
            'status':'metadata_only_no_torch_or_weights_loaded','hashes_verified':False,
            'quality_verified':False,'old_language_behavior_guaranteed_unchanged':False,
            'cuda_initialized':False,'comparison':'bitwise bytes; signed zero and NaN payloads are not normalized',
            'deadline_policy':'cooperative checks between bounded chunks; not an OS hard timeout',
            'base':{'bytes':BASE.size,'sha256':BASE.sha256},'candidate':{'bytes':CANDIDATE.size,'sha256':CANDIDATE.sha256}}
        if args.execute_cpu:
            started=time.monotonic()
            result=_execute(*inputs,started+args.deadline_seconds,args.chunk_mib*1024**2)
            report.update(status='read_only_cpu_parameter_audit_completed',hashes_verified=True,
                elapsed_seconds=time.monotonic()-started,cpu_threads=1,**result)
            publish(output,report)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except Exception as exc:
        value={'status':'audit_blocked_or_failed','error_type':type(exc).__name__}
        if isinstance(exc,AuditError):value['reason']=str(exc)
        print(json.dumps(value),file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
