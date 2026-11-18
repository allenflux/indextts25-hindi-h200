#!/usr/bin/env python3
"""CPU-only inspect/stage of fixed ComfyUI for an isolated workflow test.

This tool never imports ComfyUI/Torch, binds a port, launches a process serving
HTTP, downloads, installs, or accepts credentials. Staging is not permission to
launch: actual ops differences, plugin writes and served-directory boundaries
still need the operator's separate audit. Plugin symlinks do not enforce OS
read-only access. Only fixed GPL framework files are copied.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('_stage_reviewed_t8_compat', Path(__file__).with_name('check_t8_compat.py'))
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)
COMFY_REVISION = '8ff6dc384ba5c410266b40e137799e049459d4f2'
COMFY_REPOSITORY = 'Comfy-Org/ComfyUI'
SOURCE_PINS = {
    'comfy/cli_args.py': 'ba89699d6b887bf9987295d7213769f612efa445',
    'main.py': '3c6e41d353aad7c91cec47a2580a940a641be48f',
    'folder_paths.py': '6723e52c867c57003e9ba4ed53ae58f8d79a7752',
    'app/database/db.py': '053df288332b7d92ea109c4b9ed00e36d479369c',
    'nodes.py': '9bbba5a29e159b01102e5b1de00ed87be64ca539',
    'LICENSE': 'f288702d2fa16d3cdf0035b15a9fcbc552cd88e7',
    'comfy/ops.py': 'f4ac760c946293b095ada0b5e6e16d8b5b4f3cb6',
}
PLUGIN_NAMES = {'audio_cache': 'comfyui-audio-cache', 'audio_crypt': 'comfyui-audiocrypt'}
T8_LINK_NAME = 't8-hindi-compat'
PROJECT_DIRECTORY = 'indextts25-hindi-acceptance-20261117'
REVIEWED_OPS_SHA256 = '68611fba7c1d8750f6bd7266d5c6a40deb5c74ac8930f68c2572f4d314d299c8'
OPS_NEEDLE = b'        bias = post_cast(s, "bias", bias, bias_dtype, prefetch["resident"], update_weight)\n'
OPS_ADDITION = (b'        # LoRA requantization may restore storage dtype; the returned bias must match compute dtype.\n'
                b'        if return_weights and bias.dtype != bias_dtype:\n'
                b'            bias = bias.to(dtype=bias_dtype)\n')
PLUGIN_FILE_PINS = {
    'audio_cache': {'nodes.py': '64f7bb314f8c67c03de560e6642036142e139d51fe591ecc7655ac615a9110d4',
                    '__init__.py': '883ad34bc80cef36a6f2f45953f3cd84a4acb538d95879119977320eb6ade783'},
    'audio_crypt': {'audio_crypt.py': 'a4a50f9b3dbb34274532bdd465b12af0a5902fda2994369b0edee6951cead52e',
                    '__init__.py': 'a0e935892569d0166a9a018cdbc032817749736fbab93bd8b75c73900f26cf56'},
}
STATE_ROOTS = {'input', 'output', 'temp', 'user', 'models', 'custom_nodes', '.git'}
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_FILE = 16 * 1024 * 1024
MAX_ENTRIES = 10000
MAX_RECORD = 4 * 1024 * 1024
RECORD_NAME = 'COMFY_VALIDATION_STAGE.json'
CACHE_LEAVES = {
    'XDG_CACHE_HOME': 'xdg', 'XDG_CONFIG_HOME': 'config', 'HF_HOME': 'huggingface',
    'HF_HUB_CACHE': 'huggingface/hub', 'HUGGINGFACE_HUB_CACHE': 'huggingface/hub',
    'MODELSCOPE_CACHE': 'modelscope', 'TORCH_HOME': 'torch', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
    'NUMBA_CACHE_DIR': 'numba', 'TIKTOKEN_CACHE_DIR': 'tiktoken', 'TRITON_CACHE_DIR': 'triton',
    'CUDA_CACHE_PATH': 'cuda', 'MPLCONFIGDIR': 'matplotlib', 'TMPDIR': 'tmp',
}


class StageError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise StageError(message)


def real_directory(value):
    path = Path(value).absolute()
    require(path.is_dir() and not path.is_symlink() and path.resolve(strict=True) == path,
            'Directory must exist with no symlink components')
    return path


def regular(path):
    require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), 'Expected regular non-symlink file')


def bounded(path, limit=MAX_FILE):
    regular(path)
    require(path.stat().st_size <= limit, 'Bounded file is too large')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'File grew beyond bounded read')
    return data


def object_sha(kind, data):
    return hashlib.sha1(kind.encode() + b' ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and not any(ord(c) < 32 for c in value), 'Unsafe source path')
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and '..' not in path.parts
            and '.git' not in path.parts and str(path) == value.rstrip('/'), 'Unsafe source path')
    return path


def excluded(value):
    path = safe_relative(value)
    return path.parts[0] in STATE_ROOTS or path.name == 'extra_model_paths.yaml' or path.parts[0].startswith('.env')


def tree_sha(entries):
    """Authenticate the entire original ls-tree inventory to the fixed commit."""
    root = {}
    previous = None
    require(isinstance(entries, list) and 0 < len(entries) <= MAX_ENTRIES, 'Invalid source inventory')
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {'path', 'mode', 'blob'}, 'Invalid source entry schema')
        name, mode, blob = entry['path'], entry['mode'], entry['blob']
        path = safe_relative(name)
        require(previous is None or previous < name.encode('utf-8'), 'Source inventory must be unique and sorted')
        previous = name.encode('utf-8')
        require(mode in {'100644', '100755'} and isinstance(blob, str) and re.fullmatch(r'[0-9a-f]{40}', blob), 'Symlinks/submodules/special source entries are forbidden')
        node = root
        for part in path.parts[:-1]:
            require(part not in node or isinstance(node[part], dict), 'Source file/directory collision')
            node = node.setdefault(part, {})
        require(path.parts[-1] not in node, 'Source collision')
        node[path.parts[-1]] = (mode, blob)
    def digest(node):
        pieces = []
        for name, value in sorted(node.items(), key=lambda x: x[0].encode() + (b'/' if isinstance(x[1], dict) else b'')):
            mode, blob = ('40000', digest(value)) if isinstance(value, dict) else value
            pieces.append(mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(blob))
        return object_sha('tree', b''.join(pieces))
    return digest(root)


def authenticate_source(source):
    require(isinstance(source, dict) and set(source) == {'commit_object_base64', 'entries'}, 'Invalid framework source record')
    raw = base64.b64decode(source['commit_object_base64'], validate=True)
    require(len(raw) <= 1024 * 1024 and object_sha('commit', raw) == COMFY_REVISION, 'Framework commit object mismatch')
    first = raw.split(b'\n', 1)[0]
    require(re.fullmatch(rb'tree [0-9a-f]{40}', first), 'Invalid fixed commit tree')
    require(tree_sha(source['entries']) == first[5:].decode(), 'Framework tree identity mismatch')
    entries = {entry['path']: entry for entry in source['entries']}
    require(all(entries.get(name, {}).get('blob') == pin for name, pin in SOURCE_PINS.items()), 'Framework primary source pins mismatch')
    return entries


def source_record(root):
    require(compat._git(root, 'rev-parse', 'HEAD').decode().strip() == COMFY_REVISION, 'Framework HEAD differs from the fixed commit')
    require(Path(compat._git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve() == root, 'Framework path must be the Git root')
    raw = compat._git(root, 'cat-file', 'commit', COMFY_REVISION)
    listing = compat._git(root, 'ls-tree', '-r', '-z', '--full-tree', COMFY_REVISION)
    require(len(listing) <= MAX_RECORD, 'Framework tree listing is too large')
    entries = []
    for row in listing.split(b'\0'):
        if not row:
            continue
        header, path = row.split(b'\t', 1)
        mode, kind, blob = header.decode('ascii').split(' ')
        require(kind == 'blob', 'Framework submodules are unsupported')
        entries.append({'path': path.decode('utf-8'), 'mode': mode, 'blob': blob})
    entries.sort(key=lambda e: e['path'].encode('utf-8'))
    source = {'commit_object_base64': base64.b64encode(raw).decode('ascii'), 'entries': entries}
    authenticate_source(source)
    return source


def python_inventory(root):
    """Read only Python source inside these two explicitly authorized packages."""
    values, total = [], 0
    for directory, subdirs, files in os.walk(root, followlinks=False):
        subdirs[:] = sorted(d for d in subdirs if d not in {'.git', '__pycache__', '.venv', 'venv'})
        for name in subdirs:
            require(not (Path(directory) / name).is_symlink(), 'Plugin source directory symlinks are unsupported')
        for name in sorted(files):
            if not name.endswith('.py'):
                continue
            path = Path(directory) / name
            data = bounded(path)
            total += len(data)
            require(total <= MAX_FILE and len(values) < 500, 'Plugin Python source inventory exceeds its bound')
            values.append({'path': path.relative_to(root).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    require(any(e['path'] == '__init__.py' for e in values), 'Plugin entrypoint is missing')
    return sorted(values, key=lambda e: e['path'])


def candidate_options(args):
    """Legacy calls remain unchanged; new mode requires the explicit caller pins."""
    if getattr(args, 'candidate_manifest', None) is None and getattr(args, 'candidate_manifest_sha256', None) is None:
        return {}
    require(getattr(args, 'candidate_revision', None) is None, 'Candidate revision and reviewed manifest modes are mutually exclusive')
    options = compat.candidate_options(args)
    path, digest = options['candidate_manifest_path'], options['candidate_manifest_sha256']
    require(path is not None and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
            'Reviewed candidate manifest path and SHA must both be explicit')
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file() and path.stat().st_size <= 64 * 1024,
            'Reviewed candidate manifest must be a canonical bounded regular file')
    return {'candidate_manifest_path': path, 'candidate_manifest_sha256': digest}


def plan(args, *, _existing_destination=False):
    work, comfy, t8 = (real_directory(value) for value in (args.workspace_root, args.comfy_root, args.t8_root))
    require(work == comfy / 'models' / PROJECT_DIRECTORY, 'WORK must be the dedicated fixed project directory inside the persistent models mount')
    require(type(args.preserve_reviewed_ops) is bool, 'Ops policy must be an explicit boolean')
    require(type(args.port) is int and 1024 <= args.port <= 65535 and args.port != 8188, 'Use a declared high port other than production 8188; no reservation is made')
    destination = Path(args.destination).absolute()
    require(destination != work and destination.is_relative_to(work), 'Destination must be a dedicated WORK child')
    require(destination.parent.is_dir() and destination.parent.resolve(strict=True) == destination.parent, 'Destination parent must exist without symlinks')
    require((destination.is_dir() and destination.resolve(strict=True) == destination) if _existing_destination
            else (not destination.exists() and not destination.is_symlink()), 'Staging destination must be fresh and canonical')
    roots, plugins = {}, {}
    for label, name in PLUGIN_NAMES.items():
        expected = comfy / 'custom_nodes' / name
        root = real_directory(getattr(args, label + '_root'))
        require(root == expected, 'Plugin reference must be the explicitly allowed installed path')
        require(not destination.is_relative_to(root) and not root.is_relative_to(destination), 'Stage/source overlap is forbidden')
        roots[label] = str(root)
        plugins[label] = python_inventory(root)
        observed = {item['path']: item['sha256'] for item in plugins[label]}
        require(all(observed.get(name) == digest for name, digest in PLUGIN_FILE_PINS[label].items()),
                'Reviewed plugin definition source changed')
    require(not destination.is_relative_to(t8) and not t8.is_relative_to(destination), 'Stage/T8 overlap is forbidden')
    options = candidate_options(args)
    require(not options or options['candidate_manifest_path'].is_relative_to(work), 'Reviewed candidate manifest must be inside WORK')
    identity = compat.verify_prepared(t8, args.candidate_revision, **options)
    source = source_record(comfy)
    ops = comfy / 'comfy/ops.py'
    require(ops.is_file(), 'Installed ops.py is missing')
    host_ops = bounded(ops)
    original_ops = compat._git(comfy, 'cat-file', 'blob', COMFY_REVISION + ':comfy/ops.py')
    require(object_sha('blob', original_ops) == SOURCE_PINS['comfy/ops.py'], 'Fixed ops source mismatch')
    if args.preserve_reviewed_ops:
        expected_ops = reviewed_ops(original_ops)
        require(host_ops == expected_ops, 'Host ops differs from the exact reviewed three-line change')
    paths = {name: str(destination / relative) for name, relative in {
        'core': 'core', 'base': 'base', 'input': 'base/input', 'output': 'base/output',
        'temp': 'base/temp', 'user': 'base/user', 'database': 'base/user/validation.db',
        'runtime': 'runtime', 'cache': 'runtime/cache'}.items()}
    paths.update(python=str(Path(sys.executable).resolve()), t8=str(t8))
    caches = {key: str(Path(paths['cache']) / value) for key, value in CACHE_LEAVES.items()}
    cli = [paths['python'], str(Path(paths['core']) / 'main.py'), '--listen', '127.0.0.1', '--port', str(args.port),
           '--base-directory', paths['base'], '--input-directory', paths['input'], '--output-directory', paths['output'],
           '--temp-directory', paths['base'], '--user-directory', paths['user'], '--database-url', 'sqlite:///' + paths['database'],
           '--disable-all-custom-nodes', '--whitelist-custom-nodes', T8_LINK_NAME, *PLUGIN_NAMES.values(),
           '--disable-api-nodes', '--disable-auto-launch', '--cache-none']
    record = {'format': 'comfy-isolated-validation-stage-v1', 'framework_repository': COMFY_REPOSITORY,
            'framework_revision': COMFY_REVISION, 'workspace_root': str(work), 'destination': str(destination),
            'candidate_revision': args.candidate_revision, 'port': args.port, 'port_reserved': False,
            'paths': paths, 'plugin_roots': roots, 'plugin_python_sources': plugins,
            't8_identity': {k: v for k, v in identity.items() if k != 'manifest_data'},
            'framework_source': source, 'host_ops_sha256': hashlib.sha256(bounded(ops)).hexdigest(),
            'preserve_reviewed_ops': args.preserve_reviewed_ops,
            'ops_policy': 'exact reviewed three-line bias dtype overlay' if args.preserve_reviewed_ops else 'fixed pristine Git archive; host local ops changes are not copied',
            'cache_environment': caches, 'launch_cli': cli, 'served_scratch_exception': False,
            'symlinks_enforce_os_read_only': False, 'source_reference_audit_required': True,
            'served_directories_audit_required': True, 'ops_audit_required': True,
            'frontend_cache_policy_audit_required': True, 'launch_audit_required': True,
            'runtime_equivalence_verified': False, 'model_assets_verified': False,
            'gpu_executed': False, 'server_launched': False, 'workflow_executed': False}
    if options:
        record.update(candidate_manifest=str(options['candidate_manifest_path']),
                      candidate_manifest_sha256=options['candidate_manifest_sha256'],
                      candidate_gpt_sha256=identity['manifest_data']['files']['gpt.pth']['sha256'])
    return record


def reviewed_ops(original):
    require(original.count(OPS_NEEDLE) == 1, 'Fixed ops insertion location drift')
    changed = original.replace(OPS_NEEDLE, OPS_NEEDLE + OPS_ADDITION)
    require(hashlib.sha256(changed).hexdigest() == REVIEWED_OPS_SHA256, 'Reviewed ops output pin mismatch')
    return changed


def copy_core(comfy, core, source):
    entries = authenticate_source(source)
    expected = {name: entry for name, entry in entries.items() if not excluded(name)}
    directories = {parent.as_posix() for name in entries for parent in PurePosixPath(name).parents if parent.parts}
    archive = compat._git(comfy, 'archive', '--format=tar', COMFY_REVISION)
    require(len(archive) <= MAX_ARCHIVE, 'Framework archive exceeds bound')
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as stream:
        members = stream.getmembers()
        require(len(members) <= MAX_ENTRIES * 2, 'Too many archive entries')
        seen, total = set(), 0
        for member in members:
            name = str(safe_relative(member.name))
            require(name not in seen, 'Duplicate source archive member')
            seen.add(name)
            require(member.isdir() or member.isreg(), 'Source archive special files/symlinks are forbidden')
            require(0 <= member.size <= MAX_FILE, 'Source archive entry too large')
            total += member.size
            if member.isreg():
                require(name in entries, 'Unexpected archive source file')
            else:
                require(name in directories, 'Unexpected archive source directory')
        require(total <= MAX_ARCHIVE and expected.keys() <= seen, 'Expanded archive inventory/size mismatch')
        core.mkdir(mode=0o700)
        for member in members:
            name = str(safe_relative(member.name))
            if member.isdir() or excluded(name):
                continue
            path = core.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(b'blob ' + str(member.size).encode() + b'\0')
            with stream.extractfile(member) as incoming, path.open('xb') as outgoing:
                remaining = member.size
                while remaining:
                    chunk = incoming.read(min(1024 * 1024, remaining))
                    require(chunk and len(chunk) <= remaining, 'Truncated source archive entry')
                    outgoing.write(chunk); digest.update(chunk); remaining -= len(chunk)
                require(not incoming.read(1), 'Oversized source archive entry')
            require(digest.hexdigest() == expected[name]['blob'], 'Fixed framework file Git blob mismatch')


def verify_core(core, source, *, preserve_reviewed_ops=False):
    entries = authenticate_source(source)
    expected = {name: entry for name, entry in entries.items() if not excluded(name)}
    actual = set()
    for directory, subdirs, files in os.walk(core, followlinks=False):
        for name in subdirs:
            require(not (Path(directory) / name).is_symlink(), 'Framework source symlink appeared')
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(core).as_posix()
            require(relative in expected, 'Unexpected staged framework file')
            data = bounded(path)
            if preserve_reviewed_ops and relative == 'comfy/ops.py':
                require(data.count(OPS_ADDITION) == 1 and hashlib.sha256(data).hexdigest() == REVIEWED_OPS_SHA256,
                        'Staged reviewed ops drift')
                data = data.replace(OPS_ADDITION, b'')
            require(object_sha('blob', data) == expected[relative]['blob'], 'Staged framework content drift')
            actual.add(relative)
    require(actual == expected.keys(), 'Staged framework source inventory mismatch')


def stage(args):
    record = plan(args)
    root = Path(record['destination'])
    root.mkdir(mode=0o700)
    copy_core(Path(args.comfy_root), Path(record['paths']['core']), record['framework_source'])
    if args.preserve_reviewed_ops:
        staged_ops = Path(record['paths']['core']) / 'comfy/ops.py'
        staged_ops.write_bytes(reviewed_ops(bounded(staged_ops)))
    for name in ('base', 'input', 'output', 'temp', 'user', 'runtime', 'cache'):
        Path(record['paths'][name]).mkdir(parents=True, exist_ok=True, mode=0o700)
    for value in record['cache_environment'].values():
        Path(value).mkdir(parents=True, exist_ok=True, mode=0o700)
    custom = Path(record['paths']['base']) / 'custom_nodes'
    custom.mkdir()
    (custom / T8_LINK_NAME).symlink_to(record['paths']['t8'], target_is_directory=True)
    for label, name in PLUGIN_NAMES.items():
        (custom / name).symlink_to(record['plugin_roots'][label], target_is_directory=True)
    with (root / RECORD_NAME).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return verify_stage(root, **candidate_options(args))


def verify_stage(root: Path, *, candidate_manifest_path=None, candidate_manifest_sha256=None):
    root = real_directory(root)
    record = json.loads(bounded(root / RECORD_NAME, MAX_RECORD))
    require(isinstance(record, dict) and record.get('format') == 'comfy-isolated-validation-stage-v1'
            and record.get('framework_revision') == COMFY_REVISION and record.get('framework_repository') == COMFY_REPOSITORY,
            'Stage identity/schema mismatch')
    reviewed = 'candidate_manifest' in record or 'candidate_manifest_sha256' in record
    if reviewed:
        require(candidate_manifest_path is not None and candidate_manifest_sha256 is not None,
                'Reviewed stage requires independently supplied candidate manifest path and SHA')
        from types import SimpleNamespace
        expected = candidate_options(SimpleNamespace(candidate_revision=None,
            candidate_manifest=candidate_manifest_path, candidate_manifest_sha256=candidate_manifest_sha256))
        require(record.get('candidate_revision') is None and record.get('candidate_manifest') == str(expected['candidate_manifest_path'])
                and record.get('candidate_manifest_sha256') == expected['candidate_manifest_sha256'],
                'Stage reviewed candidate identity differs from caller pins')
    else:
        require(candidate_manifest_path is None and candidate_manifest_sha256 is None,
                'Legacy stage cannot acquire an unrecorded candidate manifest')
    # Rebuild the exact original plan without requiring the already-created
    # destination to be fresh; do not import or execute any referenced source.
    from types import SimpleNamespace
    args = SimpleNamespace(workspace_root=Path(record['workspace_root']), comfy_root=Path(record['plugin_roots']['audio_cache']).parents[1],
        t8_root=Path(record['paths']['t8']), destination=root, candidate_revision=record['candidate_revision'], port=record['port'],
        audio_cache_root=Path(record['plugin_roots']['audio_cache']), audio_crypt_root=Path(record['plugin_roots']['audio_crypt']),
        preserve_reviewed_ops=record['preserve_reviewed_ops'],
        candidate_manifest=candidate_manifest_path, candidate_manifest_sha256=candidate_manifest_sha256)
    # A temporary in-process validation parameter, never a public CLI bypass.
    wanted = _existing_plan(args)
    require(json.dumps(record, sort_keys=True, allow_nan=False) == json.dumps(wanted, sort_keys=True, allow_nan=False),
            'Stage manifest or referenced source identity changed')
    verify_core(real_directory(record['paths']['core']), record['framework_source'], preserve_reviewed_ops=record['preserve_reviewed_ops'])
    for name in ('base', 'input', 'output', 'temp', 'user', 'runtime', 'cache'):
        real_directory(record['paths'][name])
    for value in record['cache_environment'].values():
        real_directory(value)
    custom = Path(record['paths']['base']) / 'custom_nodes'
    require({p.name for p in custom.iterdir()} == {T8_LINK_NAME, *PLUGIN_NAMES.values()}, 'Unexpected staged custom plugin')
    targets = {T8_LINK_NAME: record['paths']['t8'], **{name: record['plugin_roots'][label] for label, name in PLUGIN_NAMES.items()}}
    for name, target in targets.items():
        path = custom / name
        require(path.is_symlink() and os.readlink(path) == target and path.resolve(strict=True) == Path(target), 'Staged plugin reference changed')
    return record


def _existing_plan(args):
    return plan(args, _existing_destination=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('inspect', 'stage'):
        command = sub.add_parser(action)
        for name in ('workspace-root', 'comfy-root', 't8-root', 'audio-cache-root', 'audio-crypt-root', 'destination'):
            command.add_argument('--' + name, type=Path, required=True)
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument('--candidate-revision')
        selection.add_argument('--candidate-manifest', type=Path)
        command.add_argument('--candidate-manifest-sha256')
        command.add_argument('--port', required=True, type=int)
        command.add_argument('--preserve-reviewed-ops', action='store_true', help='Apply only the exact reviewed fixed-SHA three-line ops compatibility change in the independent copy')
    verify = sub.add_parser('verify')
    verify.add_argument('--stage-root', required=True, type=Path)
    verify.add_argument('--candidate-manifest', type=Path)
    verify.add_argument('--candidate-manifest-sha256')
    args = parser.parse_args(argv)
    try:
        record = verify_stage(args.stage_root, **candidate_options(args)) if args.action == 'verify' else stage(args) if args.action == 'stage' else plan(args)
        print(json.dumps({'status': 'verified_cpu_stage' if args.action != 'inspect' else 'verified_cpu_plan',
                          **{k: v for k, v in record.items() if k not in {'framework_source', 'plugin_python_sources'}}},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except (StageError, compat.CompatibilityError, OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(json.dumps({'status': 'blocked_not_launched', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
