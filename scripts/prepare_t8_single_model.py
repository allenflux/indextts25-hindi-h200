#!/usr/bin/env python3
"""CPU prototype: one truthful best100 default under the original T8 node IDs.

No deployment, GPU, downloads, installation or credentials. The reviewed dual
profile is unchanged. This separate profile keeps model_name=IndexTTS-2.5 and
empty custom_model_path, requires the one fixed candidate's full 26 assets,
and blocks the node's model-bundle download paths. Old languages also use the
candidate; their quality is NOT established by preparing this code.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

SPEC = importlib.util.spec_from_file_location('_single_model_reviewed_dual', Path(__file__).with_name('check_t8_compat.py'))
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)
REVISION = '117c7922e4dcedc3f9885f3f6efa5be8f54c7919'
PROFILE = 'T8_SINGLE_MODEL_PROTOTYPE.json'
DOWNLOADER = 'services/downloader.py'
DOWNLOADER_SHA = '792581e68dc81dcb6fd4757e845a0334de72654eadd8bdca3a752e65f3200b72'
DOWNLOADER_BLOB = '4eadf054e587de4bada954b26a68e7d75434e0b7'
OFFLINE_REASON = '单候选实验模式不支持自动下载或修复；请离线准备并校验固定 best100 的完整26项资产，HF公开13文件包不是完整T8目录。'
OUTPUT_PINS = {'services/model_store.py': '85e36ac2dbcca4350d486746ce0bcce4d1a56c2fd2f4d948cd8862f36fe9d8ce', 'nodes_v3.py': 'b280b6067fb9a686ac280a1b4f65fca59d989ef41abc09bde181b41cdebd61b6', 'services/downloader.py': 'a613462b46fa29c5f2259659bcf1217e77c10c48fb3e9d604db037e6ab53a0f8'}


class SingleModelError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SingleModelError(message)


def fixed_revision(value):
    require(value == REVISION, 'This CPU prototype is frozen to the reviewed public best100 revision')
    return value


def replace_function(source, name, replacement):
    return compat._replace_span(source, compat._function(ast.parse(source), name), replacement)


def insert_guard(source, function, statement):
    first = function.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        first = function.body[1]
    lines = source.splitlines(keepends=True)
    indent = ' ' * first.col_offset
    lines.insert(first.lineno - 1, '\n'.join(indent + line for line in statement.splitlines()) + '\n')
    return ''.join(lines)


def loader_class(source):
    return compat._one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                        and n.name == 'T8IndexTTS25ModelLoader'), 'Missing exact original loader class')


def transform(contents):
    """Original bounded transforms on the already verified dual-profile bytes."""
    result = {}
    source = contents['services/model_store.py'].decode()
    assignment = compat._one((n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'MANIFEST_PATH' for t in n.targets)), 'Missing manifest path')
    source = compat._replace_span(source, assignment.value, 'PLUGIN_ROOT / "manifests" / "hindi_best100_experimental.json"')
    source = replace_function(source, 'load_manifest', '''def load_manifest(model_dir: Path | str | None = None) -> dict:
    # No size routing or fallback: both legacy and HI requests use this identity.
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))''')
    source = replace_function(source, 'model_options', '''def model_options() -> list[str]:
    # Retain the existing workflow value even when assets are missing.
    return [MODEL_FOLDER_NAME]''')
    source = replace_function(source, 'resolve_model', '''def resolve_model(model_name: str, custom_model_path: str = "") -> Path:
    if custom_model_path.strip():
        return Path(custom_model_path.strip().strip('"')).expanduser().resolve()
    if model_name != MODEL_FOLDER_NAME:
        raise ValueError("单候选模式默认模型名必须为 IndexTTS-2.5。")
    import folder_paths
    # Other configured roots and same-named backups never participate.
    return (Path(folder_paths.models_dir) / "TTS" / MODEL_FOLDER_NAME).expanduser().resolve()''')
    tree = ast.parse(source)
    url = compat._one((n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'MODEL_REPOSITORY_URL' for t in n.targets)), 'Missing repository URL')
    source = compat._replace_span(source, url.value, repr('https://huggingface.co/' + compat.CANDIDATE_REPOSITORY))
    validator = compat._function(ast.parse(source), 'validate_model_dir')
    source = insert_guard(source, validator, 'include_auxiliary = True  # All 26 files are mandatory in this profile.')
    # The dual prototype already forces verify_hashes=True; preserve it.
    report = compat._one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'ValidationReport'), 'Missing validation report')
    require_valid = compat._function(report, 'require_valid')
    raises = compat._one((n for n in ast.walk(require_valid) if isinstance(n, ast.Raise)), 'Missing validation error')
    message = 'IndexTTS 2.5 best100候选完整目录校验失败：'
    replacement = ('raise FileNotFoundError(' + repr(message) + ' + str(self.model_dir) + "；" + "；".join(details) + ' + repr('。' + OFFLINE_REASON) + ')')
    source = compat._replace_span(source, raises, replacement)
    result['services/model_store.py'] = source.encode()

    source = contents['nodes_v3.py'].decode()
    cls = loader_class(source)
    schema = compat._function(cls, 'define_schema')
    description = compat._one((n.value for n in ast.walk(schema) if isinstance(n, ast.keyword) and n.arg == 'description'), 'Missing loader description')
    source = compat._replace_span(source, description, repr('加载已合并的best100实验GPT及匹配IndexTTS2.5资产；保留原节点与工作流字段。原语言和HI均使用候选，不保证旧语言质量。此模式离线且强制核验26项资产。'))
    hints = {
        'verify_hashes': '本候选模式始终完整核验全部26资产SHA；此旧字段保留但不能关闭校验。',
        'download_missing': OFFLINE_REASON,
        'accept_model_license': '模型许可仍适用；此旧字段不会启用下载，也不会代为接受任何访问条件。',
        'custom_model_path': '留空使用标准models/TTS/IndexTTS-2.5候选目录；非空目录同样必须通过固定候选26资产校验。',
    }
    for field, hint in hints.items():
        schema = compat._function(loader_class(source), 'define_schema')
        call = compat._one((n for n in ast.walk(schema) if isinstance(n, ast.Call) and n.args
                            and isinstance(n.args[0], ast.Constant) and n.args[0].value == field), 'Missing exact loader input ' + field)
        tooltip = compat._one((n.value for n in call.keywords if n.arg == 'tooltip'), 'Missing loader tooltip ' + field)
        source = compat._replace_span(source, tooltip, repr(hint))
    for method, statement in (
        ('validate_inputs', 'if download_missing:\n    return ' + repr(OFFLINE_REASON)),
        ('execute', 'if download_missing:\n    raise ValueError(' + repr(OFFLINE_REASON) + ')')):
        function = compat._function(loader_class(source), method)
        source = insert_guard(source, function, statement)
    execute = compat._function(loader_class(source), 'execute')
    final_return = compat._one((n for n in execute.body if isinstance(n, ast.Return)), 'Missing exact loader return')
    source = insert_guard(source, type('ReturnLocation', (), {'body': [final_return]})(),
        'info = (f"best100 experimental | repository={manifest[\'modelRepository\']} | "\n'
        '        f"revision={manifest[\'modelRevision\']} | gpt_sha256={manifest[\'files\'][\'gpt.pth\'][\'sha256\']} | ") + info')
    result['nodes_v3.py'] = source.encode()

    source = contents[DOWNLOADER].decode()
    for name in ('ensure_model_bundle', 'download_main_model', 'download_auxiliary_models', '_download_hf_file_cancellable'):
        function = compat._function(ast.parse(source), name)
        source = insert_guard(source, function, 'raise RuntimeError(' + repr(OFFLINE_REASON) + ')')
    # The surviving verify-only CLI must identify the candidate truthfully.
    constants = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant)
                 and isinstance(n.value, str) and '正式模型' in n.value]
    require(len(constants) == 3, 'Expected exact original downloader identity messages')
    for node in sorted(constants, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        message = ('离线校验 IndexTTS 2.5 best100实验候选完整26项资产；此模式不支持下载'
                   if node.value == '下载并校验 IndexTTS 2.5 正式模型'
                   else node.value.replace('正式模型', 'best100实验候选'))
        source = compat._replace_span(source, node, repr(message))
    main_function = compat._function(ast.parse(source), 'main')
    success = compat._one((n for n in ast.walk(main_function) if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        and n.value.func.id == 'print' and n.value.args and isinstance(n.value.args[0], ast.Constant)
        and n.value.args[0].value == '>> IndexTTS 2.5 best100实验候选 SHA-256 校验通过。'),
        'Missing exact verify-only success receipt')
    receipt = ('manifest = load_manifest()\n' + ' ' * success.col_offset +
        'print(f">> best100实验候选 SHA-256 校验通过 | repository={manifest[\'modelRepository\']} | "\n' +
        ' ' * (success.col_offset + 6) + 'f"revision={manifest[\'modelRevision\']} | gpt_sha256={manifest[\'files\'][\'gpt.pth\'][\'sha256\']}")')
    source = compat._replace_span(source, success, receipt)
    result[DOWNLOADER] = source.encode()
    for name, content in result.items():
        compile(content, name, 'exec')
    return result


def profile(revision):
    return {'format': 't8-single-model-cpu-prototype-v1', 'policy': 'candidate-only',
        'node_repository': compat.NODE_REPOSITORY, 'node_revision': compat.NODE_REVISION,
        'candidate_repository': compat.CANDIDATE_REPOSITORY, 'candidate_revision': fixed_revision(revision),
        'gpt_sha256': compat.GPT_SHA256, 'gpt_bytes': compat.GPT_BYTES,
        'manifest': compat.CANDIDATE_MANIFEST, 'default_model_name': 'IndexTTS-2.5',
        'default_root': 'folder_paths.models_dir/TTS/IndexTTS-2.5', 'automatic_downloads': False,
        'fallback_to_original_gpt': False, 'required_assets': 26, 'force_sha256': True,
        'supported_language_slots': compat.LANGUAGES + ['HI'], 'cpu_prototype_only': True,
        'gpu_validated': False, 'old_language_quality_verified': False, 'production_deployed': False,
        'transformed_sha256': dict(OUTPUT_PINS), 'original_downloader_sha256': DOWNLOADER_SHA}


def original_downloader(path):
    data = compat._read(path)
    require(hashlib.sha256(data).hexdigest() == DOWNLOADER_SHA and compat._blob(data) == DOWNLOADER_BLOB,
            'Pinned original downloader source differs')
    return data


def prepare(root, destination, revision):
    fixed_revision(revision)
    original_downloader(Path(root) / DOWNLOADER)
    compat.prepare(Path(root), Path(destination), revision)
    compat.verify_prepared(Path(destination), revision)
    contents = {name: compat._read(Path(destination) / name) for name in ('nodes_v3.py', 'services/model_store.py')}
    contents[DOWNLOADER] = original_downloader(Path(destination) / DOWNLOADER)
    changed = transform(contents)
    for name, data in changed.items():
        require(hashlib.sha256(data).hexdigest() == OUTPUT_PINS.get(name), 'Deterministic single-profile source pin mismatch')
        (Path(destination) / name).write_bytes(data)
    with (Path(destination) / PROFILE).open('x', encoding='utf-8') as stream:
        json.dump(profile(revision), stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
    return verify_copy(Path(destination), revision)


def read_json(path, limit=128 * 1024):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate profile/manifest JSON key")
            value[key] = item
        return value
    return json.loads(compat._read(path, limit), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(SingleModelError("Nonfinite JSON")))


def verify_copy(root, revision):
    fixed_revision(revision)
    root = Path(root).absolute()
    require(root.is_dir() and root.resolve(strict=True) == root, 'Use a real canonical prepared directory')
    observed = read_json(root / PROFILE)
    require(json.dumps(observed, sort_keys=True, allow_nan=False) == json.dumps(profile(revision), sort_keys=True, allow_nan=False), 'Single-model profile identity/schema differs')
    require(read_json(root / compat.AUDIT, 64 * 1024) == compat._audit(revision), 'Original dual preparation provenance changed')
    for name, pin in compat.PINNED_BLOBS.items():
        data = compat._read(root / name)
        if name in OUTPUT_PINS:
            require(hashlib.sha256(data).hexdigest() == OUTPUT_PINS[name], 'Single-model source drift: ' + name)
        elif name in compat.PATCHED_SHA256:
            require(hashlib.sha256(data).hexdigest() == compat.PATCHED_SHA256[name], 'Retained language support source drift: ' + name)
        else:
            require(compat._blob(data) == pin, 'Retained base identity/license source drift: ' + name)
    require(hashlib.sha256(compat._read(root / DOWNLOADER)).hexdigest() == OUTPUT_PINS[DOWNLOADER], 'Offline downloader source drift')
    base = read_json(root / compat.BASE_MANIFEST)
    manifest = read_json(root / compat.CANDIDATE_MANIFEST)
    require(manifest == compat.candidate_manifest(base, revision), 'Fixed candidate manifest content differs')
    require(len(manifest['files']) == 26, 'All fixed candidate assets must be present in the manifest')
    return {'status': 'cpu_verified_single_model_prototype_not_deployed', **profile(revision), 'manifest_data': manifest}


def verify_assets(root, model_dir, revision):
    info = verify_copy(root, revision)
    model_dir = Path(model_dir).absolute()
    require(model_dir.is_dir() and model_dir.resolve(strict=True) == model_dir, 'Use a real canonical model directory')
    # Check all structure/sizes before reading any large GPT. A public 13-file
    # release must fail promptly rather than hashing 3 GB before missing bpe.
    for name, metadata in info['manifest_data']['files'].items():
        path = model_dir.joinpath(*compat._safe_member(name).parts)
        require(path.is_file(), 'Incomplete 26-asset candidate directory: missing ' + name)
        compat._regular(path)
        require(path.stat().st_size == metadata['size'], 'Candidate asset size mismatch: ' + name)
    verified = compat.verify_assets(model_dir, info['manifest_data'])
    return {'status': 'cpu_verified_single_model_assets_not_gpu_validated', 'asset_identity': verified,
            **{key: value for key, value in info.items() if key not in {'manifest_data', 'status'}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'verify-copy', 'verify-assets'):
        command = commands.add_parser(name)
        command.add_argument('--node-root', required=True, type=Path)
        command.add_argument('--candidate-revision', required=True)
        if name == 'prepare': command.add_argument('--destination', required=True, type=Path)
        if name == 'verify-assets': command.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.node_root, args.destination, args.candidate_revision) if args.action == 'prepare' else verify_assets(args.node_root, args.model_dir, args.candidate_revision) if args.action == 'verify-assets' else verify_copy(args.node_root, args.candidate_revision)
        print(json.dumps({key: value for key, value in result.items() if key != 'manifest_data'}, ensure_ascii=False, sort_keys=True))
        return 0
    except (SingleModelError, compat.CompatibilityError, OSError, ValueError, SyntaxError, KeyError) as exc:
        print(json.dumps({'status': 'blocked_not_deployed', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
