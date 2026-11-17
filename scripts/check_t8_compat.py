#!/usr/bin/env python3
"""Inspect pinned T8 code or prepare an isolated experimental Hindi code copy.

No model loading, network, package installation, production edits or GPU work.
The original manifest remains the base default; the fresh copy selects its
separate candidate manifest by fixed GPT size, then requires full asset hashes.
Model assets are NOT staged by this command.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile

NODE_REPOSITORY = "T8mars/comfyui-indextts25-t8"
NODE_REVISION = "1ff425ece89ef9824f58f0e4dbfb057991814e88"
CANDIDATE_REPOSITORY = "fluxa0223/IndexTTS-2.5-Hindi-best100-experimental"
GPT_SHA256 = "423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021"
GPT_BYTES = 3259538991
HI_AUTO_TOKEN_LIMIT = 60  # Conservative experimental starting limit, not tuned.
BASE_MANIFEST = "manifests/model_2_5.json"
CANDIDATE_MANIFEST = "manifests/hindi_best100_experimental.json"
AUDIT = "T8_HINDI_COMPATIBILITY.json"
LANGUAGES = ["ZH", "EN", "JA", "ES", "AR"]
PINNED_BLOBS = {
    "nodes_v3.py": "c081c7b65f3824806f7c7ad0ca34b7d898349904",
    "runtime/inference_adapter.py": "27b4e7bf804ea8d62166e57fc4dabc85cf36735e",
    "runtime/text_planner.py": "149f28cc3e2d2d2b746e2567f699d66e148432a8",
    "services/model_store.py": "fb4e12bd06c753751b12f03b5f1540a999f1437c",
    BASE_MANIFEST: "0213b2d0fe4b8ce38683d2650141868d753ecfb8",
    "LICENSE": "a41d45fac40811452bcb5160877544f22829207b",
    "LICENSE_ZH.txt": "519ba4429a9404a686990f201e55f9a3b2bc93f6",
}
# Filled from deterministic transformations of the seven pinned public blobs.
PATCHED_SHA256 = {'nodes_v3.py': '7ec24c7ba191bc24ded9a2d98d58d8ba7a79642c3357fba42d4ec856b1222854', 'runtime/inference_adapter.py': '0d282f3fc971086e9814b86b74b90d1a096e4406103a186662098d60f6f4e663', 'runtime/text_planner.py': '8b235d22d6f4bca5d9a5cd839edb367100d3c22cffa3a2f916e7e799d6ceeb67', 'services/model_store.py': '995e71b3ac4ab2a59d171feb2172d7b223d974c4218f6244929a965baaad744d'}
MAX_SOURCE_FILE = 16 * 1024 * 1024
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_ENTRIES = 5000


class CompatibilityError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise CompatibilityError(message)


def _regular(path: Path):
    _require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),
             f"Expected a regular, non-symlink file: {path.name}")
    for parent in path.parents:
        _require(not parent.is_symlink(), "Symlink parent is not allowed")


def _read(path: Path, maximum=MAX_SOURCE_FILE):
    _regular(path)
    _require(path.stat().st_size <= maximum, f"File exceeds bounded read: {path.name}")
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
    _require(len(value) <= maximum, "File grew beyond bounded read")
    return value


def _blob(value: bytes):
    return hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()


def _sha(path: Path):
    _regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(items, message):
    values = list(items)
    _require(len(values) == 1, message)
    return values[0]


def _replace_span(source, node, replacement):
    lines = source.splitlines(keepends=True)
    start = sum(len(line.encode("utf-8")) for line in lines[:node.lineno - 1]) + node.col_offset
    end = sum(len(line.encode("utf-8")) for line in lines[:node.end_lineno - 1]) + node.end_col_offset
    raw = source.encode("utf-8")
    return (raw[:start] + replacement.encode("utf-8") + raw[end:]).decode("utf-8")


def _function(tree, name):
    return _one((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == name), f"Expected one function: {name}")


def patch_sources(contents: dict[str, bytes]):
    """Pure AST-located, exact-shape transforms; no imported upstream code."""
    result = {}
    source = contents["nodes_v3.py"].decode("utf-8")
    cls = _one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                and n.name == "T8IndexTTS25Generate"), "Missing exact Generate class")
    schema = _function(cls, "define_schema")
    node_ids = [n.value for n in ast.walk(schema) if isinstance(n, ast.keyword)
                and n.arg == "node_id"]
    _require(len(node_ids) == 1 and ast.literal_eval(node_ids[0]) == "T8_IndexTTS25_Generate",
             "Generate Node ID drift")
    combo = _one((n for n in ast.walk(schema) if isinstance(n, ast.Call)
                  and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "language"),
                 "Expected one Generate language input")
    options = _one((n.value for n in combo.keywords if n.arg == "options"), "Missing language options")
    _require(ast.literal_eval(options) == LANGUAGES, "Unexpected Generate language options")
    source = _replace_span(source, options, repr(LANGUAGES + ["HI"]))
    loader = _one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                   and n.name == "T8IndexTTS25ModelLoader"), "Missing exact ModelLoader class")
    execute = _function(loader, "execute")
    load = _one((n for n in ast.walk(execute) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "load_manifest"),
                "Missing exact ModelLoader manifest selection")
    _require(not load.args and not load.keywords, "ModelLoader manifest call drift")
    result["nodes_v3.py"] = _replace_span(source, load, "load_manifest(model_dir)").encode()

    source = contents["runtime/inference_adapter.py"].decode("utf-8")
    function = _function(ast.parse(source), "run_inference")
    guard = _one((n for n in function.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and isinstance(n.test.left, ast.Call) and isinstance(n.test.left.func, ast.Attribute)
                  and isinstance(n.test.left.func.value, ast.Name)
                  and n.test.left.func.value.id == "language" and n.test.left.func.attr == "upper"),
                 "Missing exact runtime language guard")
    _require(len(guard.test.ops) == 1 and isinstance(guard.test.ops[0], ast.NotIn)
             and len(guard.test.comparators) == 1
             and ast.literal_eval(guard.test.comparators[0]) == set(LANGUAGES), "Runtime guard drift")
    result["runtime/inference_adapter.py"] = _replace_span(
        source, guard.test.comparators[0], '{"ZH", "EN", "JA", "ES", "AR", "HI"}').encode()

    source = contents["runtime/text_planner.py"].decode("utf-8")
    tree = ast.parse(source)
    limits = _one((n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "LANGUAGE_AUTO_LIMITS" for t in n.targets)),
                  "Missing planner limits")
    original = {"ZH": 120, "EN": 60, "JA": 100, "ES": 60, "AR": 80}
    _require(ast.literal_eval(limits.value) == original, "Planner limits drift")
    result["runtime/text_planner.py"] = _replace_span(
        source, limits.value, repr({**original, "HI": HI_AUTO_TOKEN_LIMIT})).encode()

    source = contents["services/model_store.py"].decode("utf-8")
    tree = ast.parse(source)
    assignment = _one((n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "MANIFEST_PATH" for t in n.targets)),
                      "Missing manifest assignment")
    _require(ast.unparse(assignment.value) == "PLUGIN_ROOT / 'manifests' / 'model_2_5.json'",
             "Manifest path drift")
    load = _function(tree, "load_manifest")
    _require(not load.args.args and not load.args.kwonlyargs, "Original manifest signature drift")
    # Size is a routing hint only. validate_model_dir still requires every SHA.
    selection = f'''def load_manifest(model_dir: Path | str | None = None) -> dict:
    selected = MANIFEST_PATH
    if model_dir is not None:
        gpt = Path(model_dir) / "gpt.pth"
        if gpt.is_file() and gpt.stat().st_size == {GPT_BYTES}:
            selected = PLUGIN_ROOT / "manifests" / "hindi_best100_experimental.json"
    return json.loads(selected.read_text(encoding="utf-8"))'''
    source = _replace_span(source, load, selection)
    for name in ("validate_model_dir", "model_fingerprint"):
        function = _function(ast.parse(source), name)
        call = _one((n for n in ast.walk(function) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "load_manifest"),
                    f"Missing exact manifest call in {name}")
        _require(not call.args and not call.keywords, "Original manifest call drift")
        source = _replace_span(source, call, "load_manifest(model_dir)")
    tree = ast.parse(source)
    function = _function(tree, "validate_model_dir")
    _require(not (isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant)),
             "Validator docstring/layout drift")
    line = function.body[0].lineno - 1
    lines = source.splitlines(keepends=True)
    lines.insert(line, "    verify_hashes = True  # Isolated dual-manifest policy: never skip actual asset hashes.\n")
    result["services/model_store.py"] = "".join(lines).encode()
    for name, value in result.items():
        compile(value, name, "exec")
    return result


def _revision(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
             "Candidate revision must be the real immutable 40-character HF commit")
    return value


def candidate_manifest(base: dict, revision: str):
    revision = _revision(revision)
    _require(isinstance(base.get("files"), dict) and "gpt.pth" in base["files"], "Invalid base manifest")
    result = copy.deepcopy(base)
    # Preserve the actual source of every unchanged asset when the global
    # repository switches to our GPT-only derivative. Retain upstream overrides.
    for name, metadata in result["files"].items():
        if name != "gpt.pth":
            metadata.setdefault("sourceRepository", base["modelRepository"])
            metadata.setdefault("sourceRevision", base["modelRevision"])
    result.update(modelRepository=CANDIDATE_REPOSITORY, modelRevision=revision)
    result["files"]["gpt.pth"] = {"size": GPT_BYTES, "sha256": GPT_SHA256,
                                 "sourceRepository": CANDIDATE_REPOSITORY, "sourceRevision": revision}
    result["experimentalDerivative"] = {
        "format": "t8-hindi-best100-derivative-v1", "qualityGatePassed": False,
        "baseModelRepository": base["modelRepository"], "baseModelRevision": base["modelRevision"],
        "baseGPT": copy.deepcopy(base["files"]["gpt.pth"]),
        "nodeRepository": NODE_REPOSITORY, "nodeRevision": NODE_REVISION,
        "hiAutoTokenLimit": HI_AUTO_TOKEN_LIMIT, "fullHashValidationRequired": True,
        "scope": "isolated compatibility experiment; not a production acceptance",
    }
    return result


def inspect_node(root: Path):
    root = Path(root).absolute()
    _require(root.is_dir() and not root.is_symlink(), "Node root must be a real directory")
    contents = {name: _read(root / name) for name in PINNED_BLOBS}
    for name, expected in PINNED_BLOBS.items():
        _require(_blob(contents[name]) == expected, f"Pinned original source mismatch: {name}")
    patched = patch_sources(contents)
    return {"status": "verified_original_pinned_source", "node_repository": NODE_REPOSITORY,
            "node_revision": NODE_REVISION, "source_blobs": dict(PINNED_BLOBS),
            "planned_files": sorted(patched), "hi_auto_token_limit": HI_AUTO_TOKEN_LIMIT,
            "weights_alone_add_hi": False}


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    command = ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull,
               "-c", "protocol.allow=never", "-c", "submodule.recurse=false", "-C", str(root), *args]
    return subprocess.run(command, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=45, env=env).stdout


def _safe_member(name):
    path = PurePosixPath(name)
    _require(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name
             and str(path) == name.rstrip("/") and ".git" not in path.parts, "Unsafe source archive path")
    return path


def _audit(revision):
    return {"format": "t8-hindi-isolated-compatibility-v1", "node_repository": NODE_REPOSITORY,
            "node_revision": NODE_REVISION, "candidate_repository": CANDIDATE_REPOSITORY,
            "candidate_revision": _revision(revision), "candidate_gpt_sha256": GPT_SHA256,
            "candidate_gpt_bytes": GPT_BYTES, "manifest": CANDIDATE_MANIFEST,
            "hi_auto_token_limit": HI_AUTO_TOKEN_LIMIT, "quality_gate_passed": False,
            "production_modified": False, "manifest_selection": "fixed_gpt_size_route_then_full_sha256",
            "source_blobs": dict(PINNED_BLOBS),
            "patched_sha256": dict(PATCHED_SHA256)}


def prepare(root: Path, destination: Path, revision: str):
    """Copy only the fixed Git archive into a fresh independent code directory."""
    _revision(revision)
    root = Path(root).absolute()
    inspect_node(root)
    _require(_git(root, "rev-parse", "HEAD").decode().strip() == NODE_REVISION, "Node HEAD is not fixed revision")
    _require(Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() == root.resolve(),
             "Node root must be the Git repository root")
    destination = Path(destination).absolute()
    _require(not destination.exists() and not destination.is_symlink(), "Destination must be fresh")
    _require(destination.parent.is_dir() and destination.parent.resolve() == destination.parent,
             "Destination parent must exist and have no symlink components")
    _require(not destination.is_relative_to(root) and not root.is_relative_to(destination), "Source and copy must be independent")
    archive = _git(root, "archive", "--format=tar", NODE_REVISION)
    _require(len(archive) <= MAX_ARCHIVE, "Source archive exceeds bounded copy")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        members = stream.getmembers()
        _require(len(members) <= MAX_ENTRIES, "Too many source entries")
        seen = set()
        total = 0
        for member in members:
            path = _safe_member(member.name)
            _require(str(path) not in seen, "Duplicate source archive entry")
            seen.add(str(path))
            _require(member.isdir() or member.isreg(), "Source symlinks/special files are forbidden")
            _require(member.size <= MAX_SOURCE_FILE, "Source entry too large")
            total += member.size
        _require(total <= MAX_ARCHIVE, "Expanded source archive too large")
        destination.mkdir(mode=0o700)
        for member in members:
            path = destination.joinpath(*_safe_member(member.name).parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as incoming, path.open("xb") as outgoing:
                    outgoing.write(incoming.read(MAX_SOURCE_FILE + 1))
    inspect_node(destination)
    contents = {name: _read(destination / name) for name in PINNED_BLOBS}
    transformed = patch_sources(contents)
    for name, value in transformed.items():
        _require(hashlib.sha256(value).hexdigest() == PATCHED_SHA256.get(name), "Deterministic patch pin mismatch")
        (destination / name).write_bytes(value)
    base = json.loads(contents[BASE_MANIFEST])
    with (destination / CANDIDATE_MANIFEST).open("x", encoding="utf-8") as stream:
        json.dump(candidate_manifest(base, revision), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (destination / AUDIT).open("x", encoding="utf-8") as stream:
        json.dump(_audit(revision), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return verify_prepared(destination, revision)


def verify_prepared(root: Path, candidate_revision: str | None = None):
    root = Path(root).absolute()
    _require(root.is_dir() and root.resolve() == root, "Prepared node root must be a real independent directory")
    audit = json.loads(_read(root / AUDIT, 64 * 1024))
    revision = _revision(candidate_revision if candidate_revision is not None else audit.get("candidate_revision"))
    _require(audit == _audit(revision), "Compatibility audit identity/schema mismatch")
    for name, expected in PINNED_BLOBS.items():
        data = _read(root / name)
        if name in PATCHED_SHA256:
            _require(hashlib.sha256(data).hexdigest() == PATCHED_SHA256[name], f"Patched source drift: {name}")
        else:
            _require(_blob(data) == expected, f"Retained pinned source drift: {name}")
    base = json.loads(_read(root / BASE_MANIFEST))
    manifest = json.loads(_read(root / CANDIDATE_MANIFEST, 128 * 1024))
    _require(manifest == candidate_manifest(base, revision), "Derivative manifest differs from fixed candidate/base identity")
    return {"status": "verified_isolated_code_copy", **audit, "model_assets_verified": False,
            "gpu_tested": False, "manifest_data": manifest}


def selected_manifest(root: Path, model_dir: Path, candidate_revision: str,
                      *, variant: str | None = None):
    """Select only one of the two reviewed local manifests; this is not authorization.

    The caller must subsequently run verify_assets on every selected asset.
    Unknown or missing GPT sizes route to base and fail its actual validation.
    """
    info = verify_prepared(root, candidate_revision)
    gpt = Path(model_dir) / "gpt.pth"
    selected = "hindi" if gpt.is_file() and gpt.stat().st_size == GPT_BYTES else "base"
    _require(variant is None or variant in {"base", "hindi"}, "Unknown requested model variant")
    _require(variant is None or variant == selected, "Model directory size selector differs from requested variant")
    manifest = info["manifest_data"] if selected == "hindi" else json.loads(_read(Path(root) / BASE_MANIFEST))
    return {"variant": selected, "selector": "gpt_size_hint_only", "assets_verified": False,
            "manifest": CANDIDATE_MANIFEST if selected == "hindi" else BASE_MANIFEST,
            "manifest_data": manifest}


def verify_assets(model_dir: Path, manifest: dict):
    """Read and hash every manifest asset, including auxiliary files, before load."""
    model_dir = Path(model_dir).absolute()
    _require(model_dir.is_dir() and model_dir.resolve() == model_dir, "Asset root must have no symlink components")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "Manifest assets missing")
    total = 0
    for name, metadata in files.items():
        relative = _safe_member(name)
        path = model_dir.joinpath(*relative.parts)
        _regular(path)
        _require(path.stat().st_size == metadata["size"], f"Asset size mismatch: {name}")
        _require(_sha(path) == metadata["sha256"], f"Asset SHA256 mismatch: {name}")
        total += metadata["size"]
    return {"status": "verified_all_manifest_asset_hashes", "files": len(files), "bytes": total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="Read-only pin/AST check of the original node")
    inspect.add_argument("--node-root", required=True, type=Path)
    prep = sub.add_parser("prepare", help="Create a fresh independent code copy; stage no models")
    prep.add_argument("--node-root", required=True, type=Path)
    prep.add_argument("--destination", required=True, type=Path)
    prep.add_argument("--candidate-revision", required=True)
    verify = sub.add_parser("verify-copy", help="Read-only strict check of the isolated code copy")
    verify.add_argument("--node-root", required=True, type=Path)
    verify.add_argument("--candidate-revision", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            result = inspect_node(args.node_root)
        elif args.action == "prepare":
            result = prepare(args.node_root, args.destination, args.candidate_revision)
        else:
            result = verify_prepared(args.node_root, args.candidate_revision)
        result.pop("manifest_data", None)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (CompatibilityError, OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
