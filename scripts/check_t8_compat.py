#!/usr/bin/env python3
"""Inspect pinned T8 code or prepare an isolated experimental Hindi code copy.

No model loading, network, package installation, production edits or GPU work.
Historical best100 behavior remains unchanged. Explicit reviewed candidates use
a SHA-bound registration marker, never a size selector. Every selected model
requires complete asset hashes. Models are not copied; bind-candidate can write
one new selector only after all candidate assets pass.
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
# One fixed public Git-tree asset exceeds the ordinary source-file bound.
LARGE_SOURCE_ENTRIES = {
    "indextts/utils/maskgct/models/codec/facodec/modules/JDC/bst.t7": {
        "size": 21029926, "blob": "d6cf419f9e964af2fd4e3a110c0f0af5fa6e6a43"},
}
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


def _source_entry_pin(member):
    """Only the exact fixed-tree path/length may exceed the usual file bound."""
    pinned = LARGE_SOURCE_ENTRIES.get(member.name)
    if pinned is not None:
        _require(member.isreg() and member.size == pinned["size"], "Pinned source entry size/type mismatch")
    else:
        _require(0 <= member.size <= MAX_SOURCE_FILE, "Source entry too large")
    return pinned


def _copy_source_entry(incoming, outgoing, member):
    pinned = _source_entry_pin(member)
    digest = hashlib.sha1(b"blob " + str(member.size).encode() + b"\0")
    remaining = member.size
    while remaining:
        chunk = incoming.read(min(1024 * 1024, remaining))
        _require(chunk and len(chunk) <= remaining, "Truncated or oversized source entry")
        outgoing.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    _require(not incoming.read(1), "Source entry length mismatch")
    if pinned is not None:
        _require(digest.hexdigest() == pinned["blob"], "Pinned source entry Git blob mismatch")


def _audit(revision):
    return {"format": "t8-hindi-isolated-compatibility-v1", "node_repository": NODE_REPOSITORY,
            "node_revision": NODE_REVISION, "candidate_repository": CANDIDATE_REPOSITORY,
            "candidate_revision": _revision(revision), "candidate_gpt_sha256": GPT_SHA256,
            "candidate_gpt_bytes": GPT_BYTES, "manifest": CANDIDATE_MANIFEST,
            "hi_auto_token_limit": HI_AUTO_TOKEN_LIMIT, "quality_gate_passed": False,
            "production_modified": False, "manifest_selection": "fixed_gpt_size_route_then_full_sha256",
            "source_blobs": dict(PINNED_BLOBS),
            "patched_sha256": dict(PATCHED_SHA256)}


# Explicit reviewed-candidate mode is separate from the historical fixed best100.
REVIEWED_MARKER = "T8_REVIEWED_CANDIDATE.json"
REVIEWED_MANIFEST = "manifests/reviewed_complete_gpt.json"
REVIEWED_REGISTRATION = "manifests/reviewed_candidate_registration.json"
REVIEWED_FORMAT = "t8-reviewed-complete-gpt-v1"
ORIGINAL_GPT_SHA256 = "43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85"
ORIGINAL_GPT_BYTES = 3259599833
DOWNLOADER_PATH = "services/downloader.py"
DOWNLOADER_BLOB = "4eadf054e587de4bada954b26a68e7d75434e0b7"
OFFLINE_ERROR = "受审完整GPT模式禁止自动下载或修复；请离线准备并核验全部26项资产。"


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate candidate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(CompatibilityError("Nonfinite candidate JSON")))


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def candidate_options(args):
    path = getattr(args, "candidate_manifest", None)
    digest = getattr(args, "candidate_manifest_sha256", None)
    return {} if path is None and digest is None else {"candidate_manifest_path": path, "candidate_manifest_sha256": digest}


def read_registration(path, expected_sha256):
    _require(path is not None and isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
             "Explicit candidate registration path and independently recorded SHA256 are both required")
    raw = _read(Path(path), 64 * 1024)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Reviewed registration SHA256 mismatch")
    value = _strict_json(raw)
    keys = {"format", "candidate_id", "repository", "revision", "training_profile", "adapter_sha256", "gpt_sha256",
            "gpt_bytes", "export_complete_sha256", "export_provenance_sha256", "quality_gate_passed"}
    _require(isinstance(value, dict) and set(value) == keys and value["format"] == REVIEWED_FORMAT,
             "Unsupported reviewed complete-GPT registration schema")
    _require(isinstance(value["candidate_id"], str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,79}", value["candidate_id"]), "Unsafe candidate ID")
    _require(value["training_profile"] == "attention_lora_only_v1" and value["quality_gate_passed"] is False,
             "Only explicitly experimental pure-attention merged candidates are supported")
    for name in ("adapter_sha256", "gpt_sha256", "export_complete_sha256", "export_provenance_sha256"):
        _require(isinstance(value[name], str) and re.fullmatch(r"[0-9a-f]{64}", value[name]), "Invalid candidate digest: " + name)
    _require(type(value["gpt_bytes"]) is int and 0 < value["gpt_bytes"] <= 8 * 1024**3
             and value["gpt_sha256"] != ORIGINAL_GPT_SHA256, "Candidate must identify a bounded distinct complete GPT")
    repository, revision = value["repository"], value["revision"]
    _require((repository is None and revision is None) or
             (isinstance(repository, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", repository)
              and isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)),
             "Repository/revision must both be null for local artifacts, or an explicitly pinned HF pair")
    return value, raw


def reviewed_candidate_manifest(base, registration):
    _require(len(base.get("files", {})) == 26 and base["files"].get("gpt.pth", {}).get("sha256") == ORIGINAL_GPT_SHA256
             and base["files"]["gpt.pth"].get("size") == ORIGINAL_GPT_BYTES, "Pinned original 26-asset base identity differs")
    result = copy.deepcopy(base)
    for name, metadata in result["files"].items():
        if name != "gpt.pth":
            metadata.setdefault("sourceRepository", base["modelRepository"])
            metadata.setdefault("sourceRevision", base["modelRevision"])
    repository = registration["repository"] or "local-reviewed/" + registration["candidate_id"]
    revision = registration["revision"] or "sha256:" + registration["gpt_sha256"]
    result.update(modelRepository=repository, modelRevision=revision)
    result["files"]["gpt.pth"] = {"size": registration["gpt_bytes"], "sha256": registration["gpt_sha256"],
                                 "sourceRepository": repository, "sourceRevision": revision}
    result["experimentalDerivative"] = {"format": REVIEWED_FORMAT, "qualityGatePassed": False,
        "baseModelRepository": base["modelRepository"], "baseModelRevision": base["modelRevision"],
        "baseGPT": copy.deepcopy(base["files"]["gpt.pth"]), "nodeRepository": NODE_REPOSITORY, "nodeRevision": NODE_REVISION,
        "trainingProfile": registration["training_profile"], "adapterSHA256": registration["adapter_sha256"],
        "exportCompleteSHA256": registration["export_complete_sha256"], "exportProvenanceSHA256": registration["export_provenance_sha256"],
        "fullHashValidationRequired": True, "automaticDownloads": False, "hiAutoTokenLimit": HI_AUTO_TOKEN_LIMIT,
        "tensorAuditVerifiedByThisTool": False, "scope": "explicit reviewed complete-GPT experiment; not production acceptance"}
    return result


def _insert_guard(source, function, text):
    first = function.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        first = function.body[1]
    lines = source.splitlines(keepends=True)
    lines.insert(first.lineno - 1, "\n".join(" " * first.col_offset + line for line in text.splitlines()) + "\n")
    return "".join(lines)


def reviewed_patch_sources(contents, downloader, registration, registration_sha256):
    changed = patch_sources(contents)
    manifest_raw = _encoded(reviewed_candidate_manifest(_strict_json(contents[BASE_MANIFEST]), registration))
    source = changed["services/model_store.py"].decode()
    load = _function(ast.parse(source), "load_manifest")
    selection = f'''def load_manifest(model_dir: Path | str | None = None) -> dict:
    import stat
    selected = MANIFEST_PATH
    expected = {hashlib.sha256(contents[BASE_MANIFEST]).hexdigest()!r}
    if model_dir is not None:
        marker = Path(model_dir) / {REVIEWED_MARKER!r}
        if marker.exists() or marker.is_symlink():
            if marker.is_symlink() or not stat.S_ISREG(marker.lstat().st_mode) or marker.stat().st_size > 65536:
                raise ValueError("Invalid reviewed candidate selector")
            with marker.open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != {registration_sha256!r}:
                raise ValueError("Candidate selector differs from the reviewed registration")
            selected = PLUGIN_ROOT / {REVIEWED_MANIFEST!r}
            expected = {hashlib.sha256(manifest_raw).hexdigest()!r}
    if selected.is_symlink() or not stat.S_ISREG(selected.lstat().st_mode) or selected.stat().st_size > 131072:
        raise ValueError("Invalid fixed model manifest")
    with selected.open("rb") as stream:
        data = stream.read(131073)
    if len(data) > 131072 or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Fixed model manifest changed")
    return json.loads(data.decode("utf-8"))'''
    source = _replace_span(source, load, selection)
    function = _function(ast.parse(source), "validate_model_dir")
    _require("include_auxiliary" in {arg.arg for arg in function.args.kwonlyargs}, "Validator auxiliary contract changed")
    source = _insert_guard(source, function, "include_auxiliary = True  # Reviewed mode always validates all 26 assets.")
    report = _one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ValidationReport"), "Missing original validation report")
    function = _function(report, "require_valid")
    raised = _one((n for n in ast.walk(function) if isinstance(n, ast.Raise)), "Validation error contract changed")
    source = _replace_span(source, raised, 'raise FileNotFoundError(' + repr(OFFLINE_ERROR) + ' + str(self.model_dir) + str(self.missing) + str(self.mismatched))')
    changed["services/model_store.py"] = source.encode()
    source = changed["nodes_v3.py"].decode()
    loader = _one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "T8IndexTTS25ModelLoader"), "Missing loader")
    for name, guard in (("validate_inputs", 'if download_missing:\n    return ' + repr(OFFLINE_ERROR)),
                        ("execute", 'if download_missing:\n    raise ValueError(' + repr(OFFLINE_ERROR) + ')')):
        function = _function(loader, name)
        _require("download_missing" in {arg.arg for arg in function.args.args}, "Loader download guard contract changed")
        source = _insert_guard(source, function, guard)
        loader = _one((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "T8IndexTTS25ModelLoader"), "Missing loader")
    changed["nodes_v3.py"] = source.encode()
    source = downloader.decode()
    for name in ("ensure_model_bundle", "download_main_model", "download_auxiliary_models", "_download_hf_file_cancellable"):
        source = _insert_guard(source, _function(ast.parse(source), name), "raise RuntimeError(" + repr(OFFLINE_ERROR) + ")")
    # The surviving verify-only CLI must not label the derivative 'official'.
    literals = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str) and "正式模型" in n.value]
    _require(len(literals) == 3, "Downloader identity wording changed")
    for node in sorted(literals, key=lambda item: (item.lineno, item.col_offset), reverse=True):
        message = ("离线校验 IndexTTS 2.5 原版或受审完整GPT；自动下载停用"
                   if node.value == "下载并校验 IndexTTS 2.5 正式模型" else node.value.replace("正式模型", "原版或受审完整GPT"))
        source = _replace_span(source, node, repr(message))
    changed[DOWNLOADER_PATH] = source.encode()
    for name, value in changed.items():
        compile(value, name, "exec")
    return changed, manifest_raw


def _reviewed_audit(registration, digest, changed):
    return {"format": "t8-reviewed-complete-gpt-code-v1", "node_repository": NODE_REPOSITORY, "node_revision": NODE_REVISION,
        "candidate_repository": registration["repository"], "candidate_revision": registration["revision"],
        "candidate_gpt_sha256": registration["gpt_sha256"], "candidate_gpt_bytes": registration["gpt_bytes"],
        "candidate_manifest_sha256": digest, "registration": registration, "manifest": REVIEWED_MANIFEST,
        "selector_marker": REVIEWED_MARKER, "manifest_selection": "reviewed_registration_marker_then_full_sha256",
        "source_blobs": dict(PINNED_BLOBS), "original_downloader_blob": DOWNLOADER_BLOB,
        "patched_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in changed.items()},
        "quality_gate_passed": False, "production_modified": False, "tensor_audit_verified": False}


def _reviewed_originals(root):
    contents = {name: _read(Path(root) / name) for name in PINNED_BLOBS}
    for name, blob in PINNED_BLOBS.items():
        _require(_blob(contents[name]) == blob, "Pinned original source mismatch: " + name)
    downloader = _read(Path(root) / DOWNLOADER_PATH)
    _require(_blob(downloader) == DOWNLOADER_BLOB, "Original downloader source differs")
    return contents, downloader


def _original_blob(root, name):
    # Retain bounded, pinned source preimages to independently recompute the
    # deterministic patch; these are licensed source, not model weights.
    return Path(root) / ".reviewed-source" / name


def _reviewed_prepare(root, destination, registration_path, digest):
    registration, raw = read_registration(registration_path, digest)
    contents, downloader = _reviewed_originals(root)
    changed, manifest_raw = reviewed_patch_sources(contents, downloader, registration, digest)
    _copy_pristine(root, destination)
    destination = Path(destination).absolute()
    for name, data in {**contents, DOWNLOADER_PATH: downloader}.items():
        path = _original_blob(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    for name, value in changed.items():
        (destination / name).write_bytes(value)
    for name, data in ((REVIEWED_MANIFEST, manifest_raw), (REVIEWED_REGISTRATION, raw), (AUDIT, _encoded(_reviewed_audit(registration, digest, changed)))):
        with (destination / name).open("xb") as stream:
            stream.write(data)
    return _reviewed_verify(destination, registration_path, digest)


def _reviewed_verify(root, registration_path, digest):
    registration, raw = read_registration(registration_path, digest)
    root = Path(root).absolute()
    _require(root.is_dir() and root.resolve(strict=True) == root, "Prepared node root must be canonical")
    contents, downloader = _reviewed_originals(root / ".reviewed-source")
    changed, manifest_raw = reviewed_patch_sources(contents, downloader, registration, digest)
    audit = _reviewed_audit(registration, digest, changed)
    _require(_read(root / REVIEWED_REGISTRATION, 64 * 1024) == raw and _read(root / REVIEWED_MANIFEST, 128 * 1024) == manifest_raw,
             "Embedded reviewed manifest/registration changed")
    _require(_encoded(_strict_json(_read(root / AUDIT, 128 * 1024))) == _encoded(audit), "Reviewed candidate audit differs")
    for name, data in contents.items():
        _require(_read(root / name) == changed.get(name, data), "Reviewed prepared source drift: " + name)
    _require(_read(root / DOWNLOADER_PATH) == changed[DOWNLOADER_PATH], "Reviewed downloader source drift")
    return {"status": "verified_reviewed_complete_gpt_code_copy", **audit,
            "model_assets_verified": False, "gpu_tested": False, "manifest_data": _strict_json(manifest_raw),
            "base_manifest_data": _strict_json(contents[BASE_MANIFEST])}


def bind_candidate(root, model_dir, registration_path, digest, workspace):
    """Only install an explicit selector after every real candidate asset passes."""
    info = _reviewed_verify(root, registration_path, digest)
    model_dir, workspace = Path(model_dir).absolute(), Path(workspace).absolute()
    _require(workspace.is_dir() and workspace.resolve(strict=True) == workspace
             and re.fullmatch(r"indextts25-hindi-[A-Za-z0-9][A-Za-z0-9_-]{0,127}", workspace.name),
             "WORK must be a dedicated canonical indextts25-hindi-* project directory")
    _require(model_dir.resolve(strict=True) == model_dir and model_dir != workspace and model_dir.is_relative_to(workspace),
             "Selector destination must be a dedicated canonical model directory inside WORK")
    for protected in (Path(root).resolve(strict=True), Path(registration_path).resolve(strict=True)):
        _require(not (model_dir == protected or model_dir.is_relative_to(protected) or protected.is_relative_to(model_dir)),
                 "Selector destination overlaps prepared source or registration")
    assets = verify_assets(model_dir, info["manifest_data"])
    _, raw = read_registration(registration_path, digest)
    marker = model_dir / REVIEWED_MARKER
    with marker.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {"status": "reviewed_candidate_marker_installed", "registration_sha256": digest,
            "assets": assets, "gpu_tested": False, "production_modified": False}


def _copy_pristine(root: Path, destination: Path):
    """Copy only the fixed Git archive into a fresh independent code directory."""
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
            _source_entry_pin(member)
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
                    _copy_source_entry(incoming, outgoing, member)
    inspect_node(destination)
    return {name: _read(destination / name) for name in PINNED_BLOBS}


def prepare(root: Path, destination: Path, revision: str | None = None, *,
            candidate_manifest_path=None, candidate_manifest_sha256=None):
    if candidate_manifest_path is not None or candidate_manifest_sha256 is not None:
        _require(revision is None, "Legacy revision and explicit registration are mutually exclusive")
        return _reviewed_prepare(root, destination, candidate_manifest_path, candidate_manifest_sha256)
    _revision(revision)
    contents = _copy_pristine(root, destination)
    destination = Path(destination).absolute()
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


def verify_prepared(root: Path, candidate_revision: str | None = None, *,
                    candidate_manifest_path=None, candidate_manifest_sha256=None):
    if candidate_manifest_path is not None or candidate_manifest_sha256 is not None:
        _require(candidate_revision is None, "Legacy revision and explicit registration are mutually exclusive")
        return _reviewed_verify(root, candidate_manifest_path, candidate_manifest_sha256)
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


def selected_manifest(root: Path, model_dir: Path, candidate_revision: str | None = None,
                      *, variant: str | None = None, candidate_manifest_path=None, candidate_manifest_sha256=None):
    if candidate_manifest_path is not None or candidate_manifest_sha256 is not None:
        _require(candidate_revision is None, "Legacy revision and explicit registration are mutually exclusive")
        info = _reviewed_verify(root, candidate_manifest_path, candidate_manifest_sha256)
        marker = Path(model_dir) / REVIEWED_MARKER
        selected = "base"
        if marker.exists() or marker.is_symlink():
            _require(hashlib.sha256(_read(marker, 64 * 1024)).hexdigest() == candidate_manifest_sha256,
                     "Model selector differs from the explicitly reviewed registration")
            selected = "hindi"
        _require(variant is None or variant in {"base", "hindi"}, "Unknown requested model variant")
        _require(variant is None or selected == variant, "Reviewed model selector differs from requested variant")
        manifest = info["manifest_data"] if selected == "hindi" else _strict_json(_read(Path(root) / BASE_MANIFEST))
        return {"variant": selected, "selector": "reviewed_registration_marker_then_full_sha256",
                "registration_sha256": candidate_manifest_sha256, "assets_verified": False,
                "manifest": REVIEWED_MANIFEST if selected == "hindi" else BASE_MANIFEST, "manifest_data": manifest}
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
    verify = sub.add_parser("verify-copy", help="Read-only strict check of the isolated code copy")
    verify.add_argument("--node-root", required=True, type=Path)
    for command in (prep, verify):
        group = command.add_mutually_exclusive_group(required=True)
        group.add_argument("--candidate-revision")
        group.add_argument("--candidate-manifest", type=Path)
        command.add_argument("--candidate-manifest-sha256")
    assets = sub.add_parser("verify-assets", help="Read-only full hash validation of the explicitly selected variant")
    assets.add_argument("--node-root", type=Path, required=True)
    assets.add_argument("--model-dir", type=Path, required=True)
    assets.add_argument("--model-variant", choices=("base", "hindi"), required=True)
    group = assets.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate-revision")
    group.add_argument("--candidate-manifest", type=Path)
    assets.add_argument("--candidate-manifest-sha256")
    bind = sub.add_parser("bind-candidate", help="Verify all candidate assets, then write one fresh reviewed selector inside WORK")
    for name in ("node-root", "model-dir", "workspace-root", "candidate-manifest"):
        bind.add_argument("--" + name, type=Path, required=True)
    bind.add_argument("--candidate-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            result = inspect_node(args.node_root)
        elif args.action == "prepare":
            result = prepare(args.node_root, args.destination, args.candidate_revision, **candidate_options(args))
        elif args.action == "verify-copy":
            result = verify_prepared(args.node_root, args.candidate_revision, **candidate_options(args))
        elif args.action == "verify-assets":
            selection = selected_manifest(args.node_root, args.model_dir, args.candidate_revision,
                                          variant=args.model_variant, **candidate_options(args))
            result = {"selection": {key:value for key,value in selection.items() if key != "manifest_data"},
                      "assets": verify_assets(args.model_dir, selection["manifest_data"]), "gpu_tested": False}
        else:
            result = bind_candidate(args.node_root, args.model_dir, args.candidate_manifest,
                                    args.candidate_manifest_sha256, args.workspace_root)
        result.pop("manifest_data", None)
        result.pop("base_manifest_data", None)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (CompatibilityError, OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
