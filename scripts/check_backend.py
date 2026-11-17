#!/usr/bin/env python3
"""Inspect pinned local sources and optional weights without importing upstream code.

This is a static preflight, never a trainer, downloader, or GPU probe.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class PreflightError(ValueError):
    pass


def strict_json(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError("Duplicate JSON key: " + key)
            obj[key] = value
        return obj
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise PreflightError(f"Missing or oversized JSON: {path}")
    try:
        value = strict_json(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise PreflightError(f"Invalid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"Expected JSON object: {path}")
    return value


def git_read(repo, *args):
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.untrackedCache=false", "-C", str(repo), *args], check=True,
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"Cannot inspect local git tree: {repo}") from exc
    return result.stdout.strip()


def check_tree(path, spec):
    root = Path(path).resolve(strict=True)
    if git_read(root, "rev-parse", "--show-toplevel") != str(root):
        raise PreflightError(f"Pass the repository root, not a subdirectory: {root}")
    head = git_read(root, "rev-parse", "HEAD")
    if head != spec["commit"]:
        raise PreflightError(f"Unexpected commit in {root}: {head}; expected {spec['commit']}")
    if git_read(root, "status", "--porcelain", "--untracked-files=all"):
        raise PreflightError(f"Source tree has modified/untracked files: {root}; use a clean isolated checkout")
    for relative in spec["required_paths"]:
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise PreflightError(f"Required source path is missing or escapes repository: {relative}")
    return root


def source_ast(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise PreflightError(f"Oversized source file: {path}")
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (ValueError, SyntaxError, UnicodeError) as exc:
        raise PreflightError(f"Cannot statically parse source: {path}") from exc


def cli_destinations(path):
    """Read literal argparse declarations; never execute a dependency module."""
    names = set()
    for node in ast.walk(source_ast(path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flags = [item.value for item in node.args
                 if isinstance(item, ast.Constant) and isinstance(item.value, str)]
        long_flag = next((flag for flag in flags if flag.startswith("--")), None)
        if long_flag is None:
            continue
        dest = next((kw.value.value for kw in node.keywords
                     if kw.arg == "dest" and isinstance(kw.value, ast.Constant)), None)
        names.add(dest or long_flag[2:].replace("-", "_"))
    if not names:
        raise PreflightError(f"No statically readable argparse fields: {path}")
    return names


def check_hindi_slot(tokenizer_path):
    tree = source_ast(tokenizer_path)
    languages = None
    default_languages = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LANGUAGES" for t in node.targets):
            try:
                languages = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise PreflightError("LANGUAGES is not a literal table") from exc
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_encoding":
            positional = node.args.args
            offset = len(positional) - len(node.args.defaults)
            for index, default in enumerate(node.args.defaults, offset):
                if positional[index].arg == "num_languages":
                    default_languages = ast.literal_eval(default)
    if not isinstance(languages, dict) or list(languages).index("hi") != 17 or default_languages != 99:
        raise PreflightError("Reviewed Hindi row/control-token prerequisites have changed")
    return {"code": "hi", "embedding_row": 17, "control_token": "<|hi|>",
            "support_or_quality_verified": False}


def check_rights_record(path, backend_commit):
    record = read_json(path)
    allowed = {"backend_commit", "permission_confirmed", "permission_basis"}
    if set(record) != allowed or record.get("backend_commit") != backend_commit:
        raise PreflightError("Rights record schema or backend commit does not match upstream.lock.json")
    basis = record.get("permission_basis")
    if record.get("permission_confirmed") is not True or not isinstance(basis, str) or len(basis.strip()) < 12:
        raise PreflightError("Backend rights are not confirmed in the supplied record")
    return {"record_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "scope": "User-supplied rights declaration; this tool cannot verify a legal permission grant."}


def contained_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PreflightError("Model manifest paths must be safe relative paths")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise PreflightError(f"Model path escapes root or is not a file: {relative}")
    return path


def check_weights(model_dir, manifest_path, model_spec):
    root = Path(model_dir).resolve(strict=True)
    manifest = read_json(manifest_path)
    if set(manifest) != {"repository", "snapshot_revision", "files"}:
        raise PreflightError("Weight manifest needs exactly repository,snapshot_revision,files")
    if manifest["repository"] != model_spec["repository"] or not re.fullmatch(r"[0-9a-f]{40}", str(manifest["snapshot_revision"])):
        raise PreflightError("Pin the actual IndexTeam/IndexTTS-2.5 immutable snapshot revision")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(model_spec["minimum_files"]).issubset(files):
        raise PreflightError("Weight manifest is missing minimum 2.5 files")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise PreflightError(f"Expected SHA256 for model file: {relative}")
        digest = hashlib.sha256()
        with contained_file(root, relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise PreflightError(f"Model SHA256 mismatch: {relative}")
    config = contained_file(root, "config.yaml").read_text(encoding="utf-8")
    if not re.search(r"(?m)^version:\s*[\"']?2\.5[\"']?\s*(?:#.*)?$", config):
        raise PreflightError("config.yaml must explicitly declare version: 2.5; inspect the actual 2.5 model config")
    return {"repository": manifest["repository"], "snapshot_revision": manifest["snapshot_revision"],
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "verified_sha256_files": len(files),
            "auxiliary_assets_complete": False,
            "note": "File hashes verify supplied files, not trusted provenance. Independently verify model origin/terms and perform offline runtime setup; auxiliary assets may be missing."}


def inspect(official_repo, backend_repo, model_dir=None, rights_record=None,
            weight_manifest=None, source_only=False):
    lock = read_json(ROOT / "upstream.lock.json")
    official = check_tree(official_repo, lock["official"])
    backend = check_tree(backend_repo, lock["backend"])
    slot = check_hindi_slot(official / "indextts/utils/tokenizer.py")
    fields = cli_destinations(backend / "scripts/train.py")
    needed = {"lang", "trainable_mode", "train_text_rows", "train_heads", "amp", "max_steps", "normalizer", "case", "final_export"}
    if not needed.issubset(fields):
        raise PreflightError("Reviewed training CLI schema no longer matches")
    report = {"status": "source_verified", "official_commit": lock["official"]["commit"],
              "backend_commit": lock["backend"]["commit"], "language": slot,
              "static_inputs_verified": False, "ready_for_training": False,
              "gpu_training_verified": False, "runtime_backend_imported": False,
              "backend_license": lock["backend"]["license_status"]}
    if not source_only:
        if not all((model_dir, rights_record, weight_manifest)):
            raise PreflightError("Full static preflight requires --model-dir, --rights-record and --weight-manifest; otherwise use --source-only")
        report["rights"] = check_rights_record(rights_record, lock["backend"]["commit"])
        report["weights"] = check_weights(model_dir, weight_manifest, lock["model"])
        report["status"] = "static_inputs_verified"
        report["static_inputs_verified"] = True
        report["remaining_gates"] = ["Resolved dependency lock", "Offline actual backend check_setup", "Tokenizer/frontend parity and feature integrity", "Small-sample overfit", "100 optimizer steps on H200", "Evaluation and adapter parity"]
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-repo", required=True, type=Path)
    p.add_argument("--backend-repo", required=True, type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--rights-record", type=Path)
    p.add_argument("--weight-manifest", type=Path)
    p.add_argument("--source-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = inspect(**vars(args))
    except (PreflightError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "gpu_training_verified": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
