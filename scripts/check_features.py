#!/usr/bin/env python3
"""Audit real feature arrays; optionally compare every row with the approved backend frontend.

NumPy is imported only when the user runs this gate in the prepared environment.
No models or corpora are downloaded. The script never starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import io
import os
from pathlib import Path
import sys
import zipfile

if __package__:
    from .check_backend import ROOT, PreflightError, contained_file, inspect, read_json, strict_json
else:
    from check_backend import ROOT, PreflightError, contained_file, inspect, read_json, strict_json

MAX_LINE = 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line) > MAX_LINE or not line.strip():
                raise PreflightError(f"Blank/oversized manifest line: {path}:{line_number}")
            try:
                value = strict_json(line)
            except ValueError as exc:
                raise PreflightError(f"Invalid JSON: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise PreflightError(f"Manifest row is not an object: {path}:{line_number}")
            yield value


def local_feature(path, base, root):
    if not isinstance(path, str) or not path or "\x00" in path:
        raise PreflightError("Feature path must be a nonempty local path")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(base) / candidate
    candidate = candidate.resolve(strict=True)
    if not candidate.is_relative_to(Path(root).resolve()) or not candidate.is_file() or candidate.suffix != ".npz":
        raise PreflightError(f"Feature path must be a confined NPZ file: {path}")
    return candidate


def inspect_npz(path, record, np, limits, expected_tokens=None):
    expected_keys = {"codes", "text_ids", "spk_emb", "emo_vec"}
    try:
        with Path(path).open("rb") as stream:
            payload = stream.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise PreflightError(f"NPZ file exceeds safety bound: {path}")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if {item.filename for item in entries} != {key + ".npy" for key in expected_keys} or len(entries) != 4:
                raise PreflightError(f"Unexpected NPZ members: {path}")
            if any(item.file_size > 8 * 1024 * 1024 for item in entries) or sum(item.file_size for item in entries) > 16 * 1024 * 1024:
                raise PreflightError(f"NPZ expansion exceeds safety bound: {path}")
            for item in entries:
                if item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) or item.flag_bits & 1:
                    raise PreflightError(f"Unsupported NPZ compression/encryption: {path}")
                key = item.filename[:-4]
                dtype_expected = "int32" if key in ("codes", "text_ids") else "float32"
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise PreflightError(f"Unsupported NPY header version: {path}")
                    if dtype.hasobject or dtype != np.dtype(dtype_expected) or len(shape) != 1:
                        raise PreflightError(f"Unsafe NPY dtype/shape: {path}")
                    expected_length = {"spk_emb": 192, "emo_vec": limits["emotion_dim"]}.get(key)
                    upper = limits["max_code_tokens"] if key == "codes" else limits["max_text_tokens"]
                    if expected_length is not None:
                        if shape != (expected_length,):
                            raise PreflightError(f"Invalid NPY embedding shape: {path}")
                    elif not 1 <= shape[0] <= upper:
                        raise PreflightError(f"NPY token shape exceeds configured limit: {path}")
                    byte_count = shape[0] * dtype.itemsize
                    if byte_count > 8 * 1024 * 1024 or byte_count != item.file_size - stream.tell():
                        raise PreflightError(f"NPY payload does not match bounded header shape: {path}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            data = {key: archive[key] for key in expected_keys}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise PreflightError(f"Cannot safely read feature NPZ: {path}: {exc}") from exc
    for key, dtype, shape in (("codes", "int32", None), ("text_ids", "int32", None),
                              ("spk_emb", "float32", (192,)), ("emo_vec", "float32", (limits["emotion_dim"],))):
        array = data[key]
        if array.dtype != np.dtype(dtype) or array.ndim != 1 or (shape is not None and array.shape != shape):
            raise PreflightError(f"Invalid {key} dtype/shape: {path}")
        if not np.isfinite(array).all():
            raise PreflightError(f"Nonfinite {key}: {path}")
    codes, text = data["codes"], data["text_ids"]
    if not 8 <= len(codes) <= limits["max_code_tokens"] or not 1 <= len(text) <= limits["max_text_tokens"]:
        raise PreflightError(f"Actual feature lengths exceed model limits: {path}")
    if (codes < 0).any() or (codes >= limits["codebook_size"]).any():
        raise PreflightError(f"Reserved or out-of-range semantic codec ID: {path}")
    if (text < 0).any() or (text >= limits["number_text_tokens"]).any():
        raise PreflightError(f"Out-of-range text token ID: {path}")
    if type(record.get("code_len")) is not int or type(record.get("text_len")) is not int or record["code_len"] != len(codes) or record["text_len"] != len(text):
        raise PreflightError(f"Manifest lengths disagree with actual arrays: {path}")
    if expected_tokens is not None and text.tolist() != list(expected_tokens):
        raise PreflightError(f"Cached token IDs differ from freshly encoded Hindi text: {path}")
    return {"text_len": len(text), "code_len": len(codes), "sha256": hashlib.sha256(payload).hexdigest()}


def check_model_limits(cfg, args):
    gpt = cfg.gpt
    actual = {"number_text_tokens": gpt.number_text_tokens, "max_text_tokens": gpt.max_text_tokens,
              "max_code_tokens": gpt.max_mel_tokens, "emotion_dim": gpt.model_dim,
              "codebook_size": cfg.semantic_codec.codebook_size}
    reserved = (gpt.number_mel_codes, gpt.start_mel_token, gpt.stop_mel_token)
    if any(type(value) is not int or value < 1 for value in (*actual.values(), *reserved)):
        raise PreflightError("Actual model feature capacities must be positive integers")
    if reserved != (actual["codebook_size"] + 2, actual["codebook_size"], actual["codebook_size"] + 1):
        raise PreflightError("Actual codec/GPT reserved start/stop IDs do not match reviewed 2.5 format")
    for key in ("number_text_tokens", "emotion_dim", "codebook_size"):
        if getattr(args, key) != actual[key]:
            raise PreflightError("Feature limit does not match actual hashed model config: " + key)
    if args.max_text_tokens > actual["max_text_tokens"] or args.max_code_tokens > actual["max_code_tokens"] or args.max_prompt_code_tokens > actual["max_code_tokens"]:
        raise PreflightError("Requested feature length exceeds actual model config")
    return actual


def backend_frontend(args):
    if not all((args.official_repo, args.backend_repo, args.model_dir, args.rights_record, args.weight_manifest)):
        raise PreflightError("Tokenizer parity requires all local backend, rights and model preflight arguments")
    preflight = inspect(args.official_repo, args.backend_repo, args.model_dir, args.rights_record, args.weight_manifest)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.official_repo.resolve()))
    sys.path.insert(0, str(args.backend_repo.resolve()))
    try:
        from itts25ft import env
        env.bootstrap(str(args.official_repo.resolve()))
        preflight["actual_feature_limits"] = check_model_limits(env.load_model_config(args.model_dir), args)
        from itts25ft.lang import resolve
        from itts25ft.textfront import TextFrontend, TextFrontendConfig
        frontend = TextFrontend(args.model_dir, resolve("hi"),
                                TextFrontendConfig(normalizer="none", case="none", apply_pronunciation=False))
        frontend._hi_h200_preflight = preflight
        return frontend
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
        raise PreflightError("Approved backend frontend cannot run offline on CPU: " + str(exc)) from exc


def validate_prepared(directory, audio_root):
    sys.path.insert(0, str(ROOT / "src"))
    from hi_h200.data import validate
    from hi_h200.text import DataError
    try:
        return validate(directory, audio_root)
    except DataError as exc:
        raise PreflightError("Prepared audio/text/rights/split validation failed: " + str(exc)) from exc


def audit(args, np, frontend=None):
    validate_prepared(args.prepared, args.audio_root)
    manifest = args.manifest.resolve(strict=True)
    pair_path = args.pairs.resolve(strict=True)
    feature_root = args.feature_root.resolve(strict=True)
    prepared_path = (args.prepared / (args.split + ".jsonl")).resolve(strict=True)
    prepared = {}
    for row in rows(prepared_path):
        uid = row.get("id")
        if not isinstance(uid, str) or uid in prepared or row.get("split") != args.split or row.get("language") != "hi":
            raise PreflightError("Invalid/duplicate prepared row or wrong split/language")
        prepared[uid] = row
    if not prepared:
        raise PreflightError("Prepared split is empty")
    limits = {key: getattr(args, key) for key in ("number_text_tokens", "max_text_tokens", "max_code_tokens", "max_prompt_code_tokens", "codebook_size", "emotion_dim")}
    for value in limits.values():
        if type(value) is not int or value < 1:
            raise PreflightError("Model feature limits must be positive integers from actual config")
    features = {}
    by_path = {}
    for record in rows(manifest):
        uid = record.get("id")
        if uid not in prepared or uid in features:
            raise PreflightError("Feature IDs must exactly match the prepared split with no duplicates")
        original = prepared[uid]
        if record.get("speaker") != original.get("speaker_id") or record.get("text") != original.get("text_norm") or type(record.get("lang")) is not int or record.get("lang") != 17 or record.get("lang_alias") != "hi":
            raise PreflightError(f"Feature metadata changed speaker/text/Hindi language: {uid}")
        path = local_feature(record.get("features"), manifest.parent, feature_root)
        if path in by_path:
            raise PreflightError("Feature file reused for two clip IDs")
        encoded = frontend.encode(record["text"]) if frontend is not None else None
        inspected = inspect_npz(path, record, np, limits, encoded)
        features[uid] = {"path": path, "speaker": record["speaker"], **inspected}
        by_path[path] = uid
    if set(features) != set(prepared):
        raise PreflightError("Backend preprocessing skipped prepared clips; do not train")
    seen = set()
    pair_ids = set()
    for pair in rows(pair_path):
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise PreflightError("Pair IDs must be nonempty unique opaque strings")
        target = local_feature(pair.get("target_features"), pair_path.parent, feature_root)
        prompt = local_feature(pair.get("prompt_features"), pair_path.parent, feature_root)
        if target not in by_path or prompt not in by_path or target == prompt:
            raise PreflightError("Pair target identity or distinct prompt clip is invalid")
        uid = by_path[target]
        if uid in seen:
            raise PreflightError("Pairs must contain each prepared target exactly once")
        if pair.get("target_id", uid) != uid or pair.get("prompt_id", by_path[prompt]) != by_path[prompt]:
            raise PreflightError("Explicit target/prompt IDs disagree with feature identity")
        pair_ids.add(pair_id)
        prompt_row = features[by_path[prompt]]
        if pair.get("speaker") != features[uid]["speaker"] or prompt_row["speaker"] != features[uid]["speaker"]:
            raise PreflightError("Prompt/target are not the same speaker")
        if type(pair.get("lang")) is not int or type(pair.get("prompt_lang")) is not int or pair.get("lang") != 17 or pair.get("prompt_lang") != 17 or pair.get("lang_alias") != "hi" or pair.get("cross_lingual") is not False:
            raise PreflightError("Unexpected pair language metadata")
        if type(pair.get("text_len")) is not int or type(pair.get("code_len")) is not int or pair.get("text_len") != features[uid]["text_len"] or pair.get("code_len") != features[uid]["code_len"]:
            raise PreflightError("Pair lengths disagree with actual target arrays")
        if prompt_row["code_len"] > args.max_prompt_code_tokens:
            raise PreflightError("Prompt exceeds explicit prompt-code limit")
        seen.add(uid)
    if seen != set(prepared):
        raise PreflightError("Pairs do not cover every prepared target")
    inventory = [{"id": uid, "path": str(features[uid]["path"]), "sha256": features[uid]["sha256"]} for uid in sorted(features)]
    report = {"status": "feature_arrays_verified", "split": args.split, "clips": len(features),
              "manifest": str(manifest), "manifest_sha256": sha256(manifest),
              "pairs": str(pair_path), "pairs_sha256": sha256(pair_path),
              "prepared": str(prepared_path), "prepared_sha256": sha256(prepared_path), "feature_root": str(feature_root),
              "prepared_dataset_root": str(args.prepared.resolve()),
              "prepared_dataset_sha256": {name: sha256(args.prepared / name) for name in ["manifest.json", "train.jsonl", "val.jsonl", "test.jsonl", "backend_train.jsonl", "backend_val.jsonl", "backend_test.jsonl"]},
              "features": inventory, "limits": limits,
              "tokenizer_parity_verified": frontend is not None,
              "backend_preflight": getattr(frontend, "_hi_h200_preflight", None),
              "frontend": {"normalizer": "none", "case": "none", "apply_pronunciation": False},
              "gpu_training_verified": False, "ready_for_training": False,
              "remaining_gate": "Offline backend check_setup, small-sample overfit, H200 smoke and release evaluations remain."}
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pairs", "feature-root", "prepared", "audio-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    for name in ("number-text-tokens", "max-text-tokens", "max-code-tokens", "max-prompt-code-tokens", "codebook-size", "emotion-dim"):
        p.add_argument("--" + name, type=int, required=True)
    p.add_argument("--structural-only", action="store_true", help="Skip backend imports; such reports cannot render a training config")
    for name in ("official-repo", "backend-repo", "model-dir", "rights-record", "weight-manifest"):
        p.add_argument("--" + name, type=Path)
    args = p.parse_args(argv)
    try:
        try:
            import numpy as np
        except ImportError as exc:
            raise PreflightError("NumPy is required for actual NPZ validation; run in the separately prepared backend environment") from exc
        frontend = None if args.structural_only else backend_frontend(args)
        report = audit(args, np, frontend)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (PreflightError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "gpu_training_verified": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "clips": report["clips"], "tokenizer_parity_verified": report["tokenizer_parity_verified"], "report": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
