#!/usr/bin/env python3
"""Render a checked experimental Hindi config; never launch the external trainer.

JSON is a YAML subset accepted by the reviewed backend OmegaConf loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__:
    from .check_backend import ROOT, PreflightError, cli_destinations, inspect, read_json, strict_json
else:
    from check_backend import ROOT, PreflightError, cli_destinations, inspect, read_json, strict_json


def validate_config(config, allowed):
    unknown = set(config) - set(allowed)
    if unknown:
        raise PreflightError("Unknown backend fields (upstream silently ignores them): " + ", ".join(sorted(unknown)))
    required = {"lang", "trainable_mode", "train_text_rows", "train_text_embedding", "train_all_lang_rows", "preserve_emotion", "train_heads", "amp", "normalizer", "case", "final_export", "text_row_init", "max_steps", "batch_size", "grad_accumulation"}
    if not required.issubset(config):
        raise PreflightError("Template is missing required experimental safety settings: " + ", ".join(sorted(required - set(config))))
    expected = {"lang": "hi", "trainable_mode": "lora", "train_text_rows": True,
                "train_text_embedding": False, "train_all_lang_rows": False,
                "preserve_emotion": True, "train_heads": True, "amp": "bf16",
                "normalizer": "none", "case": "none", "final_export": "adapter",
                "text_row_init": "keep"}
    for key, value in expected.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise PreflightError(f"Required audited starting setting: {key}={value!r}")
    for key in ("batch_size", "grad_accumulation", "max_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise PreflightError(f"{key} must be a positive integer")
    for key, value in config.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise PreflightError(f"Nonfinite value in configuration: {key}")
    if config.get("lang_alias") or config.get("base_checkpoint") or config.get("init_from") or config.get("resume"):
        raise PreflightError("Initial smoke/overfit must use original model weights and a fresh run")
    return config


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or any(token in str(path) for token in ("::", "@", "\n", "\r")):
        raise PreflightError("Pair manifest must be a local file with no backend selector delimiters")
    speakers, features, ids = set(), set(), set()
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line) > 1024 * 1024 or not line.strip():
                raise PreflightError(f"Blank/oversized pair row: {path}:{line_number}")
            try:
                row = strict_json(line)
            except ValueError as exc:
                raise PreflightError(f"Invalid pair JSONL: {path}:{line_number}") from exc
            if not isinstance(row, dict) or not {"id", "speaker", "prompt_features", "target_features", "lang", "lang_alias", "prompt_lang", "cross_lingual"}.issubset(row):
                raise PreflightError(f"Expected audited feature-pair schema: {path}:{line_number}")
            uid, speaker = row["id"], row["speaker"]
            if not isinstance(uid, str) or not uid or uid in ids or not isinstance(speaker, str) or not speaker:
                raise PreflightError("Pairs need unique target IDs and nonempty speaker IDs")
            if row["lang"] != 17 or row["prompt_lang"] != 17 or row["lang_alias"] != "hi" or row["cross_lingual"] is not False:
                raise PreflightError("Unexpected language in initial Hindi pair manifest")
            pair_files = []
            for field in ("target_features", "prompt_features"):
                value = row[field]
                if not isinstance(value, str) or not value:
                    raise PreflightError("Pair features must be local NPZ paths")
                candidate = Path(value)
                candidate = (candidate if candidate.is_absolute() else path.parent / candidate).resolve(strict=True)
                if not candidate.is_file() or candidate.suffix != ".npz":
                    raise PreflightError("Pair feature path is not an NPZ file")
                pair_files.append(candidate)
                features.add(candidate)
            if pair_files[0] == pair_files[1]:
                raise PreflightError("Prompt and target must be different clips")
            ids.add(uid)
            speakers.add(speaker)
            rows += 1
    if rows == 0:
        raise PreflightError(f"Empty pair manifest: {path}")
    return path, {"speakers": speakers, "features": features, "ids": ids, "rows": rows}


def feature_report(path, pairs, split, preflight):
    record = read_json(path)
    if record.get("status") != "feature_arrays_verified" or record.get("split") != split or record.get("tokenizer_parity_verified") is not True:
        raise PreflightError("Feature audit must pass actual-array and backend-tokenizer parity for this split")
    if record.get("frontend") != {"normalizer": "none", "case": "none", "apply_pronunciation": False}:
        raise PreflightError("Feature frontend audit does not match configuration")
    if record.get("pairs_sha256") != file_sha256(pairs):
        raise PreflightError("Feature report does not match current pair-manifest bytes")
    for field in ("manifest", "prepared"):
        candidate = Path(record.get(field, "")).resolve(strict=True)
        if record.get(field + "_sha256") != file_sha256(candidate):
            raise PreflightError("Feature report is stale for " + field)
    dataset_root = Path(record.get("prepared_dataset_root", "")).resolve(strict=True)
    dataset_hashes = record.get("prepared_dataset_sha256")
    expected_dataset = {"manifest.json", "train.jsonl", "val.jsonl", "test.jsonl", "backend_train.jsonl", "backend_val.jsonl", "backend_test.jsonl"}
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != expected_dataset:
        raise PreflightError("Feature report lacks full prepared-dataset audit binding")
    for name, expected_hash in dataset_hashes.items():
        if expected_hash != file_sha256(dataset_root / name):
            raise PreflightError("Prepared dataset changed after feature audit")
    identity = record.get("backend_preflight")
    if not isinstance(identity, dict) or any(identity.get(key) != preflight.get(key) for key in ("official_commit", "backend_commit")) or identity.get("weights", {}).get("manifest_sha256") != preflight.get("weights", {}).get("manifest_sha256"):
        raise PreflightError("Feature tokenizer audit used a different source or weight manifest")
    root = Path(record.get("feature_root", "")).resolve(strict=True)
    inventory = record.get("features")
    if not isinstance(inventory, list) or not inventory:
        raise PreflightError("Missing feature audit inventory")
    paths = set()
    for item in inventory:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise PreflightError("Invalid feature audit inventory entry")
        candidate = Path(item["path"]).resolve(strict=True)
        if not candidate.is_relative_to(root) or candidate in paths or item.get("sha256") != file_sha256(candidate):
            raise PreflightError("Feature file escapes audit root, duplicates another ID, or changed after audit")
        paths.add(candidate)
    return record, paths


def output_path(path, protected):
    path = Path(path).resolve()
    for root in protected:
        root = Path(root).resolve()
        if path == root or path.is_relative_to(root) or root.is_relative_to(path):
            raise PreflightError(f"Run output must be isolated from source and original weights: {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-repo", required=True, type=Path)
    p.add_argument("--backend-repo", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--rights-record", required=True, type=Path)
    p.add_argument("--weight-manifest", required=True, type=Path)
    p.add_argument("--template", type=Path, default=ROOT / "configs/hindi-h200.example.json")
    p.add_argument("--train-pairs", required=True, type=Path)
    p.add_argument("--val-pairs", required=True, type=Path)
    p.add_argument("--train-feature-report", required=True, type=Path)
    p.add_argument("--val-feature-report", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--profile", choices=("smoke", "overfit"), default="smoke")
    args = p.parse_args(argv)
    try:
        report = inspect(args.official_repo, args.backend_repo, args.model_dir,
                         args.rights_record, args.weight_manifest)
        allowed = cli_destinations(args.backend_repo / "scripts/train.py")
        config = validate_config(read_json(args.template), allowed)
        train, train_meta = manifest_path(args.train_pairs)
        val, val_meta = manifest_path(args.val_pairs)
        train_report, train_features = feature_report(args.train_feature_report, train, "train", report)
        val_report, val_features = feature_report(args.val_feature_report, val, "val", report)
        if train_meta["features"] != train_features or val_meta["features"] != val_features:
            raise PreflightError("Pairs do not exactly cover the audited feature inventory")
        if train_meta["speakers"] & val_meta["speakers"] or train_features & val_features or train_meta["ids"] & val_meta["ids"]:
            raise PreflightError("Training/validation speaker, ID or feature leakage")
        if train_meta["rows"] < config["batch_size"] * config["grad_accumulation"]:
            raise PreflightError("Training split has too few targets for one complete accumulated optimizer update per epoch")
        if train == val:
            raise PreflightError("Training and validation pair files must be different; use the prepared speaker split")
        run_dir = output_path(args.output_dir, (args.official_repo, args.backend_repo, args.model_dir))
        if run_dir.exists() and any(run_dir.iterdir()):
            raise PreflightError("Initial smoke/overfit run directory must be new or empty")
        config.update(repo=str(args.official_repo.resolve()), model_dir=str(args.model_dir.resolve()),
                      output_dir=str(run_dir), train_manifests=[str(train) + "::hi"],
                      val_manifests=[str(val) + "::hi"], max_steps=100 if args.profile == "smoke" else 300)
        validate_config(config, allowed)
        destination = args.output.resolve()
        provenance = destination.with_suffix(destination.suffix + ".preflight.json")
        if destination.exists() or provenance.exists():
            raise PreflightError("Refusing to overwrite existing config or provenance")
        if not destination.parent.is_dir():
            raise PreflightError("Create the config output parent explicitly before rendering")
        text = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        report.update(profile=args.profile, config_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      note="Configuration rendered only. Run the documented feature, tokenizer, offline-runtime and GPU gates before scaling.",
                      feature_audits={"train_report_sha256": file_sha256(args.train_feature_report), "val_report_sha256": file_sha256(args.val_feature_report)},
                      train_heads_note="Both mel_head and text_head train; exported adapter omits text_head. Validate synthesis and adapter parity.")
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(text)
        with provenance.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (PreflightError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "gpu_training_verified": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "rendered_not_trained", "config": str(destination), "provenance": str(provenance)}, ensure_ascii=False))
    print("Backend train.py requires --output-dir on its CLI even when the config includes it.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
