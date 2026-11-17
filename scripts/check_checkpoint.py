#!/usr/bin/env python3
"""Bind a run and inspect an explicitly trusted local checkpoint without resuming it.

This tool never launches a trainer. Real backend save/restore and H200 interruption
round trips remain separate gates. Tensor loading always uses weights_only=True on CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __package__:
    from .check_backend import ROOT, PreflightError, inspect, read_json, cli_destinations
    from .render_backend_config import feature_report, file_sha256, manifest_path, validate_config
else:
    from check_backend import ROOT, PreflightError, inspect, read_json, cli_destinations
    from render_backend_config import feature_report, file_sha256, manifest_path, validate_config

SCHEMA = {"trainable", "optimizer", "scheduler", "scaler", "step", "epoch", "data_batches", "baselines", "spec"}
LIMITATIONS = [
    "No Python/NumPy/Torch/CUDA RNG state is saved by the reviewed backend; continuation is not bit-exact.",
    "Optimizer parameter IDs have no saved names; this inspection cannot prove their mapping to a reconstructed model.",
    "No actual backend load, forward/backward, resumed update, or H200 interruption round trip is performed.",
    "Only completed periodic/STOP checkpoints are candidates. SIGTERM/SIGINT have no automatic-save handler.",
    "Environment-lock contents are hash-bound; installed packages/driver/kernel equivalence must still be verified on the host.",
]


def positive_int(value, name, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise PreflightError(name + " must be a nonnegative/positive integer as appropriate")
    return value


def expected_spec(config):
    return {"mode": config["trainable_mode"], "lora_rank": config["lora_rank"],
            "lora_alpha": config["lora_alpha"], "lora_dropout": config.get("lora_dropout", 0.0),
            "lora_last_n_layers": config.get("lora_last_n_layers", 0),
            "train_last_n_layers": config.get("train_last_n_layers", 6),
            "train_text_embedding": config["train_text_embedding"], "train_text_rows": config["train_text_rows"],
            "train_heads": config["train_heads"], "preserve_emotion": config["preserve_emotion"],
            "extra_train": [], "extra_freeze": []}


def validate_state(state, config, tensor_metadata):
    """Validate primitives/tensor metadata; injected metadata enables honest CPU unit tests."""
    if not isinstance(state, dict) or set(state) != SCHEMA:
        raise PreflightError("Checkpoint does not match the reviewed backend state schema")
    step = positive_int(state["step"], "step")
    epoch = positive_int(state["epoch"], "epoch", True)
    batches = positive_int(state["data_batches"], "data_batches")
    epochs = positive_int(config.get("epochs"), "configured epochs")
    max_steps = positive_int(config.get("max_steps"), "configured max_steps")
    accumulation = positive_int(config.get("grad_accumulation"), "configured grad_accumulation")
    if epoch >= epochs or step >= max_steps:
        raise PreflightError("Completed/final checkpoint is not a mid-run continuation candidate; do not extend this run by silently changing its schedule")
    if batches % accumulation or batches // accumulation > step or (epoch == 0 and batches // accumulation != step):
        raise PreflightError("Checkpoint data cursor does not match completed optimizer boundaries")
    if state["spec"] != expected_spec(config):
        raise PreflightError("Saved trainable spec differs from the frozen run configuration")
    if config.get("amp") != "bf16" or state["scaler"] is not None:
        raise PreflightError("This audited resume route requires BF16 with scaler=None")
    trainable = state["trainable"]
    if not isinstance(trainable, dict) or not trainable or len(trainable) > 10000:
        raise PreflightError("Missing or oversized trainable tensor dictionary")
    signature = []
    required = {"lang_embedding.weight", "text_embedding.weight", "mel_head.weight", "text_head.weight"}
    if not required.issubset(trainable) or not any(name.endswith(".lora_A") for name in trainable) or not any(name.endswith(".lora_B") for name in trainable):
        raise PreflightError("Expected Hindi embedding, both heads and LoRA tensor names are absent")
    for name, value in trainable.items():
        meta = tensor_metadata(value)
        if not isinstance(name, str) or not name or meta is None or not meta["finite"] or not meta["floating"] or not meta["shape"] or any(d < 1 for d in meta["shape"]):
            raise PreflightError("Nonfinite, nonfloating or invalid trainable tensor: " + str(name))
        signature.append({"name": name, "shape": list(meta["shape"]), "dtype": meta["dtype"]})
    optimizer = state["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"} or not isinstance(optimizer["state"], dict) or not isinstance(optimizer["param_groups"], list):
        raise PreflightError("Invalid optimizer state schema")
    parameter_ids = []
    for group in optimizer["param_groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list) or not group["params"]:
            raise PreflightError("Optimizer parameter group is missing IDs")
        parameter_ids.extend(group["params"])
    if any(type(value) is not int or value < 0 for value in parameter_ids) or len(set(parameter_ids)) != len(parameter_ids) or len(parameter_ids) != len(trainable):
        raise PreflightError("Optimizer parameter ID coverage disagrees with trainable tensor count")
    if set(optimizer["state"]) != set(parameter_ids):
        raise PreflightError("Optimizer moments are incomplete for the saved trainable parameters")
    for item in optimizer["state"].values():
        if not isinstance(item, dict) or not {"step", "exp_avg", "exp_avg_sq"}.issubset(item):
            raise PreflightError("Missing AdamW step or moment tensors")
        first, second = tensor_metadata(item["exp_avg"]), tensor_metadata(item["exp_avg_sq"])
        if first is None or second is None or first["shape"] != second["shape"] or first["dtype"] != second["dtype"] or not first["floating"] or not second["floating"] or not first["finite"] or not second["finite"] or not first["shape"] or any(d < 1 for d in first["shape"]):
            raise PreflightError("Optimizer moment dtype/shape/finiteness is invalid")
        counter = item["step"]
        counter_meta = tensor_metadata(counter)
        if counter_meta is not None:
            if counter_meta["shape"] not in ((), (1,)) or not counter_meta["finite"]:
                raise PreflightError("Optimizer step must be a finite scalar")
            counter = counter_meta.get("scalar")
        if type(counter) not in (int, float) or not math.isfinite(counter) or counter != int(counter) or not 0 <= counter <= step:
            raise PreflightError("Optimizer step must be an integer-like scalar within saved global progress")
    scheduler = state["scheduler"]
    if not isinstance(scheduler, dict) or scheduler.get("last_epoch") != step or type(scheduler.get("last_epoch")) is not int:
        raise PreflightError("Scheduler cursor does not match optimizer step")
    if not isinstance(state["baselines"], dict):
        raise PreflightError("Invalid validation baseline state")
    visited = 0
    def finite_tree(value, depth=0):
        nonlocal visited
        visited += 1
        if depth > 24 or visited > 200000:
            raise PreflightError("Checkpoint metadata nesting/size exceeds inspection bound")
        meta = tensor_metadata(value)
        if meta is not None:
            if not meta["finite"]:
                raise PreflightError("Nonfinite tensor in optimizer/scheduler/checkpoint state")
        elif value is None or isinstance(value, (str, bool, int)):
            return
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise PreflightError("Nonfinite checkpoint scalar")
        elif isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, (str, int)):
                    raise PreflightError("Unsupported checkpoint metadata key")
                finite_tree(child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value:
                finite_tree(child, depth + 1)
        else:
            raise PreflightError("Unsupported non-tensor checkpoint metadata object")
    finite_tree(state)
    return {"step": step, "epoch": epoch, "data_batches": batches,
            "trainable_tensors": len(trainable),
            "ordered_trainable_signature_sha256": hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest(),
            "optimizer_name_mapping_verified": False, "model_tensor_shapes_verified": False, "rng_state_restored": False,
            "actual_backend_roundtrip_verified": False, "gpu_training_verified": False}


def load_trusted_checkpoint(path, trusted, config):
    if trusted is not True:
        raise PreflightError("Explicit --trusted-own-checkpoint is required; never inspect an unknown .pt file")
    path = Path(path).resolve(strict=True)
    if not re.fullmatch(r"step[0-9]+\.pt", path.name) or not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024 ** 3:
        raise PreflightError("Use a completed local stepN.pt checkpoint, at most 32 GiB; never .tmp or a final adapter")
    try:
        import torch
    except ImportError as exc:
        raise PreflightError("CPU Torch is required in the separately prepared backend environment") from exc
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise PreflightError("Restricted CPU checkpoint load failed; there is no unsafe-pickle fallback: " + str(exc)) from exc
    def metadata(value):
        if not torch.is_tensor(value):
            return None
        return {"shape": tuple(value.shape), "dtype": str(value.dtype),
                "floating": bool(value.is_floating_point()), "finite": bool(torch.isfinite(value).all().item()),
                "scalar": value.item() if value.numel() == 1 else None}
    result = validate_state(state, config, metadata)
    if path.stem != "step" + str(result["step"]):
        raise PreflightError("Checkpoint filename step differs from its saved step")
    return result


def add_inventory(inventory, path, category):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise PreflightError("Run artifact is not a file: " + str(path))
    key = str(path)
    inventory[key] = {"path": key, "sha256": file_sha256(path), "category": category}


def verify_inventory(inventory):
    if not isinstance(inventory, list) or not inventory:
        raise PreflightError("Missing run artifact inventory")
    seen = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "category"} or not isinstance(item["path"], str) or item["path"] in seen:
            raise PreflightError("Invalid/duplicate run inventory entry")
        path = Path(item["path"]).resolve(strict=True)
        if str(path) != item["path"] or item["sha256"] != file_sha256(path):
            raise PreflightError("Frozen run artifact changed or moved: " + str(path))
        seen.add(item["path"])


def save_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def bind_run(args):
    config = read_json(args.config)
    preflight = inspect(config["repo"], args.backend_repo, config["model_dir"], args.rights_record, args.weight_manifest)
    validate_config(config, cli_destinations(args.backend_repo / "scripts/train.py"))
    original = read_json(args.preflight)
    if original.get("config_sha256") != file_sha256(args.config) or original.get("backend_commit") != preflight["backend_commit"] or original.get("official_commit") != preflight["official_commit"] or original.get("weights", {}).get("manifest_sha256") != preflight["weights"]["manifest_sha256"]:
        raise PreflightError("Original rendered-config preflight does not match this run")
    inventory = {}
    for path, category in ((args.config,"config"),(args.preflight,"preflight"),(args.environment_lock,"environment"),(args.rights_record,"rights"),(args.weight_manifest,"model-manifest"),(ROOT / "upstream.lock.json","source-lock")):
        add_inventory(inventory, path, category)
    feature_reports = {}
    for split, report_path in (("train",args.train_feature_report),("val",args.val_feature_report)):
        selectors = config[split + "_manifests"]
        if not isinstance(selectors, list) or len(selectors) != 1 or not selectors[0].endswith("::hi"):
            raise PreflightError("Resume binder requires one audited Hindi-only pair manifest per split")
        pairs, _ = manifest_path(selectors[0][:-4])
        report, _ = feature_report(report_path, pairs, split, preflight)
        if original.get("feature_audits", {}).get(split + "_report_sha256") != file_sha256(report_path):
            raise PreflightError("Feature report differs from the original rendered-config preflight")
        feature_reports[split] = str(Path(report_path).resolve())
        for path in (report_path,pairs,report["manifest"],report["prepared"]):
            add_inventory(inventory,path,"data")
        for item in report["features"]:
            add_inventory(inventory,item["path"],"feature")
        for relative in report["prepared_dataset_sha256"]:
            add_inventory(inventory,Path(report["prepared_dataset_root"]) / relative,"prepared")
    run_dir = Path(config["output_dir"]).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise PreflightError("Bind before launching a fresh run; output_dir must be absent or empty")
    record = {"schema_version":1,"kind":"run_binding","created_utc":datetime.now(timezone.utc).isoformat(),
              "config":str(args.config.resolve()),"preflight":str(args.preflight.resolve()),
              "backend_repo":str(args.backend_repo.resolve()),"rights_record":str(args.rights_record.resolve()),
              "weight_manifest":str(args.weight_manifest.resolve()),"feature_reports":feature_reports,
              "run_dir":str(run_dir),"inventory":list(inventory.values()),
              "runtime_roundtrip_verified":False,"limitations":LIMITATIONS}
    save_json(args.output,record)
    return {"status":"run_bound_not_started","artifacts":len(inventory),"runtime_roundtrip_verified":False}


def check_run_binding(path):
    binding = read_json(path)
    if binding.get("schema_version") != 1 or binding.get("kind") != "run_binding":
        raise PreflightError("Expected an original run_binding sidecar")
    verify_inventory(binding.get("inventory"))
    config = read_json(binding["config"])
    inspect(config["repo"],binding["backend_repo"],config["model_dir"],binding["rights_record"],binding["weight_manifest"])
    return binding, config


def seal(args):
    binding, config = check_run_binding(args.binding)
    checkpoint = args.checkpoint.resolve(strict=True)
    run_dir = Path(binding["run_dir"])
    if checkpoint.parent != run_dir / "checkpoints":
        raise PreflightError("Checkpoint must belong to the bound run/checkpoints directory")
    before_sha256 = file_sha256(checkpoint)
    result = load_trusted_checkpoint(checkpoint,args.trusted_own_checkpoint,config)
    if before_sha256 != file_sha256(checkpoint):
        raise PreflightError("Checkpoint changed during inspection; stop training and use a stable copied snapshot")
    generated = {}
    for name in ("train_args.json","text_rows.json","text_token_counts_hi.json"):
        add_inventory(generated,run_dir / name,"generated-run-state")
    train_args = read_json(run_dir / "train_args.json")
    for key,value in config.items():
        if key != "resume" and train_args.get(key) != str(value):
            raise PreflightError("Recorded actual train_args differ from frozen config: " + key)
    if train_args.get("init_from", "") or train_args.get("resume_data_step", "-1") != "-1":
        raise PreflightError("Initialization/cursor override is outside this continuation route")
    text_rows = read_json(run_dir / "text_rows.json")
    if text_rows.get("lang") != "hi" or not isinstance(text_rows.get("rows"),list) or not text_rows["rows"] or any(type(value) is not int or value < 0 for value in text_rows["rows"]):
        raise PreflightError("Missing/invalid saved Hindi text-row selection")
    record = {"schema_version":1,"kind":"checkpoint_seal","binding":str(args.binding.resolve()),
              "binding_sha256":file_sha256(args.binding),"checkpoint":str(checkpoint),
              "checkpoint_sha256":before_sha256,"generated_inventory":list(generated.values()),
              "inspection":result,"limitations":LIMITATIONS}
    save_json(args.output,record)
    return {"status":"checkpoint_inspected_not_resumed",**result}


def verify(args):
    record = read_json(args.sidecar)
    if record.get("schema_version") != 1 or record.get("kind") != "checkpoint_seal":
        raise PreflightError("Expected a checkpoint_seal sidecar")
    if file_sha256(record["binding"]) != record.get("binding_sha256"):
        raise PreflightError("Original run binding changed")
    _,config = check_run_binding(record["binding"])
    verify_inventory(record.get("generated_inventory"))
    checkpoint = args.checkpoint.resolve(strict=True)
    if str(checkpoint) != record.get("checkpoint") or file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise PreflightError("Explicit resume checkpoint does not match the sealed checkpoint")
    result = load_trusted_checkpoint(checkpoint,args.trusted_own_checkpoint,config)
    if file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise PreflightError("Checkpoint changed during verification")
    if result != record.get("inspection"):
        raise PreflightError("Checkpoint metadata no longer matches its sealed inspection")
    return {"status":"checkpoint_verified_not_resumed",**result,"limitations":LIMITATIONS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    bind = sub.add_parser("bind-run",help="Freeze source/model/data/config/environment identity before a fresh run")
    for name in ("config","preflight","backend-repo","rights-record","weight-manifest","environment-lock","train-feature-report","val-feature-report","output"):
        bind.add_argument("--"+name,type=Path,required=True)
    binding_seal = sub.add_parser("seal",help="Inspect an own completed periodic/STOP checkpoint and create its audit sidecar")
    for name in ("binding","checkpoint","output"):
        binding_seal.add_argument("--"+name,type=Path,required=True)
    binding_seal.add_argument("--trusted-own-checkpoint",action="store_true")
    verify_parser = sub.add_parser("verify",help="Verify unchanged run/checkpoint before a separate explicit --resume command")
    for name in ("sidecar","checkpoint"):
        verify_parser.add_argument("--"+name,type=Path,required=True)
    verify_parser.add_argument("--trusted-own-checkpoint",action="store_true")
    args = parser.parse_args(argv)
    try:
        result = {"bind-run":bind_run,"seal":seal,"verify":verify}[args.command](args)
    except (PreflightError,OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps({"status":"blocked","reason":str(exc),"runtime_roundtrip_verified":False},ensure_ascii=False),file=sys.stderr)
        return 2
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
