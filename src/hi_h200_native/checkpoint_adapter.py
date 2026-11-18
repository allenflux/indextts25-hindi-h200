"""Select a completed native checkpoint as a standard adapter, without resuming.

Default is stdlib metadata/hash preflight. --execute performs strict CPU validation
against the pinned base and recorded run/data/source identity. No download or GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

CODE = ("model.py", "data.py", "engine.py", "train.py")
STATE_KEYS = {"schema_version", "kind", "reason", "config", "identity", "runtime", "step", "sampler",
              "parameter_groups", "trainable", "optimizer", "scheduler", "scaler", "rng", "model_identity", "adapter_spec"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def read_object(path, maximum=16 * 1024 * 1024):
    from hi_h200.data import load_json
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Missing or oversized native metadata")
    value = load_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a native metadata object")
    return value


def destination(path, workspace, protected):
    path = Path(path).absolute()
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    if path.exists() or path.is_symlink() or not target.is_relative_to(workspace):
        raise ValueError("Adapter output must be a new directory inside WORK")
    if any(target == root or target.is_relative_to(root) or root.is_relative_to(target) for root in protected):
        raise ValueError("Adapter output overlaps protected run/source/model/data")
    return target


def preflight(args):
    from .train import read_config, _dataset_paths
    from .synth import verify_source, OFFICIAL_COMMIT
    workspace = args.workspace_root.resolve(strict=True)
    config_path, run_manifest, checkpoint = (path.resolve(strict=True) for path in (args.config, args.run_manifest, args.checkpoint))
    if any(not path.is_relative_to(workspace) for path in (config_path, run_manifest, checkpoint)):
        raise ValueError("Checkpoint/config/run manifest must be inside this project's WORK")
    if not run_manifest.is_file() or not 0 < run_manifest.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("Missing or oversized native run manifest")
    if not re.fullmatch("[0-9a-f]{64}", args.run_manifest_sha256) or digest(run_manifest) != args.run_manifest_sha256:
        raise ValueError("Independent run-manifest SHA256 mismatch")
    recorded = read_object(run_manifest)
    if set(recorded) != {"schema_version", "kind", "identity", "limitations"} or type(recorded["schema_version"]) is not int or recorded["schema_version"] != 1 or recorded["kind"] != "hi_h200_native_run":
        raise ValueError("Unknown native run manifest")
    identity = recorded["identity"]
    if not isinstance(identity, dict) or set(identity) != {"config_sha256", "resolved_config", "datasets", "adapter_spec", "native_code_sha256"}:
        raise ValueError("Run identity schema differs from the original native CLI")
    options = read_config(config_path)
    if digest(config_path) != identity["config_sha256"] or options != identity["resolved_config"]:
        raise ValueError("Original config identity changed")
    if identity["native_code_sha256"] != {name:digest(Path(__file__).with_name(name)) for name in CODE}:
        raise ValueError("Checkpoint-bound native source identity changed")
    run = Path(options["output_dir"])
    if run_manifest != run.parent / (run.name + ".native-run.json") or checkpoint.parent != run / "checkpoints":
        raise ValueError("Checkpoint is outside the recorded native run")
    if type(args.expected_step) is not int or not 1 <= args.expected_step <= options["loop"]["max_updates"] or checkpoint.name != "step" + str(args.expected_step).zfill(8) + ".pt":
        raise ValueError("Explicit selected step differs from the run/filename")
    if not checkpoint.is_file() or not 0 < checkpoint.stat().st_size <= 32 * 1024**3:
        raise ValueError("Native checkpoint exceeds the existing 32 GiB bound")
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")
    if sidecar_path.resolve(strict=True).parent != checkpoint.parent:
        raise ValueError("Checkpoint sidecar escapes the selected run")
    sidecar = read_object(sidecar_path, 4096)
    if set(sidecar) != {"schema_version", "sha256", "step"} or type(sidecar["schema_version"]) is not int or sidecar["schema_version"] != 1 or type(sidecar["step"]) is not int or sidecar["step"] != args.expected_step or sidecar["sha256"] != digest(checkpoint):
        raise ValueError("Native checkpoint SHA256 sidecar/step mismatch")
    dataset_configs = [Path(options[key]).resolve(strict=True) for key in ("train_dataset", "val_dataset")]
    if any(not path.is_relative_to(workspace) for path in dataset_configs):
        raise ValueError("Native dataset config files must stay inside WORK")
    paths = [_dataset_paths(path) for path in dataset_configs]
    for values in paths:
        verify_source(values["official_repo"])
        if any(not path.is_relative_to(workspace) for key,path in values.items() if key not in ("model_dir", "model_manifest")):
            raise ValueError("Source/data must be in WORK; original model assets may use the authorized read-only share")
    if paths[0]["official_repo"] != paths[1]["official_repo"]:
        raise ValueError("Train/validation official source differs")
    datasets = identity["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != {"train", "val"} or any(not isinstance(value, dict) or value.get("official_commit") != OFFICIAL_COMMIT or value.get("kind") == "tiny_random_official_cpu_test" for value in datasets.values()):
        raise ValueError("Production conversion requires the recorded verified full native dataset/base identity")
    if not isinstance(args.selection, str) or not 1 <= len(args.selection) <= 512 or any(unicodedata.category(c) in {"Cc","Cf","Cs"} for c in args.selection):
        raise ValueError("Provide a bounded, explicit selection rationale")
    output = destination(args.output_dir, workspace, [run, config_path, run_manifest, checkpoint, *dataset_configs, *[p for values in paths for p in values.values()], Path(__file__).resolve().parents[2]])
    return dict(options=options, identity=identity, paths=paths[0], checkpoint=checkpoint, sidecar=sidecar, output=output,
        metadata={"status":"checkpoint_adapter_metadata_preflight_passed", "source_checkpoint_sha256":sidecar["sha256"],
          "source_sidecar_sha256":digest(sidecar_path), "run_manifest_sha256":args.run_manifest_sha256,
          "run_identity_sha256":hashlib.sha256(canonical(identity)).hexdigest(), "selected_step":args.expected_step,
          "selection":args.selection, "selection_ranking_recomputed":False, "checkpoint_tensor_validation":False,
          "adapter_emitted":False, "training_executed":False, "quality_verified":False})


def validate_checkpoint(state, model, identity, options, expected_step, dataset_length):
    """CPU-only strict schema/state gate; does not restore RNG or continue training."""
    import torch
    from .engine import _finite_tree, _parameter_groups, _runtime
    if not isinstance(state, dict) or set(state) != STATE_KEYS or type(state["schema_version"]) is not int or state["schema_version"] != 1 or state["kind"] != "hi_h200_native_training":
        raise ValueError("Unknown/partial native training checkpoint schema")
    _finite_tree(state)
    if state["identity"] != identity or state["config"] != options["loop"] or state["model_identity"] != model.native_identity or state["adapter_spec"] != model.adapter_spec() or identity["adapter_spec"] != model.adapter_spec():
        raise ValueError("Checkpoint source/config/base/adapter identity mismatch")
    step, sampler = state["step"], state["sampler"]
    if type(step) is not int or step != expected_step or not isinstance(state["reason"], str) or not state["reason"]:
        raise ValueError("Invalid selected optimizer boundary")
    if not isinstance(sampler, dict) or set(sampler) != {"epoch","offset","micro_steps","dataset_length"} or any(type(v) is not int for v in sampler.values()):
        raise ValueError("Missing native sampler state")
    if sampler["dataset_length"] != dataset_length or sampler["epoch"] < 0 or not 0 <= sampler["offset"] <= dataset_length or sampler["micro_steps"] != step * options["loop"]["grad_accumulation"] or sampler["epoch"] * dataset_length + sampler["offset"] != sampler["micro_steps"]:
        raise ValueError("Checkpoint is not a complete matching optimizer boundary")
    if not isinstance(state["runtime"], dict) or set(state["runtime"]) != set(_runtime()) or state["scaler"] is not None:
        raise ValueError("Native runtime/scaler schema mismatch")
    optimizer_options = options["optimizer"]
    optimizer = torch.optim.AdamW(model.optimizer_groups(optimizer_options["weight_decay"]), lr=optimizer_options["learning_rate"], foreach=False)
    groups = _parameter_groups(model, optimizer)
    if state["parameter_groups"] != groups:
        raise ValueError("Optimizer trainable parameter ordering differs")
    saved, expected = state["optimizer"], optimizer.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(expected) or len(saved["param_groups"]) != len(groups):
        raise ValueError("Incomplete AdamW state")
    named = dict(model.named_parameters())
    ids = []
    for names, group, fresh in zip(groups, saved["param_groups"], expected["param_groups"]):
        if set(group) - {"initial_lr"} != set(fresh) or group["params"] != fresh["params"] or len(names) != len(group["params"]):
            raise ValueError("AdamW group schema/ordering differs")
        ids.extend(group["params"])
        for name, number in zip(names, group["params"]):
            entry = saved["state"].get(number, {})
            if set(entry) != {"step","exp_avg","exp_avg_sq"} or not torch.is_tensor(entry["step"]) or entry["step"].numel() != 1 or entry["step"].item() != step:
                raise ValueError("Missing/inconsistent AdamW parameter step")
            for key in ("exp_avg", "exp_avg_sq"):
                if not torch.is_tensor(entry[key]) or entry[key].shape != named[name].shape or entry[key].dtype != named[name].dtype:
                    raise ValueError("AdamW tensor shape/dtype differs")
    if set(saved["state"]) != set(ids):
        raise ValueError("AdamW state inventory differs")
    # This is adapter extraction, not resume: never load optimizer/scheduler/RNG
    # or claim they were restored. Their complete top-level presence and finite
    # trees were checked, and only recognized tensor/model state is exported.
    if not isinstance(state["scheduler"], dict) or state["scheduler"].get("last_epoch") != step:
        raise ValueError("Incomplete/inconsistent recorded scheduler state")
    if not isinstance(state["rng"], dict) or set(state["rng"]) != {"python","numpy","torch_cpu","torch_cuda"}:
        raise ValueError("Missing complete recorded RNG schema")
    model.load_trainable_state(state["trainable"])  # Complete names/dtype/shape/finite/protected-row checks.
    return {"schema_version":1,"kind":"hi_h200_native_adapter","step":step,
            "base_identity":model.native_identity,"adapter_spec":model.adapter_spec(),"trainable":model.trainable_state()}


def execute(plan):
    import torch
    from .data import load_native_dataset
    from .train import split_identity
    from .model import load_native_model
    options, paths = plan["options"], plan["paths"]
    train, val = (load_native_dataset(options[key]) for key in ("train_dataset", "val_dataset"))
    if split_identity(train, val) != plan["identity"]["datasets"]:
        raise ValueError("Current dataset identity differs from the original run")
    lora = options["lora"]
    model = load_native_model(paths["official_repo"], paths["model_dir"], paths["model_manifest"],
        lora_rank=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        text_rows=() if lora.get("training_profile") == "attention_lora_only_v1" else train.text_rows,
        text_loss_weight=lora["text_loss_weight"],
        training_profile=lora.get("training_profile", "legacy_rows_heads_v1"))
    state = torch.load(plan["checkpoint"], map_location="cpu", weights_only=True)
    if digest(plan["checkpoint"]) != plan["sidecar"]["sha256"]:
        raise ValueError("Source checkpoint changed during restricted CPU inspection")
    payload = validate_checkpoint(state, model, plan["identity"], options, plan["sidecar"]["step"], len(train))
    return publish_adapter(payload, model, plan["output"], plan["metadata"])


def publish_adapter(payload, model, output, metadata):
    """Publish only into a fresh caller-validated private output directory."""
    from .engine import atomic_torch_save
    from .model import load_native_adapter
    from .train import write_manifest
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    adapter = atomic_torch_save(output / "adapter.pt", payload)
    loaded = load_native_adapter(model, adapter, digest(adapter))
    report = dict(metadata, status="selected_checkpoint_adapter_complete", checkpoint_tensor_validation=True,
        adapter_emitted=True, adapter_sha256=loaded["sha256"], adapter_file="adapter.pt", adapter_load_verified=True,
        resume_state_restored=False, implementation_sha256=digest(__file__))
    write_manifest(output / "selection.json", report)
    write_manifest(output / "COMPLETE.json", {"schema_version":1,"files":{p.name:digest(p) for p in output.iterdir()}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace-root","config","run-manifest","checkpoint","output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    try:
        plan = preflight(args)
        if args.execute:
            os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
            report = execute(plan)
        else:
            report = plan["metadata"]
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(json.dumps({"status":"blocked_or_failed","failure_type":type(exc).__name__,"reason":str(exc),"adapter_emitted":False,"output_may_be_incomplete":args.output_dir.exists()}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
