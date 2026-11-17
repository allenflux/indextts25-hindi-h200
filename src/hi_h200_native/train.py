"""Explicit original training CLI. No community training package is used.

A runnable configuration needs verified official assets and native feature
provenance. --help works without Torch; there is no implicit downloader or GPU job.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {"schema_version", "train_dataset", "val_dataset", "output_dir", "loop", "optimizer", "lora"}
LOOP_KEYS = {"max_updates", "grad_accumulation", "seed", "device", "precision", "grad_clip",
             "save_interval", "validation_interval", "max_validation_examples"}
OPTIMIZER_KEYS = {"learning_rate", "weight_decay", "warmup_updates", "min_lr_ratio"}
LORA_KEYS = {"rank", "alpha", "dropout", "text_loss_weight"}


def read_object(path):
    from hi_h200.data import load_json
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Missing or oversized native training config")
    result = load_json(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Native training config must be a JSON object")
    return result


def path_value(value, parent, *, must_exist=True):
    if not isinstance(value, str) or not value or "://" in value or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in value):
        raise ValueError("Native training paths must be explicit local paths")
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve(strict=must_exist)


def read_config(path):
    path = Path(path).resolve(strict=True)
    value = read_object(path)
    if set(value) != CONFIG_KEYS or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unknown or missing native training fields")
    for name, keys in (("loop", LOOP_KEYS), ("optimizer", OPTIMIZER_KEYS), ("lora", LORA_KEYS)):
        if not isinstance(value[name], dict) or set(value[name]) != keys:
            raise ValueError("Unknown or missing native " + name + " settings")
    for name in ("train_dataset", "val_dataset"):
        value[name] = str(path_value(value[name], path.parent))
    value["output_dir"] = str(path_value(value["output_dir"], path.parent, must_exist=False))
    optimizer, lora = value["optimizer"], value["lora"]
    for name in ("learning_rate", "weight_decay", "min_lr_ratio"):
        number = optimizer[name]
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("Nonfinite/invalid optimizer setting: " + name)
    if optimizer["learning_rate"] <= 0 or optimizer["weight_decay"] < 0 or not 0 <= optimizer["min_lr_ratio"] <= 1:
        raise ValueError("Invalid optimizer range")
    if type(optimizer["warmup_updates"]) is not int:
        raise ValueError("warmup_updates must be an integer")
    if type(lora["rank"]) is not int or lora["rank"] < 1:
        raise ValueError("LoRA rank must be a positive integer")
    for name in ("alpha", "dropout", "text_loss_weight"):
        number = lora[name]
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("Nonfinite/invalid LoRA setting: " + name)
    if lora["alpha"] <= 0 or not 0 <= lora["dropout"] < 1 or lora["text_loss_weight"] <= 0:
        raise ValueError("Native initial route requires positive LoRA alpha/text loss and dropout in [0,1)")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_paths(config_path):
    config_path = Path(config_path)
    options = read_object(config_path)
    expected = {"official_repo", "model_dir", "model_manifest", "audio_root", "prepared_dir",
                "feature_root", "feature_manifest", "provenance"}
    return {key: path_value(options[key], config_path.parent) for key in expected}


def protect_output(output, protected):
    output = Path(output).resolve()
    for root in protected:
        root = Path(root).resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Training output overlaps source/model/data: " + str(output))
    return output


@contextlib.contextmanager
def launch_lease(output):
    """Acquire before model loading, preventing two launches for one run."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lease = output.parent / ("." + output.name + ".native-launch.lock")
    descriptor = os.open(lease, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another native launch owns this run") from exc
        yield


def split_identity(train, validation):
    if train.identity.get("split") != "train" or validation.identity.get("split") != "val":
        raise ValueError("Training requires explicit train and held-out val feature splits")
    shared = ("official_commit", "model_snapshot_revision", "model_manifest_sha256", "tokenizer_sha256", "config_sha256", "frontend")
    if any(train.identity.get(key) != validation.identity.get(key) for key in shared):
        raise ValueError("Training/validation source, weights or frontend differ")
    for field in ("speaker_ids", "clip_ids", "audio_hashes", "text_hashes", "feature_hashes"):
        if getattr(train, field) & getattr(validation, field):
            raise ValueError("Training/validation leakage in " + field)
    return {"train": train.identity, "val": validation.identity}


def write_manifest(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def train(config_path, *, resume=None, stop_after_updates=None, allow_cuda=False):
    config_path = Path(config_path).resolve(strict=True)
    config_hash = file_sha256(config_path)
    options = read_config(config_path)
    if options["loop"]["device"] != "cpu" and not allow_cuda:
        raise ValueError("CUDA launch requires explicit --allow-cuda after shared-resource checks; this CLI does not allocate pods")
    if options["loop"]["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Optional heavy imports happen only after all cheap launch flags/schema.
    import torch
    from .data import load_native_dataset
    from .engine import LoopConfig, StatefulTrainer, StopController, atomic_torch_save, cosine_scheduler, seed_everything
    from .model import load_native_model

    loop_config = LoopConfig(**options["loop"]).validate()
    optimizer_config, lora = options["optimizer"], options["lora"]
    if not 0 <= optimizer_config["warmup_updates"] < loop_config.max_updates:
        raise ValueError("Scheduler warmup must be smaller than the unchanged update plan")
    if resume is not None:
        resume = Path(resume).resolve(strict=True)
        if not resume.is_file() or not resume.with_suffix(resume.suffix + ".sha256.json").is_file():
            raise ValueError("Explicit resume checkpoint and its hash sidecar must already exist")
    train_paths, val_paths = (_dataset_paths(options[name]) for name in ("train_dataset", "val_dataset"))
    protected = [*train_paths.values(), *val_paths.values(), ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    output = protect_output(options["output_dir"], protected)
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run output must be absent or empty")
    if (output / "STOP").exists():
        raise ValueError("Review and remove your STOP marker before launching/resuming")

    with launch_lease(output):
        train_data = load_native_dataset(options["train_dataset"])
        val_data = load_native_dataset(options["val_dataset"])
        datasets = split_identity(train_data, val_data)
        seed_everything(loop_config.seed)
        if loop_config.device == "cpu":
            torch.use_deterministic_algorithms(True)
        model = load_native_model(train_paths["official_repo"], train_paths["model_dir"], train_paths["model_manifest"],
                                  lora_rank=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                  text_rows=train_data.text_rows, text_loss_weight=lora["text_loss_weight"])
        model.to(torch.device(loop_config.device))
        optimizer = torch.optim.AdamW(model.optimizer_groups(optimizer_config["weight_decay"]),
                                      lr=optimizer_config["learning_rate"], foreach=False)
        scheduler = cosine_scheduler(optimizer, max_updates=loop_config.max_updates,
                                     warmup_updates=optimizer_config["warmup_updates"],
                                     min_lr_ratio=optimizer_config["min_lr_ratio"])
        identity = {"config_sha256": config_hash, "resolved_config": options, "datasets": datasets,
                    "adapter_spec": model.adapter_spec(),
                    "native_code_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                           for name in ("model.py", "data.py", "engine.py", "train.py")}}
        if file_sha256(config_path) != config_hash:
            raise ValueError("Training config changed during preflight")
        manifest = {"schema_version": 1, "kind": "hi_h200_native_run", "identity": identity,
                    "limitations": ["No speech-quality or old-language regression acceptance is implied by training loss.",
                                    "No compatibility with community adapters, ComfyUI, vLLM or accelerated inference is established.",
                                    "SIGKILL/host loss cannot guarantee a new checkpoint; retain prior durable checkpoints."]}
        manifest_path = output.parent / (output.name + ".native-run.json")
        if resume is None:
            write_manifest(manifest_path, manifest)
        elif read_object(manifest_path) != manifest:
            raise ValueError("Original native run manifest changed or is missing")
        if loop_config.device.startswith("cuda:"):
            torch.cuda.reset_peak_memory_stats(torch.device(loop_config.device))
        trainer = StatefulTrainer(model, optimizer, scheduler, train_data, loop_config, identity,
                                  validation_dataset=val_data)
        stop = StopController()
        with stop.signals():
            result = trainer.run(output, resume=resume, stop=stop, stop_after_updates=stop_after_updates,
                                 on_update=lambda _trainer, values: print(json.dumps({"status": "real_optimizer_update", **values}, allow_nan=False), flush=True))
        if trainer.step and result.get("updates_this_run", 0) > 0:
            directory = output / "adapters"
            if directory.is_symlink() or not directory.resolve().is_relative_to(output):
                raise ValueError("Adapter directory escapes the run root")
            adapter = directory / ("step" + str(trainer.step).zfill(8) + ".pt")
            payload = {"schema_version": 1, "kind": "hi_h200_native_adapter", "step": trainer.step,
                       "base_identity": model.native_identity, "adapter_spec": model.adapter_spec(),
                       "trainable": model.trainable_state()}
            if adapter.exists():
                raise ValueError("Refusing to overwrite an existing adapter")
            atomic_torch_save(adapter, payload)
            result.update(adapter=str(adapter), adapter_sha256=file_sha256(adapter))
        result.update(native_training=True, hindi_quality_verified=False, waveform_inference_verified=False,
                      run_manifest=str(manifest_path))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="Explicit completed native checkpoint; never auto/fallback")
    parser.add_argument("--stop-after-updates", type=int, help="Cooperative absolute update boundary for a real recovery rehearsal")
    parser.add_argument("--allow-cuda", action="store_true", help="Operator has separately checked existing GPU sharing/storage; no resource is allocated here")
    args = parser.parse_args(argv)
    try:
        result = train(args.config, resume=args.resume, stop_after_updates=args.stop_after_updates, allow_cuda=args.allow_cuda)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "blocked_or_failed", "reason": str(exc),
                          "training_success_asserted": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
