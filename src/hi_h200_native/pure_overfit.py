"""Fixed four-TRAIN-example mechanism gate for a newly trained pure LoRA.

Default metadata does not import Torch or load tensor/model state. Explicit CUDA
execution evaluates the same sorted first four training examples before/after the
adapter, with FP32 master weights and BF16 autocast. No training/merge/ASR occurs.
Mean total-loss ratio <= 0.9 is a small-set mechanism check, never a quality gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from . import synth
from .assets import MODEL_REPO, MODEL_REVISION
from .train import _dataset_paths

BASE_GPT_SHA256 = "43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85"
PROFILE = "attention_lora_only_v1"
COUNT = 4
MAX_RATIO = 0.9
EXPECTED_STEP = 100


def require(value, message):
    if not value:
        raise ValueError(message)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def preflight(args):
    from .data import read_object
    from .adapter_synth import checked_adapter
    workspace = args.workspace_root.resolve(strict=True)
    dataset_config = args.dataset.resolve(strict=True)
    require(dataset_config.is_relative_to(workspace), "Dataset config must stay inside WORK")
    options = read_object(dataset_config)
    require(options.get("split") == "train", "This fixed gate accepts only the training split")
    paths = _dataset_paths(dataset_config)
    require(all(path.is_relative_to(workspace) for name, path in paths.items() if name not in ("model_dir", "model_manifest")),
            "Source/data must stay in WORK; original assets may use the authorized read-only share")
    source = synth.verify_source(paths["official_repo"])
    require(paths["model_manifest"].is_relative_to(paths["model_dir"]), "Original manifest must be inside original assets")
    manifest = read_object(paths["model_manifest"])
    require(manifest.get("repository") == MODEL_REPO and manifest.get("snapshot_revision") == MODEL_REVISION
            and isinstance(manifest.get("files"), dict) and manifest["files"].get("gpt.pth") == BASE_GPT_SHA256,
            "The gate requires the fixed original official 2.5 base, never a merged candidate")
    adapter = checked_adapter(args.adapter, args.adapter_sha256)
    require(adapter.is_relative_to(workspace), "Adapter must stay inside WORK")
    output = synth._independent_output(args.output_dir, workspace, (*paths.values(), dataset_config, adapter, Path(__file__).resolve().parents[2]))
    report = {"schema_version": 1, "status": "pure_overfit_metadata_only_tensor_validation_pending",
        "training_profile": PROFILE, "official_commit": synth.OFFICIAL_COMMIT,
        "base_model_revision": MODEL_REVISION, "base_gpt_sha256": BASE_GPT_SHA256,
        "dataset_config_sha256": synth.sha256(dataset_config), "adapter_sha256": args.adapter_sha256,
        "adapter_bytes": adapter.stat().st_size, "selection": "sorted_first_four_train_examples",
        "sample_count": COUNT, "expected_adapter_step": EXPECTED_STEP, "seed": args.seed, "maximum_mean_loss_ratio": MAX_RATIO,
        "dataset_fully_validated": False, "adapter_tensor_validation": False, "models_loaded": False,
        "adapter_training_dataset_binding_verified": False,
        "dataset_binding_limitation": "Operator-supplied TRAIN input is safety-validated; the adapter does not record training dataset identity. Verify its original run identity separately.",
        "cuda_access_attempted": False, "evaluation_executed": False, "mechanism_gate_passed": False,
        "training_executed": False, "quality_verified": False, "generalization_verified": False,
        "implementation_sha256": synth.sha256(Path(__file__)),
        "native_code_sha256": {name: synth.sha256(Path(__file__).with_name(name))
                               for name in ("model.py", "data.py", "engine.py", "train.py")}}
    return dict(dataset_config=dataset_config, paths=paths, source=source, adapter=adapter,
                output=output, report=report)


def aggregate(before, after):
    require(len(before) == len(after) == COUNT, "Exactly four fixed examples are required")
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in (*before, *after)),
            "Loss must be finite and positive for every example")
    first, last = sum(before) / COUNT, sum(after) / COUNT
    ratio = last / first
    return {"mean_before_loss": first, "mean_after_loss": last, "mean_loss_ratio": ratio,
            "maximum_mean_loss_ratio": MAX_RATIO, "mechanism_gate_passed": ratio <= MAX_RATIO and last < first}


def compare_four(model, dataset, adapter, adapter_sha256, *, device, seed, deadline):
    """Real tensor evaluation core; production supplies the fully validated dataset."""
    import contextlib
    import torch
    from .engine import seed_everything
    from .model import load_native_adapter, preserved_state_digest
    require(model.training_profile == PROFILE, "Only the independent pure-attention profile is allowed")
    model.assert_trainable_profile()
    require(dataset.identity.get("split") == "train" and len(dataset) >= COUNT,
            "Four or more validated TRAIN examples are required")
    require(dataset._ids == sorted(dataset._ids), "Native dataset order must be the fixed sorted ID order")
    names = set(model.trainable_state())
    require(names and all(name.endswith((".lora_A", ".lora_B")) for name in names), "Unexpected non-A/B adapter tensor")
    require(all(torch.count_nonzero(value).item() == 0 for name, value in model.trainable_state().items() if name.endswith(".lora_B")),
            "Before evaluation requires a fresh zero-delta base, not an already loaded adapter")
    frozen = preserved_state_digest(model)
    require(frozen == model.native_preserved_sha256, "Frozen base changed before the comparison")
    model.eval().to(device=device)  # Keep FP32 master state; autocast only the CUDA forward.
    rows = [{"index": index} for index in range(COUNT)]
    for phase in ("before", "after"):
        if phase == "after":
            loaded = load_native_adapter(model, adapter, adapter_sha256)
        for index in range(COUNT):
            require(time.monotonic() < deadline, "Cooperative evaluation deadline reached")
            seed_everything(seed)
            example = dataset[index]  # Native dataset rehashes the target and prompt NPZ.
            require(example["id"] == dataset._ids[index], "Dataset order changed after validation")
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
            with torch.inference_mode(), autocast:
                result = model(example)
            values = {key: float(result[key].detach().float().cpu()) for key in ("loss", "code_loss", "text_loss")}
            require(all(math.isfinite(value) and value > 0 for value in values.values()), "Nonfinite or nonpositive actual loss")
            rows[index][phase] = values
            require(time.monotonic() < deadline, "Cooperative evaluation deadline reached")
        require(preserved_state_digest(model) == frozen, "Frozen original tensor changed during evaluation/adapter load")
    summary = aggregate([row["before"]["loss"] for row in rows], [row["after"]["loss"] for row in rows])
    return {**summary, "cases": rows, "adapter_step": loaded["step"], "frozen_state_sha256": frozen,
            "frozen_state_unchanged": True, "trainable_tensors": len(names), "dropout_enabled": False}


def execute(args, plan):
    require(args.allow_cuda, "Execution requires explicit --allow-cuda")
    require("torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "Use a fresh process without an existing CUDA context")
    admission = synth.resource_admission(args)
    output, report = plan["output"], dict(plan["report"])
    output.mkdir(mode=0o700, exist_ok=False)
    stage = "runtime_import"
    try:
        with synth.isolated_runtime(output, plan["source"], admission[0]["uuid"]):
            import torch
            from .adapter_synth import read_pure_payload
            from .data import load_native_dataset
            from .engine import seed_everything
            from .model import load_native_model, verify_official_assets
            report["cuda_access_attempted"] = True
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(),
                    "One explicit BF16-capable GPU is required")
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            stage = "dataset_validation"
            dataset = load_native_dataset(plan["dataset_config"])
            require(dataset.identity.get("split") == "train" and len(dataset) >= COUNT, "Need at least four validated TRAIN examples")
            paths = plan["paths"]
            config, identity = verify_official_assets(paths["official_repo"], paths["model_dir"], paths["model_manifest"])
            payload, parameters = read_pure_payload(plan["adapter"], args.adapter_sha256, identity, config)
            require(payload["step"] == EXPECTED_STEP, "This fixed gate requires the completed 100-update adapter")
            report.update(adapter_spec_sha256=identity_hash(payload["adapter_spec"]),
                          lora=payload["adapter_spec"]["lora"], text_loss_weight=parameters["text_loss_weight"])
            del payload
            require(parameters["lora_dropout"] == 0, "This fixed pilot gate requires the declared zero-dropout profile")
            current = synth.resource_admission(args)
            require(current[0]["uuid"] == admission[0]["uuid"], "Selected GPU changed before model loading")
            admission.extend(current)
            seed_everything(args.seed)
            stage = "model_loading"
            model = load_native_model(paths["official_repo"], paths["model_dir"], paths["model_manifest"], **parameters)
            require(len(model.lora_targets) == 48 and len(model.trainable_state()) == 96, "Expected exactly 48 modules and 96 A/B tensors")
            report.update(dataset_fully_validated=True, models_loaded=True,
                          dataset_identity_sha256=identity_hash(dataset.identity), base_identity_sha256=identity_hash(identity))
            stage = "fixed_four_evaluation"
            result = compare_four(model, dataset, plan["adapter"], args.adapter_sha256, device="cuda:0", seed=args.seed,
                                  deadline=time.monotonic() + args.max_seconds)
            require(synth.sha256(plan["dataset_config"]) == report["dataset_config_sha256"], "Dataset config changed during evaluation")
            report.update(result, status="fixed_train_mechanism_gate_passed" if result["mechanism_gate_passed"] else "fixed_train_mechanism_gate_failed",
                adapter_tensor_validation=True, evaluation_executed=True, precision="fp32_master_bf16_autocast",
                resource_admission_snapshots=[{key: value for key, value in sample.items() if key not in ("uuid", "name")} for sample in admission],
                gpu_identity_sha256=hashlib.sha256(admission[0]["uuid"].encode()).hexdigest(), deadline_scope="Cooperative checks between forwards; external parent timeout is still required")
            synth._write_json(output / "result.json", report)
            return report
    except BaseException as exc:
        # Never serialize exception text, example identifiers, transcripts or paths.
        synth._write_json(output / "failure.json", dict(report, status="evaluation_failed_not_verified",
                          failure_stage=stage, failure_type=type(exc).__name__, mechanism_gate_passed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace-root", "dataset", "adapter", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-seconds", type=int, default=300, help="Cooperative forward-phase limit, not an OS timeout")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    for name in ("min-free-gpu-mib", "min-free-ram-mib", "max-utilization"):
        parser.add_argument("--" + name, type=float)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**31 or not 1 <= args.max_seconds <= 600:
        parser.error("Use a bounded seed and cooperative limit of 1..600 seconds")
    try:
        plan = preflight(args)
        report = execute(args, plan) if args.execute else plan["report"]
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 3 if args.execute and not report["mechanism_gate_passed"] else 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "blocked_not_verified", "failure_type": type(exc).__name__,
                          "training_executed": False, "quality_verified": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
