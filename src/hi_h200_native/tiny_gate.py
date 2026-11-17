"""Low-memory integration gate with the actual official tiny GPT.

This is a math/state/resource probe using random small weights and synthetic
token/code/embedding inputs. It never loads a pretrained TTS checkpoint or
corpus, and it cannot establish Hindi speech quality.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[2]
STOP_CONTENT = "hi_h200_native.tiny_gate.owned-cooperative-stop.v1\n"


class TinyGateError(ValueError):
    pass


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise TinyGateError("Missing/oversized gate result JSON")
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda _value: (_ for _ in ()).throw(TinyGateError("Nonfinite gate JSON")))
    if not isinstance(value, dict):
        raise TinyGateError("Gate result must be a JSON object")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(args):
    if args.device not in ("cpu", "cuda:0"):
        raise TinyGateError("Use cpu or the single visible logical device cuda:0")
    if args.device == "cuda:0":
        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not args.allow_cuda or not mask or "," in mask or any(c.isspace() for c in mask):
            raise TinyGateError("CUDA requires --allow-cuda and exactly one explicit CUDA_VISIBLE_DEVICES entry")
    for name in ("updates", "stop_update", "memory_limit_mib", "min_free_mib"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise TinyGateError(name + " must be positive")
    if not 1 <= args.stop_update < args.updates <= 20:
        raise TinyGateError("Use 1 <= stop-update < updates <= 20 for this bounded tiny gate")
    if not 128 <= args.memory_limit_mib <= 2048 or args.min_free_mib < args.memory_limit_mib + 1024:
        raise TinyGateError("Allocator cap must be 128..2048 MiB, with at least cap+1024 MiB device headroom")
    for name in ("atol", "rtol"):
        value = getattr(args, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e-2:
            raise TinyGateError("Comparison tolerances must be finite and within [0,1e-2]")
    return args


def _safe_output(path, official_repo):
    from .train import protect_output
    return protect_output(path, [official_repo, SOURCE_ROOT / "src",
                                 SOURCE_ROOT / "tests", SOURCE_ROOT / "scripts"])


def _load_checkpoint(path):
    import torch
    path = Path(path).resolve(strict=True)
    sidecar = _read_json(path.with_suffix(path.suffix + ".sha256.json"))
    before = _sha256(path)
    if sidecar.get("sha256") != before:
        raise TinyGateError("Gate checkpoint SHA256 does not match its completed sidecar")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TinyGateError("Restricted checkpoint read failed; no unsafe fallback") from exc
    if _sha256(path) != before or not isinstance(state, dict) or state.get("kind") != "hi_h200_native_training":
        raise TinyGateError("Checkpoint changed during inspection or has the wrong kind")
    return state


def compare_states(expected, actual, *, atol, rtol):
    """Report tensor differences while requiring exact cursor, RNG and identity."""
    import torch
    results = {}
    for field in ("schema_version", "kind", "config", "identity", "runtime", "step", "sampler",
                  "parameter_groups", "model_identity", "adapter_spec", "scaler"):
        if expected.get(field) != actual.get(field):
            raise TinyGateError("Continuation differs in exact field: " + field)

    def compare_tree(first, second, name, *, exact):
        stats = {"tensors": 0, "max_absolute_error": 0.0, "max_relative_error": 0.0,
                 "all_tensors_exact": True}
        def visit(a, b, location):
            if torch.is_tensor(a):
                if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                    raise TinyGateError("Tensor shape/dtype changed: " + location)
                if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
                    raise TinyGateError("Nonfinite comparison tensor: " + location)
                stats["tensors"] += 1
                identical = torch.equal(a, b)
                stats["all_tensors_exact"] = stats["all_tensors_exact"] and identical
                if a.numel() and a.is_floating_point():
                    delta = (a.double() - b.double()).abs()
                    absolute = float(delta.max())
                    relative = float((delta / a.double().abs().clamp_min(1e-12)).max())
                    stats["max_absolute_error"] = max(stats["max_absolute_error"], absolute)
                    stats["max_relative_error"] = max(stats["max_relative_error"], relative)
                    good = identical if exact else torch.allclose(a, b, atol=atol, rtol=rtol)
                    if not good:
                        raise TinyGateError(f"Tensor comparison outside tolerance at {location}: max_abs={absolute}, max_rel={relative}")
                elif not identical:
                    raise TinyGateError("Nonfloating tensor differs: " + location)
            elif isinstance(a, dict):
                if not isinstance(b, dict) or set(a) != set(b):
                    raise TinyGateError("Dictionary keys changed: " + location)
                for key in a:
                    visit(a[key], b[key], location + "." + str(key))
            elif isinstance(a, (tuple, list)):
                if type(a) is not type(b) or len(a) != len(b):
                    raise TinyGateError("Sequence changed: " + location)
                for index, (left, right) in enumerate(zip(a, b)):
                    visit(left, right, location + "." + str(index))
            elif type(a) is not type(b) or a != b:
                raise TinyGateError("Scalar metadata differs: " + location)
        visit(first, second, name)
        return stats

    for field in ("trainable", "optimizer", "scheduler"):
        results[field] = compare_tree(expected[field], actual[field], field, exact=False)
    # RNG byte states and Python/NumPy scalar state must remain exact even when
    # CUDA floating-point reductions are allowed a small numeric tolerance.
    results["rng"] = compare_tree(expected["rng"], actual["rng"], "rng", exact=True)
    results["atol"], results["rtol"] = atol, rtol
    return results


def _segment(args):
    _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import random
    import torch
    from .engine import LoopConfig, StatefulTrainer, StopController, cosine_scheduler, seed_everything
    from .model import OFFICIAL_COMMIT, make_tiny_official_model, verify_official_source
    official = verify_official_source(args.official_repo)
    output = _safe_output(args.output_dir, official)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    config = LoopConfig(max_updates=args.updates, grad_accumulation=2, seed=71,
                        device=args.device, precision="bf16" if device.type == "cuda" else "fp32",
                        save_interval=args.updates, validation_interval=args.updates,
                        max_validation_examples=1).validate()
    resource = {"torch_allocator_cap_mib": None, "whole_device_or_context_memory_cap": False}
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        free, total = torch.cuda.mem_get_info(device)
        if free < args.min_free_mib * 1024**2:
            raise TinyGateError("Insufficient current shared-GPU headroom for tiny gate")
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib * 1024**2 / total, device)
        torch.cuda.reset_peak_memory_stats(device)
        resource.update(torch_allocator_cap_mib=args.memory_limit_mib,
                        initial_device_free_mib=free / 1024**2,
                        device_total_mib=total / 1024**2,
                        gpu_name=torch.cuda.get_device_name(device),
                        compute_capability=list(torch.cuda.get_device_capability(device)))
    model = make_tiny_official_model(official, seed=111, lora_dropout=.2).to(device)
    seed_everything(222)
    optimizer = torch.optim.AdamW(model.optimizer_groups(.02), lr=.003, foreach=False)
    scheduler = cosine_scheduler(optimizer, max_updates=args.updates, warmup_updates=1)
    observed_dtypes = set()

    class ProbeFeatures:
        def __len__(self):
            return 5
        def __getitem__(self, index):
            return {"id": "synthetic-" + str(index),
                    "text_ids": torch.tensor([3, 4 + index % 3, 5]),
                    "codes": torch.tensor([2, 3, 4 + index % 4, 5]),
                    "spk_emb": torch.full((192,), index / 10 + random.random() / 100),
                    "emo_vec": torch.randn(16) / 100 + float(np.random.random()) / 100}

    def observed_loss(actual_model, batch):
        result = actual_model.loss(batch)
        dtype = str(result["code_logits"].dtype)
        observed_dtypes.add(dtype)
        expected = torch.bfloat16 if device.type == "cuda" else torch.float32
        if result["code_logits"].dtype != expected:
            raise TinyGateError("Actual logits did not use the requested BF16/FP32 compute path")
        return result

    identity = {"purpose": "actual-official-tiny-GPT-state-gate-not-pretrained-Hindi",
                "official_commit": OFFICIAL_COMMIT,
                "native_code_sha256": {name: _sha256(Path(__file__).with_name(name))
                                       for name in ("tiny_gate.py", "engine.py", "model.py")},
                "allocator_policy": {"cap_mib": args.memory_limit_mib, "min_free_mib": args.min_free_mib}}
    trainer = StatefulTrainer(model, optimizer, scheduler, ProbeFeatures(), config, identity,
                              loss_fn=observed_loss)
    stop = StopController()
    owned_marker = STOP_CONTENT + str(os.getpid()) + "\n"

    def monitor(loop, _metrics):
        if device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(device)
            if free < args.min_free_mib * 1024**2:
                stop.request("shared-gpu-headroom")
        if args.phase == "stop" and loop.step == args.stop_update:
            with (output / "STOP").open("x", encoding="utf-8") as stream:
                stream.write(owned_marker)

    resume = args.resume
    if args.phase == "resume":
        if resume is None:
            raise TinyGateError("Resume segment requires an explicit completed native checkpoint")
        seed_everything(999)  # Real checkpoint load must replace all four RNGs.
    elif resume is not None:
        raise TinyGateError("Only the resume segment accepts --resume")
    with stop.signals():
        result = trainer.run(output, resume=resume, stop=stop, on_update=monitor)
    wanted_step = args.stop_update if args.phase == "stop" else args.updates
    wanted_status = "stopped_at_optimizer_boundary" if args.phase == "stop" else "completed"
    if result.get("step") != wanted_step or result.get("status") != wanted_status or stop.reason == "shared-gpu-headroom":
        raise TinyGateError("Segment did not reach its requested real optimizer boundary: " + json.dumps(result))
    state = _load_checkpoint(result["checkpoint"])
    if state["step"] != wanted_step or state["sampler"]["micro_steps"] != 2 * wanted_step:
        raise TinyGateError("Checkpoint did not preserve the actual accumulated-update cursor")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free, _total = torch.cuda.mem_get_info(device)
        resource.update(peak_torch_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
                        peak_torch_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
                        final_device_free_mib=free / 1024**2)
    report = {"schema_version": 1, "phase": args.phase, "status": "segment_verified",
              "device": args.device, "actual_logit_dtypes": sorted(observed_dtypes),
              "actual_model_class": type(model.base).__name__, "official_commit": OFFICIAL_COMMIT,
              "checkpoint": str(Path(result["checkpoint"]).resolve()),
              "checkpoint_sha256": _sha256(result["checkpoint"]),
              "step": wanted_step, "updates_this_run": result["updates_this_run"],
              "sampler": state["sampler"], "resource": resource,
              "owned_stop_content": owned_marker if args.phase == "stop" else None,
              "pretrained_model_loaded": False, "hindi_quality_verified": False,
              "waveform_inference_verified": False}
    path = output.parent / (output.name + "." + args.phase + "-result.json")
    _write_json(path, report)
    return report


def run_gate(args):
    """Orchestrate three fresh OS processes; the parent never creates CUDA context."""
    _validate_args(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .model import verify_official_source
    official = verify_official_source(args.official_repo)
    output = _safe_output(args.output_dir, official)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise TinyGateError("Gate output must be absent or empty; never reuse a previous gate")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports = {}

    def child(phase, run_output, resume=None):
        command = [sys.executable, "-B", "-m", "hi_h200_native.tiny_gate", "segment",
                   "--official-repo", str(official), "--output-dir", str(run_output),
                   "--device", args.device, "--updates", str(args.updates),
                   "--stop-update", str(args.stop_update), "--memory-limit-mib", str(args.memory_limit_mib),
                   "--min-free-mib", str(args.min_free_mib), "--atol", str(args.atol),
                   "--rtol", str(args.rtol), "--phase", phase]
        if args.allow_cuda:
            command.append("--allow-cuda")
        if resume is not None:
            command.extend(("--resume", str(resume)))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"])
        log = output / (phase + ".process.log")
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise TinyGateError(f"{phase} subprocess failed with exit {result.returncode}; inspect its local log: {log}")
        report = _read_json(run_output.parent / (run_output.name + "." + phase + "-result.json"))
        if report.get("status") != "segment_verified" or report.get("phase") != phase or report.get("device") != args.device:
            raise TinyGateError("Subprocess report did not verify the requested segment")
        reports[phase] = report
        return report

    continuous = child("continuous", output / "continuous")
    stopped = child("stop", output / "split")
    marker = output / "split/STOP"
    if marker.is_symlink() or not marker.is_file() or marker.read_text(encoding="utf-8") != stopped["owned_stop_content"]:
        raise TinyGateError("Expected owned STOP marker is missing/changed; refusing to remove another stop request")
    marker.unlink()
    resumed = child("resume", output / "split", Path(stopped["checkpoint"]))
    expected, actual = _load_checkpoint(continuous["checkpoint"]), _load_checkpoint(resumed["checkpoint"])
    comparison = compare_states(expected, actual, atol=args.atol, rtol=args.rtol)
    if stopped["updates_this_run"] != args.stop_update or resumed["updates_this_run"] != args.updates - args.stop_update:
        raise TinyGateError("Resume did not continue from the saved optimizer step")
    import torch
    if torch.cuda.is_initialized():
        raise TinyGateError("Orchestration process unexpectedly created a CUDA context")
    report = {"schema_version": 1, "status": "tiny_gate_passed",
              "device": args.device, "cuda_bf16_verified": args.device == "cuda:0",
              "separate_process_stop_resume_verified": True,
              "optimizer_updates_continuous": args.updates, "optimizer_updates_before_stop": args.stop_update,
              "optimizer_updates_after_resume": args.updates - args.stop_update,
              "comparison": comparison, "segments": reports,
              "elapsed_wall_seconds": time.monotonic() - started,
              "orchestrator_cuda_context_created": False,
              "input_kind": "synthetic token/code/speaker/emotion probes; no speech corpus",
              "model_kind": "actual official UnifiedVoice with small random weights",
              "pretrained_model_loaded": False, "hindi_quality_verified": False,
              "waveform_inference_verified": False,
              "limitations": ["CUDA numerical tolerance is reported, not a general bitwise-reproducibility promise.",
                              "PyTorch allocator cap excludes CUDA context/library allocations and other processes.",
                              "SIGKILL/eviction cannot guarantee a new checkpoint; this gate uses cooperative STOP."]}
    _write_json(output / "gate-result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "segment"):
        p = sub.add_parser(command, help="Three-process gate" if command == "run" else "Internal single segment")
        p.add_argument("--official-repo", required=True, type=Path)
        p.add_argument("--output-dir", required=True, type=Path)
        p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
        p.add_argument("--allow-cuda", action="store_true")
        p.add_argument("--updates", type=int, default=6)
        p.add_argument("--stop-update", type=int, default=3)
        p.add_argument("--memory-limit-mib", type=int, default=512)
        p.add_argument("--min-free-mib", type=int, default=4096)
        p.add_argument("--atol", type=float, default=1e-5)
        p.add_argument("--rtol", type=float, default=1e-4)
        if command == "segment":
            p.add_argument("--phase", choices=("continuous", "stop", "resume"), required=True)
            p.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_gate(args) if args.command == "run" else _segment(args)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "tiny_gate_failed_or_blocked", "reason": str(exc),
                          "cuda_bf16_verified": False, "pretrained_model_loaded": False,
                          "hindi_quality_verified": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "device": result["device"],
                      "cuda_bf16_verified": result.get("cuda_bf16_verified", False),
                      "pretrained_model_loaded": False, "hindi_quality_verified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
