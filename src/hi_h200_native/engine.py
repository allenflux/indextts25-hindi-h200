"""Original batch-one training loop with explicit, complete continuation state.

Only real PyTorch losses/backward/optimizer updates are accepted. No model, corpus
or third-party training implementation is downloaded or invoked by this module.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import datetime
import fcntl
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import time
from typing import Callable, Mapping

import numpy as np
import torch


class TrainingError(ValueError):
    """A failed invariant; callers must retain the previous completed checkpoint."""


@dataclass(frozen=True)
class LoopConfig:
    max_updates: int = 100
    grad_accumulation: int = 1
    seed: int = 1234
    device: str = "cpu"
    precision: str = "fp32"
    grad_clip: float = 1.0
    save_interval: int = 25
    validation_interval: int = 25
    max_validation_examples: int = 8

    def validate(self):
        for name in ("max_updates", "grad_accumulation", "save_interval",
                     "validation_interval", "max_validation_examples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise TrainingError(name + " must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise TrainingError("seed must be an integer in [0, 2**31)")
        if type(self.grad_clip) not in (int, float) or not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise TrainingError("grad_clip must be finite and positive")
        if self.device != "cpu" and not (isinstance(self.device, str) and re.fullmatch(r"cuda:[0-9]+", self.device)):
            raise TrainingError("Use cpu or an explicit cuda:N device")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise TrainingError("Only explicit CPU and CUDA devices are supported")
        if (device.type, self.precision) not in (("cpu", "fp32"), ("cuda", "bf16")):
            raise TrainingError("CPU requires fp32; CUDA requires bf16")
        if device.type == "cuda":
            mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not mask or "," in mask or any(c.isspace() for c in mask) or device.index != 0:
                raise TrainingError("Native single-GPU training requires one explicit CUDA_VISIBLE_DEVICES entry and device cuda:0")
            if torch.cuda.device_count() != 1:
                raise TrainingError("Exactly one GPU must be visible to the native trainer")
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise TrainingError("Requested CUDA BF16 is unavailable")
        return self


class StopController:
    """Signal handlers set a flag; all serialization happens in the main loop."""
    def __init__(self):
        self.requested = False
        self.reason = None

    def request(self, reason="requested"):
        self.requested = True
        self.reason = str(reason)

    @contextlib.contextmanager
    def signals(self):
        old = {}
        def handle(number, _frame):
            self.request("signal:" + signal.Signals(number).name)
        try:
            for number in (signal.SIGINT, signal.SIGTERM):
                old[number] = signal.getsignal(number)
                signal.signal(number, handle)
            yield self
        finally:
            for number, handler in old.items():
                signal.signal(number, handler)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainingError("Identity and configuration must be finite JSON primitives") from exc


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_state():
    if torch.cuda.is_initialized() and torch.cuda.device_count() != 1:
        raise TrainingError("Refusing to capture RNG for multiple visible CUDA devices")
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"kind": n[0], "keys": n[1].tolist(), "position": int(n[2]),
                  "has_gauss": int(n[3]), "cached_gaussian": float(n[4])},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise TrainingError("Missing complete RNG state")
    n = state["numpy"]
    if set(n) != {"kind", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise TrainingError("Malformed NumPy RNG state")
    random.setstate(state["python"])
    np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    cuda = state["torch_cuda"]
    if cuda:
        if not torch.cuda.is_initialized() or len(cuda) != torch.cuda.device_count():
            raise TrainingError("CUDA RNG device topology differs from the checkpoint")
        torch.cuda.set_rng_state_all(cuda)


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TrainingError("Unsupported checkpoint object type: " + type(value).__name__)


def _finite_tree(value, depth=0):
    if depth > 40:
        raise TrainingError("Checkpoint metadata nesting exceeds the safety bound")
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            raise TrainingError("Nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise TrainingError("Unsupported checkpoint dictionary key")
        for child in value.values():
            _finite_tree(child, depth + 1)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise TrainingError("Nonfinite checkpoint scalar")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise TrainingError("Unsupported checkpoint metadata object")


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    temporary = sidecar.with_name("." + sidecar.name + "." + str(os.getpid()) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "sha256": digest_file(path),
                       "step": payload["step"]}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, sidecar)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _parameter_groups(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        current = []
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None or name in seen:
                raise TrainingError("Optimizer must cover each trainable parameter exactly once")
            seen.add(name)
            current.append(name)
        groups.append(current)
    if seen != set(names.values()):
        raise TrainingError("Optimizer does not cover the complete trainable parameter set")
    return groups


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_move(child, device) for child in value)
    # FeatureExample dataclasses may be used by callers; model accepts mappings.
    if hasattr(value, "__dataclass_fields__"):
        return {name: _move(getattr(value, name), device) for name in value.__dataclass_fields__}
    return value


def _runtime():
    try:
        transformers = package_metadata.version("transformers")
    except package_metadata.PackageNotFoundError:
        transformers = None
    return {"transformers": transformers, "torch_cuda_build": str(torch.version.cuda), "python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None,
            "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_initialized() else None,
            "cuda_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_initialized() else None,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


class StatefulTrainer:
    """Sequential batch-one loop; no worker or prefetched-data state is hidden."""
    schema_version = 1

    def __init__(self, model, optimizer, scheduler, dataset, config: LoopConfig,
                 identity: dict, *, loss_fn: Callable | None = None, validation_dataset=None):
        self.config = config.validate()
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__") or len(dataset) < 1:
            raise TrainingError("Training dataset must be a nonempty, indexable collection")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.dataset, self.validation_dataset = dataset, validation_dataset
        self.identity = json.loads(_canonical(identity))
        if not self.identity:
            raise TrainingError("An explicit source/model/data identity is required")
        self.loss_fn = loss_fn or (lambda model, batch: model.loss(batch))
        self.parameter_groups = _parameter_groups(model, optimizer)
        self.device = torch.device(config.device)
        if any(parameter.device != self.device for parameter in self.model.parameters()):
            raise TrainingError("Move the model to its target device before constructing the optimizer")
        self.step = self.epoch = self.offset = self.micro_steps = 0
        self.last_checkpoint = None
        self._at_boundary = True

    def _next(self):
        if self.offset >= len(self.dataset):
            self.epoch += 1
            self.offset = 0
        order = list(range(len(self.dataset)))
        random.Random(self.config.seed + self.epoch).shuffle(order)
        value = self.dataset[order[self.offset]]
        self.offset += 1
        self.micro_steps += 1
        return _move(value, self.device)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else contextlib.nullcontext()

    def _loss(self, batch):
        output = self.loss_fn(self.model, batch)
        if not isinstance(output, dict) or "loss" not in output:
            raise TrainingError("Model loss must return a dictionary with an actual scalar loss tensor")
        loss = output["loss"]
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss).item():
            raise TrainingError("Loss must be a finite scalar tensor")
        return output

    def checkpoint_state(self, reason):
        if not self._at_boundary or self.micro_steps != self.step * self.config.grad_accumulation:
            raise TrainingError("Checkpoints are allowed only after complete optimizer updates")
        state = {
            "schema_version": self.schema_version, "kind": "hi_h200_native_training",
            "reason": str(reason), "config": asdict(self.config), "identity": self.identity,
            "runtime": _runtime(), "step": self.step,
            "model_identity": getattr(self.model, "native_identity", None),
            "adapter_spec": self.model.adapter_spec() if hasattr(self.model, "adapter_spec") else None,
            "sampler": {"epoch": self.epoch, "offset": self.offset, "micro_steps": self.micro_steps,
                        "dataset_length": len(self.dataset)},
            "parameter_groups": self.parameter_groups,
            "trainable": {name: parameter.detach().cpu().clone()
                          for name, parameter in self.model.named_parameters() if parameter.requires_grad},
            "optimizer": _cpu_tree(self.optimizer.state_dict()),
            "scheduler": _cpu_tree(self.scheduler.state_dict()) if self.scheduler is not None else None,
            "scaler": None, "rng": rng_state(),
        }
        _finite_tree(state)
        return state

    def save(self, output_dir, reason):
        state = self.checkpoint_state(reason)
        root = Path(output_dir).resolve()
        directory = root / "checkpoints"
        if directory.is_symlink() or not directory.resolve().is_relative_to(root):
            raise TrainingError("Checkpoint directory escapes the run root")
        path = directory / ("step" + str(self.step).zfill(8) + ".pt")
        if path.exists():
            raise TrainingError("Refusing to overwrite an existing step checkpoint: " + str(path))
        self.last_checkpoint = atomic_torch_save(path, state)
        return self.last_checkpoint

    def load(self, checkpoint):
        path = Path(checkpoint).resolve(strict=True)
        if not path.is_file() or path.suffix != ".pt" or path.stat().st_size > 32 * 1024**3:
            raise TrainingError("Resume requires an explicit completed checkpoint no larger than 32 GiB")
        sidecar_path = path.with_suffix(path.suffix + ".sha256.json")
        if not sidecar_path.is_file() or sidecar_path.stat().st_size > 4096:
            raise TrainingError("Missing or oversized checkpoint SHA256 sidecar")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict) or set(sidecar) != {"schema_version", "sha256", "step"} or type(sidecar["step"]) is not int:
            raise TrainingError("Malformed checkpoint SHA256 sidecar")
        before = digest_file(path)
        if sidecar.get("schema_version") != 1 or sidecar.get("sha256") != before:
            raise TrainingError("Checkpoint SHA256 sidecar mismatch")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise TrainingError("Restricted checkpoint load failed; no unsafe fallback: " + str(exc)) from exc
        if digest_file(path) != before:
            raise TrainingError("Checkpoint changed during inspection")
        expected = {"schema_version", "kind", "reason", "config", "identity", "runtime", "step", "sampler",
                    "parameter_groups", "trainable", "optimizer", "scheduler", "scaler", "rng", "model_identity", "adapter_spec"}
        if not isinstance(state, dict) or set(state) != expected or state["schema_version"] != self.schema_version or state["kind"] != "hi_h200_native_training":
            raise TrainingError("Checkpoint schema differs from the native trainer")
        if state["config"] != asdict(self.config) or state["identity"] != self.identity:
            raise TrainingError("Source/model/data/config identity changed; refusing continuation")
        if state["model_identity"] != getattr(self.model, "native_identity", None) or state["adapter_spec"] != (self.model.adapter_spec() if hasattr(self.model, "adapter_spec") else None):
            raise TrainingError("Reconstructed native base or adapter specification differs")
        if state["runtime"] != _runtime() or state["parameter_groups"] != self.parameter_groups:
            raise TrainingError("Runtime or optimizer parameter ordering differs")
        _finite_tree(state)
        step, sampler = state["step"], state["sampler"]
        if type(step) is not int or not 0 <= step <= self.config.max_updates or sidecar.get("step") != step:
            raise TrainingError("Invalid saved optimizer step")
        if not isinstance(sampler, dict) or set(sampler) != {"epoch", "offset", "micro_steps", "dataset_length"}:
            raise TrainingError("Missing sampler state")
        if any(type(sampler[key]) is not int for key in sampler) or sampler["dataset_length"] != len(self.dataset):
            raise TrainingError("Sampler dataset changed")
        epoch, offset, micros = sampler["epoch"], sampler["offset"], sampler["micro_steps"]
        if epoch < 0 or not 0 <= offset <= len(self.dataset) or micros != step * self.config.grad_accumulation or epoch * len(self.dataset) + offset != micros:
            raise TrainingError("Sampler cursor differs from completed optimizer boundaries")
        named = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if not isinstance(state["trainable"], dict) or set(state["trainable"]) != set(named):
            raise TrainingError("Saved trainable names differ from the reconstructed model")
        for name, value in state["trainable"].items():
            if not torch.is_tensor(value) or value.shape != named[name].shape or value.dtype != named[name].dtype:
                raise TrainingError("Saved parameter shape/dtype differs: " + name)
        if state["scaler"] is not None or (state["scheduler"] is None) != (self.scheduler is None):
            raise TrainingError("Precision/scheduler state differs")
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(state["trainable"][name].to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.step, self.epoch, self.offset, self.micro_steps = step, epoch, offset, micros
        self.optimizer.zero_grad(set_to_none=True)
        self._at_boundary = True
        restore_rng(state["rng"])
        self.last_checkpoint = path
        return {"status": "resumed_state_loaded", "step": step, "cursor": sampler}

    def evaluate(self):
        if self.validation_dataset is None:
            return {}
        if len(self.validation_dataset) < 1:
            raise TrainingError("Validation dataset is empty")
        rng = rng_state()
        was_training = self.model.training
        total = count = 0
        try:
            self.model.eval()
            with torch.no_grad():
                for index in range(min(len(self.validation_dataset), self.config.max_validation_examples)):
                    with self._autocast():
                        output = self._loss(_move(self.validation_dataset[index], self.device))
                    total += float(output["loss"].detach().cpu())
                    count += 1
        finally:
            self.model.train(was_training)
            restore_rng(rng)
        return {"validation_loss_mean_per_example": total / count, "validation_examples": count}

    def run(self, output_dir, *, resume=None, stop=None, stop_after_updates=None, on_update=None):
        output = Path(output_dir).resolve()
        if output.exists() and not output.is_dir():
            raise TrainingError("Output path is not a directory")
        if resume is None and output.exists() and any(output.iterdir()):
            raise TrainingError("Fresh training requires an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        if stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 1):
            raise TrainingError("stop_after_updates must be a positive absolute update number")
        stop = stop or StopController()
        lock_path = output / ".native-trainer.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TrainingError("Another trainer already holds this output directory") from exc
            if resume is not None:
                self.load(resume)
            elif any(path.name != lock_path.name for path in output.iterdir()):
                raise TrainingError("Fresh training requires an empty output directory")
            if (output / "STOP").exists():
                raise TrainingError("STOP is already present; review and remove your stop marker before launch/resume")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            starting_step = self.step
            if self.step == self.config.max_updates:
                return {"status": "already_complete", "step": self.step, "checkpoint": str(self.last_checkpoint), "updates_this_run": 0}
            if stop.requested or (stop_after_updates is not None and self.step >= stop_after_updates):
                checkpoint = self.last_checkpoint or self.save(output, "stopped_before_update")
                return {"status": "stopped_at_optimizer_boundary", "step": self.step, "checkpoint": str(checkpoint), "updates_this_run": 0}
            log_path = output / "updates.jsonl"
            if log_path.is_symlink():
                raise TrainingError("Training log must not be a symlink")
            with log_path.open("a", encoding="utf-8") as log:
                while self.step < self.config.max_updates:
                    update_started = time.perf_counter()
                    self._at_boundary = False
                    losses = []
                    for _ in range(self.config.grad_accumulation):
                        batch = self._next()
                        with self._autocast():
                            result = self._loss(batch)
                            loss = result["loss"]
                        if not loss.requires_grad:
                            raise TrainingError("Loss is detached; refusing a fake training update")
                        (loss / self.config.grad_accumulation).backward()
                        losses.append(float(loss.detach().cpu()))
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    if not any(p.grad is not None for p in parameters):
                        raise TrainingError("No trainable gradients were produced")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip, error_if_nonfinite=True)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    self._at_boundary = True
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    compute_seconds = time.perf_counter() - update_started
                    metrics = {"step": self.step, "loss": sum(losses) / len(losses),
                               "update_compute_seconds": compute_seconds,
                               "micro_examples": self.config.grad_accumulation,
                               "gradient_norm": float(norm.detach().cpu()),
                               "epoch": self.epoch, "offset": self.offset,
                               "learning_rates": [float(g["lr"]) for g in self.optimizer.param_groups]}
                    if self.step % self.config.validation_interval == 0 or self.step == self.config.max_updates:
                        metrics.update(self.evaluate())
                    if on_update is not None:
                        on_update(self, metrics)
                    marker = output / "STOP"
                    requested = stop.requested or marker.exists() or (stop_after_updates is not None and self.step >= stop_after_updates)
                    reason = "cooperative_stop" if requested else "complete" if self.step == self.config.max_updates else "periodic"
                    if requested or self.step == self.config.max_updates or self.step % self.config.save_interval == 0:
                        checkpoint = self.save(output, reason)
                        metrics["checkpoint"] = str(checkpoint)
                        metrics["checkpoint_reason"] = reason
                    wall_seconds = time.perf_counter() - update_started
                    metrics.update(update_wall_seconds=wall_seconds, optimizer_updates_per_second=1 / wall_seconds,
                                   examples_per_second=self.config.grad_accumulation / wall_seconds)
                    if self.device.type == "cuda":
                        metrics.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
                    log.write(_canonical(metrics) + "\n")
                    log.flush()
                    if requested:
                        return {"status": "stopped_at_optimizer_boundary", "step": self.step,
                                "checkpoint": str(self.last_checkpoint), "reason": stop.reason or "STOP/update limit", "updates_this_run": self.step - starting_step}
        return {"status": "completed", "step": self.step, "checkpoint": str(self.last_checkpoint), "updates_this_run": self.step - starting_step}


def cosine_scheduler(optimizer, *, max_updates, warmup_updates=0, min_lr_ratio=0.1):
    if type(max_updates) is not int or max_updates < 1 or type(warmup_updates) is not int or not 0 <= warmup_updates < max_updates:
        raise TrainingError("Scheduler needs 0 <= warmup_updates < max_updates")
    if type(min_lr_ratio) not in (int, float) or not math.isfinite(min_lr_ratio) or not 0 <= min_lr_ratio <= 1:
        raise TrainingError("min_lr_ratio must be in [0,1]")
    def multiplier(step):
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        fraction = min(1.0, max(0.0, (step - warmup_updates) / (max_updates - warmup_updates)))
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
