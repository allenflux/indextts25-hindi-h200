"""Original batch-one training integration with the licensed official 2.5 GPT.

No community training backend is imported. Full pretrained checkpoints and source
are verified separately; tiny CPU tests instantiate the actual official class.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_COMMIT = "d9e41aac89fd00b3d71497fddb287b7f24613712"
MODEL_REPOSITORY = "IndexTeam/IndexTTS-2.5"
TOKENIZER_FILE = "multilingual_zh_ja_yue_char_del.tiktoken"
MINIMUM_FILES = {"config.yaml", "gpt.pth", TOKENIZER_FILE, "LICENSE"}


class NativeModelError(ValueError):
    """A native integration prerequisite is missing or inconsistent."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise NativeModelError("Missing or oversized JSON manifest")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise NativeModelError("Duplicate JSON manifest key")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(NativeModelError("Nonfinite JSON value")))
    if not isinstance(value, dict):
        raise NativeModelError("Expected JSON object")
    return value


def verify_official_source(official_repo):
    root = Path(official_repo).resolve(strict=True)
    def git(*args):
        command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
                   "-c", "core.untrackedCache=false", "-C", str(root), *args]
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise NativeModelError("Cannot verify pinned local official source") from exc
    if git("rev-parse", "--show-toplevel") != str(root) or git("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise NativeModelError("Official source must be the exact reviewed commit/repository root")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise NativeModelError("Official source has modified/untracked files; use a clean isolated checkout")
    for name in ("LICENSE", "LICENSE_ZH.txt", "indextts/gpt/model_v2.py", "indextts/utils/tokenizer.py"):
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise NativeModelError("Official required source/license file is missing or escapes its root")
    return root


def _model_file(root, name):
    if not isinstance(name, str) or not name or any(ord(c) < 32 for c in name) or "\\" in name or ":" in name:
        raise NativeModelError("Model manifest paths must be safe relative POSIX paths")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise NativeModelError("Model manifest path escapes its root")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise NativeModelError("Model manifest path is not a confined local file")
    return path


def verify_official_assets(official_repo, model_dir, model_manifest):
    """Verify local source, source notices, immutable snapshot and all supplied hashes.

    Hashes establish consistency, not ownership or model provenance. Operators must
    separately review the official/model/data terms. No backend or weights load.
    """
    verify_official_source(official_repo)
    root = Path(model_dir).resolve(strict=True)
    manifest = _json_object(model_manifest)
    if set(manifest) != {"repository", "snapshot_revision", "files"} or manifest["repository"] != MODEL_REPOSITORY:
        raise NativeModelError("Model manifest must identify the official 2.5 repository and complete file hashes")
    revision = manifest["snapshot_revision"]
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise NativeModelError("Pin the actual immutable model snapshot revision")
    files = manifest["files"]
    if not isinstance(files, dict) or not MINIMUM_FILES.issubset(files):
        raise NativeModelError("Model manifest is missing minimum 2.5 files")
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise NativeModelError("Every supplied model file needs a SHA256")
        if file_sha256(_model_file(root, name)) != digest:
            raise NativeModelError("Model SHA256 mismatch: " + name)
    try:
        import yaml
        config = yaml.safe_load(_model_file(root, "config.yaml").read_text(encoding="utf-8"))
    except (ImportError, ValueError) as exc:
        raise NativeModelError("PyYAML and a safe readable model config are required") from exc
    if not isinstance(config, dict) or str(config.get("version")) != "2.5":
        raise NativeModelError("Actual model config must declare version: 2.5; a directory name is insufficient")
    if not isinstance(config.get("gpt"), dict) or not isinstance(config.get("semantic_codec"), dict):
        raise NativeModelError("Model config lacks GPT/semantic codec capacities")
    gpt, codec = config["gpt"], config["semantic_codec"]
    for name in ("layers", "model_dim", "heads", "max_text_tokens", "max_mel_tokens", "number_text_tokens", "number_mel_codes"):
        if type(gpt.get(name)) is not int or gpt[name] < 1:
            raise NativeModelError("Actual GPT capacities must be positive integers: " + name)
    if gpt["layers"] > 64 or gpt["model_dim"] > 4096 or gpt["number_text_tokens"] > 1_000_000 or gpt["number_mel_codes"] > 1_000_000:
        raise NativeModelError("Model architecture exceeds this audited loader's allocation bounds")
    if gpt["model_dim"] % gpt["heads"]:
        raise NativeModelError("GPT model dimension must be divisible by attention heads")
    size = codec.get("codebook_size")
    if type(size) is not int or size < 1 or (gpt.get("start_mel_token"), gpt.get("stop_mel_token"), gpt["number_mel_codes"]) != (size, size + 1, size + 2):
        raise NativeModelError("Codec and GPT reserved semantic IDs are inconsistent")
    reserved = (gpt.get("start_text_token"), gpt.get("stop_text_token"))
    if any(type(value) is not int or not 0 <= value < gpt["number_text_tokens"] for value in reserved) or reserved[0] == reserved[1]:
        raise NativeModelError("Actual GPT text boundary IDs are invalid")
    identity = {"official_commit": OFFICIAL_COMMIT, "model_repository": MODEL_REPOSITORY,
                "model_snapshot_revision": revision, "model_manifest_sha256": file_sha256(model_manifest),
                "model_files": dict(files), "config_sha256": files["config.yaml"],
                "tokenizer_sha256": files[TOKENIZER_FILE]}
    return config, identity


def import_official_model(official_repo):
    root = verify_official_source(official_repo)
    existing = sys.modules.get("indextts.gpt.model_v2")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise NativeModelError("A different IndexTTS source is already imported in this process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("indextts.gpt.model_v2")
    except ImportError as exc:
        raise NativeModelError("Official GPT dependencies are missing: " + str(exc)) from exc
    if not Path(inspect.getfile(module.UnifiedVoice)).resolve().is_relative_to(root):
        raise NativeModelError("Imported GPT does not belong to the verified source checkout")
    return module.UnifiedVoice


def load_verified_base(official_repo, model_dir, model_manifest):
    config, identity = verify_official_assets(official_repo, model_dir, model_manifest)
    cls = import_official_model(official_repo)
    kwargs = dict(config["gpt"])
    kwargs.update(spk_cond_mode="campplus", use_accel=False)
    base = cls(**kwargs)
    path = _model_file(Path(model_dir).resolve(strict=True), "gpt.pth")
    expected_digest = identity["model_files"]["gpt.pth"]
    if file_sha256(path) != expected_digest:
        raise NativeModelError("GPT weights changed after asset verification")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != expected_digest:
            raise NativeModelError("GPT weights changed during restricted loading")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise NativeModelError("GPT checkpoint must be a tensor state dictionary")
        if any(value.is_complex() or (value.is_floating_point() and not torch.isfinite(value).all()) for value in state.values()):
            raise NativeModelError("GPT checkpoint contains complex or nonfinite tensors")
        base.load_state_dict(state, strict=True)
    except Exception as exc:
        raise NativeModelError("Strict restricted CPU GPT load failed; no partial or unsafe fallback: " + str(exc)) from exc
    base.native_identity = identity
    return base, config, identity


class Conv1DLoRA(nn.Module):
    """Low-rank residual for HF Conv1D's [input, output] matrix orientation."""
    def __init__(self, original, rank, alpha, dropout):
        super().__init__()
        if original.weight.ndim != 2 or original.bias.ndim != 1 or original.weight.shape[1] != original.bias.numel():
            raise NativeModelError("Unsupported Conv1D matrix layout")
        self.base = original
        width_in, width_out = original.weight.shape
        if type(rank) is not int or not 1 <= rank <= min(width_in, width_out):
            raise NativeModelError("LoRA rank must fit the actual input/output dimensions")
        self.scale = float(alpha) / rank
        if not math.isfinite(self.scale) or self.scale <= 0 or not 0 <= dropout < 1:
            raise NativeModelError("Invalid LoRA scaling or dropout")
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(original.weight.new_empty((rank, width_in)))
        self.lora_B = nn.Parameter(original.weight.new_zeros((width_out, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale


def _row_gradient(parameter, rows):
    mask = torch.zeros(parameter.shape[0], 1, device=parameter.device, dtype=parameter.dtype)
    mask[list(rows)] = 1
    # The buffer-free hook constructs its mask on the gradient device after .to().
    def hook(gradient):
        return gradient * mask.to(device=gradient.device, dtype=gradient.dtype)
    return parameter.register_hook(hook)


class NativeHindiModel(nn.Module):
    """Batch-one real teacher-forcing loss; no padded-batch or synthesis claim."""
    def __init__(self, base, *, lora_rank=8, lora_alpha=16.0, lora_dropout=0.0,
                 hindi_row=17, text_rows=(), text_loss_weight=0.1):
        super().__init__()
        if base.spk_cond_mode != "campplus" or getattr(base, "use_accel", False):
            raise NativeModelError("Native training requires unaccelerated CAMPPlus-mode official GPT")
        if type(hindi_row) is not int or not 0 <= hindi_row < base.lang_embedding.num_embeddings:
            raise NativeModelError("Hindi language embedding row is out of range")
        if not math.isfinite(text_loss_weight) or text_loss_weight < 0:
            raise NativeModelError("Text auxiliary loss weight must be finite and nonnegative")
        rows = tuple(sorted(set(text_rows)))
        if any(type(row) is not int or row in (base.start_text_token, base.stop_text_token) or not 0 <= row < base.number_text_tokens for row in rows):
            raise NativeModelError("Text trainable rows must be valid nonreserved raw token IDs")
        self.base, self.hindi_row, self.text_rows = base, hindi_row, rows
        base.gpt.config.use_cache = False
        self.text_loss_weight = float(text_loss_weight)
        self.lora_config = {"rank": lora_rank, "alpha": float(lora_alpha), "dropout": float(lora_dropout)}
        self.lora_targets = []
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        for index, block in enumerate(base.gpt.h):
            for attribute in ("c_attn", "c_proj"):
                original = getattr(block.attn, attribute)
                if original.__class__.__name__ != "Conv1D":
                    raise NativeModelError("Expected the reviewed HF Conv1D attention module")
                setattr(block.attn, attribute, Conv1DLoRA(original, lora_rank, lora_alpha, lora_dropout))
                self.lora_targets.append(f"gpt.h.{index}.attn.{attribute}")
        self._row_hooks = []
        base.lang_embedding.weight.requires_grad_(True)
        self._row_hooks.append(_row_gradient(base.lang_embedding.weight, (hindi_row,)))
        if rows:
            base.text_embedding.weight.requires_grad_(True)
            self._row_hooks.append(_row_gradient(base.text_embedding.weight, rows))
        for head in (base.mel_head, base.text_head):
            for parameter in head.parameters():
                parameter.requires_grad_(True)

    def optimizer_groups(self, weight_decay=0.01):
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise NativeModelError("Weight decay must be finite and nonnegative")
        no_decay = {id(self.base.lang_embedding.weight), id(self.base.text_embedding.weight)}
        ordinary, protected = [], []
        for parameter in self.parameters():
            if parameter.requires_grad:
                (protected if id(parameter) in no_decay else ordinary).append(parameter)
        return [{"params": ordinary, "weight_decay": float(weight_decay)},
                {"params": protected, "weight_decay": 0.0}]

    def trainable_state(self):
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_trainable_state(self, state):
        expected = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if not isinstance(state, dict) or set(state) != set(expected):
            raise NativeModelError("Native adapter does not cover exactly all trainable parameters, including both heads")
        for name, parameter in expected.items():
            value = state[name]
            if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or not value.is_floating_point() or not torch.isfinite(value).all():
                raise NativeModelError("Invalid native trainable tensor: " + name)
            protected_rows = {"base.lang_embedding.weight": (self.hindi_row,),
                              "base.text_embedding.weight": self.text_rows}
            if name in protected_rows:
                permitted = set(protected_rows[name])
                for start in range(0, parameter.shape[0], 1024):
                    stop = min(start + 1024, parameter.shape[0])
                    keep = [row - start for row in range(start, stop) if row not in permitted]
                    if keep and not torch.equal(value[start:stop].detach().cpu()[keep], parameter[start:stop].detach().cpu()[keep]):
                        raise NativeModelError("Native state changes a protected language/text embedding row: " + name)
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def adapter_spec(self):
        return {"format": "hi_h200_native_batch1_v1", "hindi_row": self.hindi_row,
                "text_rows": list(self.text_rows), "text_loss_weight": self.text_loss_weight,
                "lora": dict(self.lora_config), "lora_targets": list(self.lora_targets),
                "heads": ["mel_head", "text_head"], "model_dim": self.base.model_dim}

    def _batch(self, batch):
        def item(name):
            return batch[name] if isinstance(batch, Mapping) else getattr(batch, name)
        device = self.base.text_embedding.weight.device
        text, codes = torch.as_tensor(item("text_ids"), device=device), torch.as_tensor(item("codes"), device=device)
        for value, name, maximum, reserved, capacity in (
            (text, "text", self.base.max_text_tokens, (self.base.start_text_token, self.base.stop_text_token), self.base.number_text_tokens),
            (codes, "codes", self.base.max_mel_tokens, (), self.base.start_mel_token)):
            if value.ndim == 2 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.ndim != 1 or not 1 <= value.numel() <= maximum or value.dtype not in (torch.int32, torch.int64):
                raise NativeModelError(name + " must be one unpadded integer sequence within model limits")
            if (value < 0).any() or (value >= capacity).any() or any((value == token).any() for token in reserved):
                raise NativeModelError(name + " includes reserved/out-of-range raw IDs; review text punctuation, do not silently strip")
        text, codes = text.reshape(1, -1).long(), codes.reshape(1, -1).long()
        if text.numel() + codes.numel() + 6 > self.base.gpt.config.n_positions:
            raise NativeModelError("Combined unpadded context exceeds actual GPT positional capacity")
        dtype = self.base.spk_emb_proj.weight.dtype
        speaker = torch.as_tensor(item("spk_emb"), device=device, dtype=dtype)
        emotion = torch.as_tensor(item("emo_vec"), device=device, dtype=dtype)
        if speaker.numel() != 192 or tuple(speaker.shape) not in ((192,), (1, 192)):
            raise NativeModelError("Prompt speaker embedding must have shape [192] or [1,192]")
        if emotion.numel() != self.base.model_dim or tuple(emotion.shape) not in ((self.base.model_dim,), (1, self.base.model_dim)):
            raise NativeModelError("Target projected emotion must have shape [model_dim] or [1,model_dim]")
        if not torch.isfinite(speaker).all() or not torch.isfinite(emotion).all():
            raise NativeModelError("Speaker/emotion embeddings must be finite")
        return text, codes, speaker.reshape(1,192), emotion.reshape(1,-1)

    def training_inputs(self, batch):
        text, codes, speaker, emotion = self._batch(batch)
        projected = self.base.spk_emb_proj(speaker).unsqueeze(1)
        condition = torch.cat((projected + emotion.unsqueeze(1), projected.new_zeros((1, 2, self.base.model_dim))), dim=1)
        language = text.new_tensor([self.hindi_row])
        # Delegate the exact text framing/position/language convention to the
        # official public method, then preserve its full [START,text,STOP] context.
        _, prefix, attention = self.base.prepare_gpt_inputs(condition, text, language)
        if not torch.all(attention == 1):
            raise NativeModelError("Only unpadded single-example prefixes are supported")
        text_embeddings = prefix[:, condition.shape[1]:]
        mel_input = torch.cat((codes.new_full((1,1), self.base.start_mel_token), codes), dim=1)
        mel_embeddings = self.base.mel_embedding(mel_input) + self.base.mel_pos_embedding(mel_input)
        text_targets = torch.cat((text, text.new_full((1,1), self.base.stop_text_token)), dim=1)
        code_targets = torch.cat((codes, codes.new_full((1,1), self.base.stop_mel_token)), dim=1)
        return {"condition": condition, "text_embeddings": text_embeddings, "mel_embeddings": mel_embeddings,
                "text_targets": text_targets, "code_targets": code_targets, "prefix": prefix,
                "mel_input": mel_input}

    def forward(self, batch):
        data = self.training_inputs(batch)
        text_logits, code_logits = self.base.get_logits(data["condition"], data["text_embeddings"], self.base.text_head,
                                                        data["mel_embeddings"], self.base.mel_head)
        text_logits = text_logits[:, :, :data["text_targets"].shape[1]]
        text_loss = F.cross_entropy(text_logits.float(), data["text_targets"])
        code_loss = F.cross_entropy(code_logits.float(), data["code_targets"])
        loss = code_loss + self.text_loss_weight * text_loss
        if not torch.isfinite(loss):
            raise NativeModelError("Nonfinite native teacher-forcing loss")
        return {"loss": loss, "text_loss": text_loss, "code_loss": code_loss,
                "text_logits": text_logits, "code_logits": code_logits,
                "text_targets": data["text_targets"], "code_targets": data["code_targets"]}

    loss = forward


def load_native_model(official_repo, model_dir, model_manifest, **kwargs):
    base, config, identity = load_verified_base(official_repo, model_dir, model_manifest)
    model = NativeHindiModel(base, **kwargs)
    model.native_identity, model.native_config = identity, config
    model.native_model_dir = str(Path(model_dir).resolve())
    model.native_model_manifest = str(Path(model_manifest).resolve())
    model.native_preserved_sha256 = preserved_state_digest(model)
    return model


def make_tiny_official_model(official_repo, *, seed=None, lora_dropout=0.0):
    """Actual small official GPT for CPU math/resume tests, never a pretrained TTS.

    Random weights and synthetic code/token IDs test integration and gradients;
    these tests do not establish Hindi audio quality or real-checkpoint parity.
    """
    if seed is not None:
        torch.manual_seed(seed)
    cls = import_official_model(official_repo)
    kwargs = dict(layers=1, model_dim=16, heads=2, max_text_tokens=24, max_mel_tokens=24,
                  max_conditioning_inputs=3, number_text_tokens=64, number_mel_codes=34,
                  start_text_token=0, stop_text_token=1, start_mel_token=32, stop_mel_token=33,
                  checkpointing=False, spk_cond_mode="campplus", use_accel=False,
                  emo_condition_module={"output_size":16,"linear_units":32,"attention_heads":2,
                                        "num_blocks":1,"input_layer":"linear","perceiver_mult":1})
    base = cls(**kwargs)
    model = NativeHindiModel(base, lora_rank=2, lora_alpha=4.0, lora_dropout=lora_dropout,
                             text_rows=tuple(range(2,16)))
    frozen = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            value = parameter.detach().cpu().contiguous()
            frozen.update(name.encode())
            frozen.update(str(tuple(value.shape)).encode())
            frozen.update(value.numpy().tobytes())
    model.native_identity = {"kind": "tiny_random_official_cpu_test", "official_commit": OFFICIAL_COMMIT,
                             "model_dim": 16, "number_text_tokens": 64, "number_mel_codes": 34,
                             "frozen_base_sha256": frozen.hexdigest()}
    model.native_config = {"version": "2.5", "gpt": kwargs, "semantic_codec": {"codebook_size": 32}}
    model.native_preserved_sha256 = preserved_state_digest(model)
    return model


def load_native_adapter(model, adapter_path, expected_sha256):
    """Apply a complete native adapter after verifying its explicit local digest.

    The caller constructs the same pinned, verified base first. Loading on CPU
    uses only the restricted Torch loader and never falls back to arbitrary pickle.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise NativeModelError("An independently recorded adapter SHA256 is required")
    path = Path(adapter_path).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 ** 3:
        raise NativeModelError("Native adapter must be a nonempty local file below 16 GiB")
    if file_sha256(path) != expected_sha256:
        raise NativeModelError("Native adapter SHA256 mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise NativeModelError("Restricted CPU adapter load failed; no unsafe fallback: " + str(exc)) from exc
    if file_sha256(path) != expected_sha256:
        raise NativeModelError("Native adapter changed during restricted loading")
    required = {"schema_version", "kind", "step", "base_identity", "adapter_spec", "trainable"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != 1 or payload["kind"] != "hi_h200_native_adapter":
        raise NativeModelError("Native adapter schema does not match this implementation")
    if type(payload["step"]) is not int or payload["step"] < 1:
        raise NativeModelError("Native adapter must record at least one optimizer update")
    if not hasattr(model, "native_identity") or payload["base_identity"] != model.native_identity:
        raise NativeModelError("Native adapter was trained with a different or unverified base identity")
    if payload["adapter_spec"] != model.adapter_spec():
        raise NativeModelError("Native adapter settings/row selection do not match the constructed model")
    model.load_trainable_state(payload["trainable"])
    return {"status": "native_adapter_loaded", "step": payload["step"], "sha256": expected_sha256,
            "synthesis_verified": False, "gpu_training_verified": False}


def preserved_state_digest(model):
    """Hash all frozen state and every unselected embedding row, in bounded chunks."""
    trainable = {name.removeprefix("base.") for name, parameter in model.named_parameters() if parameter.requires_grad}
    selected = {"lang_embedding.weight": {model.hindi_row}, "text_embedding.weight": set(model.text_rows)}
    digest = hashlib.sha256()
    def add_bytes(value):
        flat = value.detach().reshape(-1)
        for offset in range(0, flat.numel(), 1_000_000):
            chunk = flat[offset:offset+1_000_000].cpu().contiguous().view(torch.uint8)
            digest.update(chunk.numpy().tobytes())
    for name, value in sorted(model.base.state_dict().items()):
        if name in trainable and name not in selected:
            continue
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        if name in selected:
            allowed = selected[name]
            digest.update(json.dumps(sorted(allowed)).encode())
            for start in range(0, value.shape[0], 1024):
                stop = min(start + 1024, value.shape[0])
                keep = [row - start for row in range(start, stop) if row not in allowed]
                if keep:
                    add_bytes(value[start:stop].detach().cpu()[keep])
        else:
            add_bytes(value)
    return digest.hexdigest()
