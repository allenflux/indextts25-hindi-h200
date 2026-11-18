"""Offline HI-only, unmerged pure-attention native LoRA inference.

Default: hash/metadata checks only, without Torch or CUDA. This standalone route
is not a ComfyUI LoRA node. Original-language requests are rejected before any
adapter access; run them through a separate untouched original-model instance.
Closing the private Hindi process disposes its adapter, rather than hot-restoring
shared weights. No model files are merged, overwritten, downloaded or installed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from . import synth
from .assets import MODEL_REPO, MODEL_REVISION

PROFILE = "attention_lora_only_v1"
FORMAT = "hi_h200_native_attention_lora_only_v1"
MAX_ADAPTER_BYTES = 512 * 1024**2


class AdapterSynthesisError(ValueError):
    pass


def require(value, message):
    if not value:
        raise AdapterSynthesisError(message)


def require_hindi(language):
    require(language == "hi", "This adapter route is HI-only; use a separate untouched original instance for other languages")


def checked_adapter(path, expected_sha256):
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
            "An independently recorded adapter SHA256 is required")
    path = Path(path).resolve(strict=True)
    require(path.is_file() and 0 < path.stat().st_size <= MAX_ADAPTER_BYTES,
            "Pure adapter must be a bounded local file, at most 512 MiB")
    require(synth.sha256(path) == expected_sha256, "Adapter SHA256 mismatch")
    return path


def training_base_identity(original, manifest_path, asset_identity):
    """Keep the exact training-manifest bytes; never substitute a T8 manifest."""
    original, manifest_path = Path(original).resolve(strict=True), Path(manifest_path).resolve(strict=True)
    require(manifest_path.is_relative_to(original), "Training base manifest must be inside original assets")
    manifest = synth._read(manifest_path)
    require(set(manifest) == {"repository", "snapshot_revision", "files"}
            and manifest["repository"] == MODEL_REPO and manifest["snapshot_revision"] == MODEL_REVISION,
            "Training base manifest must identify the fixed official snapshot")
    files = manifest["files"]
    require(isinstance(files, dict) and set(synth.CORE).issubset(files), "Training base manifest lacks core files")
    for name, digest in files.items():
        synth.safe_relative(name)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                and asset_identity["files"].get(name) == digest,
                "Training/synthesis manifest payload identity differs: " + name)
    require(asset_identity["repository"] == MODEL_REPO and asset_identity["snapshot_revision"] == MODEL_REVISION,
            "Synthesis snapshot differs from training base")
    return {"official_commit": synth.OFFICIAL_COMMIT, "model_repository": MODEL_REPO,
            "model_snapshot_revision": MODEL_REVISION, "model_manifest_sha256": synth.sha256(manifest_path),
            "model_files": dict(files), "config_sha256": files["config.yaml"],
            "tokenizer_sha256": files[synth.CORE[2]]}


def pure_parameters(spec, config):
    keys = {"format", "training_profile", "hindi_row", "text_rows", "text_loss_weight", "lora",
            "lora_targets", "heads", "model_dim", "frozen_embeddings"}
    require(isinstance(spec, dict) and set(spec) == keys and spec["format"] == FORMAT
            and spec["training_profile"] == PROFILE and type(spec["hindi_row"]) is int
            and spec["hindi_row"] == 17 and spec["text_rows"] == [] and spec["heads"] == []
            and spec["frozen_embeddings"] is True, "Only the complete pure-attention profile is accepted")
    gpt = config["gpt"]
    require(type(spec["model_dim"]) is int and spec["model_dim"] == gpt["model_dim"], "Adapter model dimension differs")
    targets = [f"gpt.h.{index}.attn.{name}" for index in range(gpt["layers"]) for name in ("c_attn", "c_proj")]
    require(spec["lora_targets"] == targets, "Adapter attention target inventory differs")
    lora = spec["lora"]
    require(isinstance(lora, dict) and set(lora) == {"rank", "alpha", "dropout"}, "Invalid LoRA construction settings")
    require(type(lora["rank"]) is int and 1 <= lora["rank"] <= min(64, gpt["model_dim"]), "LoRA rank exceeds the inference bound")
    for name, lower, upper in (("alpha", 0, 4096), ("dropout", 0, 1)):
        value = lora[name]
        require(type(value) in (int, float) and math.isfinite(value)
                and (lower < value <= upper if name == "alpha" else lower <= value < upper),
                "Invalid bounded LoRA " + name)
    weight = spec["text_loss_weight"]
    require(type(weight) in (int, float) and math.isfinite(weight) and 0 <= weight <= 100,
            "Invalid recorded auxiliary loss weight")
    return dict(lora_rank=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                hindi_row=17, text_rows=(), text_loss_weight=weight, training_profile=PROFILE)


def read_pure_payload(path, expected_sha256, expected_identity, config):
    """Execution-only restricted CPU read; validation precedes model mutation."""
    path = checked_adapter(path, expected_sha256)
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(synth.sha256(path) == expected_sha256, "Adapter changed during restricted CPU loading")
    require(isinstance(payload, dict) and set(payload) == {"schema_version", "kind", "step", "base_identity", "adapter_spec", "trainable"}
            and type(payload["schema_version"]) is int and payload["schema_version"] == 1
            and payload["kind"] == "hi_h200_native_adapter" and type(payload["step"]) is int and payload["step"] >= 1,
            "Unsupported or incomplete native adapter schema")
    require(payload["base_identity"] == expected_identity, "Adapter exact original base identity differs")
    parameters = pure_parameters(payload["adapter_spec"], config)
    return payload, parameters


def load_pure_into(model, adapter, expected_sha256):
    """Strict attach to a fresh CPU FP32 native model, used by the real runtime."""
    from .model import load_native_adapter, preserved_state_digest
    require(model.training_profile == PROFILE, "A fresh pure-attention native model is required")
    require(not hasattr(model.base, "inference_model"), "Attach before creating the official inference wrapper")
    require(all(parameter.device.type == "cpu" and str(parameter.dtype) == "torch.float32"
                for parameter in model.parameters()), "Attach requires CPU FP32 original parameters")
    payload, _ = read_pure_payload(adapter, expected_sha256, model.native_identity, model.native_config)
    require(payload["adapter_spec"] == model.adapter_spec(), "Adapter specification differs from the private model")
    del payload
    result = load_native_adapter(model, adapter, expected_sha256)
    require(preserved_state_digest(model) == model.native_preserved_sha256, "Adapter changed a frozen original tensor")
    return result


def attach_private_gpt(tts, native, *, device, dtype):
    """Replace only this private object's GPT, then build its real stock wrapper."""
    from .model import Conv1DLoRA, preserved_state_digest
    require(native.training_profile == PROFILE and not hasattr(native.base, "inference_model"),
            "Expected fresh unmerged pure native GPT")
    require(preserved_state_digest(native) == native.native_preserved_sha256, "Private original state changed")
    old = tts.gpt
    require(old is not native.base, "Original and Hindi GPT objects must be independent")
    old_storage = {value.untyped_storage().data_ptr() for value in old.parameters()}
    require(not any(value.untyped_storage().data_ptr() in old_storage for value in native.parameters()),
            "Original and Hindi GPT tensors must not share storage")
    # Release this process's redundant original GPT before moving the new one.
    # No object from the original-language service is accepted or modified here.
    tts.gpt = None
    del old
    gc.collect()
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    native.eval()
    native.base.to(device=device, dtype=dtype)
    native.base.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=dtype == torch.bfloat16)
    wrapper = native.base.inference_model
    require(wrapper.transformer is native.base.gpt and wrapper.lm_head[1] is native.base.mel_head,
            "Official inference wrapper contains stale model references")
    require(all(isinstance(dict(native.base.named_modules())[name], Conv1DLoRA) for name in native.lora_targets),
            "Unmerged attention modules disappeared")
    tts.gpt = native.base
    tts._hi_pure_adapter_owner = native
    return {"unmerged": True, "training_profile": PROFILE, "lora_modules": len(native.lora_targets),
            "official_inference_wrapper_rebuilt": True, "kv_cache": wrapper.kv_cache,
            "original_language_instance_touched": False, "hot_unload_supported": False}


def metadata_preflight(args):
    require_hindi(args.language)  # Must precede even looking at adapter paths.
    workspace = args.workspace_root.resolve(strict=True)
    source = synth.verify_source(args.official_repo)
    original = args.model_dir.resolve(strict=True)
    adapter = checked_adapter(args.adapter, args.adapter_sha256)
    manifest = args.model_manifest.resolve(strict=True)
    training_manifest = args.training_base_manifest.resolve(strict=True)
    prompt = args.prompt.resolve(strict=True)
    require(all(path.is_relative_to(workspace) for path in (source, original, adapter, manifest, training_manifest, prompt)),
            "All source, assets, manifests, adapter and prompt must stay in WORK")
    require(manifest.is_relative_to(original), "Synthesis manifest must be inside original assets")
    require(not adapter.is_relative_to(original) and not adapter.is_relative_to(source), "Adapter must be independent from original assets/source")
    output = synth._independent_output(args.output_dir, workspace, (source, original, adapter, prompt))
    require(output is not None, "A fresh output directory is required")
    config, assets = synth.verify_synth_assets(original, manifest)
    identity = training_base_identity(original, training_manifest, assets)
    text = synth.reviewed_hindi_text(source, args.text)
    prompt_info = synth.wav_info(prompt, prompt=True)
    dependencies = synth.dependency_report(workspace, source)
    report = {"status": "unmerged_adapter_metadata_verified_tensor_validation_pending", "inference_route": "hi_only_unmerged_pure_attention_lora",
        "official_commit": synth.OFFICIAL_COMMIT, "asset_identity": assets, "training_base_identity": identity,
        "adapter_sha256": args.adapter_sha256, "adapter_bytes": adapter.stat().st_size,
        "adapter_tensor_validation": False, "adapter_profile_verified": False, "variant": "original",
        "language": "hi", "seed": args.seed, "constructor_flags": dict(synth.DISABLED_FLAGS, use_bf16=True),
        "input": {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "raw_text_sha256": hashlib.sha256(args.text.encode()).hexdigest(), "prompt": prompt_info},
        "dependencies": dependencies, "models_loaded": False, "cuda_accessed": False, "synthesis_executed": False,
        "quality_verified": False, "comfyui_adapter_supported": False, "hot_unload_supported": False,
        "close_policy": "Dispose the private Hindi process; original-language process never loads this adapter",
        "memory_note": "The original TTS constructor and an additional CPU FP32 GPT are temporarily present; admission is observation, not reservation"}
    state = dict(workspace=workspace, source=source, original=original, output=output, config=config, text=text, prompt=prompt,
                 adapter=adapter, training_manifest=training_manifest)
    return report, state


def execute_once(args, report, state):
    require_hindi(args.language)
    def setup(tts, runtime_state, runtime_report):
        from .model import load_native_model
        payload, parameters = read_pure_payload(state["adapter"], args.adapter_sha256,
                                               report["training_base_identity"], state["config"])
        spec_sha = hashlib.sha256(json.dumps(payload["adapter_spec"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        del payload
        native = load_native_model(state["source"], state["original"], state["training_manifest"], **parameters)
        require(native.native_identity == report["training_base_identity"], "Current training base identity changed")
        loaded = load_pure_into(native, state["adapter"], args.adapter_sha256)
        current = synth.resource_admission(args)
        require(current[0]["uuid"] == runtime_report["admission_gpu_uuid"], "GPU changed before adapter attachment")
        import torch
        attached = attach_private_gpt(tts, native, device="cuda:0", dtype=torch.bfloat16)
        runtime_report.update(adapter_tensor_validation=True, adapter_profile_verified=True, adapter_spec_sha256=spec_sha,
                              adapter_step=loaded["step"], adapter_loaded_without_merge=True)
        return attached
    # Reuse all original resource/offline/input/seed/output gates. The original
    # GPT path remains truthful; separate adapter identity is recorded above.
    from types import SimpleNamespace
    selected_args = SimpleNamespace(**vars(args), variant="original", no_kv_cache=False)
    return synth.execute_once(selected_args, report, state, model_setup=setup)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace-root", "official-repo", "model-dir", "model-manifest", "training-base-manifest", "adapter", "prompt", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--language", default="hi", choices=("hi",))
    parser.add_argument("--text", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--max-mel-tokens", type=int, default=256)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--min-free-gpu-mib", type=float)
    parser.add_argument("--min-free-ram-mib", type=float)
    parser.add_argument("--max-utilization", type=float)
    args = parser.parse_args(argv)
    if args.execute and not args.allow_cuda:
        parser.error("--execute requires --allow-cuda and explicit resource thresholds")
    if not 0 <= args.seed < 2**32 or not 8 <= args.max_text_tokens <= 120 or not 8 <= args.max_mel_tokens <= 512:
        parser.error("Use uint32 seed and bounded text/code generation limits")
    try:
        report, state = metadata_preflight(args)
        if args.execute:
            report = execute_once(args, report, state)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return 2 if report["dependencies"]["missing"] else 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "blocked_not_generated", "reason": str(exc), "synthesis_executed": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
