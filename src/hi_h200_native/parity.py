"""Explicit FP32 native-adapter versus stock merged-GPT teacher-forcing parity.

Default preflight hashes local inputs; no model is constructed or downloaded.
This checks one real cached-feature batch, not synthesis quality or old languages.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

RTOL = ATOL = 1e-4  # Fixed before seeing results; there is no tolerance CLI override.


def stock_forward(stock, batch, text_loss_weight):
    """Independently frame the real stock public APIs; never wrap it with LoRA."""
    import torch
    from torch.nn import functional as F
    device = stock.text_embedding.weight.device
    text, codes = (batch[key].to(device).reshape(1, -1).long() for key in ("text_ids", "codes"))
    speaker = stock.spk_emb_proj(batch["spk_emb"].to(device).reshape(1, 192)).unsqueeze(1)
    condition = torch.cat((speaker + batch["emo_vec"].to(device).reshape(1, 1, stock.model_dim),
                           speaker.new_zeros((1, 2, stock.model_dim))), dim=1)
    _, prefix, attention = stock.prepare_gpt_inputs(condition, text, text.new_tensor([17]))
    if not torch.all(attention == 1):
        raise ValueError("Parity requires an unpadded single feature batch")
    mel = torch.cat((codes.new_full((1, 1), stock.start_mel_token), codes), dim=1)
    text_logits, code_logits = stock.get_logits(condition, prefix[:, 3:], stock.text_head,
        stock.mel_embedding(mel) + stock.mel_pos_embedding(mel), stock.mel_head)
    text_targets = torch.cat((text, text.new_full((1, 1), stock.stop_text_token)), dim=1)
    code_targets = torch.cat((codes, codes.new_full((1, 1), stock.stop_mel_token)), dim=1)
    text_logits = text_logits[:, :, :text_targets.shape[1]]
    def ce(logits, targets):
        return F.cross_entropy(logits.transpose(1, 2).reshape(-1, logits.shape[1]), targets.reshape(-1))
    text_loss, code_loss = ce(text_logits, text_targets), ce(code_logits, code_targets)
    return dict(text_logits=text_logits, code_logits=code_logits, text_targets=text_targets,
                code_targets=code_targets, text_loss=text_loss, code_loss=code_loss,
                loss=code_loss + text_loss_weight * text_loss)


def compare_outputs(native, stock):
    import torch
    metrics, passed = {}, True
    for name in ("text_logits", "code_logits", "text_loss", "code_loss", "loss"):
        actual, reference = native[name].detach().cpu(), stock[name].detach().cpu()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
        if actual.shape != reference.shape or not finite:
            raise ValueError("Nonfinite or shape-mismatched parity tensors: " + name)
        delta = (actual - reference).abs()
        close = torch.allclose(actual, reference, rtol=RTOL, atol=ATOL)
        metrics[name] = {"shape":list(actual.shape), "max_abs_error":delta.max().item(),
                         "max_relative_error":(delta / reference.abs().clamp_min(1e-8)).max().item(), "within_tolerance":close}
        if name.endswith("logits"):
            equal = torch.equal(actual.argmax(1), reference.argmax(1))
            metrics[name]["argmax_equal"] = equal
            passed &= equal
        else:
            metrics[name].update(native=actual.item(), stock=reference.item())
        passed &= close
    for name in ("text_targets", "code_targets"):
        equal = torch.equal(native[name].cpu(), stock[name].cpu())
        metrics[name] = {"exact_equal":equal}
        passed &= equal
    return {"passed":bool(passed), "rtol":RTOL, "atol":ATOL, "reference":"stock", "metrics":metrics}


def preflight(args):
    from .data import file_sha256, read_object
    from .model import verify_official_assets
    from .export import verify_export_bundle
    from .train import _dataset_paths, protect_output
    workspace = args.workspace_root.resolve(strict=True)
    dataset, adapter, bundle = (path.resolve(strict=True) for path in (args.dataset, args.adapter, args.merged_bundle))
    paths = _dataset_paths(dataset)
    if not dataset.is_relative_to(workspace) or any(not paths[key].is_relative_to(workspace) for key in paths if key not in ("model_dir", "model_manifest")):
        raise ValueError("Source/data must stay in WORK; only original model assets may use the authorized read-only share")
    if not paths["model_manifest"].is_relative_to(paths["model_dir"]):
        raise ValueError("Original model manifest must be within its model directory")
    if not adapter.is_relative_to(workspace) or not bundle.is_relative_to(workspace):
        raise ValueError("Adapter and merged bundle must stay in WORK")
    output = args.output.absolute()
    output = output.parent.resolve(strict=True) / output.name
    if output.exists() or output.is_symlink() or not output.is_relative_to(workspace):
        raise ValueError("Parity JSON must be a fresh path inside WORK")
    protect_output(output, [*paths.values(), dataset, adapter, bundle])
    config, identity = verify_official_assets(paths["official_repo"], paths["model_dir"], paths["model_manifest"])
    merged = verify_export_bundle(bundle)
    provenance = read_object(bundle / "provenance.json")
    if provenance.get("kind") != "hi_h200_native_merged_official_gpt" or identity.get("kind") == "tiny_random_official_cpu_test" or provenance.get("base_identity") != identity:
        raise ValueError("Production parity requires the exact full verified original base identity")
    if file_sha256(adapter) != args.adapter_sha256 or provenance.get("adapter_sha256") != args.adapter_sha256 or provenance.get("gpt_sha256") != merged["gpt_sha256"]:
        raise ValueError("Adapter/merged GPT SHA256 identity mismatch")
    return paths, config, identity, provenance, output, {"implementation_sha256":file_sha256(__file__), "status":"parity_preflight_passed", "models_loaded":False,
        "synthesis_quality_verified":False, "full_pretrained_forward_parity_verified":False,
        "dataset_config_sha256":file_sha256(dataset), "adapter_sha256":args.adapter_sha256,
        "merged_gpt_sha256":merged["gpt_sha256"], "base_identity":identity,
        "provenance_sha256":file_sha256(bundle / "provenance.json"), "rtol":RTOL, "atol":ATOL}


def execute(args, paths, config, identity, provenance, report):
    import torch
    from .data import load_native_dataset, file_sha256
    from .model import load_native_model, load_native_adapter, import_official_model
    if args.device != "cpu" and not args.allow_cuda:
        raise ValueError("CUDA parity requires explicit --allow-cuda and separately checked shared GPU headroom")
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda:0":
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one already-authorized CUDA device")
        torch.cuda.manual_seed_all(1234)
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    if file_sha256(args.adapter) != args.adapter_sha256:
        raise ValueError("Adapter changed during restricted load")
    spec = payload["adapter_spec"]
    canonical = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != provenance["adapter_spec_sha256"]:
        raise ValueError("Adapter specification differs from merged provenance")
    del payload
    dataset = load_native_dataset(args.dataset)
    if file_sha256(args.dataset) != report["dataset_config_sha256"]:
        raise ValueError("Dataset config changed during verification")
    if not 0 <= args.index < len(dataset):
        raise ValueError("Feature index outside verified dataset")
    batch = dataset[args.index]
    native = load_native_model(paths["official_repo"], paths["model_dir"], paths["model_manifest"],
        lora_rank=spec["lora"]["rank"], lora_alpha=spec["lora"]["alpha"], lora_dropout=spec["lora"]["dropout"],
        text_rows=spec["text_rows"], text_loss_weight=spec["text_loss_weight"])
    load_native_adapter(native, args.adapter, args.adapter_sha256)
    native.to(device=args.device, dtype=torch.float32).eval()
    with torch.inference_mode():
        expected = {key:value.detach().cpu() for key, value in native(batch).items()}
    del native
    gc.collect()
    if args.device != "cpu":
        torch.cuda.empty_cache()
    stock = import_official_model(paths["official_repo"])(**dict(config["gpt"], spk_cond_mode="campplus", use_accel=False))
    checkpoint = args.merged_bundle / "gpt.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(saved) != {"model"} or any("lora_" in key or ".base." in key for key in saved["model"]):
        raise ValueError("Merged checkpoint is not a complete stock-only model container")
    if file_sha256(checkpoint) != report["merged_gpt_sha256"]:
        raise ValueError("Merged GPT changed during restricted load")
    stock.load_state_dict(saved["model"], strict=True)
    del saved
    stock.to(device=args.device, dtype=torch.float32).eval()
    with torch.inference_mode():
        actual = stock_forward(stock, batch, spec["text_loss_weight"])
    comparison = compare_outputs(expected, actual)
    report.update(status="full_pretrained_single_batch_parity_passed" if comparison["passed"] else "parity_failed",
        models_loaded=True, device=args.device, dtype="float32", tf32=False, strict_deterministic=True, seed=1234,
        dataset_identity=dataset.identity, feature_index=args.index, comparison=comparison,
        scope="One verified cached feature batch; no synthesis or old-language regression claim",
        full_pretrained_forward_parity_verified=comparison["passed"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "adapter", "merged-bundle", "workspace-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.device != "cpu" and not args.allow_cuda:
        parser.error("CUDA execution requires explicit --allow-cuda")
    sys.dont_write_bytecode = True
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if not args.execute or args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        paths, config, identity, provenance, output, report = preflight(args)
        if args.execute:
            report = execute(args, paths, config, identity, provenance, report)
        from .train import write_manifest
        write_manifest(output, report)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(json.dumps({"status":"blocked_or_failed", "failure_type":type(exc).__name__, "reason":str(exc),
                          "full_pretrained_forward_parity_verified":False}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report.get("comparison", {"passed":True})["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
