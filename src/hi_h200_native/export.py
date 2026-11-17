"""Export a complete official-format GPT replacement from the original native route.

The output is a fresh local bundle. It never edits the base checkpoint, source,
ComfyUI configuration, or a pre-existing output. Both trained heads are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys

import torch

from hi_h200.data import load_json

from .model import (Conv1DLoRA, NativeHindiModel, NativeModelError, OFFICIAL_COMMIT,
                    file_sha256, import_official_model, load_native_adapter,
                    load_native_model, preserved_state_digest, verify_official_assets,
                    verify_official_source)

NON_ENDORSEMENT = (
    "Any modifications made to the original model in this Derivative Work are not endorsed, "
    "warranted, or guaranteed by the original right-holder of the original model, and the "
    "original right-holder disclaims all liability related to this Derivative Work.\n"
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _check_destination(path, protected):
    destination = Path(path).resolve()
    if destination.exists() or not destination.parent.is_dir():
        raise NativeModelError("Export output must be absent with an explicitly created parent directory")
    for value in protected:
        root = Path(value).resolve()
        if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
            raise NativeModelError("Export must be isolated from the original model/source/protected directory")
    return destination


def merge_official_state(model):
    """Produce all stock GPT tensors without changing the in-memory native model."""
    if not isinstance(model, NativeHindiModel) or not hasattr(model, "native_preserved_sha256"):
        raise NativeModelError("Export requires a verified native model with an initial preserved-state digest")
    if preserved_state_digest(model) != model.native_preserved_sha256:
        raise NativeModelError("Frozen state or an unselected language/text embedding row changed")
    modules = {name: module for name, module in model.base.named_modules() if isinstance(module, Conv1DLoRA)}
    if set(modules) != set(model.lora_targets) or not modules:
        raise NativeModelError("Native LoRA target inventory changed or is incomplete")
    state = {}
    for name, value in model.base.state_dict().items():
        if not torch.is_tensor(value) or value.is_complex():
            raise NativeModelError("Official state contains an unsupported non-tensor/complex value")
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise NativeModelError("Export requires finite FP32 master weights; do not cast the trained model before export")
        state[name] = value.detach().cpu().clone()
    for name, module in modules.items():
        spec = model.lora_config
        if module.lora_A.shape[0] != spec["rank"] or module.lora_B.shape[1] != spec["rank"] or module.scale != spec["alpha"] / spec["rank"] or module.dropout.p != spec["dropout"]:
            raise NativeModelError("Actual LoRA rank/scaling/dropout differs from the recorded adapter settings")
        keys = [name + suffix for suffix in (".base.weight", ".base.bias", ".lora_A", ".lora_B")]
        if not all(key in state for key in keys):
            raise NativeModelError("Missing complete Conv1D/LoRA state for " + name)
        weight, bias, left, right = [state.pop(key) for key in keys]
        # HF Conv1D stores [input, output]; native A is [rank,input], B [output,rank].
        merged = weight + (left.T @ right.T) * module.scale
        if not torch.isfinite(merged).all():
            raise NativeModelError("Nonfinite merged Conv1D weight")
        state[name + ".weight"] = merged
        state[name + ".bias"] = bias
    if any(".lora_A" in name or ".lora_B" in name or any(name.startswith(target + ".base.") for target in modules) for name in state):
        raise NativeModelError("Native wrapper names remain in the stock state")
    return state


def _official_schema(model, official_repo, state):
    if not hasattr(model, "native_config") or str(model.native_config.get("version")) != "2.5":
        raise NativeModelError("Original verified 2.5 constructor configuration is missing")
    kwargs = dict(model.native_config["gpt"])
    kwargs.update(spk_cond_mode="campplus", use_accel=False)
    cls = import_official_model(official_repo)
    # The real official class supplies the expected schema without allocating a
    # second full pretrained model. Actual physical reload/parity is a separate gate.
    with torch.device("meta"):
        expected = cls(**kwargs)
    template = expected.state_dict()
    if set(template) != set(state):
        raise NativeModelError("Merged state is not complete stock official GPT state; missing/extra keys")
    for name, tensor in state.items():
        if tensor.shape != template[name].shape or tensor.dtype != template[name].dtype:
            raise NativeModelError("Merged stock tensor shape/dtype mismatch: " + name)
    expected.load_state_dict(state, strict=True, assign=True)
    return len(template)


def _exclusive_bytes(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def verify_export_bundle(directory):
    root = Path(directory).resolve(strict=True)
    complete = root / "COMPLETE.json"
    if not complete.resolve().is_relative_to(root) or not complete.is_file() or complete.stat().st_size > 1024 * 1024:
        raise NativeModelError("Export bundle is incomplete; COMPLETE.json is required")
    record = load_json(complete.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or set(record) != {"schema_version", "files"} or record["schema_version"] != 1 or not isinstance(record["files"], dict):
        raise NativeModelError("Invalid export completion inventory")
    required = {"gpt.pth", "LICENSE", "LICENSE_ZH.txt", "NON_ENDORSEMENT.txt", "README.md", "provenance.json"}
    if not required.issubset(record["files"]) or set(record["files"]) - required - {"MODEL_LICENSE"}:
        raise NativeModelError("Export inventory is missing required files or includes unsupported content")
    if {item.name for item in root.iterdir()} != set(record["files"]) | {"COMPLETE.json"}:
        raise NativeModelError("Unexpected or partial files in export bundle")
    for name, digest in record["files"].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or file_sha256(path) != digest:
            raise NativeModelError("Export file changed or escaped its bundle: " + name)
    return {"status": "complete_native_gpt_bundle", "gpt_sha256": record["files"]["gpt.pth"],
            "files": len(record["files"]), "full_pretrained_forward_parity_verified": False}


def export_merged_gpt(model, official_repo, output_dir, *, adapter_path, adapter_sha256, protected_roots=()):
    """Merge a verified adapter into all official GPT tensors in a fresh bundle.

    Call only after the run's data/license/resource gates. This verifies stock keys
    and shapes; it does not establish speech quality, old-language regression or
    full pretrained forward/synthesis parity.
    """
    if not isinstance(model, NativeHindiModel) or not hasattr(model, "native_identity"):
        raise NativeModelError("A verified native model is required for merged export")
    source = verify_official_source(official_repo)
    protected = [source, *protected_roots]
    if hasattr(model, "native_model_dir"):
        protected.append(model.native_model_dir)
    destination = _check_destination(output_dir, protected)
    if model.native_identity.get("official_commit") != OFFICIAL_COMMIT:
        raise NativeModelError("Native base source identity does not match the export source")
    tiny = model.native_identity.get("kind") == "tiny_random_official_cpu_test"
    if not tiny:
        if not all(hasattr(model, field) for field in ("native_model_dir", "native_model_manifest")):
            raise NativeModelError("Export requires the verified original model directory/manifest")
        config, identity = verify_official_assets(source, model.native_model_dir, model.native_model_manifest)
        if identity != model.native_identity or config != model.native_config:
            raise NativeModelError("Original model/source/config identity changed before export")
    loaded = load_native_adapter(model, adapter_path, adapter_sha256)
    state = merge_official_state(model)
    tensors = _official_schema(model, source, state)
    # Reserve an absent directory exclusively. A killed export never receives its
    # final COMPLETE marker; even gpt.pth is published only after its full fsync.
    destination.mkdir(mode=0o700, exist_ok=False)
    for name in ("LICENSE", "LICENSE_ZH.txt"):
        _exclusive_bytes(destination / name, (source / name).read_bytes())
    _exclusive_bytes(destination / "NON_ENDORSEMENT.txt", NON_ENDORSEMENT.encode())
    model_files = model.native_identity.get("model_files", {})
    if not tiny and "LICENSE" in model_files:
        _exclusive_bytes(destination / "MODEL_LICENSE", (Path(model.native_model_dir) / "LICENSE").read_bytes())
    scope = "tiny random official CPU test, not a pretrained Hindi model" if tiny else "Hindi-intended merged official GPT replacement; acceptance still required"
    readme = ("# 原创 native GPT 替换件\n\n" + scope + "\n\n"
              "gpt.pth 使用官方加载器支持的 {model: state_dict} 容器；没有optimizer、语料、提示音频或训练清单。"
              "它只替换GPT，其他固定版本的官方资产仍需另备。原始权重和现有ComfyUI目录未被覆盖。\n\n"
              "共享attention与两个head已合并；全局替换可能影响旧语言。Hindi使用隔离实例，其他语言继续使用原始模型。"
              "发布前完成真实预训练模型重载/前向、语音合成和旧语言回归；本导出器不声称这些验收已通过。\n\n"
              "只有COMPLETE.json存在且全部哈希验证成功才是完整导出。中断留下的目录不可作为完成产物。\n")
    _exclusive_bytes(destination / "README.md", readme.encode())
    partial = destination / ".gpt.pth.partial"
    with partial.open("xb") as stream:
        torch.save({"model": state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, destination / "gpt.pth")  # Atomic and exclusive, never replaces an existing file.
    partial.unlink()
    provenance = {"schema_version": 1, "kind": "hi_h200_native_merged_official_gpt",
                  "scope": scope, "official_commit": OFFICIAL_COMMIT,
                  "base_identity": model.native_identity, "adapter_sha256": adapter_sha256,
                  "native_implementation_files": {name: file_sha256(Path(__file__).with_name(name)) for name in ("model.py", "export.py")},
                  "adapter_spec_sha256": hashlib.sha256(_canonical(model.adapter_spec())).hexdigest(),
                  "optimizer_updates": loaded["step"], "gpt_sha256": file_sha256(destination / "gpt.pth"),
                  "preserved_base_sha256": model.native_preserved_sha256,
                  "merge_semantics": "evaluation/inference with native LoRA dropout disabled",
                  "official_tensor_count": tensors, "checkpoint_container": "model",
                  "official_key_shape_check": "actual official meta constructor, strict=True",
                  "full_pretrained_forward_parity_verified": False, "synthesis_verified": False,
                  "old_language_behavior_guaranteed_unchanged": False}
    _exclusive_bytes(destination / "provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
    inventory = {item.name: file_sha256(item) for item in destination.iterdir() if item.is_file()}
    marker = json.dumps({"schema_version": 1, "files": inventory}, sort_keys=True, indent=2).encode() + b"\n"
    marker_partial = destination / ".COMPLETE.partial"
    _exclusive_bytes(marker_partial, marker)
    os.link(marker_partial, destination / "COMPLETE.json")
    marker_partial.unlink()
    result = verify_export_bundle(destination)
    result.update(output_dir=str(destination), gpt_checkpoint=str(destination / "gpt.pth"),
                  provenance=str(destination / "provenance.json"), optimizer_updates=loaded["step"])
    return result


def export_from_adapter(official_repo, model_dir, model_manifest, adapter_path, adapter_sha256, output_dir, *, protected_roots=()):
    """Standalone CPU route; no trainer/optimizer is required to reconstruct export."""
    _check_destination(output_dir, [official_repo, model_dir, *protected_roots])
    path = Path(adapter_path).resolve(strict=True)
    if not isinstance(adapter_sha256, str) or not re.fullmatch("[0-9a-f]{64}", adapter_sha256) or not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 ** 3:
        raise NativeModelError("A bounded local adapter and independently recorded SHA256 are required")
    if file_sha256(path) != adapter_sha256:
        raise NativeModelError("Adapter digest mismatch before reading its construction settings")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise NativeModelError("Restricted CPU adapter read failed; no unsafe fallback: " + str(exc)) from exc
    if file_sha256(path) != adapter_sha256 or not isinstance(payload, dict) or payload.get("kind") != "hi_h200_native_adapter":
        raise NativeModelError("Adapter changed or is not a native adapter")
    spec = payload.get("adapter_spec")
    if not isinstance(spec, dict) or not isinstance(spec.get("lora"), dict):
        raise NativeModelError("Native adapter construction settings are missing")
    parameters = {"lora_rank": spec["lora"]["rank"], "lora_alpha": spec["lora"]["alpha"],
                  "lora_dropout": spec["lora"]["dropout"], "hindi_row": spec["hindi_row"],
                  "text_rows": spec["text_rows"], "text_loss_weight": spec["text_loss_weight"]}
    del payload
    model = load_native_model(official_repo, model_dir, model_manifest, **parameters)
    return export_merged_gpt(model, official_repo, output_dir, adapter_path=path,
                             adapter_sha256=adapter_sha256, protected_roots=protected_roots)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-repo", "model-dir", "model-manifest", "adapter", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--protect-root", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    try:
        result = export_from_adapter(args.official_repo, args.model_dir, args.model_manifest,
                                     args.adapter, args.adapter_sha256, args.output, protected_roots=args.protect_root)
    except (NativeModelError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
