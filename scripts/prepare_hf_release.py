#!/usr/bin/env python3
"""Prepare a whitelist-only experimental best100 release; never upload or log in.

Default: local byte/hash metadata verification only, no writes or Torch imports.
Tests use small fixtures for safety checks, not a substitute for the real model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hi_h200.data import load_json
from hi_h200_native import asr_limited_eval as asr
from hi_h200_native import synth
from hi_h200_native.assets import MODEL_REPO, MODEL_REVISION, selected_assets

STEP = 100
GPT_SHA = "423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021"
TOKENIZER = "multilingual_zh_ja_yue_char_del.tiktoken"
PAYLOADS = {"gpt.pth", "config.yaml", TOKENIZER, "LICENSE", "LICENSE_ZH.txt", "MODEL_LICENSE",
            "NON_ENDORSEMENT.txt", "README.md", "DATA_ATTRIBUTION.md", "release.json", "evaluation.json"}
FILES = PAYLOADS | {"manifest.json", "SHA256SUMS"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
# Bound to official commit d9e41... Git blobs a41d45f... and 519ba44..., independently checked.
CODE_LICENSES = {"LICENSE": (10554, "53875532237e0a97b17a41721556ff57e93193fc566e48c3a0febe6e35bb5343"),
                 "LICENSE_ZH.txt": (7336, "366175ecf329274358801b8365922677c52f25e643dcf00f0bc15857b5b2777f")}
RELEASE_NOTE = "Hash-bound experimental merged GPT only; tensor contents are not reloaded by this preparation tool. See README and complete license terms."
NON_ENDORSEMENT = ("Any modifications made to the original model in this Derivative Work are not endorsed, "
    "warranted, or guaranteed by the original right-holder of the original model, and the "
    "original right-holder disclaims all liability related to this Derivative Work.\n")
ATTRIBUTION = """# Training data attribution and changes

AI4Bharat; Ashwin Sankar et al. (2024), IndicVoices-R: Unlocking a Massive
Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS.
https://arxiv.org/abs/2409.05356
https://huggingface.co/datasets/ai4bharat/indicvoices_r
Pinned revision: 5f4495c91d500742a58d1be2ab07d77f73c0acf8; Hindi only.
Data-card license: CC-BY-4.0 https://creativecommons.org/licenses/by/4.0/

Changes: selected Hindi clips; complete-utterance mono PCM16 resampling;
NFC/whitespace text normalization preserving marks/joiners; model fine-tuning.
No source audio, transcripts, speaker identifiers or data shards are distributed.
The source authors' consent statements are not independent verification of every
individual personality/privacy right or permission for arbitrary future cloning.
The model remains subject to the included bilibili model license; the original
preparation code's MIT license does not replace model or data obligations.
"""
GAP_REASONS = {"source_failed", "source_not_executed", "source_attempt_started", "source_frontend_unverified",
               "audio_missing", "audio_preflight_failed", "asr_or_audio_runtime_failed", "missing_pair_condition", "no_paired_probes"}


class ReleaseError(ValueError):
    pass


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def bounded_json(path, *, with_digest=False):
    if not path.is_file() or not 0 < path.stat().st_size <= 2 * 1024 * 1024:
        raise ReleaseError("Missing or oversized JSON")
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if not 0 < len(raw) <= 2 * 1024 * 1024:
        raise ReleaseError("JSON grew beyond its bound")
    value = load_json(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError("Expected JSON object")
    return (value, hashlib.sha256(raw).hexdigest()) if with_digest else value


def number(value, *, integer=False, positive=False):
    if (type(value) is not int if integer else type(value) not in (int, float)) or not math.isfinite(value) or value < (1 if positive else 0):
        raise ReleaseError("Invalid finite aggregate number")
    return value


def sanitize_evaluation(value):
    keys = {"status", "cases", "asr_computed_cases", "counts", "complete_pairs_by_language", "coverage_gap_counts",
            "calibrated_reference_available", "metrics", "primary_metric", "metrics_scope", "asr_executed", "download",
            "not_full_regression_passed", "quality_gate_passed", "manual_listening_completed", "limitations"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ReleaseError("Only the reviewed aggregate public ASR schema is accepted; private fields are forbidden")
    if (value["status"] != "limited_asr_diagnostics" or value["asr_executed"] is not True
            or value["download"] is not False or value["not_full_regression_passed"] is not True
            or value["quality_gate_passed"] is not False or value["manual_listening_completed"] is not False
            or value["primary_metric"] != "strict.cer" or value["metrics_scope"] != "References and complete baseline/merged pairs only"
            or value["limitations"] != list(asr.LIMITATIONS)):
        raise ReleaseError("ASR provenance/limitations differ from the reviewed public schema")
    cases = number(value["cases"], integer=True, positive=True)
    computed = number(value["asr_computed_cases"], integer=True)
    if not computed <= cases <= 40:
        raise ReleaseError("Invalid bounded ASR case counts")
    output = {key: value[key] for key in ("status", "cases", "asr_computed_cases", "primary_metric", "metrics_scope")}
    for key in ("counts", "complete_pairs_by_language", "coverage_gap_counts", "calibrated_reference_available", "metrics"):
        if not isinstance(value[key], dict) or set(value[key]) != set(asr.LANGUAGES):
            raise ReleaseError("Aggregate language inventory mismatch")
        output[key] = {}
    for lang in asr.LANGUAGES:
        counts = value["counts"][lang]
        if not isinstance(counts, dict) or set(counts) != {"reference", "baseline", "merged"}:
            raise ReleaseError("Aggregate condition count mismatch")
        output["counts"][lang] = {key: number(count, integer=True) for key, count in counts.items()}
        pairs = number(value["complete_pairs_by_language"][lang], integer=True)
        if pairs > min(counts["baseline"], counts["merged"]):
            raise ReleaseError("Impossible paired count")
        output["complete_pairs_by_language"][lang] = pairs
        gaps = value["coverage_gap_counts"][lang]
        if not isinstance(gaps, dict) or set(gaps) - GAP_REASONS:
            raise ReleaseError("Unknown/private coverage gap keys")
        output["coverage_gap_counts"][lang] = {key: number(count, integer=True) for key, count in gaps.items()}
        calibration = value["calibrated_reference_available"][lang]
        if type(calibration) is not bool or (calibration and counts["reference"] == 0):
            raise ReleaseError("Impossible language-specific reference calibration")
        output["calibrated_reference_available"][lang] = calibration
        conditions = value["metrics"][lang]
        if not isinstance(conditions, dict) or set(conditions) - {"reference", "baseline", "merged"}:
            raise ReleaseError("Unknown/private metric conditions")
        output["metrics"][lang] = {}
        for condition, views in conditions.items():
            expected = {"strict", "punctuation_only"} | ({"caseless"} if lang in {"en", "es"} else set())
            if not isinstance(views, dict) or set(views) != expected:
                raise ReleaseError("Metric views differ from the prespecified policy")
            output["metrics"][lang][condition] = {}
            for view, metric in views.items():
                if not isinstance(metric, dict) or set(metric) != {"cases", "char_edits", "reference_chars", "cer", "word_edits", "reference_words", "wer", "wer_interpretable"}:
                    raise ReleaseError("Private/unknown per-case metric fields are forbidden")
                clean = {key: number(metric[key], integer=True, positive=key in {"cases", "reference_chars", "reference_words"})
                         for key in ("cases", "char_edits", "reference_chars", "word_edits", "reference_words")}
                if clean["cases"] > (counts[condition] if condition == "reference" else pairs):
                    raise ReleaseError("Metric cases exceed the declared complete coverage")
                cer = number(metric["cer"])
                if not math.isclose(cer, clean["char_edits"] / clean["reference_chars"], rel_tol=1e-9, abs_tol=1e-12):
                    raise ReleaseError("CER does not match its public edit counts")
                interpretable = lang not in {"zh", "ja"}
                if metric["wer_interpretable"] is not interpretable:
                    raise ReleaseError("Whitespace WER interpretation mismatch")
                wer = metric["wer"]
                if (not interpretable and wer is not None) or (interpretable and not math.isclose(number(wer), clean["word_edits"] / clean["reference_words"], rel_tol=1e-9, abs_tol=1e-12)):
                    raise ReleaseError("WER does not match its declared interpretation/edit counts")
                output["metrics"][lang][condition][view] = dict(clean, cer=cer, wer=wer, wer_interpretable=interpretable)
    if sum(sum(row.values()) for row in output["counts"].values()) != cases:
        raise ReleaseError("ASR counts do not sum to the declared total")
    output.update(asr_executed=True, download=False, not_full_regression_passed=True, quality_gate_passed=False,
                  manual_listening_completed=False, limitations=list(asr.LIMITATIONS))
    return output


def local(work, value, *, exists=True, card=False):
    supplied = Path(value)
    if any(part in {".git", ".ssh", ".huggingface", "hf-home", ".hf-home"} for part in supplied.parts):
        raise ReleaseError("Authentication/Git internals are not release inputs or destinations")
    raw = supplied if supplied.is_absolute() else work / supplied
    if not exists and raw.is_symlink():
        raise ReleaseError("Release destination cannot be an existing symlink")
    path = raw.resolve(strict=exists)
    if any(part in {".git", ".ssh", ".huggingface", "hf-home", ".hf-home"} for part in path.parts):
        raise ReleaseError("Resolved release path enters authentication/Git internals")
    if not path.is_relative_to(work) and not (card and path.is_relative_to(ROOT / "docs")):
        raise ReleaseError("Release paths must stay in controlled WORK; only the reviewed repository model card may be external")
    return path


def public_text(path):
    if not path.is_file() or not 0 < path.stat().st_size <= 1024 * 1024:
        raise ReleaseError("Missing or oversized public text file")
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not 0 < len(raw) <= 1024 * 1024:
        raise ReleaseError("Public text grew beyond its bound")
    text = raw.decode("utf-8")
    if re.search(r"(?:(?:hf_|gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|\bsk-[A-Za-z0-9_-]{20,}|\bBearer\s+[A-Za-z0-9._~+/-]{10,}=*|https?://[^/\s:@]+:[^/\s@]+@|-----BEGIN .*PRIVATE KEY-----|/(?:Users|Volumes|root|home|workspace|tmp|app|mnt|content|var/tmp|run/user)/)", text):
        raise ReleaseError("Public text contains a credential-like value or private absolute path")
    return raw


def reviewed_card(path):
    data = public_text(path)
    parts = data.split(b"---\n", 2)
    if len(parts) != 3 or parts[0]:
        raise ReleaseError("Reviewed HF model card needs leading YAML frontmatter")
    import yaml
    node = yaml.compose(parts[1].decode("utf-8"), Loader=yaml.SafeLoader)
    if not isinstance(node, yaml.nodes.MappingNode) or any(not isinstance(key, yaml.nodes.ScalarNode) for key, _ in node.value):
        raise ReleaseError("Model card frontmatter must be a mapping")
    keys = [key.value for key, _ in node.value]
    if len(keys) != len(set(keys)) or "<<" in keys or keys.count("license") != 1:
        raise ReleaseError("Model card frontmatter cannot duplicate or merge license metadata")
    header = yaml.safe_load(parts[1].decode("utf-8"))
    if not isinstance(header, dict) or header.get("license") != "other":
        raise ReleaseError("Reviewed HF model card must retain top-level license: other")
    return data


def pinned_bytes(path, size, kind, digest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ReleaseError("Fixed official payload size/type mismatch")
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    actual = hashlib.sha256(data).hexdigest() if kind == "sha256" else hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if len(data) != size or actual != digest:
        raise ReleaseError("Fixed official payload hash mismatch")


def verify_fixed_payloads(root):
    assets = {asset.relative_path: asset for asset in selected_assets("synth")}
    for destination, name in (("config.yaml", "config.yaml"), (TOKENIZER, TOKENIZER), ("MODEL_LICENSE", "LICENSE")):
        asset = assets[name]
        pinned_bytes(root / destination, asset.size, asset.hash_kind, asset.digest)
    for name, (size, digest) in CODE_LICENSES.items():
        pinned_bytes(root / name, size, "sha256", digest)


def inspect(args):
    work = Path(args.work_root).resolve(strict=True)
    if not work.is_dir() or work == Path(work.anchor):
        raise ReleaseError("Use an existing dedicated controlled WORK directory")
    source = local(work, args.official_repo)
    assets = local(work, args.official_assets)
    manifest = local(work, args.asset_manifest)
    bundle = local(work, args.merged_bundle)
    card = local(work, args.model_card, card=True)
    evaluation = local(work, args.public_asr_summary)
    output = local(work, args.output, exists=False)
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output == work:
        raise ReleaseError("Release output must be fresh with an existing parent")
    if not manifest.is_relative_to(assets):
        raise ReleaseError("Official manifest must be inside the declared asset directory")
    for protected in (source, assets, bundle, card, evaluation):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ReleaseError("Release output must be isolated from every input")
    if output.is_relative_to(ROOT) and output.relative_to(ROOT).parts[0] in {"src", "scripts", "tests", "configs", "docs", "requirements"}:
        raise ReleaseError("Release output cannot be inside repository source directories")
    asset_record = bounded_json(manifest)
    allowed_assets = {item.relative_path for item in selected_assets("synth")} | {"asset-plan.json", "w2v-bert-2.0.manifest.json", "campplus.manifest.json", "bigvgan.manifest.json"}
    if set(asset_record) != {"repository", "snapshot_revision", "files"} or asset_record["repository"] != MODEL_REPO or asset_record["snapshot_revision"] != MODEL_REVISION or not isinstance(asset_record["files"], dict) or set(asset_record["files"]) - allowed_assets:
        raise ReleaseError("Official asset manifest contains an unknown snapshot or non-asset/private path")
    for name in set(asset_record["files"]) - {item.relative_path for item in selected_assets("synth")}:
        auxiliary = (assets / name).resolve(strict=True)
        if not auxiliary.is_relative_to(assets) or not auxiliary.is_file() or not 0 < auxiliary.stat().st_size <= 2 * 1024 * 1024:
            raise ReleaseError("Auxiliary official asset JSON escapes or exceeds its bound")
    candidate = bundle / "gpt.pth"
    if candidate.is_symlink() or not candidate.is_file() or not 0 < candidate.stat().st_size <= 4 * 1024 ** 3:
        raise ReleaseError("Candidate GPT must be a bounded real local file")
    synth.verify_source(source)
    _, identity = synth.verify_synth_assets(assets, manifest)
    merged = synth.verify_merged_bundle(bundle, assets, identity)
    if merged["gpt_sha256"] != GPT_SHA or type(merged["optimizer_updates"]) is not int or merged["optimizer_updates"] != STEP:
        raise ReleaseError("Only the independently recorded frozen best100 GPT is eligible")
    complete = bounded_json(bundle / "COMPLETE.json")
    files = complete.get("files", {})
    if files.get("gpt.pth") != GPT_SHA or synth.sha256(candidate) != GPT_SHA:
        raise ReleaseError("Candidate completion marker does not bind frozen best100")
    evaluation_record, evaluation_digest = bounded_json(evaluation, with_digest=True)
    public = sanitize_evaluation(evaluation_record)
    card_bytes = reviewed_card(card)
    operations = {"gpt.pth": (bundle / "gpt.pth", GPT_SHA),
                  "config.yaml": (assets / "config.yaml", identity["files"]["config.yaml"]),
                  TOKENIZER: (assets / TOKENIZER, identity["files"][TOKENIZER]),
                  "MODEL_LICENSE": (assets / "LICENSE", identity["files"]["LICENSE"]),
                  "README.md": (card, hashlib.sha256(card_bytes).hexdigest())}
    for name in ("LICENSE", "LICENSE_ZH.txt"):
        official = public_text(source / name)
        if public_text(bundle / name) != official:
            raise ReleaseError("Candidate code-license bytes differ from the fixed official checkout")
        operations[name] = (source / name, hashlib.sha256(official).hexdigest())
    if public_text(bundle / "NON_ENDORSEMENT.txt").decode() != NON_ENDORSEMENT:
        raise ReleaseError("Required non-endorsement notice differs from the reviewed literal")
    operations["NON_ENDORSEMENT.txt"] = (bundle / "NON_ENDORSEMENT.txt", hashlib.sha256(NON_ENDORSEMENT.encode()).hexdigest())
    release = dict(schema_version=1, kind="experimental_indextts25_hindi_best100", target_owner="fluxa0223",
                   checkpoint_step=STEP, gpt_sha256=GPT_SHA, official_source_commit=synth.OFFICIAL_COMMIT,
                   base_model=MODEL_REPO, base_revision=MODEL_REVISION, base_gpt_sha256=identity["files"]["gpt.pth"],
                   candidate_complete_sha256=merged["complete_sha256"], reviewed_model_card_sha256=operations["README.md"][1],
                   input_public_evaluation_sha256=evaluation_digest, model_license="bilibili Model Use License Agreement",
                   quality_gate_passed=False, native_hindi_listening_completed=False, full_old_language_regression_passed=False,
                   evaluation_source_binding_independently_verified_by_tool=False,
                   note=RELEASE_NOTE)
    generated = {"DATA_ATTRIBUTION.md": ATTRIBUTION.encode(), "release.json": encode(release), "evaluation.json": encode(public)}
    return output, operations, generated


def copy_verified(source, destination, digest, size):
    actual, count = hashlib.sha256(), 0
    with source.open("rb") as original, destination.open("xb") as copied:
        for block in iter(lambda: original.read(8 * 1024 * 1024), b""):
            count += len(block)
            if count > size:
                raise ReleaseError("Release source grew while copying")
            actual.update(block)
            copied.write(block)
        copied.flush()
        os.fsync(copied.fileno())
    if count != size or actual.hexdigest() != digest:
        raise ReleaseError("Release source changed or failed its trusted SHA256 during copy")


def materialize(output, operations, generated):
    if set(operations) & set(generated) or set(operations) | set(generated) != PAYLOADS:
        raise ReleaseError("Release payload must be exactly the literal whitelist")
    sizes = {name: path.stat().st_size for name, (path, _) in operations.items()}
    if shutil.disk_usage(output.parent).free < sum(sizes.values()) + sum(map(len, generated.values())) + 256 * 1024 * 1024:
        raise ReleaseError("Insufficient disk headroom for a new independent release copy")
    output.mkdir(mode=0o700, exist_ok=False)
    for name, (path, digest) in operations.items():
        copy_verified(path, output / name, digest, sizes[name])
    for name, content in generated.items():
        with (output / name).open("xb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
    inventory = {name: {"bytes": (output / name).stat().st_size, "sha256": synth.sha256(output / name)} for name in sorted(PAYLOADS)}
    with (output / "manifest.json").open("xb") as target:
        target.write(encode(dict(schema_version=1, kind="public_release_whitelist", files=inventory)))
        target.flush()
        os.fsync(target.fileno())
    sums = "".join(synth.sha256(output / name) + "  " + name + "\n" for name in sorted(PAYLOADS | {"manifest.json"}))
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as target:
        target.write(sums)
        target.flush()
        os.fsync(target.fileno())


def release_size_limits():
    assets = {asset.relative_path: asset for asset in selected_assets("synth")}
    fixed = {"config.yaml": assets["config.yaml"].size, TOKENIZER: assets[TOKENIZER].size,
             "MODEL_LICENSE": assets["LICENSE"].size, "DATA_ATTRIBUTION.md": len(ATTRIBUTION.encode()),
             "NON_ENDORSEMENT.txt": len(NON_ENDORSEMENT.encode()),
             "SHA256SUMS": sum(64 + 2 + len(name.encode()) + 1 for name in PAYLOADS | {"manifest.json"})}
    fixed.update({name: size for name, (size, _) in CODE_LICENSES.items()})
    maximum = {name: 2 * 1024 * 1024 for name in ("manifest.json", "release.json", "evaluation.json")}
    maximum.update({"gpt.pth": 4 * 1024 ** 3, "README.md": 1024 * 1024})
    return fixed, maximum


def verify_release_sizes(root):
    fixed, maximum = release_size_limits()
    for name in ["gpt.pth", *sorted(FILES - {"gpt.pth"})]:
        size = (root / name).stat().st_size
        if size <= 0 or (name in fixed and size != fixed[name]) or (name in maximum and size > maximum[name]):
            raise ReleaseError("Release file violates its fixed/bounded byte count")


def verify_release(root):
    if not root.is_dir() or {item.name for item in root.iterdir()} != FILES or any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ReleaseError("Release directory differs from the exact public whitelist")
    verify_release_sizes(root)
    verify_fixed_payloads(root)
    record = bounded_json(root / "manifest.json")
    if set(record) != {"schema_version", "kind", "files"} or record["schema_version"] != 1 or record["kind"] != "public_release_whitelist" or set(record["files"]) != PAYLOADS:
        raise ReleaseError("Invalid release inventory")
    for name, entry in record["files"].items():
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"} or type(entry["bytes"]) is not int or entry["bytes"] != (root / name).stat().st_size or not isinstance(entry["sha256"], str) or not SHA.fullmatch(entry["sha256"]) or synth.sha256(root / name) != entry["sha256"]:
            raise ReleaseError("Release payload size/hash mismatch")
    if record["files"]["gpt.pth"]["sha256"] != GPT_SHA:
        raise ReleaseError("Released GPT is not frozen best100")
    release = bounded_json(root / "release.json")
    release_keys = {"schema_version", "kind", "target_owner", "checkpoint_step", "gpt_sha256", "official_source_commit", "base_model", "base_revision", "base_gpt_sha256", "candidate_complete_sha256", "reviewed_model_card_sha256", "input_public_evaluation_sha256", "model_license", "quality_gate_passed", "native_hindi_listening_completed", "full_old_language_regression_passed", "evaluation_source_binding_independently_verified_by_tool", "note"}
    if set(release) != release_keys or release.get("schema_version") != 1 or release.get("kind") != "experimental_indextts25_hindi_best100" or release.get("target_owner") != "fluxa0223" or release.get("base_model") != MODEL_REPO or release.get("model_license") != "bilibili Model Use License Agreement" or release.get("note") != RELEASE_NOTE or any(release.get(key) is not False for key in ("quality_gate_passed", "native_hindi_listening_completed", "full_old_language_regression_passed", "evaluation_source_binding_independently_verified_by_tool")):
        raise ReleaseError("Unknown/private release metadata or unsupported scope")
    for key in ("base_gpt_sha256", "candidate_complete_sha256", "reviewed_model_card_sha256", "input_public_evaluation_sha256"):
        if not isinstance(release[key], str) or not SHA.fullmatch(release[key]):
            raise ReleaseError("Invalid sanitized release fingerprint")
    if release["reviewed_model_card_sha256"] != record["files"]["README.md"]["sha256"]:
        raise ReleaseError("Reviewed model card changed")
    if release.get("checkpoint_step") != STEP or release.get("gpt_sha256") != GPT_SHA or release.get("official_source_commit") != synth.OFFICIAL_COMMIT or release.get("base_revision") != MODEL_REVISION or release.get("quality_gate_passed") is not False:
        raise ReleaseError("Release identity/quality scope mismatch")
    sanitize_evaluation(bounded_json(root / "evaluation.json"))
    if (root / "DATA_ATTRIBUTION.md").read_bytes() != ATTRIBUTION.encode() or (root / "NON_ENDORSEMENT.txt").read_text() != NON_ENDORSEMENT:
        raise ReleaseError("Required public attribution/non-endorsement notice mismatch")
    reviewed_card(root / "README.md")
    sums = "".join(synth.sha256(root / name) + "  " + name + "\n" for name in sorted(PAYLOADS | {"manifest.json"}))
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != sums:
        raise ReleaseError("SHA256SUMS mismatch")
    return dict(status="verified_public_release", files=len(FILES), gpt_sha256=GPT_SHA, checkpoint_step=STEP,
                quality_gate_passed=False, uploaded=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, required=True)
    for name in ("official-repo", "official-assets", "asset-manifest", "merged-bundle", "model-card", "public-asr-summary", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--verify-release", type=Path)
    args = p.parse_args(argv)
    try:
        with synth.deny_network():
            if args.verify_release is not None:
                if args.prepare:
                    raise ReleaseError("Verification does not prepare or mutate a directory")
                work = args.work_root.resolve(strict=True)
                if not work.is_dir() or work == Path(work.anchor):
                    raise ReleaseError("Use an existing dedicated controlled WORK directory")
                result = verify_release(local(work, args.verify_release))
            else:
                if any(getattr(args, name) is None for name in ("official_repo", "official_assets", "asset_manifest", "merged_bundle", "model_card", "public_asr_summary", "output")):
                    raise ReleaseError("Preparation/plan requires all explicit local source paths")
                output, operations, generated = inspect(args)
                if args.prepare:
                    materialize(output, operations, generated)
                    result = verify_release(output)
                else:
                    result = dict(status="metadata_only", payload_files=len(PAYLOADS), payload_bytes=sum(path.stat().st_size for path, _ in operations.values()) + sum(map(len, generated.values())),
                                  checkpoint_step=STEP, gpt_sha256=GPT_SHA, output_created=False, torch_loaded=False, uploaded=False)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ReleaseError) else "Local release verification failed; private paths and exception payloads suppressed"
        print(json.dumps(dict(status="blocked", reason=reason, error_type=type(exc).__name__, uploaded=False)))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
