"""Opt-in limited multilingual ASR diagnostics; never a full regression pass.

No training, model download, or ASR occurs in the default metadata-only mode.
Private reports contain transcripts; the separately built public summary does not.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re

from hi_h200.data import load_json
from . import asr_eval as base

LANGUAGES = ("hi", "zh", "en", "ja", "es", "ar")
MAX_CASES = 40
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
LIMITATIONS = [
    "Limited diagnostics only; no full regression or TTS quality acceptance.",
    "Reference roles, native-language identity, held-out provenance, and TTS prompt/seed/parameter parity require operator verification.",
    "Calibration is language-specific; Hindi reference audio does not calibrate another language. ASR availability does not prove reference accuracy.",
    "ASR mistakes and hallucinations are not necessarily TTS errors; native human listening and broader old-language regression remain required.",
    "Primary CER uses authored/test canonical text with NFC+whitespace codepoints including spaces; internal stock casing/tagger text never replaces it.",
    "The en/es casefold-only auxiliary view is fixed in advance and does not replace primary CER or remove diacritics.",
    "Old-language generation using a Hindi prompt is crosslingual stress comparison, not calibrated native-prompt regression. Prompt language is not inferred by this tool.",
    "Whitespace WER for zh/ja is not interpretable and is not reported as a degradation decision. No automatic quality threshold is applied.",
]


def decode_settings(language):
    if language not in LANGUAGES:
        raise base.ASRError("Unsupported diagnostic language")
    return {**base.SETTINGS, "language": language}


def metric_views(reference, hypothesis, language):
    decode_settings(language)
    values = {"strict": base.scores(reference, hypothesis),
              "punctuation_only": base.scores(reference, hypothesis, True)}
    if language in {"en", "es"}:
        values["caseless"] = base.scores(base.normalized(reference).casefold(), base.normalized(hypothesis).casefold())
    for value in values.values():
        value["wer_interpretable"] = language not in {"zh", "ja"}
        if not value["wer_interpretable"]:
            value["wer"] = None
    return values


def _rows(path):
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise base.ASRError("Use a bounded private JSON manifest")
    document = load_json(path.read_text(encoding="utf-8"))
    condition = None
    if isinstance(document, dict):
        rows = document.get("cases")
        if "variant" in document:
            condition = {"original": "baseline", "merged": "merged"}.get(document["variant"])
            if condition is None:
                raise base.ASRError("Unknown synthesis variant")
    else:
        rows = document
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CASES:
        raise base.ASRError("Manifest requires 1..40 case objects")
    return rows, condition


def load_cases(root, manifests):
    cases, seen, identities = [], set(), []
    for supplied in manifests:
        path = base.contained(root, supplied, exists=True)
        rows, fallback = _rows(path)
        identities.append({"path": str(path.relative_to(root)), "sha256": base.sha256(path)})
        for row in rows:
            if not isinstance(row, dict) or len(cases) >= MAX_CASES:
                raise base.ASRError("Expected case object; combined limit is 40")
            uid, language = row.get("id"), row.get("language")
            condition = row.get("condition", fallback)
            if not isinstance(uid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", uid):
                raise base.ASRError("Case id must be a bounded safe identifier")
            if language not in LANGUAGES or condition not in {"baseline", "merged", "reference"}:
                raise base.ASRError("Unknown language or condition")
            if fallback is not None and condition != fallback:
                raise base.ASRError("Case condition differs from batch variant")
            key = (uid, language, condition)
            if key in seen:
                raise base.ASRError("Duplicate case id/language/condition")
            seen.add(key)
            text = row.get("text")
            if not base.normalized(text) or not base.normalized(text, True):
                raise base.ASRError("Case reference text is empty under the fixed metric policy")
            if "text_sha256" in row and row["text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
                raise base.ASRError("Case text identity mismatch")
            if "frontend_verified" in row and type(row["frontend_verified"]) is not bool:
                raise base.ASRError("Frontend verification metadata must be a boolean")
            source_status = row.get("status", "generated")
            if source_status not in {"generated", "failed", "not_executed", "attempt_started"}:
                raise base.ASRError("Unknown source execution status")
            record = dict(id=uid, language=language, condition=condition, text=text,
                          source_status=source_status, status="preflight_ready")
            audio = row.get("audio")
            audio_path = None
            if audio is not None:
                if not isinstance(audio, str) or not audio or audio != audio.strip() or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", audio):
                    raise base.ASRError("Audio must be a project-relative or project-contained local path")
                audio_path = base.contained(root, audio)
                record["audio_path"] = str(audio_path.relative_to(root))
            if source_status != "generated":
                record.update(status="not_evaluated", gap="source_" + source_status)
            elif row.get("frontend_verified") is False or (fallback is not None and row.get("frontend_verified") is not True):
                record.update(status="not_evaluated", gap="source_frontend_unverified")
            elif audio_path is None or not audio_path.exists():
                record.update(status="not_evaluated", gap="audio_missing")
            else:
                try:
                    if audio_path.suffix.lower() != ".wav" or not audio_path.is_file() or audio_path.stat().st_size > 2 * 1024 * 1024:
                        raise base.ASRError("Invalid bounded WAV")
                    stats = base.wav_info(audio_path)
                    if stats["sample_rate"] not in (22050, 24000) or not 1 <= stats["seconds"] <= 20:
                        raise base.ASRError("WAV requires mono PCM16, 22050/24000Hz, 1..20s")
                    record["audio"] = stats
                except Exception:
                    record.update(status="not_evaluated", gap="audio_preflight_failed")
            cases.append(record)
    grouped = _groups(cases)
    for pair in grouped.values():
        if set(pair) == {"baseline", "merged"} and all(row["status"] == "preflight_ready" for row in pair.values()):
            if pair["baseline"]["text"] != pair["merged"]["text"]:
                raise base.ASRError("Paired baseline/merged text must match exactly")
    return cases, identities


def _groups(cases):
    groups = {}
    for row in cases:
        if row["condition"] in {"baseline", "merged"}:
            groups.setdefault((row["language"], row["id"]), {})[row["condition"]] = row
    return groups


def coverage(cases, *, executed):
    expected = "asr_computed" if executed else "preflight_ready"
    gaps, complete = [], {language: 0 for language in LANGUAGES}
    eligible = set()
    for row in cases:
        if row.get("gap"):
            gaps.append({"id": row["id"], "language": row["language"], "condition": row["condition"], "reason": row["gap"]})
    groups = _groups(cases)
    for (language, uid), pair in groups.items():
        for condition in ("baseline", "merged"):
            if condition not in pair:
                gaps.append(dict(id=uid, language=language, condition=condition, reason="missing_pair_condition"))
        if set(pair) == {"baseline", "merged"} and all(row["status"] == expected for row in pair.values()):
            complete[language] += 1
            eligible.add((language, uid))
    for language in LANGUAGES:
        if not any(lang == language for lang, _ in groups):
            gaps.append(dict(language=language, reason="no_paired_probes"))
    return gaps, complete, eligible


def aggregate(cases, eligible):
    result = {}
    for language in LANGUAGES:
        result[language] = {}
        for condition in ("reference", "baseline", "merged"):
            rows = [row for row in cases if row["language"] == language and row["condition"] == condition
                    and row["status"] == "asr_computed"
                    and (condition == "reference" or (language, row["id"]) in eligible)]
            if not rows:
                continue
            summary = {}
            for view in rows[0]["metrics"]:
                selected = [row["metrics"][view] for row in rows]
                ce, nc = sum(x["char_edits"] for x in selected), sum(x["reference_chars"] for x in selected)
                we, nw = sum(x["word_edits"] for x in selected), sum(x["reference_words"] for x in selected)
                summary[view] = dict(cases=len(rows), char_edits=ce, reference_chars=nc, cer=ce / nc,
                                     word_edits=we, reference_words=nw, wer=we / nw if language not in {"zh", "ja"} else None,
                                     wer_interpretable=language not in {"zh", "ja"})
            result[language][condition] = summary
    return result


def public_summary(cases, *, executed):
    gaps, pairs, eligible = coverage(cases, executed=executed)
    gap_counts = {language: dict(Counter(gap["reason"] for gap in gaps if gap["language"] == language)) for language in LANGUAGES}
    counts = {language: {condition: sum(row["language"] == language and row["condition"] == condition for row in cases)
                         for condition in ("reference", "baseline", "merged")} for language in LANGUAGES}
    return dict(status="limited_asr_diagnostics" if executed else "metadata_only", cases=len(cases),
                asr_computed_cases=sum(row["status"] == "asr_computed" for row in cases), counts=counts,
                complete_pairs_by_language=pairs, coverage_gap_counts=gap_counts,
                calibrated_reference_available={language: any(row["language"] == language and row["condition"] == "reference"
                    and row["status"] == "asr_computed" for row in cases) for language in LANGUAGES},
                metrics=aggregate(cases, eligible), primary_metric="strict.cer", metrics_scope="References and complete baseline/merged pairs only",
                asr_executed=executed, download=False, not_full_regression_passed=True, quality_gate_passed=False,
                manual_listening_completed=False, limitations=list(LIMITATIONS))


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(args, root, folder, cases, identities):
    output = base.contained(root, args.output)
    if output.exists() or output.is_symlink():
        raise base.ASRError("Result directory must be fresh")
    if args.device == "cuda" and (not args.allow_cuda or type(args.gpu_id) is not int or not 0 <= args.gpu_id <= 255):
        raise base.ASRError("CUDA requires explicit --allow-cuda and a single --gpu-id after resource review")
    runnable = [row for row in cases if row["status"] == "preflight_ready"]
    if runnable:
        base.package()
        model_path = base.verify_model(folder)
    output.mkdir(parents=True, mode=0o700)
    executed = False
    if runnable:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.device == "cuda" else ""
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        for key, child in (("NUMBA_CACHE_DIR", "numba"), ("TORCH_HOME", "torch"), ("TIKTOKEN_CACHE_DIR", "tiktoken")):
            os.environ[key] = str(output / "cache" / child)
        with base.deny_network():
            import torch
            import whisper
            base.runtime_guard(torch, whisper, args.device)
            torch.set_num_threads(2)
            torch.manual_seed(0)
            model = whisper.load_model(str(model_path), device="cuda:0" if args.device == "cuda" else "cpu", download_root=str(folder))
            base.verify_model(folder)
            executed = True
            for row in runnable:
                try:
                    path = base.contained(root, row["audio_path"], exists=True)
                    if base.sha256(path) != row["audio"]["file_sha256"]:
                        raise base.ASRError("WAV changed after preflight")
                    result = model.transcribe(base.waveform(path), **decode_settings(row["language"]))
                    row["hypothesis"] = result.get("text")
                    row["metrics"] = metric_views(row["text"], row["hypothesis"], row["language"])
                    row["status"] = "asr_computed"
                except Exception:
                    row.update(status="not_evaluated", gap="asr_or_audio_runtime_failed")
    summary = public_summary(cases, executed=executed)
    gaps, _, _ = coverage(cases, executed=executed)
    private = dict(schema_version=1, manifests=identities, cases=cases, coverage_gaps=gaps, summary=summary,
                   model_sha256=base.MODEL_SHA, model_bytes=base.MODEL_BYTES, model_verified=bool(runnable),
                   required_whisper_version=base.VERSION, license="MIT", source=base.SOURCE, license_url=base.LICENSE_URL,
                   device=args.device, seed=0, decode_settings={lang: decode_settings(lang) for lang in LANGUAGES},
                   runtime_versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "openai-whisper")} if runnable else {},
                   reference_role_verified_by_tool=False, tts_prompt_seed_parameter_parity_verified_by_tool=False)
    _write(output / "private-asr.json", private)
    _write(output / "public-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Repeat for original/merged batch results and private reference cases")
    parser.add_argument("--model-dir", type=Path, default=Path("assets-local/whisper-small"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--gpu-id", type=int)
    args = parser.parse_args(argv)
    try:
        root = base.workspace(args.project_root)
        folder = base.contained(root, args.model_dir)
        cases, identities = load_cases(root, args.input)
        if args.execute:
            if args.output is None:
                raise base.ASRError("Explicit execution requires a fresh --output directory")
            result = execute(args, root, folder, cases, identities)
        else:
            result = public_summary(cases, executed=False)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, base.ASRError) else "Diagnostic preflight/runtime failed; private paths and transport details suppressed"
        result = dict(status="blocked", reason=reason, error_type=type(exc).__name__, quality_gate_passed=False,
                      not_full_regression_passed=True, asr_executed=False, download=False)
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
