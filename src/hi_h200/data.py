"""Auditable WAV/JSONL validation and deterministic speaker-disjoint splits."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import uuid
import unicodedata
import wave

from .text import DataError, normalize_hindi

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
SPLITS = ("train", "val", "test")
DEFAULT_SETTINGS = {"sample_rate": 24000, "min_seconds": 1.0, "max_seconds": 20.0,
                    "min_chars": 2, "max_chars": 400, "allow_as_spoken_numbers": False}


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError("duplicate JSON key")
        result[key] = value
    return result


def load_json(text):
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=lambda _: (_ for _ in ()).throw(DataError("nonfinite JSON value")))
    except (ValueError, TypeError, RecursionError) as error:
        raise DataError(f"invalid JSON: {error}") from error


def read_jsonl(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if len(line) > 1_000_000:
                    raise DataError(f"line {line_number}: exceeds 1 MB character limit")
                if not line.strip():
                    raise DataError(f"line {line_number}: blank lines are forbidden")
                row = load_json(line)
                if not isinstance(row, dict):
                    raise DataError(f"line {line_number}: expected JSON object")
                yield row
    except (OSError, UnicodeError) as error:
        raise DataError(f"cannot read UTF-8 JSONL: {error}") from error


def _identifier(value, field):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise DataError(f"{field} must be 1..128 safe identifier characters")
    return value


def _metadata(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise DataError(f"{field} must be a nonempty string of at most 512 characters")
    if any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value):
        raise DataError(f"{field} contains control characters")
    return value


def _settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise DataError("settings keys differ from schema version 1")
    for key in ("sample_rate", "min_chars", "max_chars"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise DataError(f"{key} must be a positive integer")
    for key in ("min_seconds", "max_seconds"):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise DataError(f"{key} must be positive and finite")
    if settings["min_seconds"] > settings["max_seconds"] or settings["max_seconds"] > 60:
        raise DataError("duration limits must be ordered and at most 60 seconds")
    if settings["min_chars"] > settings["max_chars"] or settings["max_chars"] > 10000:
        raise DataError("text limits must be ordered and max_chars at most 10000")
    if type(settings["allow_as_spoken_numbers"]) is not bool:
        raise DataError("allow_as_spoken_numbers must be boolean")
    return settings


def audio_info(relative, audio_root, settings):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise DataError("audio must be a relative POSIX file path, never a URL")
    if any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in relative):
        raise DataError("audio path contains control/format characters")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or relative != str(path):
        raise DataError("audio path must be canonical and stay inside audio-root")
    try:
        root = Path(audio_root).resolve(strict=True)
        target = (root / relative).resolve(strict=True)
        if not root.is_dir() or not target.is_relative_to(root) or not target.is_file():
            raise DataError("audio resolves outside audio-root or is not a file")
        if target.suffix.lower() != ".wav":
            raise DataError("only PCM WAV files are accepted; convert explicitly before preparation")
        if target.stat().st_size > 100_000_000:
            raise DataError("audio WAV exceeds 100 MB")
        with wave.open(str(target), "rb") as handle:
            rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
            if handle.getcomptype() != "NONE" or channels != 1 or width != 2:
                raise DataError("audio must be uncompressed mono signed 16-bit PCM WAV")
            if rate != settings["sample_rate"]:
                raise DataError(f"audio sample rate must equal {settings['sample_rate']}")
            frames = handle.getnframes()
            seconds = frames / rate
            if not settings["min_seconds"] <= seconds <= settings["max_seconds"]:
                raise DataError("audio duration is outside configured limits")
            digest, actual_bytes = hashlib.sha256(), 0
            while chunk := handle.readframes(65536):
                digest.update(chunk)
                actual_bytes += len(chunk)
            if actual_bytes != frames * channels * width:
                raise DataError("audio WAV is truncated")
        return {"duration_seconds": seconds, "sample_rate": rate, "num_frames": frames,
                "audio_sha256": digest.hexdigest()}
    except (OSError, wave.Error, EOFError, RuntimeError) as error:
        raise DataError(f"cannot inspect WAV: {error}") from error


def inspect_rows(rows, audio_root, settings, *, prepared=False):
    settings = _settings(settings)
    result, ids, audio_hashes, text_hashes = [], set(), {}, {}
    for index, row in enumerate(rows, 1):
        try:
            clip_id = _identifier(row.get("id"), "id")
            speaker = _identifier(row.get("speaker_id"), "speaker_id")
            if clip_id in ids:
                raise DataError("duplicate id")
            if row.get("language") != "hi":
                raise DataError("language must equal hi")
            if row.get("consent") is not True:
                raise DataError("consent=true is required; record rights before processing")
            if "numbers_as_spoken" in row and type(row["numbers_as_spoken"]) is not bool:
                raise DataError("numbers_as_spoken must be boolean")
            raw_text = row.get("text_raw" if prepared else "text")
            norm = normalize_hindi(raw_text, allow_as_spoken_numbers=settings["allow_as_spoken_numbers"],
                                   numbers_as_spoken=row.get("numbers_as_spoken", False),
                                   min_chars=settings["min_chars"], max_chars=settings["max_chars"])
            info = audio_info(row.get("audio"), audio_root, settings)
            text_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if info["audio_sha256"] in audio_hashes:
                raise DataError(f"duplicate PCM audio with id {audio_hashes[info['audio_sha256']]}")
            if text_hash in text_hashes:
                raise DataError(f"duplicate normalized text with id {text_hashes[text_hash]}")
            clean = {"id": clip_id, "speaker_id": speaker, "audio": row["audio"], "language": "hi",
                     "text_raw": raw_text, "text_norm": norm, "text_sha256": text_hash,
                     "source": _metadata(row.get("source"), "source"),
                     "license": _metadata(row.get("license"), "license"), "consent": True,
                     "numbers_as_spoken": row.get("numbers_as_spoken", False), **info}
            if prepared:
                for key, value in clean.items():
                    if row.get(key) != value:
                        raise DataError(f"prepared field {key} differs from recomputed value")
                clean.update({key: row.get(key) for key in ("split", "prompt_id", "prompt_audio")})
            result.append(clean)
            ids.add(clip_id)
            audio_hashes[info["audio_sha256"]] = clip_id
            text_hashes[text_hash] = clip_id
        except DataError as error:
            raise DataError(f"row {index}: {error}") from error
    if not result:
        raise DataError("dataset is empty")
    speakers = {}
    for row in result:
        speakers.setdefault(row["speaker_id"], []).append(row)
    if len(speakers) < 3:
        raise DataError("at least three speakers are required for nonempty train/val/test")
    if any(len(clips) < 2 for clips in speakers.values()):
        raise DataError("each speaker requires at least two distinct clips for prompt/target pairing")
    return result


def split_rows(rows, seed, val_ratio, test_ratio):
    for ratio in (val_ratio, test_ratio):
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio < 1:
            raise DataError("split ratios must be finite and between 0 and 1")
    if val_ratio + test_ratio >= 1:
        raise DataError("val_ratio + test_ratio must be less than 1")
    speakers = sorted({row["speaker_id"] for row in rows},
                      key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest())
    count = len(speakers)
    n_val, n_test = max(1, round(count * val_ratio)), max(1, round(count * test_ratio))
    if n_val + n_test >= count:
        raise DataError("requested split ratios leave no training speaker")
    assignment = {speaker: "val" if index < n_val else "test" if index < n_val + n_test else "train"
                  for index, speaker in enumerate(speakers)}
    groups = {split: [] for split in SPLITS}
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    for speaker, clips in sorted(by_speaker.items()):
        clips.sort(key=lambda row: row["id"])
        split = assignment[speaker]
        for index, row in enumerate(clips):
            prompt = clips[(index + 1) % len(clips)]
            groups[split].append({**row, "split": split, "prompt_id": prompt["id"],
                                  "prompt_audio": prompt["audio"]})
    return groups


def _summary(groups):
    return {split: {"clips": len(rows), "speakers": len({row["speaker_id"] for row in rows}),
                    "seconds": sum(row["duration_seconds"] for row in rows)}
            for split, rows in groups.items()}


def prepare(input_path, audio_root, output, *, seed=42, val_ratio=.1, test_ratio=.1, **options):
    settings = _settings({**DEFAULT_SETTINGS, **options})
    rows = inspect_rows(read_jsonl(input_path), audio_root, settings)
    groups = split_rows(rows, seed, val_ratio, test_ratio)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise DataError("output already exists; choose a fresh directory to prevent overwrites")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    manifest = {"schema_version": 1, "tool": "hi_h200 0.1.0", "settings": settings,
                "seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio,
                "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "counts": _summary(groups), "normalization": "Unicode NFC; whitespace collapse only",
                "pairing": "same-speaker next distinct clip sorted by id; reference stays in split",
                "deduplication": "global exact PCM SHA256 and normalized text SHA256; reject duplicates"}
    try:
        stage.mkdir()
        for split, clips in groups.items():
            with (stage / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
                for row in clips:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        for split, clips in groups.items():
            with (stage / f"backend_{split}.jsonl").open("x", encoding="utf-8") as handle:
                for row in clips:
                    backend_row = {"id": row["id"], "audio": str((Path(audio_root).resolve() / row["audio"]).resolve()),
                                   "text": row["text_norm"], "speaker": row["speaker_id"],
                                   "duration": row["duration_seconds"]}
                    handle.write(json.dumps(backend_row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise DataError("output appeared during preparation; refusing overwrite")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def validate(directory, audio_root):
    directory = Path(directory)
    try:
        manifest = load_json((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise DataError(f"cannot read manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DataError("manifest schema_version must equal 1")
    settings = _settings(manifest.get("settings", {}))
    groups, all_rows, assignment = {}, [], {}
    for split in SPLITS:
        groups[split] = list(read_jsonl(directory / f"{split}.jsonl"))
        if not groups[split]:
            raise DataError(f"{split} is empty")
        for row in groups[split]:
            if row.get("split") != split:
                raise DataError("row split differs from file name")
            speaker = _identifier(row.get("speaker_id"), "speaker_id")
            if speaker in assignment and assignment[speaker] != split:
                raise DataError("speaker leakage across splits")
            assignment[speaker] = split
        all_rows.extend(groups[split])
    checked = inspect_rows(all_rows, audio_root, settings, prepared=True)
    by_id = {row["id"]: row for row in checked}
    for row in checked:
        prompt = by_id.get(row.get("prompt_id"))
        if prompt is None or prompt["id"] == row["id"] or prompt["audio_sha256"] == row["audio_sha256"]:
            raise DataError("prompt must reference a different valid clip")
        if prompt["speaker_id"] != row["speaker_id"] or prompt["split"] != row["split"]:
            raise DataError("prompt speaker or split leakage")
        if row["prompt_audio"] != prompt["audio"]:
            raise DataError("prompt_audio differs from referenced clip")
    expected = split_rows(checked, manifest.get("seed"), manifest.get("val_ratio"), manifest.get("test_ratio"))
    for split in SPLITS:
        if groups[split] != expected[split]:
            raise DataError("split or pairing differs from deterministic manifest policy")
    for split in SPLITS:
        backend_expected = [{"id": row["id"],
                             "audio": str((Path(audio_root).resolve() / row["audio"]).resolve()),
                             "text": row["text_norm"], "speaker": row["speaker_id"],
                             "duration": row["duration_seconds"]} for row in groups[split]]
        if list(read_jsonl(directory / f"backend_{split}.jsonl")) != backend_expected:
            raise DataError("backend export differs from validated prepared rows")
    summary = _summary(groups)
    if manifest.get("counts") != summary:
        raise DataError("manifest counts differ from recomputed counts")
    return {"schema_version": 1, "status": "valid", "counts": summary,
            "checks": ["audio PCM hashes", "text NFC hashes", "speaker isolation", "distinct same-speaker prompts",
                       "path confinement", "durations", "consent metadata", "deterministic splits"]}


def build_pairs(feature_manifest, prepared, split, feature_root, output, audio_root, *,
                max_text_tokens, max_code_tokens, max_prompt_code_tokens):
    """Build one unique ordered pair per target without loading pickle/NumPy data.

    This verifies manifest metadata and file confinement. It does not certify NPZ
    contents; backend compatibility and full tokenizer checks are separate gates.
    """
    validate(prepared, audio_root)
    if split not in SPLITS:
        raise DataError("unknown split")
    for value in (max_text_tokens, max_code_tokens, max_prompt_code_tokens):
        if type(value) is not int or value <= 0:
            raise DataError("token limits must be positive integers")
    try:
        root = Path(feature_root).resolve(strict=True)
        feature_manifest = Path(feature_manifest).resolve(strict=True)
        if not root.is_dir() or not feature_manifest.is_relative_to(root):
            raise DataError("feature manifest must reside inside feature-root")
        destination = Path(output).resolve()
        if not destination.is_relative_to(root) or destination.exists():
            raise DataError("pair output must be a fresh file inside feature-root")
        expected = {row["id"]: row for row in read_jsonl(Path(prepared) / f"{split}.jsonl")}
        if not expected:
            raise DataError("prepared split is empty")
        records = {}
        for row in read_jsonl(feature_manifest):
            clip_id = _identifier(row.get("id"), "feature id")
            if clip_id not in expected or clip_id in records:
                raise DataError("unexpected or duplicate feature id")
            source = expected[clip_id]
            if row.get("speaker") != source["speaker_id"] or source.get("split") != split:
                raise DataError("feature speaker or prepared split mismatch")
            if type(row.get("lang")) is not int or row["lang"] != 17 or row.get("lang_alias") != "hi":
                raise DataError("feature language must be Hindi slot 17 / hi")
            for key, maximum in (("text_len", max_text_tokens), ("code_len", max_code_tokens)):
                if type(row.get(key)) is not int or not 0 < row[key] <= maximum:
                    raise DataError(f"feature {key} outside configured token limits")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise DataError("feature text must be nonempty")
            if row["text"] != source["text_norm"]:
                raise DataError("feature text differs from reviewed source text_norm; review backend punctuation changes and reprepare source")
            relative = row.get("features")
            if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
                raise DataError("features must name a local NPZ path")
            if any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in relative):
                raise DataError("feature path contains control/format characters")
            path = Path(relative)
            target = (path if path.is_absolute() else feature_manifest.parent / path).resolve(strict=True)
            if not target.is_relative_to(root) or not target.is_file() or target.suffix.lower() != ".npz":
                raise DataError("feature file must be NPZ confined inside feature-root")
            if not 0 < target.stat().st_size <= 100_000_000:
                raise DataError("feature file is empty or exceeds 100 MB")
            records[clip_id] = {**row, "resolved_features": target}
        if set(records) != set(expected):
            raise DataError("feature manifest did not process every prepared clip")
        groups = {}
        for row in records.values():
            groups.setdefault(row["speaker"], []).append(row)
        pairs = []
        for speaker, clips in sorted(groups.items()):
            clips.sort(key=lambda row: row["id"])
            for index, target in enumerate(clips):
                candidates = clips[index + 1:] + clips[:index]
                prompt = next((row for row in candidates
                               if row["code_len"] <= max_prompt_code_tokens
                               and row["resolved_features"] != target["resolved_features"]
                               and expected[row["id"]]["audio_sha256"] != expected[target["id"]]["audio_sha256"]), None)
                if prompt is None:
                    raise DataError("no distinct same-speaker prompt satisfies prompt token limit")
                pairs.append({"id": f"{target['id']}__prompt__{prompt['id']}",
                              "target_features": os.path.relpath(target["resolved_features"], destination.parent),
                              "prompt_features": os.path.relpath(prompt["resolved_features"], destination.parent),
                              "speaker": speaker, "lang": 17, "lang_alias": "hi", "prompt_lang": 17,
                              "cross_lingual": False, "text_len": target["text_len"], "code_len": target["code_len"]})
        if len({row["id"] for row in pairs}) != len(pairs):
            raise DataError("pair ID collision")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            for row in pairs:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    except (OSError, RuntimeError) as error:
        raise DataError(f"cannot build feature pairs: {error}") from error
    return {"status": "paired", "split": split, "targets": len(records), "pairs": len(pairs),
            "scope": "manifest metadata and path checks only; NPZ contents not loaded"}
