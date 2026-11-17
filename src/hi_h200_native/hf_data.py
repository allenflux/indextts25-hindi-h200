"""Bounded, opt-in IndicVoices-R Hindi import using the operator's existing HF login.

Default mode checks metadata/access only. No token argument, login, terms
acceptance, arbitrary repo, test shard, or whole-subset download is supported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import unicodedata
import wave

from hi_h200.data import load_json, prepare, validate
from hi_h200.text import DataError, normalize_hindi

REPOSITORY = "ai4bharat/indicvoices_r"
REVISION = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"
EXPECTED_ACCOUNT = "fluxa0223"
ENDPOINT = "https://huggingface.co"
SHARDS = (("Hindi/train-00000-of-00099.parquet", 479633889),
          ("Hindi/train-00001-of-00099.parquet", 467329797))
HINDI_FILES = 101
HINDI_BYTES = 47225888374
MAX_BYTES = 1000000000
MAX_ROWS = 4096
MAX_CANDIDATE_SPEAKERS = 32
MAX_CANDIDATES_PER_SPEAKER = 16
MAX_AUDIO_BYTES = 16 * 1024 * 1024
TARGET_RATE = 24000
ATTRIBUTION = "AI4Bharat IndicVoices-R, Sankar et al. (2024), CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ImportBlocked(ValueError):
    """A bounded import gate failed. Messages must never contain auth responses."""


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _integer(value, label, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ImportBlocked("Invalid bounded integer: " + label)
    return value


def duration_cap(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 20:
        raise ImportBlocked("max-duration-seconds must be finite and within 1..20")
    return float(value)


def rights_record(path):
    if path is None:
        raise ImportBlocked("--download requires an operator-created --rights-record; this tool never accepts terms")
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 65536:
        raise ImportBlocked("Missing/oversized operator rights record")
    try:
        record = load_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, DataError) as exc:
        raise ImportBlocked("Invalid operator rights record") from exc
    required = {"dataset_repo", "revision", "terms_accepted", "license_reviewed", "permission_basis"}
    if not isinstance(record, dict) or set(record) != required:
        raise ImportBlocked("Rights record requires exactly dataset_repo,revision,terms_accepted,license_reviewed,permission_basis")
    if (record["dataset_repo"] != REPOSITORY or record["revision"] != REVISION
            or record["terms_accepted"] is not True or record["license_reviewed"] != "CC-BY-4.0"):
        raise ImportBlocked("Operator must record prior acceptance and review for this exact dataset/revision/license")
    basis = record["permission_basis"]
    if not isinstance(basis, str) or not 12 <= len(basis.strip()) <= 2048 or any(unicodedata.category(c).startswith("C") for c in basis):
        raise ImportBlocked("Rights record needs a meaningful plain-text permission basis")
    return {"sha256": digest_file(path), "scope": "Operator declaration of accepted dataset conditions and use review; not independent proof of individual voice-cloning consent."}


@dataclass(frozen=True)
class FilePin:
    filename: str
    size: int
    sha256: str | None
    git_blob_sha1: str | None = None

    def record(self):
        return {"filename": self.filename, "size": self.size, "sha256": self.sha256, "git_blob_sha1": self.git_blob_sha1}


def auth_environment(*, require_home=False):
    """Inspect override NAMES only; never read/print their secret values."""
    if any(name in os.environ for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")):
        raise ImportBlocked("Ambient token overrides are not allowed; use only the existing project HF_HOME credential store")
    if require_home:
        home = os.environ.get("HF_HOME")
        if not home or not Path(home).is_absolute() or not Path(home).is_dir():
            raise ImportBlocked("Set HF_HOME to the existing project credential-store directory; this tool does not create credentials")


def metadata_plan(api, *, max_bytes=MAX_BYTES, max_shards=2):
    """Use only the existing credential store. Never return account/token payloads."""
    auth_environment()
    _integer(max_bytes, "max_bytes", maximum=MAX_BYTES)
    _integer(max_shards, "max_shards", maximum=2)
    try:
        account = api.whoami(token=None)
        if not isinstance(account, dict) or account.get("name") != EXPECTED_ACCOUNT:
            raise ImportBlocked("Existing HF login must match the expected operator account")
        api.auth_check(REPOSITORY, repo_type="dataset", token=None)
        info = api.dataset_info(REPOSITORY, revision=REVISION, files_metadata=True, token=None)
    except ImportBlocked:
        raise
    except Exception as exc:
        raise ImportBlocked("Existing HF authentication/read access check failed; complete login/conditions in the browser, never send a token to this tool") from exc
    if _field(info, "sha") != REVISION:
        raise ImportBlocked("Dataset metadata did not resolve the exact pinned revision")
    card = _field(info, "card_data", {})
    license_value = _field(card, "license")
    if license_value not in ("cc-by-4.0", ["cc-by-4.0"]):
        raise ImportBlocked("Actual dataset card license differs from the reviewed CC-BY-4.0 declaration")
    siblings = _field(info, "siblings", [])
    if not isinstance(siblings, (list, tuple)):
        raise ImportBlocked("Missing dataset file metadata")
    indexed = {}
    for item in siblings:
        name, size = _field(item, "rfilename"), _field(item, "size")
        if not isinstance(name, str) or name in indexed or type(size) is not int or size < 0:
            raise ImportBlocked("Invalid or duplicate dataset file metadata")
        lfs = _field(item, "lfs")
        sha = _field(lfs, "sha256") if lfs is not None else None
        if lfs is not None and (_field(lfs, "size") != size or not isinstance(sha, str) or not SHA256.fullmatch(sha)):
            raise ImportBlocked("Invalid dataset LFS size/hash metadata")
        blob = _field(item, "blob_id")
        if sha is None and (not isinstance(blob, str) or re.fullmatch(r"[0-9a-f]{40}", blob) is None):
            raise ImportBlocked("Missing non-LFS Git blob identity")
        indexed[name] = FilePin(name, size, sha, blob if sha is None else None)
    hindi = [pin for name, pin in indexed.items() if name.startswith("Hindi/") and name.endswith(".parquet")]
    if len(hindi) != HINDI_FILES or sum(pin.size for pin in hindi) != HINDI_BYTES:
        raise ImportBlocked("Pinned Hindi subset inventory differs from the independently reviewed 101-file/byte total")
    selected = []
    for name, expected_size in SHARDS[:max_shards]:
        pin = indexed.get(name)
        if pin is None or pin.size != expected_size or pin.sha256 is None:
            raise ImportBlocked("Allowed Hindi train shard size/hash metadata is missing or changed")
        selected.append(pin)
    documents = []
    for name, pin in indexed.items():
        if name == "README.md" or ("/" not in name and name.upper().startswith("LICENSE")):
            if pin.size > 1024 * 1024:
                raise ImportBlocked("Dataset license/card document exceeds 1 MiB bound")
            documents.append(pin)
    if not any(pin.filename == "README.md" for pin in documents):
        raise ImportBlocked("Pinned dataset card is missing")
    total = sum(pin.size for pin in selected) + sum(pin.size for pin in documents)
    if total > max_bytes:
        raise ImportBlocked("Planned bounded shards plus audit documents exceed --max-bytes; no files downloaded")
    report = {"schema_version": 1, "kind": "ivr_hindi_import_plan", "repository": REPOSITORY,
              "revision": REVISION, "verified_account": EXPECTED_ACCOUNT,
              "card_license": "CC-BY-4.0", "attribution": ATTRIBUTION,
              "hindi_subset_files": HINDI_FILES, "hindi_subset_bytes": HINDI_BYTES,
              "max_bytes": max_bytes, "planned_bytes": total,
              "shards": [pin.record() for pin in selected], "documents": [pin.record() for pin in documents],
              "corpus_bytes_downloaded": 0, "official_test_used": False,
              "note": "Only the two explicitly allowed train shards can be transferred; this is not a whole-Hindi download."}
    return report, tuple(selected), tuple(documents)


def reuse_directory(value):
    if value is None:
        return None
    supplied = Path(value)
    if supplied.name != "downloads" or supplied.is_symlink():
        raise ImportBlocked("--reuse-downloads must name the prior import's existing downloads directory")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ImportBlocked("--reuse-downloads is not an existing directory")
    return root


def _copy_verified_reuse(pin, output, source_root):
    """Inspect only an explicitly allowed shard path; never scan the old folder."""
    if source_root is None or pin.filename not in {name for name, _ in SHARDS}:
        return None
    source_root = reuse_directory(source_root)
    candidate = source_root / pin.filename
    if not candidate.exists() and not candidate.is_symlink():
        return None
    source = candidate.resolve(strict=True)
    if not source.is_relative_to(source_root) or not source.is_file() or source.stat().st_size != pin.size:
        raise ImportBlocked("Existing shard escapes the reuse directory or has an incorrect size")
    if not isinstance(pin.sha256, str) or not SHA256.fullmatch(pin.sha256):
        raise ImportBlocked("Existing shard reuse requires current pinned Hub SHA256 metadata")
    destination = output / "downloads" / pin.filename
    if destination.exists() or destination.is_symlink():
        raise ImportBlocked("Refusing to overwrite a new import's download file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".reuse-part")
    digest, count, created = hashlib.sha256(), 0, False
    try:
        with source.open("rb") as original, partial.open("xb") as copied:
            created = True
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                count += len(chunk)
                if count > pin.size:
                    raise ImportBlocked("Existing shard grew while re-verifying")
                digest.update(chunk)
                copied.write(chunk)
            copied.flush()
            os.fsync(copied.fileno())
        if count != pin.size or digest.hexdigest() != pin.sha256:
            raise ImportBlocked("Existing shard full SHA256 does not match current pinned Hub metadata")
        # Same-directory link is atomic and refuses an existing destination.
        os.link(partial, destination)
    finally:
        if created:
            partial.unlink(missing_ok=True)
    return destination, digest.hexdigest(), True


def download_pin(pin, output, download, *, reuse_downloads=None):
    """Authenticate only through the existing HF store; all new cache stays local."""
    auth_environment()
    output = Path(output).resolve()
    if shutil.disk_usage(output).free < pin.size + 256 * 1024 * 1024:
        raise ImportBlocked("Insufficient disk headroom for the next bounded file")
    reused = _copy_verified_reuse(pin, output, reuse_downloads)
    if reused is not None:
        return reused
    try:
        result = download(repo_id=REPOSITORY, filename=pin.filename, repo_type="dataset", revision=REVISION,
                          token=None, endpoint=ENDPOINT, cache_dir=str(output / ".cache" / "hf"),
                          local_dir=str(output / "downloads"), local_dir_use_symlinks=False)
    except Exception as exc:
        raise ImportBlocked("Pinned dataset file transfer failed; auth details and download URLs are suppressed") from exc
    path = Path(result).resolve(strict=True)
    download_root = (output / "downloads").resolve()
    if not path.is_relative_to(download_root) or not path.is_file() or path.stat().st_size != pin.size:
        raise ImportBlocked("Downloaded dataset file escapes its root or has the wrong size")
    actual = digest_file(path)
    if pin.sha256 is not None and actual != pin.sha256:
        raise ImportBlocked("Downloaded dataset LFS SHA256 mismatch")
    if pin.git_blob_sha1 is not None:
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != pin.git_blob_sha1:
            raise ImportBlocked("Downloaded dataset audit document Git blob mismatch")
    return path, actual, False


def _speaker(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ImportBlocked("Publisher speaker_id must be a nonempty string/integer, never inferred")
    if type(value) is int and value < 0:
        raise ImportBlocked("Unknown/negative publisher speaker_id")
    value = str(value)
    if not value.strip() or value != value.strip():
        raise ImportBlocked("Publisher speaker_id must not be blank or carry surrounding whitespace")
    if value.casefold() in {"unknown", "unk", "none", "null", "nan", "-1"}:
        raise ImportBlocked("Unknown publisher speaker_id")
    if not value or len(value) > 256 or any(unicodedata.category(c).startswith("C") for c in value):
        raise ImportBlocked("Invalid publisher speaker_id")
    return value


def _metadata_rows(path, start_limit):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    if not {"speaker_id", "duration", "audio"}.issubset(columns) or not {"normalized", "text", "verbatim"} & columns:
        raise ImportBlocked("Actual Parquet schema lacks publisher speaker_id/duration/audio/text fields")
    projected = [name for name in ("speaker_id", "duration", "normalized", "text", "verbatim", "lang", "id") if name in columns]
    index = 0
    for batch in parquet.iter_batches(columns=projected, batch_size=128, use_threads=False):
        for row in batch.to_pylist():
            if index >= start_limit:
                return
            yield index, row
            index += 1


def _audio_cells(path, indices):
    """Read the binary audio column incrementally; decode waveform only when selected."""
    import pyarrow.parquet as pq
    if not indices:
        return
    wanted = set(indices)
    last = max(wanted)
    index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["audio"], batch_size=1, use_threads=False):
        if index in wanted:
            yield index, batch.column(0)[0].as_py()
        if index >= last:
            return
        index += 1


def decode_audio(cell, *, max_duration_seconds=6):
    """Decode a selected embedded clip and resample its complete waveform, never trim."""
    cap = duration_cap(max_duration_seconds)
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    if not isinstance(cell, dict) or not isinstance(cell.get("bytes"), bytes):
        raise ImportBlocked("Selected audio must be embedded bytes; external audio paths are never followed")
    payload = cell["bytes"]
    if not 0 < len(payload) <= MAX_AUDIO_BYTES:
        raise ImportBlocked("Selected embedded audio exceeds the 16 MiB bound")
    try:
        info = sf.info(io.BytesIO(payload))
        if not 8000 <= info.samplerate <= 192000 or not 1 <= info.channels <= 2 or not 1 <= info.frames / info.samplerate <= cap:
            raise ImportBlocked(f"Actual clip must be complete mono/stereo audio of 1..{cap:g} seconds")
        data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except ImportBlocked:
        raise
    except Exception as exc:
        raise ImportBlocked("Selected embedded audio cannot be decoded safely") from exc
    if len(data) != info.frames or not np.isfinite(data).all():
        raise ImportBlocked("Nonfinite or incomplete decoded audio")
    mono = data.mean(axis=1, dtype="float64")
    factor = math.gcd(rate, TARGET_RATE)
    samples = resample_poly(mono, TARGET_RATE // factor, rate // factor) if rate != TARGET_RATE else mono
    if not np.isfinite(samples).all() or not 1 <= len(samples) / TARGET_RATE <= cap:
        raise ImportBlocked("Resampled audio has invalid length/values")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples * samples)))
    if peak > 1.0001 or rms < 1e-5:
        raise ImportBlocked("Selected audio is silent or would clip during PCM conversion")
    quantized_clip_count = int(np.count_nonzero((samples < -1.0) | (samples > 1.0 - 1.0 / 32768)))
    samples = np.clip(samples, -1.0, 1.0 - 1.0 / 32768)
    out = io.BytesIO()
    sf.write(out, samples, TARGET_RATE, format="WAV", subtype="PCM_16")
    wav = out.getvalue()
    with wave.open(io.BytesIO(wav), "rb") as stream:
        pcm = stream.readframes(stream.getnframes())
        frames = stream.getnframes()
    original_path = cell.get("path")
    if not isinstance(original_path, str) or len(original_path) > 2048 or any(unicodedata.category(c).startswith("C") for c in original_path):
        original_path = None
    return wav, {"source_audio_path": original_path, "source_audio_sha256": hashlib.sha256(payload).hexdigest(),
                 "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                 "source_sample_rate": rate, "source_channels": info.channels,
                 "source_frames": info.frames, "source_peak_after_downmix_resample": peak,
                 "pcm_quantization_clip_samples": quantized_clip_count, "target_sample_rate": TARGET_RATE,
                 "target_frames": frames, "duration_seconds": frames / TARGET_RATE,
                 "conversion": "whole utterance; arithmetic stereo mean if needed; scipy.resample_poly; PCM_16; no target truncation or gain normalization"}


def import_parquets(shards, output, *, speakers=6, clips_per_speaker=4, max_rows=MAX_ROWS, max_duration_seconds=6):
    """Return a genuine prepared dataset or fail; no guessed speakers/partial success."""
    cap = duration_cap(max_duration_seconds)
    _integer(speakers, "speakers", minimum=6, maximum=MAX_CANDIDATE_SPEAKERS)
    _integer(clips_per_speaker, "clips_per_speaker", minimum=4, maximum=MAX_CANDIDATES_PER_SPEAKER)
    _integer(max_rows, "max_rows", maximum=MAX_ROWS)
    output = Path(output)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=False)
    incomplete = output / "IMPORT_INCOMPLETE"
    incomplete.write_text("No prepared data may be used until import.json reports ready.\n", encoding="utf-8")
    candidates, pending, completed = {}, {}, {}
    selected_rows, pcm_seen, text_seen, examined, skipped = [], set(), set(), 0, Counter()
    seen_shards = set()
    for shard_name, path in shards:
        if len(seen_shards) >= 2 or shard_name in seen_shards or shard_name not in {name for name, _ in SHARDS}:
            raise ImportBlocked("Only the two reviewed Hindi TRAIN shards may supply this import")
        seen_shards.add(shard_name)
        selected = {}
        for index, row in _metadata_rows(path, max_rows - examined):
            examined += 1
            try:
                speaker = _speaker(row.get("speaker_id"))
                duration = row.get("duration")
                if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 1 <= duration <= cap:
                    raise ImportBlocked("duration")
                if "lang" in row and (not isinstance(row["lang"], str) or row["lang"].strip().casefold() not in {"hindi", "hi"}):
                    raise ImportBlocked("language")
                text_field = next((name for name in ("normalized", "text", "verbatim") if isinstance(row.get(name), str) and row[name].strip()), None)
                if text_field is None:
                    raise ImportBlocked("text")
                text = row[text_field]
                norm = normalize_hindi(text)
                if any(char in norm for char in ("!", chr(34))):
                    skipped["ascii_framing_punctuation"] += 1
                    continue
                if speaker not in candidates:
                    if len(candidates) >= MAX_CANDIDATE_SPEAKERS:
                        raise ImportBlocked("candidate speaker cap")
                    candidates[speaker] = 0
                if speaker in completed or candidates[speaker] >= MAX_CANDIDATES_PER_SPEAKER:
                    continue
                candidates[speaker] += 1
                selected[index] = {"speaker": speaker, "text": text, "norm": norm, "text_field": text_field,
                                   "source_text_fields": {name: row[name] for name in ("normalized", "text", "verbatim") if isinstance(row.get(name), str)},
                                   "shard": shard_name, "row_index": index,
                                   "published_id": str(row["id"]) if isinstance(row.get("id"), (str, int)) else None}
            except (ImportBlocked, DataError):
                skipped["metadata_policy"] += 1
        for index, cell in _audio_cells(path, selected):
            record = selected[index]
            speaker = record["speaker"]
            if speaker in completed:
                continue
            try:
                wav, info = decode_audio(cell, max_duration_seconds=cap)
            except ImportBlocked:
                skipped["audio_policy"] += 1
                continue
            group = pending.setdefault(speaker, [])
            if info["pcm_sha256"] in pcm_seen or record["norm"] in text_seen or any(info["pcm_sha256"] == x[2]["pcm_sha256"] or record["norm"] == x[0]["norm"] for x in group):
                skipped["duplicate"] += 1
                continue
            group.append((record, wav, info))
            if len(group) < clips_per_speaker:
                continue
            group[:] = [entry for entry in group if entry[0]["norm"] not in text_seen and entry[2]["pcm_sha256"] not in pcm_seen]
            if len(group) < clips_per_speaker:
                continue
            speaker_slug = "ivr_hi_spk_" + hashlib.sha256(speaker.encode()).hexdigest()[:20]
            for chosen, payload, inspected in group:
                identity = REPOSITORY + "@" + REVISION + ":" + chosen["shard"] + ":" + str(chosen["row_index"])
                clip_id = "ivr_hi_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
                relative = "audio/" + clip_id + ".wav"
                with (output / relative).open("xb") as stream:
                    stream.write(payload)
                selected_rows.append({"id": clip_id, "speaker_id": speaker_slug, "audio": relative,
                                      "text": chosen["text"], "language": "hi", "source": identity,
                                      "license": ATTRIBUTION, "consent": True,
                                      "consent_basis": "Operator reviewed dataset terms and license; not independent individual voice-cloning consent proof",
                                      "dataset_revision": REVISION, "source_speaker_id": speaker,
                                      "source_shard": chosen["shard"], "source_row_index": chosen["row_index"],
                                      "source_published_id": chosen["published_id"], "selected_text_field": chosen["text_field"],
                                      "source_text_fields": chosen["source_text_fields"], **inspected})
                pcm_seen.add(inspected["pcm_sha256"])
                text_seen.add(chosen["norm"])
            completed[speaker] = speaker_slug
            pending.pop(speaker)
            if len(completed) == speakers:
                break
        if len(completed) == speakers or examined >= max_rows:
            break
    if len(completed) != speakers:
        raise ImportBlocked(f"Bounded rows/shards yielded only {len(completed)} complete speakers; need {speakers}, no speaker IDs are guessed")
    raw = output / "input.jsonl"
    with raw.open("x", encoding="utf-8") as stream:
        for row in selected_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest = prepare(raw, output, output / "prepared", max_seconds=cap)
    validate(output / "prepared", output)
    report = {"schema_version": 1, "kind": "ivr_hindi_import", "status": "ready", "repository": REPOSITORY,
              "revision": REVISION, "speakers": len(completed), "clips": len(selected_rows),
              "examined_rows": examined, "skipped": dict(skipped), "counts": manifest["counts"],
              "min_duration_seconds": 1.0, "max_duration_seconds": cap, "target_audio_truncated": False,
              "official_test_used": False, "split_origin": "Speaker-held-out splits from official TRAIN shards only",
              "speaker_mapping": "Published speaker_id to stable SHA256-derived local identifier; never gender/filename inference",
              "transcript_alignment_manually_verified": False, "tts_quality_verified": False,
              "attribution": ATTRIBUTION, "license_url": LICENSE_URL,
              "input_sha256": digest_file(raw), "individual_voice_cloning_consent_independently_verified": False,
              "meaning_of_consent_field": "Operator dataset-rights/use review only; see rights record and source card."}
    write_json(output / "import.json", report)
    incomplete.unlink()
    return report


def run(args, *, api=None, download=None):
    cap = duration_cap(args.max_duration_seconds)
    auth_environment(require_home=api is None)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(endpoint=ENDPOINT)
    plan, shards, documents = metadata_plan(api, max_bytes=args.max_bytes, max_shards=args.max_shards)
    rights = rights_record(args.rights_record) if args.download else None
    reuse = reuse_directory(args.reuse_downloads) if args.download else None
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise ImportBlocked("Output must be a fresh directory; metadata and transfer runs use different directories")
    output.mkdir(parents=True, exist_ok=False)
    audit = output / "audit"
    audit.mkdir()
    plan.update(max_examined_rows=args.max_examined_rows, target_speakers=args.speakers,
                clips_per_speaker=args.clips_per_speaker, max_duration_seconds=cap,
                reuse_downloads_requested=args.reuse_downloads is not None, rights=rights)
    write_json(audit / "plan.json", plan)
    if not args.download:
        return {"status": "metadata_only", "repository": REPOSITORY, "revision": REVISION,
                "planned_bytes": plan["planned_bytes"], "max_bytes": args.max_bytes,
                "corpus_bytes_downloaded": 0, "max_duration_seconds": cap, "audit": str(audit / "plan.json")}
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    transferred, new_bytes, reused_bytes, inventory = 0, 0, 0, []
    for pin in documents:
        path, sha, reused = download_pin(pin, output, download)
        transferred += pin.size
        new_bytes += pin.size
        target = audit / pin.filename
        if target.exists():
            raise ImportBlocked("Refusing to overwrite audit documents")
        shutil.copyfile(path, target)
        inventory.append({**pin.record(), "actual_sha256": sha, "reused": reused})
    card = (audit / "README.md").read_text(encoding="utf-8")
    if "text-to-speech" not in card.casefold() and re.search(r"\btts\b", card, re.IGNORECASE) is None:
        raise ImportBlocked("Pinned README does not contain the reviewed TTS-purpose statement")
    # Pull the next fixed train shard only if the bounded selector still needs
    # it. A complete first-shard result or exhausted row cap never fetches #2.
    def local_shards():
        nonlocal transferred, new_bytes, reused_bytes
        for pin in shards:
            path, sha, reused = download_pin(pin, output, download, reuse_downloads=reuse)
            transferred += pin.size
            if reused:
                reused_bytes += pin.size
            else:
                new_bytes += pin.size
            if transferred > args.max_bytes:
                raise ImportBlocked("Verified corpus/audit file bytes exceeded the reviewed budget")
            inventory.append({**pin.record(), "actual_sha256": sha, "reused": reused})
            yield pin.filename, path
    result = None
    try:
        result = import_parquets(local_shards(), output, speakers=args.speakers,
                                 clips_per_speaker=args.clips_per_speaker, max_rows=args.max_examined_rows,
                                 max_duration_seconds=cap)
    finally:
        write_json(audit / "downloaded_files.json", {"files": inventory, "verified_file_bytes": transferred,
                                                     "newly_downloaded_file_bytes": new_bytes, "reused_file_bytes": reused_bytes,
                                                     "network_wire_bytes_measured": False,
                                                     "import_completed": result is not None,
                                                     "rights": rights, "attribution": ATTRIBUTION,
                                                     "license_url": LICENSE_URL})
    return {"status": result["status"], "clips": result["clips"], "speakers": result["speakers"],
            "counts": result["counts"], "verified_file_bytes": transferred,
            "newly_downloaded_file_bytes": new_bytes, "reused_file_bytes": reused_bytes,
            "network_wire_bytes_measured": False, "max_duration_seconds": cap,
            "prepared": str(output / "prepared"), "audio_root": str(output)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-downloads", type=Path, help="Prior import downloads directory; only the two pinned train shards are re-hashed and copied")
    p.add_argument("--max-duration-seconds", type=float, default=6.0, help="Complete-utterance duration cap, default 6 and hard maximum 20; never truncates targets")
    p.add_argument("--download", action="store_true", help="Explicit opt-in after operator reviewed/accepted dataset conditions")
    p.add_argument("--rights-record", type=Path)
    p.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    p.add_argument("--max-shards", type=int, default=2)
    p.add_argument("--max-examined-rows", type=int, default=MAX_ROWS)
    p.add_argument("--speakers", type=int, default=6)
    p.add_argument("--clips-per-speaker", type=int, default=4)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        duration_cap(args.max_duration_seconds)
        _integer(args.max_examined_rows, "max_examined_rows", maximum=MAX_ROWS)
        _integer(args.speakers, "speakers", minimum=6, maximum=MAX_CANDIDATE_SPEAKERS)
        _integer(args.clips_per_speaker, "clips_per_speaker", minimum=4, maximum=MAX_CANDIDATES_PER_SPEAKER)
        result = run(args)
    except ImportBlocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "trained": False}, ensure_ascii=False))
        return 2
    except Exception as exc:
        # Hub exceptions can contain signed URLs; never echo their body/traceback.
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__,
                          "reason": "Importer failed; auth and transport details are deliberately not printed", "trained": False}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
