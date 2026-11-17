"""Original, strict feature input for the native batch-one trainer.

This module never extracts audio features or imports the external community
backend. Provenance binds supplied files; it is not proof that extraction ran.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile

from hi_h200.data import read_jsonl, validate as validate_prepared
from hi_h200.text import DataError

OFFICIAL_COMMIT = "d9e41aac89fd00b3d71497fddb287b7f24613712"
FRONTEND = "hi_h200_native.nfc_ws_hi_prefix.v1"
PREPARED_FILES = ("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl",
                  "backend_train.jsonl", "backend_val.jsonl", "backend_test.jsonl")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class NativeDataError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path):
    from hi_h200.data import load_json
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise NativeDataError("Missing or oversized JSON object")
    try:
        value = load_json(path.read_text(encoding="utf-8"))
    except (DataError, UnicodeError) as exc:
        raise NativeDataError("Invalid native JSON object") from exc
    if not isinstance(value, dict):
        raise NativeDataError("Native JSON must be an object")
    return value


def safe_file(root, relative):
    if not isinstance(relative, str) or not relative or ":" in relative or "\\" in relative:
        raise NativeDataError("Feature path must be a relative local POSIX path")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in relative):
        raise NativeDataError("Control characters in feature path")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or str(value) != relative:
        raise NativeDataError("Unsafe feature path")
    root = Path(root).resolve(strict=True)
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.suffix != ".npz":
        raise NativeDataError("NPZ feature escapes its root or is not a file")
    return path


def model_limits(config):
    try:
        gpt, codec = config["gpt"], config["semantic_codec"]
        limits = {"text_vocab": gpt["number_text_tokens"], "text_max": gpt["max_text_tokens"],
                  "code_max": gpt["max_mel_tokens"], "emotion_dim": gpt["model_dim"],
                  "codebook": codec["codebook_size"],
                  "text_start": gpt.get("start_text_token", 0), "text_stop": gpt.get("stop_text_token", 1)}
        reserved = (gpt["number_mel_codes"], gpt["start_mel_token"], gpt["stop_mel_token"])
    except (KeyError, TypeError) as exc:
        raise NativeDataError("Missing official model capacity fields") from exc
    if str(config.get("version")) != "2.5" or any(type(value) is not int or value < 1 for value in (*[v for k,v in limits.items() if k not in ("text_start", "text_stop")], *reserved)):
        raise NativeDataError("Native route requires positive integer capacities and version 2.5")
    if any(type(limits[key]) is not int or not 0 <= limits[key] < limits["text_vocab"] for key in ("text_start", "text_stop")) or limits["text_start"] == limits["text_stop"]:
        raise NativeDataError("Invalid actual model text framing IDs")
    if reserved != (limits["codebook"] + 2, limits["codebook"], limits["codebook"] + 1):
        raise NativeDataError("Unexpected codec start/stop allocation")
    return limits


def read_npz(path, limits):
    """Validate bounded NPY headers before NumPy can allocate arrays; never pickle."""
    import numpy as np
    with Path(path).open("rb") as stream:
        payload = stream.read(32 * 1024 * 1024 + 1)
    if not payload or len(payload) > 32 * 1024 * 1024:
        raise NativeDataError("Feature payload exceeds 32 MiB bound")
    fields = {"codes": ("int32", None), "text_ids": ("int32", None),
              "spk_emb": ("float32", 192), "emo_vec": ("float32", limits["emotion_dim"])}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) != 4 or {item.filename for item in entries} != {name + ".npy" for name in fields}:
                raise NativeDataError("Unexpected NPZ members")
            if sum(item.file_size for item in entries) > 16 * 1024 * 1024:
                raise NativeDataError("NPZ expansion exceeds bound")
            for item in entries:
                if item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise NativeDataError("Unsupported NPZ encoding")
                key = item.filename[:-4]
                dtype_name, length = fields[key]
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise NativeDataError("Unsupported NPY header version")
                    shape, _, dtype = reader(stream)
                    maximum = limits["code_max"] if key == "codes" else limits["text_max"]
                    if dtype.hasobject or dtype != np.dtype(dtype_name) or len(shape) != 1:
                        raise NativeDataError("Unsafe NPY dtype/shape")
                    if (length is not None and shape != (length,)) or (length is None and not 1 <= shape[0] <= maximum):
                        raise NativeDataError("NPY shape exceeds actual model capacities")
                    if shape[0] * dtype.itemsize != item.file_size - stream.tell():
                        raise NativeDataError("NPY payload disagrees with declared shape")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in fields}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise NativeDataError("Cannot safely read native NPZ") from exc
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise NativeDataError("Nonfinite native feature")
    text, codes = arrays["text_ids"], arrays["codes"]
    if not 8 <= len(codes) <= limits["code_max"] or (codes < 0).any() or (codes >= limits["codebook"]).any():
        raise NativeDataError("Reserved/out-of-range codec token or invalid length")
    if (text < 0).any() or (text >= limits["text_vocab"]).any():
        raise NativeDataError("Out-of-range text token")
    if np.isin(text, (limits["text_start"], limits["text_stop"])).any():
        raise NativeDataError("Raw text overlaps model framing IDs; review punctuation and re-tokenize, never silently strip it")
    return arrays, hashlib.sha256(payload).hexdigest()


def official_tokenizer(official_repo, model_dir):
    """Only called after the shared verifier approves the pinned source/weights."""
    sys.path.insert(0, str(Path(official_repo).resolve()))
    from indextts.utils.tokenizer import get_tokenizer
    return get_tokenizer(multilingual=True, num_languages=99, language="hi",
                         task="transcribe", model_dir=str(model_dir))


def encoded_text(tokenizer, text):
    tokens = tokenizer.encode("<|hi|> " + text, allowed_special={"<|hi|>"})
    hi = tokenizer.encode("<|hi|>", allowed_special={"<|hi|>"})
    if len(hi) != 1 or not tokens or tokens[0] != hi[0] or any(type(value) is not int for value in tokens):
        raise NativeDataError("Tokenizer did not resolve a single Hindi prefix")
    return tokens


class NativeFeatureDataset:
    """Indexable, immutable-by-hash batch-one examples; CPU tensors on demand."""
    def __init__(self, prepared, feature_root, manifest, provenance, limits, tokenizer, asset_identity):
        self._root = Path(feature_root).resolve(strict=True)
        self._limits = limits
        self._records = {}
        self._prepared = {row["id"]: row for row in prepared}
        self._ids = sorted(self._prepared)
        entries = provenance.get("features")
        if not isinstance(entries, list) or len(entries) != len(self._prepared):
            raise NativeDataError("Provenance does not cover prepared clips")
        inventory = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "feature", "sha256", "audio_pcm_sha256", "text_sha256"}:
                raise NativeDataError("Invalid feature provenance schema")
            uid = entry["id"]
            if not isinstance(uid, str) or uid not in self._prepared or uid in inventory or not SHA256.fullmatch(str(entry["sha256"])):
                raise NativeDataError("Unknown/duplicate feature provenance ID")
            original = self._prepared[uid]
            if entry["audio_pcm_sha256"] != original["audio_sha256"] or entry["text_sha256"] != original["text_sha256"]:
                raise NativeDataError("Feature provenance audio/text hash differs from prepared source")
            inventory[uid] = entry
        paths, seen_tokens = set(), set()
        for row in read_jsonl(manifest):
            if set(row) != {"id", "speaker_id", "text_norm", "feature", "text_len", "code_len"}:
                raise NativeDataError("Invalid native feature row schema")
            uid = row["id"]
            if not isinstance(uid, str) or uid not in self._prepared or uid in self._records:
                raise NativeDataError("Unknown/duplicate native feature ID")
            original = self._prepared[uid]
            if row["speaker_id"] != original["speaker_id"] or row["text_norm"] != original["text_norm"]:
                raise NativeDataError("Feature speaker/text differs from prepared source")
            path = safe_file(self._root, row["feature"])
            if path in paths or row["feature"] != inventory[uid]["feature"]:
                raise NativeDataError("Feature path reused or differs from provenance")
            arrays, digest = read_npz(path, limits)
            expected = encoded_text(tokenizer, original["text_norm"])
            if digest != inventory[uid]["sha256"] or arrays["text_ids"].tolist() != expected:
                raise NativeDataError("Feature changed or cached tokens differ from exact native frontend")
            if type(row["text_len"]) is not int or type(row["code_len"]) is not int or row["text_len"] != len(expected) or row["code_len"] != len(arrays["codes"]):
                raise NativeDataError("Feature length metadata differs from actual arrays")
            self._records[uid] = {"path": path, "sha256": digest}
            paths.add(path)
            seen_tokens.update(expected)
        if set(self._records) != set(self._prepared):
            raise NativeDataError("Native feature manifest omitted prepared clips")
        for uid, row in self._prepared.items():
            prompt = self._prepared.get(row.get("prompt_id"))
            if prompt is None or prompt["id"] == uid or prompt["speaker_id"] != row["speaker_id"] or prompt["split"] != row["split"] or prompt["audio_sha256"] == row["audio_sha256"]:
                raise NativeDataError("Prompt is not a distinct same-speaker clip in this split")
        self.text_rows = sorted(seen_tokens)
        self.speaker_ids = frozenset(row["speaker_id"] for row in prepared)
        self.clip_ids = frozenset(self._ids)
        self.audio_hashes = frozenset(row["audio_sha256"] for row in prepared)
        self.text_hashes = frozenset(row["text_sha256"] for row in prepared)
        self.feature_hashes = frozenset(record["sha256"] for record in self._records.values())
        self.identity = {**asset_identity, "frontend": FRONTEND, "split": prepared[0]["split"],
                         "feature_manifest_sha256": file_sha256(manifest),
                         "provenance_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         "prepared_sha256": provenance["prepared_sha256"],
                         "feature_files": {uid: self._records[uid]["sha256"] for uid in self._ids},
                         "runtime_audio_feature_parity_verified": False}
        self.identity["dataset_sha256"] = hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __len__(self):
        return len(self._ids)

    def _arrays(self, uid):
        record = self._records[uid]
        resolved = record["path"].resolve(strict=True)
        if not resolved.is_relative_to(self._root) or resolved != record["path"]:
            raise NativeDataError("Feature path changed or escaped after validation")
        arrays, digest = read_npz(resolved, self._limits)
        if digest != record["sha256"]:
            raise NativeDataError("Native feature changed after dataset validation")
        return arrays

    def __getitem__(self, index):
        import torch
        uid = self._ids[index]
        row = self._prepared[uid]
        target, prompt = self._arrays(uid), self._arrays(row["prompt_id"])
        return {"id": uid, "speaker_id": row["speaker_id"], "prompt_id": row["prompt_id"],
                "text_ids": torch.from_numpy(target["text_ids"].astype("int64")),
                "codes": torch.from_numpy(target["codes"].astype("int64")),
                "spk_emb": torch.from_numpy(prompt["spk_emb"].copy()),
                "emo_vec": torch.from_numpy(target["emo_vec"].copy())}


def current_extractor_identity():
    """Recognize only this version of the original local extraction implementation."""
    from .extract import RECIPE
    return {"implementation": RECIPE,
            "source_sha256": file_sha256(Path(__file__).with_name("extract.py"))}


def load_native_dataset(config_path, *, tokenizer=None, asset_verifier=None,
                        extractor_identity_provider=None):
    """Load strict config; explicit unit-test injections are absent from the CLI.

    Production accepts only the current original extractor recipe and source hash.
    This freshness check still does not prove that the claimed extraction was run.
    """
    config_path = Path(config_path).resolve(strict=True)
    options = read_object(config_path)
    expected = {"schema_version", "prepared_dir", "audio_root", "feature_manifest", "feature_root",
                "provenance", "model_dir", "model_manifest", "official_repo", "split"}
    if set(options) != expected or (type(options.get("schema_version")) is not int or options["schema_version"] != 1) or options.get("split") not in ("train", "val", "test"):
        raise NativeDataError("Native data config schema/split is invalid")
    paths = {}
    for key in expected - {"schema_version", "split"}:
        value = options[key]
        if not isinstance(value, str) or not value or "://" in value or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
            raise NativeDataError("Native data paths must be explicit local paths")
        raw = Path(value)
        paths[key] = (raw if raw.is_absolute() else config_path.parent / raw).resolve(strict=True)
    try:
        validate_prepared(paths["prepared_dir"], paths["audio_root"])
    except DataError as exc:
        raise NativeDataError("Prepared data failed native input validation: " + str(exc)) from exc
    if asset_verifier is None:
        from .model import verify_official_assets
        asset_verifier = verify_official_assets
    actual_config, identity = asset_verifier(paths["official_repo"], paths["model_dir"], paths["model_manifest"])
    limits = model_limits(actual_config)
    provenance = read_object(paths["provenance"])
    required = {"schema_version", "kind", "official_commit", "model_snapshot_revision", "model_manifest_sha256", "tokenizer_sha256", "frontend", "language", "feature_manifest_sha256", "prepared_sha256", "extractor", "features"}
    if set(provenance) != required or (type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1) or provenance["kind"] != "hi_h200_native_features" or provenance["language"] != "hi" or provenance["frontend"] != FRONTEND:
        raise NativeDataError("Native extraction provenance schema/frontend is invalid")
    if provenance["official_commit"] != OFFICIAL_COMMIT or any(provenance[key] != identity.get(key) for key in ("official_commit", "model_snapshot_revision", "model_manifest_sha256", "tokenizer_sha256")):
        raise NativeDataError("Feature extraction used different source/model/tokenizer identity")
    extractor = provenance["extractor"]
    if not isinstance(extractor, dict) or set(extractor) != {"implementation", "source_sha256"} or not isinstance(extractor["implementation"], str) or not extractor["implementation"].strip() or not SHA256.fullmatch(str(extractor["source_sha256"])):
        raise NativeDataError("Missing declared extraction implementation identity")
    expected_extractor = (extractor_identity_provider or current_extractor_identity)()
    if extractor != expected_extractor:
        raise NativeDataError("Feature extractor recipe/source differs from the current original implementation; re-extract into a fresh directory")
    hashes = {name: file_sha256(paths["prepared_dir"] / name) for name in PREPARED_FILES}
    if provenance["prepared_sha256"] != hashes or provenance["feature_manifest_sha256"] != file_sha256(paths["feature_manifest"]):
        raise NativeDataError("Native feature provenance is stale for the prepared/feature manifests")
    if not paths["feature_manifest"].is_relative_to(paths["feature_root"]):
        raise NativeDataError("Native feature manifest must be inside feature_root")
    if tokenizer is None:
        tokenizer = official_tokenizer(paths["official_repo"], paths["model_dir"])
    prepared = list(read_jsonl(paths["prepared_dir"] / (options["split"] + ".jsonl")))
    dataset = NativeFeatureDataset(prepared, paths["feature_root"], paths["feature_manifest"], provenance,
                                   limits, tokenizer, identity)
    dataset.identity["split"] = options["split"]
    return dataset
