"""Original local-only extraction from canonical PCM WAV to audited GPT features.

The recipe follows official module contracts at d9e41aac89fd00b3d71497fddb287b7f24613712.
It resamples canonical 24 kHz PCM directly to 16 kHz. The official inference
loader first uses librosa's 22.05 kHz default; numerical parity is NOT claimed.
No community backend is imported and no download helper is called.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile
import uuid
import wave

RECIPE = "hi_h200_native.extract.pcm24k_direct16k_reference15s.v1"
SOURCE_RATE = 24000
ENCODER_RATE = 16000
REFERENCE_SECONDS = 15
AUXILIARY_FILES = frozenset({
    "codec.pth", "wav2vec2bert_stats.pt", "hf_cache/campplus_cn_common.bin",
    "hf_cache/w2v-bert-2.0/config.json", "hf_cache/w2v-bert-2.0/preprocessor_config.json",
    "hf_cache/w2v-bert-2.0/model.safetensors", "LICENSE",
})


class ExtractionError(ValueError):
    """The actual local extraction prerequisites or outputs are inconsistent."""


def read_pcm_wav(path):
    """Decode bounded mono PCM16/24 kHz WAV; never truncate target speech."""
    import numpy as np
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".wav" or path.stat().st_size > 2_000_000:
        raise ExtractionError("Expected a bounded local WAV (at most 2 MB)")
    try:
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, SOURCE_RATE, "NONE"):
                raise ExtractionError("Extraction requires mono signed PCM16 WAV at 24000 Hz")
            frames = stream.getnframes()
            if not SOURCE_RATE <= frames <= SOURCE_RATE * 20:
                raise ExtractionError("Target WAV must be 1..20 seconds; split long audio with matching text first")
            pcm = stream.readframes(frames + 1)
            if len(pcm) != frames * 2:
                raise ExtractionError("Truncated or inconsistent PCM WAV")
    except (OSError, wave.Error, EOFError) as exc:
        raise ExtractionError("Cannot decode canonical WAV") from exc
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
    return samples, {"duration_seconds": frames / SOURCE_RATE, "sample_rate": SOURCE_RATE,
                     "num_frames": frames, "audio_pcm_sha256": hashlib.sha256(pcm).hexdigest()}


def _finite_tensor(value, name):
    import torch
    if not torch.is_tensor(value) or not torch.isfinite(value).all().item():
        raise ExtractionError(name + " must be a finite tensor")
    return value


def standardize_semantic(hidden, mean, variance):
    """Normalize the actual layer-17 tensor using shipped mean and variance."""
    import torch
    _finite_tensor(hidden, "semantic hidden state")
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1:
        raise ExtractionError("Expected one semantic sequence [1,T,D]")
    width = hidden.shape[2]
    mean, variance = torch.as_tensor(mean), torch.as_tensor(variance)
    _finite_tensor(mean, "semantic mean")
    _finite_tensor(variance, "semantic variance")
    if mean.numel() != width or variance.numel() != width or not (variance > 0).all().item():
        raise ExtractionError("Semantic statistics must have one strictly positive variance per channel")
    result = (hidden.float() - mean.to(hidden.device, torch.float32).reshape(1, 1, width)) / variance.to(hidden.device, torch.float32).reshape(1, 1, width).sqrt()
    return _finite_tensor(result, "standardized semantic representation")


def _strict_state(module, path, *, wrapped=False):
    """Restricted, complete state load; never retain randomly initialized gaps."""
    import torch
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if wrapped:
            if not isinstance(value, dict) or "model" not in value:
                raise ExtractionError("Codec checkpoint must contain its official model state")
            value = value["model"]
        if not isinstance(value, dict) or not value or not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
            raise ExtractionError("Expected a complete tensor state dictionary")
        expected = module.state_dict()
        if set(value) != set(expected):
            raise ExtractionError("Checkpoint keys differ from the actual official module")
        for key, tensor in value.items():
            _finite_tensor(tensor, "checkpoint tensor " + key)
            if tensor.shape != expected[key].shape:
                raise ExtractionError("Checkpoint tensor shape mismatch: " + key)
        module.load_state_dict(value, strict=True)
    except (RuntimeError, TypeError, OSError, pickle.UnpicklingError) as exc:
        raise ExtractionError("Strict restricted module load failed; no partial/unsafe fallback") from exc


def _local_module(name, official_repo):
    module = importlib.import_module(name)
    filename = Path(inspect.getfile(module)).resolve(strict=True)
    if not filename.is_relative_to(Path(official_repo).resolve(strict=True)):
        raise ExtractionError("An imported encoder is outside the verified official checkout")
    return module


def _validate_arrays(arrays, config):
    import numpy as np
    from .data import model_limits
    limits = model_limits(config)
    if not isinstance(arrays, dict) or set(arrays) != {"codes", "text_ids", "spk_emb", "emo_vec"}:
        raise ExtractionError("Expected exactly the four native feature arrays")
    for name, dtype in (("codes", "int32"), ("text_ids", "int32"), ("spk_emb", "float32"), ("emo_vec", "float32")):
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.ndim != 1 or not np.isfinite(value).all():
            raise ExtractionError("Invalid feature dtype/shape/finite values: " + name)
    if arrays["spk_emb"].shape != (192,) or arrays["emo_vec"].shape != (limits["emotion_dim"],):
        raise ExtractionError("Actual conditioning vector dimensions differ from the model")
    codes, tokens = arrays["codes"], arrays["text_ids"]
    if not 8 <= len(codes) <= limits["code_max"] or (codes < 0).any() or (codes >= limits["codebook"]).any():
        raise ExtractionError("Semantic codes are empty/too long/reserved/out of range")
    if not 1 <= len(tokens) <= limits["text_max"] or (tokens < 2).any() or (tokens >= limits["text_vocab"]).any():
        raise ExtractionError("Text IDs are empty/too long/reserved/out of range")
    return arrays


def validate_context_lengths(text_len, code_len, capacity):
    if any(type(value) is not int or value < 1 for value in (text_len, code_len, capacity)) or text_len + code_len + 6 > capacity:
        raise ExtractionError("Teacher-forcing context exceeds the actual official GPT position capacity")


def write_npz(path, arrays, config):
    """Publish complete NumPy arrays with no overwrite and no pickle."""
    import numpy as np
    arrays = _validate_arrays(arrays, config)
    path = Path(path)
    if path.suffix != ".npz" or path.exists() or path.is_symlink():
        raise ExtractionError("Choose a fresh .npz destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".extract-", suffix=".npz.tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication which refuses any existing destination.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    from .data import file_sha256
    return file_sha256(path)


class LocalFeatureExtractor:
    """Actual frozen official encoders; production construction is from_local()."""
    def __init__(self, *, config, identity, gpt, processor, semantic_model, codec, speaker_model, mean, variance, hi_token, device="cpu"):
        import torch
        from .data import model_limits
        self.config, self.identity = config, dict(identity)
        self.limits = model_limits(config)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"} or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ExtractionError("Use available CPU or explicitly available CUDA; no implicit device fallback")
        if type(hi_token) is not int or not 2 <= hi_token < self.limits["text_vocab"]:
            raise ExtractionError("Invalid resolved Hindi control token")
        self.hi_token = hi_token
        self.processor, self.gpt = processor, gpt
        self.semantic_model, self.codec, self.speaker_model = semantic_model, codec, speaker_model
        for module in (gpt, semantic_model, codec, speaker_model):
            module.eval().to(self.device)
            module.requires_grad_(False)
        self.mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32)
        self.variance = torch.as_tensor(variance, device=self.device, dtype=torch.float32)
        _finite_tensor(self.mean, "semantic mean")
        _finite_tensor(self.variance, "semantic variance")
        if self.mean.numel() != 1024 or self.variance.numel() != 1024 or not (self.variance > 0).all().item():
            raise ExtractionError("Official 2.5 semantic statistics require 1024 finite channels and positive variances")

    @classmethod
    def from_local(cls, official_repo, model_dir, model_manifest, *, device="cpu"):
        import torch
        from .data import read_object, official_tokenizer, encoded_text
        from .model import load_verified_base
        manifest = read_object(model_manifest)
        if not isinstance(manifest.get("files"), dict) or not AUXILIARY_FILES.issubset(manifest["files"]):
            raise ExtractionError("Extraction manifest must hash all local auxiliary assets and LICENSE")
        # The shared loader verifies source commit, ALL supplied hashes and GPT strictness.
        target_device = torch.device(device)
        if target_device.type not in {"cpu", "cuda"} or (target_device.type == "cuda" and not torch.cuda.is_available()):
            raise ExtractionError("Use available CPU or explicitly available CUDA")
        gpt, config, identity = load_verified_base(official_repo, model_dir, model_manifest)
        if not AUXILIARY_FILES.issubset(identity["model_files"]):
            raise ExtractionError("Auxiliary asset identity changed before encoder loading")
        root = Path(model_dir).resolve(strict=True)
        codec_cls = _local_module("indextts.codec.models", official_repo).EnhancedCodec
        speaker_cls = _local_module("indextts.s2mel.modules.campplus.DTDNN", official_repo).CAMPPlus
        codec = codec_cls(**dict(config["semantic_codec"]))
        _strict_state(codec, root / "codec.pth", wrapped=True)
        speaker = speaker_cls(feat_dim=80, embedding_size=192)
        _strict_state(speaker, root / "hf_cache/campplus_cn_common.bin")
        try:
            from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
            folder = root / "hf_cache/w2v-bert-2.0"
            processor = SeamlessM4TFeatureExtractor.from_pretrained(str(folder), local_files_only=True)
            semantic, report = Wav2Vec2BertModel.from_pretrained(str(folder), local_files_only=True,
                                                              use_safetensors=True, output_loading_info=True)
            if any(report.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ExtractionError("Local w2v-BERT weights do not completely match the actual architecture")
            stats = torch.load(root / "wav2vec2bert_stats.pt", map_location="cpu", weights_only=True)
            if not isinstance(stats, dict) or set(stats) != {"mean", "var"}:
                raise ExtractionError("Semantic statistics must contain exactly mean and var")
        except (OSError, RuntimeError, ImportError, TypeError) as exc:
            raise ExtractionError("Local encoder dependencies/weights unavailable; no download or fallback") from exc
        from .data import file_sha256
        for name in AUXILIARY_FILES:
            if file_sha256(root / name) != identity["model_files"][name]:
                raise ExtractionError("Local auxiliary asset changed while loading: " + name)
        tokenizer = official_tokenizer(official_repo, root)
        token = encoded_text(tokenizer, "हिन्दी")[0]
        return cls(config=config, identity=identity, gpt=gpt, processor=processor,
                   semantic_model=semantic, codec=codec, speaker_model=speaker,
                   mean=stats["mean"], variance=stats["var"], hi_token=token, device=device)

    def _semantic(self, samples_16k):
        import torch
        processed = self.processor(samples_16k.detach().cpu().numpy(), sampling_rate=ENCODER_RATE,
                                   return_attention_mask=True, return_tensors="pt")
        if "input_features" not in processed or "attention_mask" not in processed:
            raise ExtractionError("Seamless feature extractor did not produce features and a mask")
        result = self.semantic_model(input_features=processed["input_features"].to(self.device),
                                     attention_mask=processed["attention_mask"].to(self.device),
                                     output_hidden_states=True)
        states = result.hidden_states
        if states is None or len(states) <= 17 or states[17].shape[-1] != 1024:
            raise ExtractionError("w2v-BERT must expose its actual 1024-channel hidden state 17")
        return standardize_semantic(states[17], self.mean, self.variance)

    def extract(self, audio_path, text_ids):
        import numpy as np
        import torch
        import torchaudio
        if not isinstance(text_ids, (list, tuple)) or not text_ids or any(type(value) is not int for value in text_ids):
            raise ExtractionError("Text IDs must be an actual integer sequence from the verified tokenizer")
        if text_ids[0] != self.hi_token or len(text_ids) > self.limits["text_max"] or any(not 2 <= value < self.limits["text_vocab"] for value in text_ids):
            raise ExtractionError("Missing Hindi prefix or text IDs exceed actual model capacities")
        samples, metadata = read_pcm_wav(audio_path)
        waveform = torch.from_numpy(samples)
        with torch.no_grad():
            target = torchaudio.functional.resample(waveform, SOURCE_RATE, ENCODER_RATE)
            semantic = self._semantic(target)
            codes, _ = self.codec.quantize(semantic)
            if not torch.is_tensor(codes) or codes.dtype not in (torch.int32, torch.int64):
                raise ExtractionError("Actual semantic codec did not return integer token IDs")
            while codes.ndim > 1 and codes.shape[0] == 1:
                codes = codes.squeeze(0)
            if codes.ndim != 1 or not 8 <= codes.numel() <= self.limits["code_max"] or (codes < 0).any().item() or (codes >= self.limits["codebook"]).any().item():
                raise ExtractionError("Codec returned invalid dimensions, lengths, or reserved/out-of-range IDs")
            validate_context_lengths(len(text_ids), codes.numel(), self.gpt.gpt.config.n_positions)
            reference = waveform[: SOURCE_RATE * REFERENCE_SECONDS]
            if reference.numel() == waveform.numel():
                ref_wave, ref_semantic = target, semantic
            else:
                ref_wave = torchaudio.functional.resample(reference, SOURCE_RATE, ENCODER_RATE)
                ref_semantic = self._semantic(ref_wave)
            bank = torchaudio.compliance.kaldi.fbank(ref_wave.unsqueeze(0).to(self.device), num_mel_bins=80,
                                                    dither=0, sample_frequency=ENCODER_RATE)
            bank = bank - bank.mean(dim=0, keepdim=True)
            spk = self.speaker_model(bank.unsqueeze(0))
            lengths = torch.tensor([ref_semantic.shape[1]], dtype=torch.long, device=self.device)
            emotion = self.gpt.get_emovec(ref_semantic, lengths)
            if spk.shape != (1, 192) or emotion.shape != (1, self.limits["emotion_dim"]):
                raise ExtractionError("Official conditioning encoders returned incompatible shapes")
            arrays = {"codes": codes.detach().cpu().numpy().astype(np.int32),
                      "text_ids": np.asarray(text_ids, dtype=np.int32),
                      "spk_emb": spk[0].detach().float().cpu().numpy(),
                      "emo_vec": emotion[0].detach().float().cpu().numpy()}
        _validate_arrays(arrays, self.config)
        metadata.update({"reference_seconds": min(metadata["duration_seconds"], REFERENCE_SECONDS),
                         "recipe": RECIPE, "numerical_parity_verified": False,
                         "text_len": len(text_ids), "code_len": len(arrays["codes"])})
        return arrays, metadata


def extract_config(config_path, *, device="cpu"):
    """Extract exactly one validated prepared split and publish its provenance."""
    from hi_h200.data import IDENTIFIER, read_jsonl, validate
    from .data import FRONTEND, OFFICIAL_COMMIT, PREPARED_FILES, encoded_text, file_sha256, model_limits, official_tokenizer, read_object
    from .model import verify_official_assets
    source = Path(config_path).resolve(strict=True)
    options = read_object(source)
    fields = {"schema_version", "prepared_dir", "audio_root", "feature_manifest", "feature_root",
              "provenance", "model_dir", "model_manifest", "official_repo", "split"}
    if set(options) != fields or type(options.get("schema_version")) is not int or options["schema_version"] != 1 or options["split"] not in {"train", "val", "test"}:
        raise ExtractionError("Invalid native extraction configuration")
    paths = {}
    for key in fields - {"schema_version", "split"}:
        value = options[key]
        if not isinstance(value, str) or not value or "://" in value or any(ord(c) < 32 for c in value):
            raise ExtractionError("Extraction paths must be explicit local paths")
        raw = Path(value)
        candidate = raw if raw.is_absolute() else source.parent / raw
        if key == "feature_root" and candidate.is_symlink():
            raise ExtractionError("Feature root cannot be a symlink")
        paths[key] = candidate.resolve()
    output = paths["feature_root"]
    if output.exists() or output.is_symlink():
        raise ExtractionError("Feature root already exists; choose a fresh directory")
    if any(not paths[key].is_relative_to(output) or paths[key] == output for key in ("feature_manifest", "provenance")) or paths["feature_manifest"] == paths["provenance"]:
        raise ExtractionError("Feature manifest and provenance must be separate paths inside the fresh feature root")
    prepared_hashes = {name: file_sha256(paths["prepared_dir"] / name) for name in PREPARED_FILES}
    validate(paths["prepared_dir"], paths["audio_root"])
    rows = list(read_jsonl(paths["prepared_dir"] / (options["split"] + ".jsonl")))
    if prepared_hashes != {name: file_sha256(paths["prepared_dir"] / name) for name in PREPARED_FILES}:
        raise ExtractionError("Prepared manifests changed during input validation")
    actual_config, _ = verify_official_assets(paths["official_repo"], paths["model_dir"], paths["model_manifest"])
    limits = model_limits(actual_config)
    tokenizer = official_tokenizer(paths["official_repo"], paths["model_dir"])
    tokens_by_id = {}
    for row in rows:
        if not isinstance(row.get("id"), str) or not IDENTIFIER.fullmatch(row["id"]):
            raise ExtractionError("Unsafe or missing prepared clip identifier")
        try:
            tokens = encoded_text(tokenizer, row["text_norm"])
            if not 1 <= len(tokens) <= limits["text_max"] or any(type(value) is not int or not 2 <= value < limits["text_vocab"] for value in tokens):
                raise ExtractionError("Reserved/out-of-range text ID or length exceeding the actual model limit")
            tokens_by_id[row["id"]] = tokens
        except ValueError as exc:
            raise ExtractionError("Text preflight failed for clip " + row["id"] + ": " + str(exc)) from exc
    # Reject every invalid transcript before constructing the heavy encoders or using a GPU.
    extractor = LocalFeatureExtractor.from_local(paths["official_repo"], paths["model_dir"], paths["model_manifest"], device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ("." + output.name + ".extract-" + uuid.uuid4().hex)
    records, inventory = [], []
    try:
        stage.mkdir()
        for row in rows:
            audio = (paths["audio_root"] / row["audio"]).resolve(strict=True)
            if not audio.is_relative_to(paths["audio_root"]):
                raise ExtractionError("Prepared audio escaped audio_root")
            tokens = tokens_by_id[row["id"]]
            arrays, metadata = extractor.extract(audio, tokens)
            if metadata["audio_pcm_sha256"] != row["audio_sha256"] or metadata["duration_seconds"] != row["duration_seconds"]:
                raise ExtractionError("PCM changed after prepared dataset validation")
            relative = "features/" + row["id"] + ".npz"
            digest = write_npz(stage / relative, arrays, extractor.config)
            records.append({"id": row["id"], "speaker_id": row["speaker_id"], "text_norm": row["text_norm"],
                            "feature": relative, "text_len": len(tokens), "code_len": len(arrays["codes"])})
            inventory.append({"id": row["id"], "feature": relative, "sha256": digest,
                              "audio_pcm_sha256": row["audio_sha256"], "text_sha256": row["text_sha256"]})
        manifest = stage / paths["feature_manifest"].relative_to(output)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        identity = extractor.identity
        provenance = {"schema_version": 1, "kind": "hi_h200_native_features", "official_commit": OFFICIAL_COMMIT,
                      "model_snapshot_revision": identity["model_snapshot_revision"],
                      "model_manifest_sha256": identity["model_manifest_sha256"], "tokenizer_sha256": identity["tokenizer_sha256"],
                      "frontend": FRONTEND, "language": "hi", "feature_manifest_sha256": file_sha256(manifest),
                      "prepared_sha256": prepared_hashes, "extractor": {"implementation": RECIPE, "source_sha256": file_sha256(__file__)},
                      "features": inventory}
        target = stage / paths["provenance"].relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if prepared_hashes != {name: file_sha256(paths["prepared_dir"] / name) for name in PREPARED_FILES}:
            raise ExtractionError("Prepared manifests changed during extraction")
        # A fresh destination prevents stale feature/cache reuse. No extraction failures are skipped.
        if output.exists() or output.is_symlink():
            raise ExtractionError("Feature output appeared while extraction was running")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"clips": len(records), "split": options["split"], "feature_manifest": str(paths["feature_manifest"]),
            "provenance": str(paths["provenance"]), "recipe": RECIPE, "runtime_audio_feature_parity_verified": False}


def main(argv=None):
    import argparse
    import re
    import sys
    parser = argparse.ArgumentParser(description="Extract native 2.5 Hindi features using verified local assets; never download models.")
    parser.add_argument("--config", required=True, type=Path, help="Native dataset JSON configuration; output feature_root must be fresh")
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, or cuda:N")
    parser.add_argument("--allow-cuda", action="store_true", help="Explicitly permit this invocation to construct and run CUDA encoders")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        parser.error("device must be cpu, cuda, or cuda:N")
    if args.device.startswith("cuda") and not args.allow_cuda:
        parser.error("CUDA extraction requires explicit --allow-cuda")
    try:
        result = extract_config(args.config, device=args.device)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print("Extraction failed: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
