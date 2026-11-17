"""CPU guards and actual official encoder tensor contracts; no pretrained claim."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "torchaudio", "transformers"))
if AVAILABLE:
    import numpy as np
    import torch
    import torchaudio
    from hi_h200_native.extract import (AUXILIARY_FILES, ExtractionError, LocalFeatureExtractor,
                                        _strict_state, extract_config, read_pcm_wav,
                                        standardize_semantic, validate_context_lengths, write_npz)


def config():
    return {"version": 2.5, "gpt": {"number_text_tokens": 64, "max_text_tokens": 24,
            "max_mel_tokens": 24, "model_dim": 16, "number_mel_codes": 34,
            "start_mel_token": 32, "stop_mel_token": 33}, "semantic_codec": {"codebook_size": 32}}


@unittest.skipUnless(AVAILABLE, "Native extraction tests need installed CPU tensor dependencies")
class ExtractionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.test_root = ROOT / ".runtime/native-extract-tests"
        cls.test_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def wav(self, name="sample.wav", *, rate=24000, channels=1, width=2, seconds=1, tone=.02):
        path = self.root / name
        values = (np.sin(np.arange(int(rate * seconds)) * tone) * 15000).astype("<i2")
        pcm = values.tobytes() if width == 2 else b"\x80" * len(values)
        if channels == 2:
            pcm = np.repeat(values[:, None], 2, axis=1).tobytes()
        with wave.open(str(path), "wb") as stream:
            stream.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            stream.writeframes(pcm)
        return path, pcm

    def arrays(self):
        return {"codes": np.arange(8, dtype=np.int32), "text_ids": np.array([3, 4], dtype=np.int32),
                "spk_emb": np.zeros(192, dtype=np.float32), "emo_vec": np.zeros(16, dtype=np.float32)}

    def test_exact_pcm_hash_and_float_scaling(self):
        path, pcm = self.wav()
        samples, info = read_pcm_wav(path)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 24000)
        self.assertEqual(info["duration_seconds"], 1)
        self.assertEqual(info["audio_pcm_sha256"], hashlib.sha256(pcm).hexdigest())
        np.testing.assert_array_equal(samples, np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)

    def test_rejects_noncanonical_and_long_target_instead_of_truncating(self):
        for index, options in enumerate(({"rate":16000}, {"channels":2}, {"width":1}, {"seconds":21})):
            path, _ = self.wav(str(index) + ".wav", **options)
            with self.subTest(options=options), self.assertRaises(ExtractionError):
                read_pcm_wav(path)

    def test_truncated_pcm_is_rejected(self):
        path, _ = self.wav()
        path.write_bytes(path.read_bytes()[:-100])
        with self.assertRaises(ExtractionError):
            read_pcm_wav(path)

    def test_full_twenty_second_target_is_preserved(self):
        path, _ = self.wav(seconds=20)
        samples, info = read_pcm_wav(path)
        self.assertEqual(len(samples), 480000)
        self.assertEqual(info["duration_seconds"], 20)

    def test_semantic_standardization_and_invalid_statistics(self):
        hidden = torch.tensor([[[2., 4.], [4., 8.]]])
        actual = standardize_semantic(hidden, torch.tensor([1., 2.]), torch.tensor([1., 4.]))
        torch.testing.assert_close(actual, torch.tensor([[[1., 1.], [3., 3.]]]))
        for variance in (torch.tensor([0., 4.]), torch.tensor([-1., 4.]), torch.tensor([float("nan"), 4.])):
            with self.assertRaises(ExtractionError):
                standardize_semantic(hidden, torch.zeros(2), variance)
        with self.assertRaises(ExtractionError):
            standardize_semantic(hidden, torch.zeros(3), torch.ones(3))

    def test_strict_state_refuses_missing_shape_and_nonfinite_parameters(self):
        source = torch.nn.Linear(2, 3)
        path = self.root / "weights.pt"
        torch.save({"model": source.state_dict()}, path)
        target = torch.nn.Linear(2, 3)
        _strict_state(target, path, wrapped=True)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)
        state = source.state_dict()
        for bad in ({"weight":state["weight"]}, {**state, "weight":torch.zeros(4, 2)}, {**state, "bias":torch.full((3,), float("nan"))}):
            torch.save({"model":bad}, path)
            with self.assertRaises(ExtractionError):
                _strict_state(target, path, wrapped=True)

    def test_restricted_load_never_falls_back_to_unsafe_pickle(self):
        path = self.root / "unsafe.pt"
        torch.save({"model":SimpleNamespace(value=1)}, path)
        with self.assertRaises(ExtractionError):
            _strict_state(torch.nn.Linear(2, 3), path, wrapped=True)

    def test_atomic_npz_roundtrip_and_refuses_overwrite(self):
        from hi_h200_native.data import model_limits, read_npz
        path = self.root / "sample.npz"
        digest = write_npz(path, self.arrays(), config())
        arrays, actual = read_npz(path, model_limits(config()))
        self.assertEqual(digest, actual)
        np.testing.assert_array_equal(arrays["codes"], np.arange(8, dtype=np.int32))
        with self.assertRaises(ExtractionError):
            write_npz(path, self.arrays(), config())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_actual_context_boundary_includes_all_six_control_positions(self):
        validate_context_lengths(4, 8, 18)
        with self.assertRaises(ExtractionError):
            validate_context_lengths(4, 8, 17)

    def test_unsafe_arrays_never_create_output(self):
        path = self.root / "bad.npz"
        for bad in ({**self.arrays(), "codes":np.arange(8, dtype=np.float32)},
                    {**self.arrays(), "codes":np.full(8, 32, dtype=np.int32)},
                    {**self.arrays(), "emo_vec":np.full(16, np.nan, dtype=np.float32)}):
            with self.assertRaises(ExtractionError):
                write_npz(path, bad, config())
            self.assertFalse(path.exists())

    def test_missing_auxiliary_hashes_fail_before_any_model_load(self):
        path = self.root / "model.json"
        path.write_text(json.dumps({"files": {"gpt.pth":"0" * 64}}))
        with mock.patch("hi_h200_native.model.load_verified_base") as loader:
            with self.assertRaises(ExtractionError):
                LocalFeatureExtractor.from_local(self.root, self.root, path)
            loader.assert_not_called()
        self.assertIn("hf_cache/w2v-bert-2.0/model.safetensors", AUXILIARY_FILES)

    def cli(self, *arguments):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([sys.executable, "-B", "-m", "hi_h200_native.extract", *arguments],
                              cwd=ROOT, env=environment, text=True, capture_output=True, timeout=30)

    def test_cli_help_is_an_actual_entrypoint(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        self.assertIn("--allow-cuda", result.stdout)

    def test_cli_cuda_is_blocked_before_any_input_or_gpu_access(self):
        result = self.cli("--config", str(self.root / "absent.json"), "--device", "cuda:0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires explicit --allow-cuda", result.stderr)
        self.assertEqual(result.stdout, "")

    @unittest.skipUnless(OFFICIAL.is_dir(), "CLI prerequisite test uses the pinned local official checkout")
    def test_cli_valid_prepared_data_blocks_missing_model_assets(self):
        from hi_h200.data import prepare
        texts = ["नमस्ते मित्र।", "आज धूप है।", "कल बारिश थी।", "यह किताब है।", "हम घर चले।", "आप कैसे हैं।"]
        rows = []
        for index, text in enumerate(texts):
            path, _ = self.wav(str(index) + ".wav", tone=.02 + .001 * index)
            rows.append({"id":"clip" + str(index), "speaker_id":"speaker" + str(index // 2),
                         "audio":path.name, "language":"hi", "text":text,
                         "source":"synthetic CPU test", "license":"synthetic test only", "consent":True})
        source = self.root / "raw.jsonl"
        source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        prepared = self.root / "prepared"
        prepare(source, self.root, prepared)
        output = self.root / "output"
        model_dir = self.root / "model"
        model_dir.mkdir()
        manifest = self.root / "model.json"
        manifest.write_text(json.dumps({"repository":"IndexTeam/IndexTTS-2.5", "snapshot_revision":"a" * 40, "files":{}}))
        options = {"schema_version":1, "prepared_dir":str(prepared), "audio_root":str(self.root),
                   "feature_manifest":str(output / "manifest.jsonl"), "feature_root":str(output),
                   "provenance":str(output / "provenance.json"), "model_dir":str(model_dir),
                   "model_manifest":str(manifest), "official_repo":str(OFFICIAL), "split":"train"}
        path = self.root / "config.json"
        path.write_text(json.dumps(options))
        result = self.cli("--config", str(path))
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing minimum 2.5 files", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(output.exists())

    def test_all_text_preflight_precedes_heavy_extraction(self):
        prepared = self.root / "prepared"
        prepared.mkdir()
        from hi_h200_native.data import PREPARED_FILES
        for name in PREPARED_FILES:
            (prepared / name).write_text("{}\n")
        row = {"id":"bad-clip", "text_norm":"नमस्ते"}
        (prepared / "train.jsonl").write_text(json.dumps(row) + "\n")
        output = self.root / "output"
        values = {"schema_version":1, "prepared_dir":str(prepared), "audio_root":str(self.root),
                  "feature_manifest":str(output / "manifest.jsonl"), "feature_root":str(output),
                  "provenance":str(output / "provenance.json"), "model_dir":str(self.root),
                  "model_manifest":str(self.root / "model.json"), "official_repo":str(self.root), "split":"train"}
        path = self.root / "extract.json"
        path.write_text(json.dumps(values))
        with mock.patch("hi_h200.data.validate"), mock.patch("hi_h200_native.model.verify_official_assets", return_value=(config(), {})), mock.patch("hi_h200_native.data.official_tokenizer"), mock.patch("hi_h200_native.data.encoded_text", return_value=[3, 0]), mock.patch.object(LocalFeatureExtractor, "from_local") as loader:
            with self.assertRaisesRegex(ExtractionError, "bad-clip"):
                extract_config(path)
            loader.assert_not_called()
            self.assertFalse(output.exists())


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), "Actual encoder contracts need the pinned official source and CPU dependencies")
class RealOfficialEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from hi_h200_native.model import import_official_model
        import_official_model(OFFICIAL)

    def test_actual_seamless_frontend_on_synthetic_pcm(self):
        from transformers import SeamlessM4TFeatureExtractor
        processor = SeamlessM4TFeatureExtractor(sampling_rate=16000)
        waveform = (np.sin(np.arange(16000) * .04) * .1).astype(np.float32)
        values = processor(waveform, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
        self.assertEqual(values["input_features"].shape[0], 1)
        self.assertEqual(values["input_features"].shape[-1], 160)
        self.assertTrue(torch.isfinite(values["input_features"]).all())
        self.assertEqual(values["attention_mask"].shape, values["input_features"].shape[:2])

    def test_actual_campplus_accepts_kaldi_mean_centered_features(self):
        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        waveform = torch.sin(torch.arange(16000, dtype=torch.float32) * .04).unsqueeze(0) * .1
        bank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
        bank -= bank.mean(dim=0, keepdim=True)
        torch.testing.assert_close(bank.mean(0), torch.zeros(80), rtol=0, atol=2e-6)
        with torch.no_grad():
            result = CAMPPlus(feat_dim=80, embedding_size=192).eval()(bank.unsqueeze(0))
        self.assertEqual(result.shape, (1, 192))
        self.assertTrue(torch.isfinite(result).all())

    def test_actual_tiny_enhanced_codec_quantizes_and_decodes(self):
        from indextts.codec.models import EnhancedCodec
        codec = EnhancedCodec(codebook_size=16, hidden_size=32, codebook_dim=8,
                              vocos_dim=32, vocos_intermediate_dim=64, vocos_num_layers=1).eval()
        with torch.no_grad():
            codes, representation = codec.quantize(torch.randn(1, 20, 32))
            decoded = codec.decode(codes)
        self.assertEqual(codes.shape, (1, 10))
        self.assertTrue(((codes >= 0) & (codes < 16)).all())
        self.assertEqual(decoded.shape, (1, 20, 32))
        self.assertTrue(torch.isfinite(representation).all())
        self.assertTrue(torch.isfinite(decoded).all())

    def test_actual_official_emotion_encoder_uses_semantic_lengths(self):
        from hi_h200_native.model import make_tiny_official_model
        base = make_tiny_official_model(OFFICIAL, seed=31).base.eval()
        with torch.no_grad():
            result = base.get_emovec(torch.randn(1, 20, 1024), torch.tensor([20]))
        self.assertEqual(result.shape, (1, 16))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
