"""Real CPU forward/backward and exact continuation with the official tiny GPT.

The token/code sequences and embeddings are synthetic; no pretrained speech or
Hindi-quality result is implied. The model itself is the official UnifiedVoice.
"""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
try:
    import numpy as np
    import torch
    from hi_h200_native.engine import (LoopConfig, StatefulTrainer, StopController, TrainingError,
                                       cosine_scheduler, rng_state, seed_everything)
    from hi_h200_native.model import OFFICIAL_COMMIT, make_tiny_official_model
    AVAILABLE = SOURCE.is_dir()
except ImportError:
    AVAILABLE = False


def same(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            same(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            same(test, a, b)
    else:
        test.assertEqual(left, right)


class StochasticSyntheticFeatures:
    """RNG-dependent features make Python/NumPy/Torch restoration observable."""
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"id": str(index),
                "text_ids": torch.tensor([3, 4 + index % 3, 5]),
                "codes": torch.tensor([2, 3, 4 + index % 4, 5]),
                "spk_emb": torch.full((192,), index / 10 + random.random() / 100),
                "emo_vec": torch.randn(16) / 100 + float(np.random.random()) / 100}


@unittest.skipUnless(AVAILABLE, "Optional real Torch and pinned official source required; no fake model substitute")
class NativeResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_deterministic)

    def setUp(self):
        (ROOT / ".runtime/native-resume-tests").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime/native-resume-tests")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def new(self, *, identity=None, loss_fn=None, base_seed=111):
        # Both the frozen random tiny base and newly initialized adapters match.
        model = make_tiny_official_model(SOURCE, seed=base_seed, lora_dropout=.2)
        seed_everything(222)
        optimizer = torch.optim.AdamW(model.optimizer_groups(.02), lr=.003, foreach=False)
        scheduler = cosine_scheduler(optimizer, max_updates=6, warmup_updates=1)
        config = LoopConfig(max_updates=6, grad_accumulation=2, seed=71, save_interval=2,
                            validation_interval=2, max_validation_examples=2)
        dataset = StochasticSyntheticFeatures()
        trainer = StatefulTrainer(model, optimizer, scheduler, dataset, config,
                                  identity or {"official_commit": OFFICIAL_COMMIT, "fixture": "synthetic-real-GPT-v1"},
                                  loss_fn=loss_fn, validation_dataset=dataset)
        return trainer

    def test_actual_official_gpt_stop_resume_matches_uninterrupted(self):
        baseline = self.new()
        initial = baseline.model.trainable_state()
        result = baseline.run(self.root / "continuous")
        self.assertEqual(result["status"], "completed")
        expected = baseline.model.trainable_state()
        expected_optimizer = baseline.optimizer.state_dict()
        expected_scheduler = baseline.scheduler.state_dict()
        expected_rng = rng_state()
        self.assertTrue(any(not torch.equal(initial[name], expected[name]) for name in initial))
        self.assertTrue(any("lora_B" in name and torch.count_nonzero(value) for name, value in expected.items()))
        stopped = self.new()
        partial = stopped.run(self.root / "split", stop_after_updates=3)
        self.assertEqual(partial["status"], "stopped_at_optimizer_boundary")
        state = torch.load(partial["checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(state["step"], 3)
        self.assertEqual(state["sampler"], {"epoch": 1, "offset": 1, "micro_steps": 6, "dataset_length": 5})
        resumed = self.new()
        seed_everything(999)  # Must be replaced by the checkpoint's three RNGs.
        final = resumed.run(self.root / "split", resume=partial["checkpoint"])
        self.assertEqual(final["step"], 6)
        same(self, expected, resumed.model.trainable_state())
        same(self, expected_optimizer, resumed.optimizer.state_dict())
        same(self, expected_scheduler, resumed.scheduler.state_dict())
        same(self, expected_rng, rng_state())
        completed = torch.load(final["checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(completed["sampler"], {"epoch": 2, "offset": 2, "micro_steps": 12, "dataset_length": 5})
        self.assertEqual(completed["reason"], "complete")

    def test_actual_stop_file_saves_optimizer_boundary(self):
        trainer = self.new()
        output = self.root / "stop"
        def request(loop, _metrics):
            if loop.step == 2:
                (output / "STOP").touch()
        result = trainer.run(output, on_update=request)
        self.assertEqual(result["step"], 2)
        state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
        self.assertEqual(state["sampler"]["micro_steps"], 4)
        self.assertEqual(state["reason"], "cooperative_stop")
        with self.assertRaisesRegex(TrainingError, "STOP is already present"):
            self.new().run(output, resume=result["checkpoint"])

    def test_sigterm_only_requests_safe_boundary_save(self):
        trainer = self.new()
        stop = StopController()
        def request(loop, _metrics):
            if loop.step == 1:
                signal.raise_signal(signal.SIGTERM)
        previous = signal.getsignal(signal.SIGTERM)
        with stop.signals():
            result = trainer.run(self.root / "signal", stop=stop, on_update=request)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(result["step"], 1)
        self.assertEqual(stop.reason, "signal:SIGTERM")
        self.assertTrue(Path(result["checkpoint"]).is_file())

    def test_changed_identity_is_rejected_before_update(self):
        partial = self.new().run(self.root / "source", stop_after_updates=1)
        changed = self.new(identity={"official_commit": OFFICIAL_COMMIT, "fixture": "different-data"})
        with self.assertRaisesRegex(TrainingError, "identity changed"):
            changed.run(self.root / "changed", resume=partial["checkpoint"])
        self.assertEqual(changed.step, 0)

    def test_changed_frozen_base_is_rejected_even_with_same_user_identity(self):
        partial = self.new().run(self.root / "source", stop_after_updates=1)
        changed = self.new(base_seed=112)
        with self.assertRaisesRegex(TrainingError, "native base"):
            changed.run(self.root / "changed-base", resume=partial["checkpoint"])
        self.assertEqual(changed.step, 0)

    def test_stop_requested_before_resume_keeps_existing_checkpoint(self):
        partial = self.new().run(self.root / "source", stop_after_updates=1)
        checkpoint = Path(partial["checkpoint"])
        original = checkpoint.read_bytes()
        stop = StopController()
        stop.request("operator-review")
        result = self.new().run(self.root / "source", resume=checkpoint, stop=stop)
        self.assertEqual(result["updates_this_run"], 0)
        self.assertEqual(result["step"], 1)
        self.assertEqual(checkpoint.read_bytes(), original)

    def test_checkpoint_corruption_is_rejected(self):
        partial = self.new().run(self.root / "source", stop_after_updates=1)
        path = Path(partial["checkpoint"])
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(TrainingError, "SHA256"):
            self.new().run(self.root / "bad", resume=path)

    def test_detached_loss_cannot_emit_training_success(self):
        trainer = self.new(loss_fn=lambda _model, _batch: {"loss": torch.tensor(1.)})
        with self.assertRaisesRegex(TrainingError, "detached"):
            trainer.run(self.root / "detached")
        self.assertEqual(trainer.step, 0)
        self.assertFalse((self.root / "detached/checkpoints").exists())

    def test_preexisting_output_is_not_modified_before_failure(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(TrainingError, "empty"):
            self.new().run(output)
        self.assertEqual({p.name for p in output.iterdir()}, {"keep.txt"})

    def test_missing_resume_never_silently_restarts(self):
        trainer = self.new()
        with self.assertRaises(OSError):
            trainer.run(self.root / "missing", resume=self.root / "missing-step.pt")
        self.assertEqual(trainer.step, 0)
