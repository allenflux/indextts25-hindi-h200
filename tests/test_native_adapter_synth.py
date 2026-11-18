"""Real tiny official CPU generation/isolated adapter tests, never speech-quality evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hi_h200_native import adapter_synth as route
from hi_h200_native.assets import MODEL_REPO, MODEL_REVISION
from hi_h200_native import synth

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get('HI_H200_OFFICIAL_REPO', ROOT / '.runtime/external-official'))
AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None
if AVAILABLE:
    import torch
    from hi_h200_native.model import (PURE_LORA_PROFILE, Conv1DLoRA, NativeModelError,
        file_sha256, make_tiny_official_model, import_official_model)


class AdapterMetadataTests(unittest.TestCase):
    def test_import_and_help_do_not_import_torch(self):
        script = "import sys; from hi_h200_native.adapter_synth import main;\ntry: main(['--help'])\nexcept SystemExit as e: assert e.code == 0\nassert 'torch' not in sys.modules; assert 'indextts' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, timeout=20, text=True, capture_output=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--training-base-manifest', result.stdout)
        self.assertIn('--allow-cuda', result.stdout)
        self.assertNotIn('--merged-bundle', result.stdout)

    def test_other_language_fails_before_paths_or_adapter_are_read(self):
        args = SimpleNamespace(language='en')  # No adapter/path fields at all.
        with patch.object(synth, 'verify_source', side_effect=AssertionError('must not access source')), \
                self.assertRaisesRegex(route.AdapterSynthesisError, 'HI-only'):
            route.metadata_preflight(args)
        with self.assertRaisesRegex(route.AdapterSynthesisError, 'HI-only'):
            route.execute_once(args, {}, {})

    def test_training_manifest_retains_exact_identity_and_must_match_all_synth_hashes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as folder:
            root = Path(folder)
            files = {name: hashlib.sha256(name.encode()).hexdigest() for name in synth.CORE}
            manifest = root / 'original-train-manifest.json'
            content = {'repository':MODEL_REPO, 'snapshot_revision':MODEL_REVISION, 'files':files}
            manifest.write_text(json.dumps(content))
            assets = {'repository':MODEL_REPO, 'snapshot_revision':MODEL_REVISION, 'files':dict(files)}
            identity = route.training_base_identity(root, manifest, assets)
            self.assertEqual(identity['model_manifest_sha256'], synth.sha256(manifest))
            self.assertEqual(identity['model_files'], files)
            manifest.write_text(json.dumps(content, indent=2))
            self.assertNotEqual(identity['model_manifest_sha256'], route.training_base_identity(root, manifest, assets)['model_manifest_sha256'])
            assets['files']['gpt.pth'] = '0' * 64
            with self.assertRaisesRegex(route.AdapterSynthesisError, 'differs'):
                route.training_base_identity(root, manifest, assets)

    def test_optional_setup_cannot_bypass_original_execute_guards(self):
        called = []
        callback = lambda *args: called.append(True)
        with self.assertRaisesRegex(synth.SynthesisError, 'allow-cuda'):
            synth.execute_once(SimpleNamespace(allow_cuda=False), {}, {}, model_setup=callback)
        with self.assertRaisesRegex(synth.SynthesisError, 'Execution needs'):
            synth.execute_once(SimpleNamespace(allow_cuda=True), {},
                {'output':None, 'text':None, 'prompt':None}, model_setup=callback)
        self.assertEqual(called, [])

    def test_bad_adapter_digest_rejects_without_tensor_loader(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as folder:
            path = Path(folder) / 'not-a-checkpoint.pt'; path.write_bytes(b'only a hash fixture')
            with self.assertRaisesRegex(route.AdapterSynthesisError, 'SHA256'):
                route.checked_adapter(path, '0' * 64)


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), 'Requires real Torch/Transformers and fixed local official source')
class UnmergedOfficialGenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.runtime')
        self.root = Path(self.tmp.name)
        self.path = self.root / 'pure-adapter.pt'
        self.native = make_tiny_official_model(OFFICIAL, seed=41, lora_dropout=.2,
                                               training_profile=PURE_LORA_PROFILE)
        self.batch = {'text_ids':torch.tensor([3,4,5]), 'codes':torch.tensor([2,3,4,5]),
                      'spk_emb':torch.linspace(-.1,.1,192), 'emo_vec':torch.linspace(-.05,.05,16)}
        optimizer = torch.optim.AdamW(self.native.optimizer_groups(0), lr=.02)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            self.native(self.batch)['loss'].backward()
            optimizer.step()
        self.payload = {'schema_version':1, 'kind':'hi_h200_native_adapter', 'step':3,
            'base_identity':self.native.native_identity, 'adapter_spec':self.native.adapter_spec(),
            'trainable':self.native.trainable_state()}
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        torch.save(self.payload, self.path)
        return file_sha256(self.path)

    def fresh(self):
        return make_tiny_official_model(OFFICIAL, seed=41, lora_dropout=.2, training_profile=PURE_LORA_PROFILE)

    def original(self):
        torch.manual_seed(41)
        return import_official_model(OFFICIAL)(**self.native.native_config['gpt']).eval()

    def generate(self, model):
        speaker = model.spk_emb_proj(self.batch['spk_emb'].reshape(1,192)).unsqueeze(1)
        condition = torch.cat((speaker+self.batch['emo_vec'].reshape(1,1,16), torch.zeros(1,2,16)),1)
        _, prefix, _ = model.prepare_gpt_inputs(condition, self.batch['text_ids'].reshape(1,-1), torch.tensor([17]))
        wrapper = model.inference_model
        wrapper.store_mel_emb(prefix)
        ids = torch.cat((torch.ones((1,prefix.shape[1]),dtype=torch.long),torch.tensor([[32]])),1)
        with torch.inference_mode():
            output = wrapper.generate(ids, attention_mask=torch.ones_like(ids), bos_token_id=32,
                pad_token_id=33, eos_token_id=33, min_new_tokens=3, max_new_tokens=3,
                do_sample=False, num_beams=1, use_cache=True, return_dict_in_generate=True, output_logits=True)
        return output.sequences.clone(), tuple(value.clone() for value in output.logits)

    def test_real_unmerged_generate_uses_lora_and_original_is_unchanged_after_disposal(self):
        baseline = self.original()
        before = {name:value.clone() for name,value in baseline.state_dict().items()}
        baseline.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
        original_sequence, original_logits = self.generate(baseline)
        private = self.fresh()
        loaded = route.load_pure_into(private, self.path, self.save())
        self.assertEqual(loaded['step'],3)
        self.assertTrue(any(value.abs().sum().item() > 0 for name,value in private.trainable_state().items() if name.endswith('lora_B')))
        baseline_ptrs = {parameter.data_ptr() for parameter in baseline.parameters()}
        self.assertFalse(any(parameter.data_ptr() in baseline_ptrs for parameter in private.parameters()))
        tts = SimpleNamespace(gpt=self.original())  # Handle only; no mocked speech generation.
        receipt = route.attach_private_gpt(tts, private, device='cpu', dtype=torch.float32)
        self.assertTrue(receipt['unmerged'])
        self.assertIs(tts.gpt.inference_model.transformer, tts.gpt.gpt)
        self.assertIs(tts.gpt.inference_model.lm_head[1], tts.gpt.mel_head)
        self.assertIsInstance(tts.gpt.gpt.h[0].attn.c_attn, Conv1DLoRA)
        calls = []
        hook = tts.gpt.gpt.h[0].attn.c_attn.register_forward_hook(lambda *args:calls.append(True))
        try:
            adapted_sequence, adapted_logits = self.generate(tts.gpt)
        finally:
            hook.remove()
        self.assertGreater(len(calls),0)
        self.assertEqual(len(adapted_logits),3)
        self.assertTrue(all(torch.isfinite(value).all().item() for value in adapted_logits))
        self.assertGreater((adapted_logits[0]-original_logits[0]).abs().max().item(),1e-6)
        del tts, private  # Process/instance disposal, not an in-place unload promise.
        restored_sequence, restored_logits = self.generate(baseline)
        torch.testing.assert_close(original_sequence,restored_sequence,rtol=0,atol=0)
        for left,right in zip(original_logits,restored_logits): torch.testing.assert_close(left,right,rtol=0,atol=0)
        for name,value in before.items(): self.assertTrue(torch.equal(value,baseline.state_dict()[name]),name)
        self.assertFalse(any(isinstance(module,Conv1DLoRA) for module in baseline.modules()))

    def test_wrong_identity_missing_tensor_and_legacy_spec_never_partially_mutate(self):
        for mode in ('identity','missing','legacy','nonfinite'):
            with self.subTest(mode=mode):
                fresh = self.fresh()
                before = {name:value.clone() for name,value in fresh.state_dict().items()}
                payload = copy.deepcopy(self.payload)
                if mode == 'identity': payload['base_identity'] = {'wrong':'base'}
                if mode == 'missing': payload['trainable'].pop(next(iter(payload['trainable'])))
                if mode == 'legacy': payload['adapter_spec']['format'] = 'hi_h200_native_batch1_v1'
                if mode == 'nonfinite': list(payload['trainable'].values())[-1].flatten()[0] = float('nan')
                torch.save(payload,self.path)
                with self.assertRaises((route.AdapterSynthesisError,NativeModelError)):
                    route.load_pure_into(fresh,self.path,file_sha256(self.path))
                for name,value in before.items(): self.assertTrue(torch.equal(value,fresh.state_dict()[name]),name)
                self.assertFalse(hasattr(fresh.base,'inference_model'))

    def test_attach_rejects_shared_original_or_already_initialized_private_model(self):
        fresh = self.fresh()
        route.load_pure_into(fresh,self.path,self.save())
        with self.assertRaisesRegex(route.AdapterSynthesisError,'independent'):
            route.attach_private_gpt(SimpleNamespace(gpt=fresh.base),fresh,device='cpu',dtype=torch.float32)
        tts = SimpleNamespace(gpt=self.original())
        route.attach_private_gpt(tts,fresh,device='cpu',dtype=torch.float32)
        with self.assertRaisesRegex(route.AdapterSynthesisError,'fresh'):
            route.attach_private_gpt(SimpleNamespace(gpt=self.original()),fresh,device='cpu',dtype=torch.float32)


if __name__ == '__main__':
    unittest.main()
