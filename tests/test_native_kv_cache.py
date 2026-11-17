"""Observe real official cached/full-context generation; do not fix upstream here.

Tiny random CPU weights are an API/math check, not pretrained or audio quality.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import torch
    from hi_h200_native.model import make_tiny_official_model, import_official_model
    from hi_h200_native.synth import configure_kv_cache_experiment, SynthesisError


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), "Requires actual Torch and pinned official source")
class OfficialKVCacheExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def stock(self):
        native = make_tiny_official_model(OFFICIAL, seed=91)
        torch.manual_seed(91)
        stock = import_official_model(OFFICIAL)(**native.native_config["gpt"]).eval()
        stock.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
        return stock

    def test_default_retains_stock_and_experiment_only_changes_real_boolean(self):
        stock = self.stock()
        tts = SimpleNamespace(gpt=stock)  # Handle only; no fake synthesis constructor.
        before = {name:value.clone() for name,value in stock.state_dict().items()}
        default = configure_kv_cache_experiment(tts, OFFICIAL)
        self.assertTrue(default["enabled"])
        self.assertFalse(default["experimental"])
        experiment = configure_kv_cache_experiment(tts, OFFICIAL, no_kv_cache=True)
        self.assertFalse(stock.inference_model.kv_cache)
        self.assertTrue(experiment["experimental"])
        self.assertFalse(experiment["quality_improvement_verified"])
        self.assertFalse(experiment["upstream_positions_modified"])
        for name, value in stock.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        with self.assertRaises(SynthesisError):
            configure_kv_cache_experiment(tts, OFFICIAL)  # Does not silently reset prior experimental state.
        stock.inference_model.kv_cache = "true"
        with self.assertRaisesRegex(SynthesisError, "boolean"):
            configure_kv_cache_experiment(tts, OFFICIAL)
        stock.inference_model.kv_cache = True
        with self.assertRaisesRegex(SynthesisError, "selector"):
            configure_kv_cache_experiment(tts, OFFICIAL, no_kv_cache=1)
        with self.assertRaisesRegex(SynthesisError, "exact reviewed"):
            configure_kv_cache_experiment(tts, ROOT)

    def test_actual_stock_generate_reproduces_upstream_position_difference(self):
        stock = self.stock()
        speaker = stock.spk_emb_proj(torch.linspace(-.1,.1,192).reshape(1,192)).unsqueeze(1)
        condition = torch.cat((speaker+torch.linspace(-.05,.05,16).reshape(1,1,16), torch.zeros(1,2,16)),1)
        _,prefix,_ = stock.prepare_gpt_inputs(condition,torch.tensor([[3,4,5]]),torch.tensor([17]))
        wrapper = stock.inference_model
        wrapper.store_mel_emb(prefix)
        ids = torch.cat((torch.ones((1,prefix.shape[1]),dtype=torch.long),torch.tensor([[32]])),1)
        records = []
        for experimental in (False, True):
            configure_kv_cache_experiment(SimpleNamespace(gpt=stock), OFFICIAL, no_kv_cache=experimental)
            positions, calls = [], []
            def capture(module, args, kwargs):
                calls.append((kwargs["input_ids"].shape[1], kwargs["attention_mask"].shape[1],
                              kwargs.get("past_key_values") is not None))
            first = wrapper.register_forward_pre_hook(capture, with_kwargs=True)
            second = stock.mel_pos_embedding.emb.register_forward_pre_hook(lambda module,args:positions.append(args[0].tolist()))
            try:
                with torch.inference_mode():
                    output = wrapper.generate(ids, attention_mask=torch.ones_like(ids), bos_token_id=32,
                        pad_token_id=33,eos_token_id=33,min_new_tokens=3,max_new_tokens=3,do_sample=False,
                        num_beams=1,use_cache=True,return_dict_in_generate=True,output_logits=True)
            finally:
                first.remove()
                second.remove()
            records.append((positions,calls,output.logits))
        self.assertEqual(prefix.shape[1], 8)
        self.assertEqual(records[0][0], [[0],[2],[3]])
        self.assertEqual(records[1][0], [[0],[0,1],[0,1,2]])
        self.assertEqual(records[0][1], [(9,9,False),(1,10,True),(1,11,True)])
        self.assertEqual(records[1][1], [(9,9,False),(10,10,False),(11,11,False)])
        torch.testing.assert_close(records[0][2][0], records[1][2][0], rtol=1e-5, atol=1e-6)
        self.assertGreater((records[0][2][1]-records[1][2][1]).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
