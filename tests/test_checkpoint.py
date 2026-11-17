"""Synthetic CPU resume-gate tests; no real Torch/backend save/restore is asserted."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import check_checkpoint as checkpoint
from scripts.check_backend import PreflightError, read_json

RUNTIME = Path(__file__).resolve().parents[1] / ".runtime" / "checkpoint-tests"


class SyntheticTensor:
    def __init__(self, shape=(2,2), finite=True, scalar=25):
        self.shape = shape
        self.finite = finite
        self.scalar = scalar
        self.dtype = "torch.float32"
    def numel(self):
        return 1 if not self.shape else math.prod(self.shape)
    def item(self):
        return self.scalar
    def is_floating_point(self):
        return True


def tensor_metadata(value):
    if not isinstance(value,SyntheticTensor):
        return None
    return {"shape":value.shape,"finite":value.finite,"floating":True,"dtype":value.dtype,"scalar":value.scalar if value.shape in ((),(1,)) else None}


def fixture():
    config = read_json(checkpoint.ROOT / "configs/hindi-h200.example.json")
    names = ["lang_embedding.weight","text_embedding.weight","mel_head.weight","text_head.weight",
             "gpt.h.0.attn.c_attn.lora_A","gpt.h.0.attn.c_attn.lora_B"]
    state = {"trainable":{name:SyntheticTensor() for name in names},
             "optimizer":{"state":{index:{"step":SyntheticTensor(()),"exp_avg":SyntheticTensor(),"exp_avg_sq":SyntheticTensor()} for index in range(len(names))},
                          "param_groups":[{"params":list(range(len(names))),"lr":5e-5,"name":"synthetic-test-group"}]},
             "scheduler":{"last_epoch":25,"base_lrs":[5e-5],"lr_lambdas":[None]},
             "scaler":None,"step":25,"epoch":0,"data_batches":25*config["grad_accumulation"],
             "baselines":{"hi":1.0},"spec":checkpoint.expected_spec(config)}
    return config,state


class CheckpointTests(unittest.TestCase):
    def test_synthetic_valid_state_explicitly_not_backend_roundtrip(self):
        config,state=fixture()
        result=checkpoint.validate_state(state,config,tensor_metadata)
        self.assertEqual(result["step"],25)
        self.assertFalse(result["actual_backend_roundtrip_verified"])
        self.assertFalse(result["optimizer_name_mapping_verified"])
        self.assertFalse(result["rng_state_restored"])

    def test_completed_or_final_cursor_cannot_extend_schedule(self):
        for field in ("epoch","step"):
            config,state=fixture()
            state[field]=config["epochs" if field=="epoch" else "max_steps"]
            with self.subTest(field=field),self.assertRaises(PreflightError):
                checkpoint.validate_state(state,config,tensor_metadata)

    def test_counter_types_and_accumulation_cursor_rejected(self):
        for field,value in (("step",True),("epoch",-1),("data_batches",199)):
            config,state=fixture()
            state[field]=value
            with self.subTest(field=field),self.assertRaises(PreflightError):
                checkpoint.validate_state(state,config,tensor_metadata)

    def test_missing_state_or_changed_spec_rejected(self):
        config,state=fixture()
        del state["optimizer"]
        with self.assertRaises(PreflightError):
            checkpoint.validate_state(state,config,tensor_metadata)
        config,state=fixture()
        state["spec"]["lora_rank"]=32
        with self.assertRaises(PreflightError):
            checkpoint.validate_state(state,config,tensor_metadata)

    def test_nonfinite_weight_optimizer_and_scalar_rejected(self):
        for target in ("weight","optimizer","scalar"):
            config,state=fixture()
            if target=="weight":
                state["trainable"]["mel_head.weight"].finite=False
            elif target=="optimizer":
                state["optimizer"]["state"][0]["exp_avg"].finite=False
            else:
                state["baselines"]["hi"]=float("nan")
            with self.subTest(target=target),self.assertRaises(PreflightError):
                checkpoint.validate_state(state,config,tensor_metadata)

    def test_optimizer_ids_and_scheduler_must_match_saved_progress(self):
        config,state=fixture()
        state["optimizer"]["param_groups"][0]["params"][0]=1
        with self.assertRaises(PreflightError):
            checkpoint.validate_state(state,config,tensor_metadata)
        config,state=fixture()
        state["scheduler"]["last_epoch"]=24
        with self.assertRaises(PreflightError):
            checkpoint.validate_state(state,config,tensor_metadata)

    def test_optimizer_step_cannot_be_a_string_or_nonscalar_tensor(self):
        for value in ("25",SyntheticTensor((2,)),SyntheticTensor((),scalar=26)):
            config,state=fixture()
            state["optimizer"]["state"][0]["step"]=value
            with self.subTest(value=repr(value)),self.assertRaises(PreflightError):
                checkpoint.validate_state(state,config,tensor_metadata)

    def test_safe_loader_flags_are_required_without_fallback(self):
        config,state=fixture()
        RUNTIME.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
            path=Path(directory)/"step25.pt"
            path.write_bytes(b"synthetic-no-actual-torch-pickle")
            fake=SimpleNamespace(load=mock.Mock(return_value=state),
                                 is_tensor=lambda value:isinstance(value,SyntheticTensor),
                                 isfinite=lambda value:SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:value.finite)))
            with mock.patch.dict(sys.modules,{"torch":fake}):
                with self.assertRaises(PreflightError):
                    checkpoint.load_trusted_checkpoint(path,False,config)
                fake.load.assert_not_called()
                result=checkpoint.load_trusted_checkpoint(path,True,config)
                self.assertFalse(result["actual_backend_roundtrip_verified"])
                fake.load.assert_called_once_with(path.resolve(),map_location="cpu",weights_only=True)
                fake.load.reset_mock(side_effect=True)
                fake.load.side_effect=RuntimeError("unsupported global")
                with self.assertRaisesRegex(PreflightError,"no unsafe-pickle fallback"):
                    checkpoint.load_trusted_checkpoint(path,True,config)
                self.assertEqual(fake.load.call_count,1)

    def test_inventory_detects_changed_config_or_data(self):
        RUNTIME.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
            path=Path(directory)/"config.json"
            path.write_text('{"max_steps":100}')
            inventory={}
            checkpoint.add_inventory(inventory,path,"config")
            checkpoint.verify_inventory(list(inventory.values()))
            path.write_text('{"max_steps":200}')
            with self.assertRaisesRegex(PreflightError,"changed or moved"):
                checkpoint.verify_inventory(list(inventory.values()))


if __name__=="__main__":
    unittest.main()
