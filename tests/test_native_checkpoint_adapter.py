"""Native real training checkpoint -> standard adapter CPU integration tests."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("HI_H200_OFFICIAL_REPO", ROOT / ".runtime/external-official"))
from hi_h200_native.checkpoint_adapter import digest, destination, preflight
AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import torch
    from hi_h200_native.engine import StatefulTrainer, LoopConfig, cosine_scheduler
    from hi_h200_native.model import make_tiny_official_model, load_native_adapter, OFFICIAL_COMMIT
    from hi_h200_native.checkpoint_adapter import validate_checkpoint, publish_adapter


class CheckpointMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_rejects_existing_protected_and_outside_workspace(self):
        protected = self.root / "run"
        protected.mkdir()
        for path in (protected, protected / "adapter", self.root.parent / "outside"):
            with self.assertRaises(ValueError):
                destination(path, self.root, (protected,))
        self.assertEqual(destination(self.root / "new", self.root, (protected,)), self.root / "new")
        for path in (ROOT / ".git/native-adapter-must-not-write", ROOT / "native-adapter-must-not-write"):
            with self.assertRaisesRegex(ValueError,"protected"):
                destination(path, ROOT.parent, (ROOT,))

    @unittest.skipUnless(OFFICIAL.is_dir(), "Pinned local official source needed, no weights")
    def test_default_metadata_cli_imports_no_torch_and_rejects_bad_run_sha(self):
        # Metadata-only fixture deliberately is not a tensor checkpoint/model.
        datasets = {}
        for split in ("train","val"):
            folder = self.root / split
            folder.mkdir()
            options = {"schema_version":1,"split":split,"official_repo":str(OFFICIAL),
                       "model_dir":str(folder),"model_manifest":str(folder / "model-manifest.json"),
                       "audio_root":str(folder),"prepared_dir":str(folder),"feature_root":str(folder),
                       "feature_manifest":str(folder / "manifest.jsonl"),"provenance":str(folder / "provenance.json")}
            for name in ("model-manifest.json","manifest.jsonl","provenance.json"):
                (folder / name).write_text("{}")
            config = folder / "dataset.json"
            config.write_text(json.dumps(options))
            datasets[split] = str(config)
        loop = dict(max_updates=1000,grad_accumulation=1,seed=71,device="cpu",precision="fp32",grad_clip=1.,save_interval=100,validation_interval=100,max_validation_examples=8)
        run = self.root / "run"
        (run / "checkpoints").mkdir(parents=True)
        config = {"schema_version":1,"train_dataset":datasets["train"],"val_dataset":datasets["val"],
                  "output_dir":str(run),"loop":loop,"optimizer":{"learning_rate":.003,"weight_decay":.02,"warmup_updates":1,"min_lr_ratio":.1},
                  "lora":{"rank":2,"alpha":4.,"dropout":.2,"text_loss_weight":.1}}
        config_path = self.root / "train.json"
        config_path.write_text(json.dumps(config))
        identity = {"config_sha256":digest(config_path),"resolved_config":config,"adapter_spec":{},
                    "datasets":{key:{"official_commit":"d9e41aac89fd00b3d71497fddb287b7f24613712"} for key in ("train","val")},
                    "native_code_sha256":{name:digest(ROOT / "src/hi_h200_native" / name) for name in ("model.py","data.py","engine.py","train.py")}}
        run_manifest = self.root / "run.native-run.json"
        run_manifest.write_text(json.dumps({"schema_version":1,"kind":"hi_h200_native_run","identity":identity,"limitations":[]}))
        checkpoint = run / "checkpoints/step00000100.pt"
        checkpoint.write_bytes(b"metadata-only-not-torch-data")
        checkpoint.with_suffix(".pt.sha256.json").write_text(json.dumps({"schema_version":1,"step":100,"sha256":digest(checkpoint)}))
        planned_output = ROOT.parent / ("checkpoint-preflight-" + self.root.name)
        arguments = ["--workspace-root",str(ROOT.parent),"--config",str(config_path),"--run-manifest",str(run_manifest),
                     "--run-manifest-sha256",digest(run_manifest),"--checkpoint",str(checkpoint),"--expected-step","100",
                     "--selection","operator selected lowest recorded validation loss","--output-dir",str(planned_output)]
        script = "import sys; from hi_h200_native.checkpoint_adapter import main; result=main("+repr(arguments)+"); assert result==0; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable,"-c",script],cwd=ROOT,text=True,capture_output=True,timeout=30,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(ROOT / "src")))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)["checkpoint_tensor_validation"])
        self.assertFalse(planned_output.exists())
        from types import SimpleNamespace
        parsed = dict(workspace_root=ROOT.parent,config=config_path,run_manifest=run_manifest,run_manifest_sha256="0"*64,
                      checkpoint=checkpoint,expected_step=100,selection="selected",output_dir=planned_output)
        with self.assertRaisesRegex(ValueError,"run-manifest SHA256"):
            preflight(SimpleNamespace(**parsed))
        parsed["run_manifest_sha256"] = digest(run_manifest)
        original_config, original_manifest = config_path.read_bytes(), run_manifest.read_bytes()
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime") as outside:
            outside_config = Path(outside) / "dataset.json"
            outside_config.write_text("{}")
            config["train_dataset"] = str(outside_config)
            config_path.write_text(json.dumps(config))
            identity["config_sha256"] = digest(config_path)
            identity["resolved_config"] = config
            run_manifest.write_text(json.dumps({"schema_version":1,"kind":"hi_h200_native_run","identity":identity,"limitations":[]}))
            parsed.update(workspace_root=self.root,run_manifest_sha256=digest(run_manifest))
            with self.assertRaisesRegex(ValueError,"dataset config files"):
                preflight(SimpleNamespace(**parsed))
        config_path.write_bytes(original_config)
        run_manifest.write_bytes(original_manifest)
        parsed.update(workspace_root=ROOT.parent,run_manifest_sha256=digest(run_manifest))
        checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError,"sidecar"):
            preflight(SimpleNamespace(**parsed))


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), "Requires real Torch and pinned official UnifiedVoice")
class ActualCheckpointAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_training_checkpoint_adapter_fresh_load_matches_logits(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        model = make_tiny_official_model(OFFICIAL,seed=77,lora_dropout=.2)
        batch = {"text_ids":torch.tensor([3,4,5]),"codes":torch.tensor([2,3,4,5]),
                 "spk_emb":torch.linspace(-.1,.1,192),"emo_vec":torch.linspace(-.05,.05,16)}
        config = LoopConfig(max_updates=3,save_interval=1,validation_interval=1,max_validation_examples=1)
        options = {"loop":asdict(config),"optimizer":{"learning_rate":.003,"weight_decay":.02,"warmup_updates":1,"min_lr_ratio":.1}}
        identity = {"fixture":"actual-official-tiny-CPU-not-pretrained","adapter_spec":model.adapter_spec()}
        optimizer = torch.optim.AdamW(model.optimizer_groups(.02),lr=.003,foreach=False)
        scheduler = cosine_scheduler(optimizer,max_updates=3,warmup_updates=1)
        trainer = StatefulTrainer(model,optimizer,scheduler,[batch],config,identity)
        result = trainer.run(self.root / "actual-run",stop_after_updates=2)
        checkpoint = Path(result["checkpoint"])
        state = torch.load(checkpoint,map_location="cpu",weights_only=True)
        self.assertEqual(state["step"],2)
        fresh = make_tiny_official_model(OFFICIAL,seed=77,lora_dropout=.2)
        payload = validate_checkpoint(state,fresh,identity,options,2,1)
        self.assertEqual(set(payload),{"schema_version","kind","step","base_identity","adapter_spec","trainable"})
        output = self.root / "selected"
        report = publish_adapter(payload,fresh,output,{"source_checkpoint_sha256":digest(checkpoint),"selected_step":2,"selection":"CPU fixture"})
        self.assertTrue(report["adapter_load_verified"])
        self.assertFalse(report["resume_state_restored"])
        self.assertEqual(json.loads((output / "adapter.pt.sha256.json").read_text())["sha256"],digest(output / "adapter.pt"))
        reloaded = make_tiny_official_model(OFFICIAL,seed=77,lora_dropout=.2)
        load_native_adapter(reloaded,output / "adapter.pt",report["adapter_sha256"])
        model.eval(); reloaded.eval()
        with torch.inference_mode():
            left,right = model(batch),reloaded(batch)
        for key in ("loss","text_logits","code_logits"):
            torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        with self.assertRaises(FileExistsError):
            publish_adapter(payload,fresh,output,{})
        for field,value in (("identity",{"changed":True}),("adapter_spec",{}),("model_identity",{})):
            bad = dict(state,**{field:value})
            with self.assertRaisesRegex(ValueError,"identity"):
                validate_checkpoint(bad,fresh,identity,options,2,1)
        partial = dict(state); partial.pop("optimizer")
        with self.assertRaisesRegex(ValueError,"partial"):
            validate_checkpoint(partial,fresh,identity,options,2,1)
        bad = dict(state,trainable=dict(state["trainable"]))
        bad["trainable"].pop("base.mel_head.weight")
        with self.assertRaises(ValueError):
            validate_checkpoint(bad,fresh,identity,options,2,1)
        bad = dict(state,trainable={key:value.clone() for key,value in state["trainable"].items()})
        next(iter(bad["trainable"].values())).reshape(-1)[0] = float("nan")
        with self.assertRaises(ValueError):
            validate_checkpoint(bad,fresh,identity,options,2,1)


if __name__ == "__main__":
    unittest.main()
