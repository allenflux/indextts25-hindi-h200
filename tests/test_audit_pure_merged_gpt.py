"""Real Torch mmap/FP32 checks on synthetic tiny official states, never real weights.

Only internal fixture Pins differ; the production CLI has no base/adapter override.
The reference candidate is produced by the existing exporter, not by this auditor.
"""
from dataclasses import replace
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tested_pure_merge_auditor', ROOT / 'scripts/audit_pure_merged_gpt.py')
a = importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name] = a;SPEC.loader.exec_module(a)
OFFICIAL = Path(os.environ.get('HI_H200_OFFICIAL_REPO', ROOT / '.runtime/external-official'))
AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None and OFFICIAL.is_dir()
if AVAILABLE:
    import torch
    from hi_h200_native.model import make_tiny_official_model, PURE_LORA_PROFILE, preserved_state_digest
    from hi_h200_native.export import merge_official_state, NON_ENDORSEMENT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path, value):
    Path(path).write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')


class ImportAndCLI(unittest.TestCase):
    def test_help_is_metadata_only_and_does_not_import_torch(self):
        script = "import runpy,sys;sys.argv=['audit','--help'];\ntry:runpy.run_path(" + repr(str(ROOT / 'scripts/audit_pure_merged_gpt.py')) + ",run_name='__main__')\nexcept SystemExit as e:assert e.code==0\nassert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-c', script], text=True, capture_output=True, timeout=20, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--candidate-sha256', result.stdout)
        self.assertIn('--execute-cpu', result.stdout)
        self.assertEqual(a.PRODUCTION.adapter_sha256, '15c080715b3b50bf38b33a70f196a9c9e987446e0a6bb87b2e0c8b7835f57dbd')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            a.main(['--base-sha256', '0' * 64])


@unittest.skipUnless(AVAILABLE, 'Optional real Torch and pinned official tiny class; no substitute')
class RealTinyAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads();torch.set_num_threads(1)
        model = make_tiny_official_model(OFFICIAL, seed=121, training_profile=PURE_LORA_PROFILE)
        with torch.no_grad():
            # Known signed-zero sentinel in the synthetic original, before freezing.
            model.base.text_head.bias[0] = 0.
        model.native_preserved_sha256 = preserved_state_digest(model)
        cls.base = {}
        targets = model.lora_targets
        for name, tensor in model.base.state_dict().items():
            if name.endswith(('.lora_A', '.lora_B')):
                continue
            key = name
            for target in targets:
                if key == target + '.base.weight':key = target + '.weight'
                if key == target + '.base.bias':key = target + '.bias'
            cls.base[key] = tensor.clone()
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                if name.endswith('.lora_B'):
                    tensor.copy_(torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape) / 1000 + .01)
        cls.candidate = merge_official_state(model)
        cls.ab = model.trainable_state()
        cls.spec = model.adapter_spec()
        cls.frozen = model.native_preserved_sha256

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='pure-merge-audit-', dir=ROOT / '.runtime')
        self.work = Path(self.tmp.name).resolve()
        for name in ('original', 'adapter', 'bundle', 'reports', 'code'):(self.work / name).mkdir()
        self.root_patch = patch.object(a, 'ROOT', self.work / 'code');self.root_patch.start()
        self.base_path = self.work / 'original/gpt.pth';torch.save(self.base, self.base_path)
        self.adapter_path = self.work / 'adapter/adapter.pt'
        self.bundle = self.work / 'bundle'
        self.identity = {'official_commit': a.OFFICIAL, 'model_repository': a.REPOSITORY, 'model_snapshot_revision': a.REVISION,
            'model_manifest_sha256': 'b' * 64, 'model_files': {'gpt.pth': sha(self.base_path), 'config.yaml': 'c' * 64, a.TOKENIZER: 'd' * 64},
            'config_sha256': 'c' * 64, 'tokenizer_sha256': 'd' * 64}
        self.payload = {'schema_version': 1, 'kind': 'hi_h200_native_adapter', 'step': 100,
            'base_identity': self.identity, 'adapter_spec': self.spec, 'trainable': self.ab}
        torch.save(self.payload, self.adapter_path)
        self.pins = a.Pins(self.base_path.stat().st_size, sha(self.base_path), self.adapter_path.stat().st_size, sha(self.adapter_path),
            'b' * 64, 'c' * 64, 'd' * 64, layers=1, model_dim=16, rank=2, alpha=4., text_loss_weight=.1,
            tensor_count=len(self.base), model_file_count=3, scope='tiny_random_official_tensor_fixture_not_pretrained')
        self.provenance = {'schema_version': 1, 'kind': 'hi_h200_native_merged_official_gpt',
            'scope': 'Hindi-intended merged official GPT replacement; acceptance still required', 'official_commit': a.OFFICIAL,
            'base_identity': self.identity, 'adapter_sha256': self.pins.adapter_sha256,
            'native_implementation_files': {'model.py': 'e' * 64, 'export.py': 'f' * 64},
            'adapter_spec_sha256': hashlib.sha256(a.canonical(self.spec)).hexdigest(), 'optimizer_updates': 100,
            'gpt_sha256': '', 'preserved_base_sha256': self.frozen,
            'merge_semantics': 'evaluation/inference with native LoRA dropout disabled', 'official_tensor_count': len(self.base),
            'checkpoint_container': 'model', 'official_key_shape_check': 'actual official meta constructor, strict=True',
            'full_pretrained_forward_parity_verified': False, 'synthesis_verified': False, 'old_language_behavior_guaranteed_unchanged': False}
        for name in ('LICENSE', 'LICENSE_ZH.txt'):(self.bundle / name).write_bytes((OFFICIAL / name).read_bytes())
        (self.bundle / 'MODEL_LICENSE').write_bytes((OFFICIAL / 'LICENSE').read_bytes()[:-1])
        (self.bundle / 'NON_ENDORSEMENT.txt').write_text(NON_ENDORSEMENT)
        (self.bundle / 'README.md').write_text('Synthetic numerical fixture; not a pretrained model or quality evidence.\n')
        self.args = SimpleNamespace(workspace_root=self.work, base=self.base_path, adapter=self.adapter_path,
            merged_bundle=self.bundle, output=self.work / 'reports/audit.json', protect_root=[], _deadline=time.monotonic() + 60)
        self.rebind(self.candidate)

    def tearDown(self):
        self.root_patch.stop();self.tmp.cleanup()

    def rebind(self, state, *, container=True):
        torch.save({'model': state} if container else state, self.bundle / 'gpt.pth')
        self.args.candidate_sha256 = sha(self.bundle / 'gpt.pth');self.args.candidate_bytes = (self.bundle / 'gpt.pth').stat().st_size
        self.provenance['gpt_sha256'] = self.args.candidate_sha256
        put_json(self.bundle / 'provenance.json', self.provenance)
        self.args.provenance_sha256 = sha(self.bundle / 'provenance.json')
        put_json(self.bundle / 'COMPLETE.json', {'schema_version': 1, 'files': {name: sha(self.bundle / name) for name in a.BUNDLE_FILES}})
        self.args.complete_sha256 = sha(self.bundle / 'COMPLETE.json')

    def prepared(self):
        self.args._deadline = time.monotonic() + 60
        return a.inspect(self.args, _pins=self.pins)

    def execute(self, prepared=None):
        return a.audit(prepared or self.prepared(), deadline=time.monotonic() + 60, chunk_bytes=256, minimum_ram_bytes=0)

    def test_metadata_never_hashes_or_loads_weights_and_reports_pending(self):
        original = a.digest
        def bounded_metadata(path, deadline):
            if path in (self.base_path, self.adapter_path, self.bundle / 'gpt.pth'):
                raise AssertionError('Default metadata must not read weight bytes')
            return original(path, deadline)
        with patch.object(a, 'digest', side_effect=bounded_metadata), patch.object(torch, 'load', side_effect=AssertionError('no tensor loading')):
            prepared = self.prepared()
        self.assertFalse(prepared['report']['all_input_hashes_verified'])
        self.assertFalse(prepared['report']['cpu_tensor_audit_executed'])
        self.assertFalse(self.args.output.exists())

    def test_actual_mmap_exporter_merge_matches_independent_formula_and_preserved_digest(self):
        original = torch.load
        with patch.object(torch, 'load', wraps=original) as loads:
            report = self.execute()
        self.assertTrue(report['audit_passed'])
        self.assertEqual(report['allowed_target_tensors'], 2)
        self.assertEqual(report['changed_target_tensors'], 2)
        self.assertTrue(report['all_target_independent_fp32_formula_bitwise_equal'])
        self.assertTrue(report['non_target_all_bitwise_equal'])
        self.assertEqual(report['preserved_state_digest'], self.frozen)
        for call in loads.call_args_list:
            self.assertEqual(call.kwargs, {'map_location': 'cpu', 'weights_only': True, 'mmap': True})
        self.assertNotIn(str(self.work), json.dumps(report))
        self.assertNotIn('tensor(', json.dumps(report))

    def test_base_as_candidate_wrong_scale_and_non_target_signed_zero_are_rejected(self):
        cases = [dict(self.base)]
        wrong = {name: tensor.clone() for name, tensor in self.candidate.items()}
        name = a.modules(self.pins)[0] + '.weight'
        wrong[name] = self.base[name] + (wrong[name] - self.base[name]) * .5;cases.append(wrong)
        other = {name: tensor.clone() for name, tensor in self.candidate.items()}
        other['text_head.bias'][0] = -0.;cases.append(other)
        for state in cases:
            with self.subTest(case=len(cases)):
                self.rebind(state);self.assertFalse(self.execute()['audit_passed'])
        self.assertFalse(self.execute()['non_target_all_bitwise_equal'])

    def test_missing_key_shape_dtype_nan_layout_and_wrong_container_fail_closed(self):
        for mode in ('missing', 'shape', 'dtype', 'nan', 'layout', 'container'):
            state = {name: tensor.clone() for name, tensor in self.candidate.items()}
            if mode == 'missing':state.pop('text_head.bias')
            elif mode == 'shape':state['text_head.bias'] = torch.zeros(1)
            elif mode == 'dtype':state['text_head.bias'] = state['text_head.bias'].double()
            elif mode == 'nan':state['text_head.bias'][0] = float('nan')
            elif mode == 'layout':state['text_embedding.weight'] = state['text_embedding.weight'].T
            self.rebind(state, container=mode != 'container')
            with self.subTest(mode=mode), self.assertRaises(a.AuditError):self.execute()

    def test_adapter_profile_inventory_and_provenance_digest_are_not_self_authenticating(self):
        prepared = self.prepared()
        for mode in ('legacy', 'extra', 'missing', 'nonfinite'):
            payload = copy.deepcopy(self.payload)
            if mode == 'legacy':payload['adapter_spec']['heads'] = ['text_head', 'mel_head']
            elif mode == 'extra':payload['trainable']['base.text_head.bias'] = torch.zeros(1)
            elif mode == 'missing':payload['trainable'].pop(next(iter(payload['trainable'])))
            else:next(iter(payload['trainable'].values())).flatten()[0] = float('nan')
            with self.subTest(mode=mode), self.assertRaises(a.AuditError):
                a.compare(prepared, self.base, self.candidate, payload, torch, 256, time.monotonic() + 10)
        prepared['provenance'] = dict(prepared['provenance'], preserved_base_sha256='0' * 64)
        result = a.compare(prepared, self.base, self.candidate, self.payload, torch, 256, time.monotonic() + 10)
        self.assertFalse(result['audit_passed']);self.assertFalse(result['provenance_preserved_digest_matches'])

    def test_metadata_sha_license_extra_file_and_output_guards(self):
        self.args.candidate_sha256 = '0' * 64
        with self.assertRaises(a.AuditError):self.prepared()
        self.rebind(self.candidate)
        self.args.complete_sha256 = '0' * 64
        with self.assertRaises(a.AuditError):self.prepared()
        self.rebind(self.candidate)
        (self.bundle / 'unexpected.log').write_text('not a bundle member')
        with self.assertRaises(a.AuditError):self.prepared()
        (self.bundle / 'unexpected.log').unlink()
        old = (self.bundle / 'LICENSE_ZH.txt').read_bytes();(self.bundle / 'LICENSE_ZH.txt').unlink()
        with self.assertRaises((a.AuditError, OSError)):self.prepared()
        (self.bundle / 'LICENSE_ZH.txt').write_bytes(old)
        for output in (self.work / 'code/new.json', self.bundle / 'new.json', self.base_path.parent / 'new.json', self.adapter_path.parent / 'new.json'):
            self.args.output = output
            with self.subTest(output=output.name), self.assertRaises(a.AuditError):self.prepared()
        self.args.output = self.work / 'reports/audit.json';self.args.output.write_text('{}')
        with self.assertRaises(a.AuditError):self.prepared()
        self.args.output.unlink()
        link = self.work / 'linked-base.pth';link.symlink_to(self.base_path);self.args.base = link
        with self.assertRaises(a.AuditError):self.prepared()

    def test_post_mmap_mutation_ram_deadline_and_production_cli_cannot_use_fixture(self):
        prepared = self.prepared();original = a.compare
        def mutate(*args):
            result = original(*args)
            with self.adapter_path.open('ab') as stream:stream.write(b'changed')
            return result
        with patch.object(a, 'compare', side_effect=mutate), self.assertRaises(a.AuditError):self.execute(prepared)
        self.args._deadline = time.monotonic() - 1
        with self.assertRaises(a.AuditError):a.inspect(self.args, _pins=self.pins)
        self.payload = dict(self.payload);torch.save(self.payload, self.adapter_path)
        self.pins = replace(self.pins, adapter_bytes=self.adapter_path.stat().st_size, adapter_sha256=sha(self.adapter_path))
        self.provenance['adapter_sha256'] = self.pins.adapter_sha256;self.rebind(self.candidate)
        with patch.object(a, 'available_ram', return_value=None), self.assertRaises(a.AuditError):
            a.audit(self.prepared(), deadline=time.monotonic() + 20, chunk_bytes=256)
        argv = ['--workspace-root', str(self.work), '--base', str(self.base_path), '--adapter', str(self.adapter_path),
            '--merged-bundle', str(self.bundle), '--candidate-sha256', self.args.candidate_sha256,
            '--candidate-bytes', str(self.args.candidate_bytes), '--complete-sha256', self.args.complete_sha256,
            '--provenance-sha256', self.args.provenance_sha256, '--output', str(self.args.output), '--execute-cpu']
        with patch.object(torch, 'load', side_effect=AssertionError('CLI must reject fixture before Torch load')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(a.main(argv), 2)
        self.assertFalse(self.args.output.exists())

    def test_report_publication_is_exclusive(self):
        target = self.work / 'reports/out.json'
        a.publish(target, {'status': 'fixture'})
        with self.assertRaises(FileExistsError):a.publish(target, {'status': 'must not overwrite'})
        self.assertEqual(json.loads(target.read_text()), {'status': 'fixture'})


if __name__ == '__main__':unittest.main()
