"""CPU contracts only: mock node calls never count as a real T8/GPU probe."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_t8_nodes as probe


class ProbeTests(unittest.TestCase):
    def test_real_node_call_contract_preserves_original_sampling_and_no_retry(self):
        calls = []
        handle = SimpleNamespace(use_bf16=True, use_fp16=False, use_cuda_kernel=False,
            use_torch_compile=False, use_accel=False, use_deepspeed=False,
            reference_cache_dir=None, model_dir=ROOT)
        audio = {'actual': 'contract sentinel, not generated audio'}
        emotion, sampling = object(), object()
        def loader(**values):
            calls.append(('loader', values)); return (handle, 'loader-info')
        def emo(**values):
            calls.append(('emotion', values)); return (emotion, 'emotion-info')
        def sample(**values):
            calls.append(('sampling', values)); return (sampling, 'sampling-info')
        def generate(**values):
            calls.append(('generate', values)); return (audio, 'generation-info', [audio])
        nodes = SimpleNamespace(**{name: SimpleNamespace(execute=fn) for name, fn in zip(
            probe.CLASSES, (loader, emo, sample, generate))})
        params = probe.node_parameters(ROOT, 'नमस्ते।', 'HI')
        state = {'four_node_calls_started':False, 'generate_node_call_attempted':False}
        self.assertEqual(probe.execute_nodes(nodes, params, audio, state=state), (audio, 'generation-info'))
        self.assertTrue(state['four_node_calls_started'])
        self.assertTrue(state['generate_node_call_attempted'])
        self.assertEqual([c[0] for c in calls], ['loader', 'emotion', 'sampling', 'generate'])
        self.assertEqual(calls[2][1], probe.SAMPLING)
        self.assertEqual(calls[3][1]['seed'], 1051829406)
        self.assertEqual(calls[3][1]['quality_retry_count'], 0)
        self.assertIs(calls[3][1]['model'], handle)
        self.assertIs(calls[3][1]['emotion'], emotion)
        self.assertIs(calls[3][1]['sampling'], sampling)
        state = {'four_node_calls_started':False, 'generate_node_call_attempted':False}
        handle.use_accel = True
        with self.assertRaisesRegex(ValueError, 'required isolated controls'):
            probe.execute_nodes(nodes, params, audio, state=state)
        self.assertTrue(state['four_node_calls_started'])
        self.assertFalse(state['generate_node_call_attempted'])
        self.assertEqual(len(calls), 5)  # loader only; no generation after rejection

    def test_literal_text_and_explicit_language_guards_preserve_marks(self):
        text = 'क़्\u200cष नमस्ते।'
        # NFC can precompose/decompose nukta; use the authored canonical sequence.
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        self.assertEqual(probe.canonical_text(text), text)
        for invalid in ('', '<|hi|>', 'two  spaces', 'x\ny', 'x'*161, '\ud800', 'x\u202ey', 'x\u2066y', 'x\ufeffy'):
            with self.subTest(text=repr(invalid)), self.assertRaises(ValueError):
                probe.canonical_text(invalid)
        with self.assertRaises(ValueError):
            probe.node_parameters(ROOT, 'hello', 'AUTO')

    def test_queue_is_read_only_local_bounded_and_rejects_busy_or_external(self):
        response = SimpleNamespace(status=200, read=lambda n: json.dumps({'queue_running': [], 'queue_pending': []}).encode())
        connection = SimpleNamespace(request=unittest.mock.Mock(), getresponse=lambda: response, close=unittest.mock.Mock())
        with patch.object(probe.http.client, 'HTTPConnection', return_value=connection) as factory:
            self.assertEqual(probe.queue_clear('http://127.0.0.1:8188')['pending'], 0)
            connection.request.assert_called_once_with('GET', '/queue')
            self.assertFalse(probe.queue_clear('http://127.0.0.1:8188')['reservation'])
            response.read = lambda n: b'{"queue_running":[1],"queue_pending":[]}'
            with self.assertRaisesRegex(ValueError, 'currently empty'):
                probe.queue_clear('http://127.0.0.1:8188')
        for url in ('https://127.0.0.1', 'http://example.com', 'http://user:pass@127.0.0.1', 'http://127.0.0.1/queue', 'http://127.0.0.1?token=x'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                probe.queue_clear(url)

    def test_metadata_only_uses_explicit_source_identity_without_model_or_resource_call(self):
        import wave
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temporary:
            root = Path(temporary)
            node, model, comfy = (root / n for n in ('node', 'model', 'comfy'))
            for folder in (node, model, comfy, comfy / 'comfy_api'):
                folder.mkdir()
            (comfy / 'folder_paths.py').write_text('')
            prompt = root / 'prompt.wav'
            with wave.open(str(prompt), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
                stream.writeframes(b'\x01\x00' * 8000)
            args = SimpleNamespace(workspace_root=ROOT.parent, node_root=node, model_dir=model,
                comfy_root=comfy, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
                candidate_revision='a'*40, model_variant='hindi', text='नमस्ते।', language='HI',
                output_dir=ROOT.parent / ('uncreated-t8-probe-' + root.name))
            manifest = {'modelRepository':'candidate', 'modelRevision':'a'*40,
                'files':{'gpt.pth':{'sha256':'b'*64, 'size':123}}}
            fake = SimpleNamespace(selected_manifest=unittest.mock.Mock(return_value={
                'variant':'hindi', 'selector':'gpt_size_hint_only', 'assets_verified':False,
                'manifest':'manifests/hindi_best100_experimental.json', 'manifest_data':manifest}),
                verify_prepared=unittest.mock.Mock(return_value={
                'manifest_data':{'files':{}}, 'candidate_revision':'a'*40}),
                verify_assets=unittest.mock.Mock(return_value={'files':26}))
            with patch.dict(sys.modules, {'check_t8_compat': fake}), patch.object(probe.synth, 'resource_admission', side_effect=AssertionError('must not query GPU')), patch.object(probe, 'execute_nodes', side_effect=AssertionError('must not load nodes')):
                report, state = probe.preflight(args)
            self.assertEqual(report['status'], 'metadata_verified_no_model_loaded')
            self.assertFalse(report['gpu_executed'])
            self.assertFalse(report['comfy_queue_submitted'])
            self.assertFalse(args.output_dir.exists())
            fake.verify_prepared.assert_called_once_with(node, 'a'*40)
            fake.selected_manifest.assert_called_once_with(node, model, 'a'*40, variant='hindi')
            self.assertEqual(report['model_variant'], 'hindi')
            self.assertEqual(report['selected_model_identity']['gpt_sha256'], 'b'*64)
            fake.selected_manifest.reset_mock()
            fake.selected_manifest.return_value.update(variant='base', manifest='manifests/model_2_5.json')
            args.model_variant, args.language = 'base', 'ZH'
            with patch.dict(sys.modules, {'check_t8_compat': fake}):
                base_report, _ = probe.preflight(args)
            self.assertEqual(base_report['model_variant'], 'base')
            fake.selected_manifest.assert_called_once_with(node, model, 'a'*40, variant='base')
            self.assertNotIn('manifest_data', report['compatibility'])
            registration = root / 'reviewed.json';registration.write_text('{}')
            digest = hashlib.sha256(registration.read_bytes()).hexdigest()
            args.candidate_revision = None
            args.candidate_manifest, args.candidate_manifest_sha256 = registration, digest
            args.model_variant, args.language = 'hindi', 'HI'
            fake.verify_prepared.reset_mock();fake.selected_manifest.reset_mock()
            manifest['modelRepository']='local-reviewed/fixture';manifest['modelRevision']='sha256:'+'e'*64
            manifest['files']['gpt.pth']['sha256']='e'*64
            fake.selected_manifest.return_value.update(variant='hindi',manifest_data=manifest,
                selector='reviewed_registration_marker_then_full_sha256')
            with patch.dict(sys.modules, {'check_t8_compat': fake}):
                reviewed, _ = probe.preflight(args)
            fake.verify_prepared.assert_called_once_with(node,None,candidate_manifest_path=registration,candidate_manifest_sha256=digest)
            fake.selected_manifest.assert_called_once_with(node,model,None,variant='hindi',candidate_manifest_path=registration,candidate_manifest_sha256=digest)
            self.assertEqual(reviewed['selected_model_identity']['gpt_sha256'],'e'*64)
            self.assertFalse(reviewed['gpu_executed'])

    def test_canonical_dedicated_comfy_output_scratch_preserves_protected_inputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temporary:
            comfy = Path(temporary) / 'ComfyUI'
            workspace = comfy / 'output' / 'dedicated-probe'
            workspace.mkdir(parents=True)
            repo, node, model = (workspace / n for n in ('repo', 'node', 'model'))
            for path in (repo, node, model):
                path.mkdir()
            prompt = workspace / 'prompt.wav'; prompt.write_bytes(b'input sentinel')
            with patch.object(probe, 'ROOT', repo):
                output, policy = probe._probe_output(workspace / 'fresh', workspace, comfy, node, model, prompt)
                self.assertEqual(output, workspace / 'fresh')
                self.assertTrue(policy['comfy_output_scratch_exception'])
                self.assertFalse(output.exists())
                for protected in (repo, node, model):
                    with self.subTest(protected=protected.name), self.assertRaises(ValueError):
                        probe._probe_output(protected / 'fresh', workspace, comfy, node, model, prompt)
                with self.assertRaises(ValueError):
                    probe._probe_output(prompt, workspace, comfy, node, model, prompt)
                with self.assertRaises(ValueError):
                    probe._probe_output(comfy / 'outside-scratch', workspace, comfy, node, model, prompt)
            self.assertEqual(prompt.read_bytes(), b'input sentinel')

    def test_comfy_source_model_user_roots_and_symlink_output_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temporary:
            root = Path(temporary); comfy = root / 'ComfyUI'; comfy.mkdir()
            output_root = comfy / 'output'; output_root.mkdir()
            for name in ('models', 'custom_nodes', 'user', 'comfy'):
                workspace = comfy / name / 'probe'; workspace.mkdir(parents=True)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'dedicated canonical output'):
                    probe._probe_output(workspace / 'fresh', workspace, comfy, root, root, root)
            with self.assertRaises(ValueError):
                probe._probe_output(output_root / 'fresh', output_root, comfy, root, root, root)
            output_root.rmdir()
            external = root / 'external'; external.mkdir()
            output_root.symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                probe._probe_output(output_root / 'probe' / 'fresh', output_root / 'probe', comfy, root, root, root)
            # Outside-Comfy policy still protects the entire Comfy directory.
            with patch.object(probe, 'ROOT', root / 'separate-repo'):
                with self.assertRaises(ValueError):
                    probe._probe_output(comfy / 'new-output', root, comfy, root / 'node', root / 'model', root / 'prompt')

    def test_base_hindi_rejected_before_paths_or_model_inspection(self):
        with patch.object(probe.synth, 'resource_admission', side_effect=AssertionError('no GPU')):
            with self.assertRaisesRegex(ValueError, 'untrained base Hindi'):
                probe.preflight(SimpleNamespace(model_variant='base', language='HI'))

    def test_actual_cpu_tensor_pcm_bounds_and_fresh_file(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional actual Torch CPU verification')
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as temporary:
            path = Path(temporary) / 'audio.wav'
            valid = {'waveform': torch.full((1, 1, 16000), .25), 'sample_rate': 16000}
            result = probe.write_audio(valid, path)
            self.assertEqual(result['seconds'], 1)
            self.assertEqual(result['file_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                probe.write_audio(valid, path)
            for value in (torch.zeros(1, 1, 16000), torch.full((1, 1, 16000), float('nan')), torch.ones(1, 1, 320001), torch.ones(1, 2, 16000)):
                with self.assertRaises(ValueError):
                    probe.write_audio({'waveform': value, 'sample_rate': 16000}, Path(temporary) / 'rejected.wav')
            self.assertFalse((Path(temporary) / 'rejected.wav').exists())


if __name__ == '__main__':
    unittest.main()
