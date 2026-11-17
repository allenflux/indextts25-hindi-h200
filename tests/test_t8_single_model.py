"""Tiny CPU default/guard tests; no real model, downloads or GPU are simulated."""
import argparse
import ast
import contextlib
import io
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('single_model_test_entry', ROOT / 'scripts/prepare_t8_single_model.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)

STORE = b'''from pathlib import Path
import hashlib, json
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PLUGIN_ROOT / "manifests" / "model_2_5.json"
MODEL_FOLDER_NAME = "IndexTTS-2.5"
MODEL_REPOSITORY_URL = "https://fixture.invalid/original"
class ValidationReport:
    def __init__(self, model_dir, missing=(), mismatched=(), hashes_verified=False):
        self.model_dir, self.missing, self.mismatched, self.hashes_verified = model_dir, tuple(missing), tuple(mismatched), hashes_verified
    @property
    def valid(self): return not self.missing and not self.mismatched
    def require_valid(self):
        if self.valid: return
        details = list(self.missing) + list(self.mismatched)
        raise FileNotFoundError("old download hint")
def load_manifest(model_dir=None): return json.loads(MANIFEST_PATH.read_text())
def model_options(): return ["fixture-old"]
def resolve_model(model_name, custom_model_path=""): raise AssertionError("old resolver must be replaced")
def configured_model_roots(): raise AssertionError("other roots must never be consulted")
def validate_model_dir(model_dir, *, verify_hashes=False, include_auxiliary=True, progress=None):
    verify_hashes = True
    model_dir = Path(model_dir)
    missing, mismatched = [], []
    for name, meta in load_manifest(model_dir)["files"].items():
        if not include_auxiliary and meta.get("group") == "auxiliary": continue
        path = model_dir / name
        if not path.is_file(): missing.append(name); continue
        data = path.read_bytes()
        if len(data) != meta["size"] or verify_hashes and hashlib.sha256(data).hexdigest() != meta["sha256"]: mismatched.append(name)
    return ValidationReport(model_dir, missing, mismatched, verify_hashes)
def model_fingerprint(model_dir): return load_manifest(model_dir)["modelRevision"]
'''
NODE = b'''class T8IndexTTS25ModelLoader:
    @classmethod
    def define_schema(cls):
        return schema(description="original", inputs=[control("verify_hashes",tooltip="old"), control("download_missing",tooltip="old"), control("accept_model_license",tooltip="old"), control("custom_model_path",tooltip="old")])
    @classmethod
    def validate_inputs(cls, model_name="IndexTTS-2.5", custom_model_path="", download_missing=False, **kwargs):
        sentinel("validation must not continue when automatic downloads requested")
        return True
    @classmethod
    def execute(cls, download_missing=False, **kwargs):
        sentinel("resolver/download/device must not execute when automatic downloads requested")
        manifest = load_manifest()
        info, handle = "CPU metadata fixture only", None
        return output(handle, info)
'''
DOWNLOAD = '''def ensure_model_bundle(*args, **kwargs):
    sentinel("bundle target mkdir/network")
def download_main_model(*args, **kwargs):
    sentinel("HF or unpinned ModelScope")
def download_auxiliary_models(*args, **kwargs):
    sentinel("unfixed auxiliary downloader")
def _download_hf_file_cancellable(*args, **kwargs):
    """Transfer fixture, never a real downloader."""
    sentinel("HTTP metadata request")
def build_parser():
    return argparse.ArgumentParser(description="下载并校验 IndexTTS 2.5 正式模型")
def main(verify_only=False):
    if verify_only:
        print(">> IndexTTS 2.5 正式模型 SHA-256 校验通过。")
        return "verification branch retained"
    ensure_model_bundle()
    print(">> 正式模型 SHA-256 校验通过。")
'''.encode()


class SingleModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='single-model-', dir=ROOT / '.runtime')
        self.work = Path(self.tmp.name)
        self.node, self.model = self.work / 'node', self.work / 'models/TTS/IndexTTS-2.5'
        self.node.mkdir(); (self.node / 'services').mkdir(); (self.node / 'manifests').mkdir()
        self.model.mkdir(parents=True)
        self.contents = {'services/model_store.py': STORE, 'nodes_v3.py': NODE, p.DOWNLOADER: DOWNLOAD}
        self.changed = p.transform(self.contents)
        self.pin_context = patch.multiple(p.compat, GPT_BYTES=9, GPT_SHA256=hashlib.sha256(b'candidate').hexdigest())
        self.pin_context.start()
        self.files = {'gpt.pth': b'candidate', **{f'asset-{i:02d}.bin': bytes([i]) for i in range(25)}}
        self.base = {'modelRepository': 'fixture/base', 'modelRevision': 'b' * 40, 'files': {}}
        for name, data in self.files.items():
            self.base['files'][name] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            (self.model / name).write_bytes(data)
        self.base['files']['asset-24.bin']['group'] = 'auxiliary'
        self.manifest = p.compat.candidate_manifest(self.base, p.REVISION)
        (self.node / p.compat.CANDIDATE_MANIFEST).write_text(json.dumps(self.manifest))
        (self.node / p.compat.BASE_MANIFEST).write_text(json.dumps(self.base))
        for name, data in self.changed.items():
            path = self.node / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        self.store = {'__file__': str(self.node / 'services/model_store.py')}
        exec(self.changed['services/model_store.py'], self.store)

    def tearDown(self):
        self.pin_context.stop(); self.tmp.cleanup()

    def test_default_keeps_old_name_empty_path_and_ignores_backup_search_order(self):
        folder = ModuleType('folder_paths'); folder.models_dir = str(self.work / 'models')
        backup = self.work / 'a-backup/IndexTTS-2.5'; backup.mkdir(parents=True)
        with patch.dict(sys.modules, {'folder_paths': folder}):
            self.assertEqual(self.store['model_options'](), ['IndexTTS-2.5'])
            self.assertEqual(self.store['resolve_model']('IndexTTS-2.5', ''), self.model)
            self.assertEqual(self.store['resolve_model']('IndexTTS-2.5', '"' + str(self.model) + '"'), self.model)
            with self.assertRaises(ValueError): self.store['resolve_model']('IndexTTS-2.5 [2]', '')
        self.assertEqual(self.store['MANIFEST_PATH'], self.node / p.compat.CANDIDATE_MANIFEST)
        self.assertEqual(self.store['load_manifest']()['modelRepository'], p.compat.CANDIDATE_REPOSITORY)
        self.assertEqual(self.store['load_manifest'](backup)['modelRevision'], p.REVISION)

    def test_only_candidate_assets_validate_without_any_original_gpt_directory(self):
        report = self.store['validate_model_dir'](self.model, verify_hashes=False, include_auxiliary=False)
        self.assertTrue(report.valid); self.assertTrue(report.hashes_verified)
        self.assertFalse((self.work / 'original-gpt').exists())
        (self.model / 'gpt.pth').write_bytes(b'wrong-gpt')  # Same size, must still fail actual SHA.
        report = self.store['validate_model_dir'](self.model, verify_hashes=False)
        self.assertEqual(report.mismatched, ('gpt.pth',))
        with self.assertRaisesRegex(FileNotFoundError, '26'): report.require_valid()

    def test_skip_auxiliary_false_and_missing_gpt_never_fall_back_to_base(self):
        (self.model / 'asset-24.bin').unlink()
        report = self.store['validate_model_dir'](self.model, verify_hashes=False, include_auxiliary=False)
        self.assertIn('asset-24.bin', report.missing)
        (self.model / 'gpt.pth').unlink()
        self.assertEqual(self.store['load_manifest'](self.model)['modelRepository'], p.compat.CANDIDATE_REPOSITORY)
        report = self.store['validate_model_dir'](self.model)
        self.assertIn('gpt.pth', report.missing)
        self.assertEqual(self.store['model_fingerprint'](self.model), p.REVISION)

    def test_all_download_flags_and_entrypoints_fail_before_side_effects(self):
        called = []
        def sentinel(value): called.append(value); raise AssertionError(value)
        loader = {'sentinel': sentinel}
        exec(self.changed['nodes_v3.py'], loader)
        cls = loader['T8IndexTTS25ModelLoader']
        self.assertIn('不支持自动下载', cls.validate_inputs(download_missing=True))
        with self.assertRaisesRegex(ValueError, '不支持自动下载'): cls.execute(download_missing=True)
        downloader = {'sentinel': sentinel, 'load_manifest': self.store['load_manifest'], 'argparse': argparse}
        exec(self.changed[p.DOWNLOADER], downloader)
        for name in ('ensure_model_bundle', 'download_main_model', 'download_auxiliary_models', '_download_hf_file_cancellable'):
            with self.subTest(entry=name), self.assertRaisesRegex(RuntimeError, '不支持自动下载'):
                downloader[name](self.model, source='modelscope')
        with self.assertRaises(RuntimeError): downloader['main']()
        receipt = io.StringIO()
        with contextlib.redirect_stdout(receipt):
            self.assertEqual(downloader['main'](verify_only=True), 'verification branch retained')
        self.assertNotIn('正式模型', downloader['build_parser']().description)
        self.assertIn('不支持下载', downloader['build_parser']().description)
        for identity in (p.compat.CANDIDATE_REPOSITORY, p.REVISION, p.compat.GPT_SHA256):
            self.assertIn(identity, receipt.getvalue())
        self.assertEqual(called, [])

    def test_loader_metadata_and_ui_are_truthful_with_unchanged_field_names(self):
        namespace = {'schema': lambda **kwargs: kwargs, 'control': lambda name, **kwargs: (name, kwargs),
                     'sentinel': lambda value: None, 'load_manifest': self.store['load_manifest'],
                     'output': lambda handle, info: (handle, info)}
        exec(self.changed['nodes_v3.py'], namespace)
        cls = namespace['T8IndexTTS25ModelLoader']
        schema = cls.define_schema()
        self.assertIn('实验', schema['description'])
        fields = {name: kwargs['tooltip'] for name, kwargs in schema['inputs']}
        self.assertEqual(set(fields), {'verify_hashes', 'download_missing', 'accept_model_license', 'custom_model_path'})
        self.assertIn('26', fields['verify_hashes'])
        info = cls.execute()[1]
        self.assertIn(p.compat.CANDIDATE_REPOSITORY, info)
        self.assertIn(p.REVISION, info)
        self.assertIn(p.compat.GPT_SHA256, info)

    def test_legacy_path_normalization_cannot_authorize_original_weights(self):
        # Existing legacy normalization passes a path to the same resolver; the
        # profile's fixed manifest remains the final authority for every path.
        other = self.work / 'original-directory'; other.mkdir()
        (other / 'gpt.pth').write_bytes(b'base')
        resolved = self.store['resolve_model']('IndexTTS-2.5', str(other))
        report = self.store['validate_model_dir'](resolved, verify_hashes=False, include_auxiliary=False)
        self.assertFalse(report.valid)
        self.assertIn('gpt.pth', report.mismatched)

    def test_fixed_revision_and_original_downloader_pin_fail_closed(self):
        for revision in ('main', 'b' * 40):
            with self.assertRaises(p.SingleModelError): p.fixed_revision(revision)
        wrong = self.work / 'wrong-downloader.py'; wrong.write_bytes(b'not pinned')
        with self.assertRaisesRegex(p.SingleModelError, 'source differs'): p.original_downloader(wrong)
        with self.assertRaises((p.SingleModelError, OSError)): p.read_json(self.node / 'missing.json')

    def test_profile_tamper_and_duplicate_json_are_rejected_before_source_or_weights(self):
        value = p.profile(p.REVISION)
        value['gpu_validated'] = 0  # False and zero must not be interchangeable metadata.
        (self.node / p.PROFILE).write_text(json.dumps(value))
        with self.assertRaisesRegex(p.SingleModelError, 'identity/schema'):
            p.verify_copy(self.node, p.REVISION)
        duplicate = self.work / 'duplicate.json'
        duplicate.write_text('{"policy":"candidate-only","policy":"other"}')
        with self.assertRaisesRegex(p.SingleModelError, 'Duplicate'):
            p.read_json(duplicate)

    def test_incomplete_release_stops_before_any_large_asset_hash(self):
        partial = self.work / 'partial-release'; partial.mkdir()
        (partial / 'gpt.pth').write_bytes(b'candidate')
        with patch.object(p, 'verify_copy', return_value={'manifest_data': self.manifest}), \
                patch.object(p.compat, '_sha', side_effect=AssertionError('must not hash before structure passes')), \
                self.assertRaisesRegex(p.SingleModelError, 'Incomplete 26-asset'):
            p.verify_assets(self.node, partial, p.REVISION)

    def test_source_profile_and_asset_cli_delegate_without_original_weight_input(self):
        fake = {'manifest_data': self.manifest, 'cpu_prototype_only': True, 'gpu_validated': False, 'status': 'copy-check'}
        with patch.object(p, 'verify_copy', return_value=fake):
            result = p.verify_assets(self.node, self.model, p.REVISION)
        self.assertEqual(result['asset_identity']['files'], 26)
        self.assertEqual(result['status'], 'cpu_verified_single_model_assets_not_gpu_validated')
        self.assertFalse(result['gpu_validated'])
        (self.model / 'asset-00.bin').write_bytes(b'tampered')
        with patch.object(p, 'verify_copy', return_value=fake), self.assertRaises((p.SingleModelError, p.compat.CompatibilityError)):
            p.verify_assets(self.node, self.model, p.REVISION)


if __name__ == '__main__':
    unittest.main()
