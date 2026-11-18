"""Tiny CPU staging tests: actual local Git objects and files, no server/GPU."""
import argparse
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('stage_comfy_test_entry', ROOT / 'scripts/stage_comfy_validation.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class ComfyStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='comfy-stage-', dir=ROOT / '.runtime')
        self.home = Path(self.temp.name)
        self.comfy = self.home / 'comfy'
        self.comfy.mkdir()
        self.files = {'main.py': b'# CPU framework fixture, not a real server\n',
            'comfy/cli_args.py': b'# CLI source fixture only\n',
            'folder_paths.py': b'# folder-path fixture only\n',
            'app/database/db.py': b'# database interface fixture only\n',
            'nodes.py': b'# node registry fixture only\n',
            'LICENSE': b'Tiny GPL license interface fixture, not the real license.\n',
            'comfy/ops.py': b'def example():\n    if bias is not None:\n' + p.OPS_NEEDLE + b'    return bias\n',
            'extra_model_paths.yaml': b'private: production paths must not be copied\n',
            'user/comfyui.db': b'private fixture database\n',
            'input/example.png': b'input fixture must not be staged\n',
            'output/tracked-fixture.txt': b'output fixture must not be staged\n',
            'models/put_models_here': b'',
            'custom_nodes/example.py': b'# tracked example not copied\n',
            '.env.example': b'fixed public fixture also excluded\n',
            'comfy_extras/example.py': b'# fixed extra module fixture\n'}
        for name, data in self.files.items():
            path = self.comfy / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=CPU Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixed tiny framework fixture')
        revision = self.git('rev-parse', 'HEAD').decode().strip()
        # Existing untracked source state is never in git archive.
        (self.comfy / 'private-untracked.txt').write_bytes(b'not a credential, fixture only')
        self.work = self.comfy / 'models' / p.PROJECT_DIRECTORY
        self.work.mkdir()
        self.t8 = self.home / 'old-work-t8'
        self.t8.mkdir()
        (self.t8 / 'identity.txt').write_bytes(b'tiny T8 interface fixture')
        self.plugins = {}
        self.plugin_pins = {}
        for label, name in p.PLUGIN_NAMES.items():
            root = self.comfy / 'custom_nodes' / name
            root.mkdir()
            (root / '__init__.py').write_bytes(b'NODE_CLASS_MAPPINGS = {}\n')
            source_name = 'nodes.py' if label == 'audio_cache' else 'audio_crypt.py'
            content = b'# existing installed source interface fixture; never copied\n'
            (root / source_name).write_bytes(content)
            self.plugins[label] = root
            self.plugin_pins[label] = {source_name: hashlib.sha256(content).hexdigest(),
                                       '__init__.py': hashlib.sha256((root / '__init__.py').read_bytes()).hexdigest()}
        self.pins = {name: p.object_sha('blob', self.files[name]) for name in p.SOURCE_PINS}
        changed = self.files['comfy/ops.py'].replace(p.OPS_NEEDLE, p.OPS_NEEDLE + p.OPS_ADDITION)
        self.constants = patch.multiple(p, COMFY_REVISION=revision, SOURCE_PINS=self.pins,
            PLUGIN_FILE_PINS=self.plugin_pins, REVIEWED_OPS_SHA256=hashlib.sha256(changed).hexdigest())
        self.constants.start()
        self.compat = patch.object(p.compat, 'verify_prepared', side_effect=self.verify_t8_fixture)
        self.compat.start()
        self.args = argparse.Namespace(workspace_root=self.work, comfy_root=self.comfy,
            t8_root=self.t8, audio_cache_root=self.plugins['audio_cache'], audio_crypt_root=self.plugins['audio_crypt'],
            candidate_revision='a' * 40, destination=self.work / 'stage-001', port=18889, preserve_reviewed_ops=False)

    def tearDown(self):
        self.compat.stop()
        self.constants.stop()
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.comfy), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def verify_t8_fixture(self, root, revision, **options):
        # Only the already-tested adjacent verifier's interface is stubbed here.
        # Framework Git authentication and source copying below are real.
        if (Path(root) / 'identity.txt').read_bytes() != b'tiny T8 interface fixture':
            raise p.compat.CompatibilityError('T8 fixture identity drift')
        if options:
            p.compat.read_registration(options['candidate_manifest_path'], options['candidate_manifest_sha256'])
            return {'status':'reviewed_tiny_T8_interface_fixture_only','candidate_revision':None,
                    'candidate_manifest_sha256':options['candidate_manifest_sha256'],
                    'manifest_data':{'files':{'gpt.pth':{'sha256':'e'*64}}},'gpu_tested':False}
        return {'status': 'tiny_T8_interface_fixture_only', 'candidate_revision': revision,
                'manifest_data': {}, 'gpu_tested': False, 'model_assets_verified': False}

    def test_reviewed_stage_requires_external_registration_pins_again_on_verify(self):
        registration = self.work / 'reviewed.json'
        value = {'format':'t8-reviewed-complete-gpt-v1','candidate_id':'tiny-interface-fixture',
            'repository':None,'revision':None,'training_profile':'attention_lora_only_v1',
            'adapter_sha256':'a'*64,'gpt_sha256':'e'*64,'gpt_bytes':123,
            'export_complete_sha256':'b'*64,'export_provenance_sha256':'c'*64,'quality_gate_passed':False}
        registration.write_text(json.dumps(value))
        digest = hashlib.sha256(registration.read_bytes()).hexdigest()
        self.args.candidate_revision = None
        self.args.candidate_manifest = registration
        self.args.candidate_manifest_sha256 = digest
        record = p.stage(self.args)
        self.assertEqual(record['candidate_manifest_sha256'], digest)
        self.assertEqual(record['candidate_gpt_sha256'], 'e'*64)
        self.assertIsNone(record['candidate_revision'])
        with self.assertRaisesRegex(p.StageError,'independently supplied'):
            p.verify_stage(self.args.destination)
        self.assertEqual(p.verify_stage(self.args.destination,candidate_manifest_path=registration,
            candidate_manifest_sha256=digest),record)
        with self.assertRaisesRegex(p.StageError,'caller pins'):
            p.verify_stage(self.args.destination,candidate_manifest_path=registration,candidate_manifest_sha256='0'*64)
        saved = self.args.destination / p.RECORD_NAME
        altered = json.loads(saved.read_text());altered['candidate_manifest_sha256']='0'*64
        saved.write_text(json.dumps(altered))
        with self.assertRaisesRegex(p.StageError,'caller pins'):
            p.verify_stage(self.args.destination,candidate_manifest_path=registration,candidate_manifest_sha256=digest)

    def test_inspect_is_read_only_and_declares_safe_paths_without_launch(self):
        before = sorted(str(path.relative_to(self.home)) for path in self.home.rglob('*'))
        record = p.plan(self.args)
        self.assertEqual(before, sorted(str(path.relative_to(self.home)) for path in self.home.rglob('*')))
        self.assertFalse(self.args.destination.exists())
        self.assertFalse(record['server_launched'])
        self.assertTrue(record['launch_audit_required'])
        self.assertFalse(record['port_reserved'])
        self.assertFalse(record['symlinks_enforce_os_read_only'])
        for name in ('base', 'input', 'output', 'temp', 'user', 'runtime', 'cache', 'database', 'core'):
            self.assertTrue(Path(record['paths'][name]).is_relative_to(self.work))
        for value in record['cache_environment'].values():
            self.assertTrue(Path(value).is_relative_to(self.work))
        cli = record['launch_cli']
        self.assertEqual(cli[cli.index('--listen') + 1], '127.0.0.1')
        self.assertIn('--cache-none', cli)
        self.assertIn('--disable-api-nodes', cli)
        self.assertNotIn('--enable-manager', cli)
        self.assertNotIn('--offline', cli)
        self.assertNotIn('--disable-assets', cli)
        self.assertEqual(cli[cli.index('--database-url') + 1], 'sqlite:///' + record['paths']['database'])

    def test_stage_authenticates_whole_git_tree_and_excludes_state_without_copying_plugins(self):
        record = p.stage(self.args)
        core = Path(record['paths']['core'])
        self.assertEqual((core / 'LICENSE').read_bytes(), self.files['LICENSE'])
        for name in ('input', 'output', 'models', 'user', 'custom_nodes', 'private-untracked.txt', 'extra_model_paths.yaml', '.env.example', '.git'):
            self.assertFalse((core / name).exists(), name)
        custom = Path(record['paths']['base']) / 'custom_nodes'
        self.assertEqual({path.name for path in custom.iterdir()}, {p.T8_LINK_NAME, *p.PLUGIN_NAMES.values()})
        for label, name in p.PLUGIN_NAMES.items():
            self.assertTrue((custom / name).is_symlink())
            self.assertEqual((custom / name).resolve(), self.plugins[label])
        self.assertEqual((self.comfy / 'comfy/ops.py').read_bytes(), self.files['comfy/ops.py'])
        self.assertFalse(Path(record['paths']['database']).exists())
        self.assertEqual(p.verify_stage(self.args.destination), record)

    def test_exact_reviewed_ops_overlay_changes_only_isolated_copy(self):
        self.args.preserve_reviewed_ops = True
        host = self.comfy / 'comfy/ops.py'
        reviewed = self.files['comfy/ops.py'].replace(p.OPS_NEEDLE, p.OPS_NEEDLE + p.OPS_ADDITION)
        host.write_bytes(reviewed)
        result = p.stage(self.args)
        self.assertEqual((Path(result['paths']['core']) / 'comfy/ops.py').read_bytes(), reviewed)
        self.assertEqual(host.read_bytes(), reviewed)
        self.assertTrue(result['preserve_reviewed_ops'])
        host.write_bytes(reviewed + b'# unreviewed change\n')
        with self.assertRaisesRegex(p.StageError, 'exact reviewed'):
            p.verify_stage(self.args.destination)

    def test_wrong_work_port_arbitrary_plugin_or_existing_destination_refused(self):
        cases = [dict(workspace_root=self.home), dict(port=8188), dict(port=0),
                 dict(audio_cache_root=self.t8), dict(destination=self.work),
                 dict(destination=self.comfy / 'output' / 'bad-stage')]
        for changes in cases:
            args = copy.copy(self.args)
            for name, value in changes.items(): setattr(args, name, value)
            with self.subTest(changes=changes), self.assertRaises(p.StageError): p.plan(args)
        self.args.destination.mkdir()
        with self.assertRaisesRegex(p.StageError, 'fresh'): p.stage(self.args)

    def test_head_commit_tree_and_file_tamper_are_not_self_signed_authority(self):
        source = p.source_record(self.comfy)
        modified = copy.deepcopy(source)
        modified['entries'][0]['blob'] = '0' * 40
        with self.assertRaisesRegex(p.StageError, 'tree identity'): p.authenticate_source(modified)
        modified = copy.deepcopy(source)
        modified['commit_object_base64'] = base64.b64encode(b'tree ' + b'0' * 40 + b'\n').decode()
        with self.assertRaisesRegex(p.StageError, 'commit object'): p.authenticate_source(modified)
        p.stage(self.args)
        core = self.args.destination / 'core'
        (core / 'main.py').write_bytes(b'# tampered source\n')
        with self.assertRaisesRegex(p.StageError, 'content drift'): p.verify_stage(self.args.destination)

    def test_reference_source_and_stage_links_are_rechecked(self):
        p.stage(self.args)
        source = self.plugins['audio_cache'] / 'nodes.py'
        before = source.read_bytes()
        source.write_bytes(before + b'# drift\n')
        with self.assertRaisesRegex(p.StageError, 'definition source changed'): p.verify_stage(self.args.destination)
        source.write_bytes(before)
        init = self.plugins['audio_crypt'] / '__init__.py'
        original_init = init.read_bytes()
        init.write_bytes(original_init + b'# drift\n')
        with self.assertRaisesRegex(p.StageError, 'definition source changed'): p.verify_stage(self.args.destination)
        init.write_bytes(original_init)
        link = self.args.destination / 'base/custom_nodes' / p.T8_LINK_NAME
        link.unlink(); link.symlink_to(self.comfy, target_is_directory=True)
        with self.assertRaisesRegex(p.StageError, 'reference changed'): p.verify_stage(self.args.destination)

    def test_source_git_replace_is_ignored_and_symlink_archive_is_rejected(self):
        original = self.git('rev-parse', 'HEAD').decode().strip()
        (self.comfy / 'unexpected.py').write_bytes(b'# replacement tree\n')
        self.git('add', 'unexpected.py')
        self.git('-c', 'user.name=CPU Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'replacement')
        replacement = self.git('rev-parse', 'HEAD').decode().strip()
        self.git('reset', '--hard', original)
        self.git('replace', original, replacement)
        p.stage(self.args)
        self.assertFalse((self.args.destination / 'core/unexpected.py').exists())
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:') as archive:
            entry = tarfile.TarInfo('main.py'); entry.type = tarfile.SYMTYPE; entry.linkname = '/outside'
            archive.addfile(entry)
        real_git = p.compat._git
        def fake_archive(root, *args):
            return output.getvalue() if args[0] == 'archive' else real_git(root, *args)
        self.args.destination = self.work / 'stage-bad-archive'
        with patch.object(p.compat, '_git', side_effect=fake_archive), self.assertRaisesRegex(p.StageError, 'special files'):
            p.stage(self.args)
        self.assertFalse((self.args.destination / p.RECORD_NAME).exists())

    def test_manifest_extra_fields_and_missing_stage_never_verify(self):
        with self.assertRaises(p.StageError): p.verify_stage(self.work / 'missing')
        p.stage(self.args)
        path = self.args.destination / p.RECORD_NAME
        record = json.loads(path.read_text())
        record['server_launched'] = True
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(p.StageError, 'manifest'): p.verify_stage(self.args.destination)


if __name__ == '__main__':
    unittest.main()
