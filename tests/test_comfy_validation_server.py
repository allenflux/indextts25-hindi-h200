"""CPU bootstrap contracts only; no framework/model/server/network execution."""
import argparse
import array
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('comfy_bootstrap_test_entry', SCRIPTS / 'comfy_validation_server.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='comfy-bootstrap-', dir=ROOT / '.runtime')
        self.work = Path(self.temp.name)
        self.core = self.work / 'core'
        (self.core / 'comfy').mkdir(parents=True)
        for name in ('options.py', 'cli_args.py'):
            (self.core / 'comfy' / name).write_text('# CPU import-order fixture, not real ComfyUI\n')
        self.prompt = self.work / 'public-prompt-fixture.wav'
        samples = array.array('h', (int(1000 * math.sin(i / 11)) for i in range(16000)))
        if sys.byteorder != 'little': samples.byteswap()
        with wave.open(str(self.prompt), 'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000); output.writeframes(samples.tobytes())
        self.prompt_sha = hashlib.sha256(self.prompt.read_bytes()).hexdigest()
        self.pin = patch.object(p, 'PUBLIC_PROMPT_SHA256', self.prompt_sha)
        self.pin.start()
        self.config = {'format': 'comfy-owned-validation-bootstrap-v1', 'stage_root': str(self.core),
            'controller_pid': os.getppid(), 'controller_id': str(uuid.uuid4()), 'nonce': 'a' * 64,
            'prompt': str(self.prompt), 'prompt_sha256': self.prompt_sha, 'graph_hashes': ['1' * 64, '2' * 64, '3' * 64]}
        self.config_path = self.work / 'bootstrap.json'
        self.config_path.write_text(json.dumps(self.config)); self.config_path.chmod(0o600)

    def tearDown(self):
        self.pin.stop()
        self.temp.cleanup()

    def test_private_exact_config_fixed_prompt_and_no_credential_fields(self):
        self.assertEqual(p.load_config(self.config_path), self.config)
        for changes in ({'token': 'not-a-real-token'}, {'nonce': 'bad'}, {'prompt_sha256': '0' * 64},
                        {'controller_pid': os.getppid() + 99999}, {'graph_hashes': ['1' * 64] * 3}):
            value = dict(self.config, **changes)
            self.config_path.write_text(json.dumps(value))
            with self.subTest(fields=sorted(changes)), self.assertRaises(ValueError): p.load_config(self.config_path)
        self.config_path.write_text(json.dumps(self.config)); self.config_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'private'): p.load_config(self.config_path)
        self.config_path.chmod(0o600)
        self.config_path.write_text('{"format":"x","format":"y"}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'): p.load_config(self.config_path)
        self.config_path.write_text(json.dumps(self.config))
        self.prompt.write_bytes(b'tampered non-wave content')
        with self.assertRaises(Exception): p.load_config(self.config_path)

    def test_argument_parsing_precedes_cli_import_and_checks_effective_stage_paths(self):
        import stage_comfy_validation as staging
        paths = {name: str(self.work / name) for name in ('base', 'input', 'output', 'user', 'database')}
        paths['core'] = str(self.core)
        stage = {'paths': paths, 'port': 18889, 'launch_cli': ['python', str(self.core / 'main.py'), '--port', '18889']}
        enabled, calls = [False], []
        options = ModuleType('comfy.options'); options.__file__ = str(self.core / 'comfy/options.py')
        def enable(): enabled[0] = True; calls.append('enable')
        options.enable_args_parsing = enable
        effective = dict(listen='127.0.0.1', port=18889, base_directory=paths['base'], input_directory=paths['input'],
            output_directory=paths['output'], temp_directory=paths['base'], user_directory=paths['user'],
            database_url='sqlite:///' + paths['database'], disable_all_custom_nodes=True, disable_api_nodes=True,
            disable_auto_launch=True, cache_none=True, enable_manager=False, enable_manager_legacy_ui=False,
            enable_assets=False, enable_asset_hashing=False, extra_model_paths_config=None, auto_launch=False,
            whitelist_custom_nodes=[staging.T8_LINK_NAME, *staging.PLUGIN_NAMES.values()])
        cli = ModuleType('comfy.cli_args'); cli.__file__ = str(self.core / 'comfy/cli_args.py'); cli.args = SimpleNamespace(**effective)
        def importer(name):
            calls.append(name)
            if name == 'comfy.options': return options
            self.assertTrue(enabled[0], 'Default args must never be cached before parsing is enabled')
            self.assertEqual(sys.argv, stage['launch_cli'][1:])
            return cli
        with patch.object(p.importlib, 'import_module', side_effect=importer), patch.object(sys, 'argv', ['old']):
            self.assertIs(p.activate_comfy_arguments(stage), cli.args)
            self.assertEqual(calls, ['comfy.options', 'enable', 'comfy.cli_args'])
            cli.args.port = 8188
            with self.assertRaisesRegex(RuntimeError, 'port'): p.activate_comfy_arguments(stage)
        with patch.dict(sys.modules, {'comfy.cli_args': cli}), self.assertRaisesRegex(RuntimeError, 'fresh process'):
            p.activate_comfy_arguments(stage)

    def test_nonce_policy_binds_exact_ordered_graphs_and_never_allows_retry_or_other_post(self):
        import run_comfy_validation as controller
        graphs = [controller.build_graph(i, self.work / ('hindi' if i == 1 else 'base'), 18889) for i in range(3)]
        config = dict(self.config, graph_hashes=[controller.digest_json(graph) for graph in graphs])
        policy = p.RequestPolicy(config); headers = {'X-Validation-Nonce': config['nonce']}
        self.assertTrue(policy.authorized('GET', '/validation/prompt.wav', {}))
        self.assertFalse(policy.authorized('GET', '/validation/state', {}))
        self.assertFalse(policy.authorized('POST', '/queue', headers, b'{}'))
        self.assertFalse(policy.authorized('GET', '/upload/image', headers))
        bodies = [json.dumps({'prompt': graph, 'client_id': config['controller_id']}).encode() for graph in graphs]
        self.assertFalse(policy.authorized('POST', '/prompt', headers, bodies[1]))
        self.assertTrue(policy.authorized('POST', '/prompt', headers, bodies[0]))
        self.assertFalse(policy.authorized('POST', '/prompt', headers, bodies[0]))
        self.assertTrue(policy.authorized('POST', '/prompt', headers, bodies[1]))
        self.assertTrue(policy.authorized('POST', '/prompt', headers, bodies[2]))
        self.assertFalse(policy.authorized('POST', '/prompt', headers, bodies[2]))
        self.assertEqual(policy.submissions, 3)
        policy = p.RequestPolicy(config)
        extra = json.dumps({'prompt': graphs[0], 'client_id': config['controller_id'], 'extra_data': {'secret': 'fixture'}}).encode()
        self.assertFalse(policy.authorized('POST', '/prompt', headers, extra))
        self.assertFalse(policy.authorized('POST', '/prompt', headers, b'x' * (64 * 1024 + 1)))
        self.assertEqual(policy.submissions, 0)

    def test_event_telemetry_is_bounded_and_never_keeps_text_audio_nonce_or_error_data(self):
        log = p.EventLog()
        value = {'prompt_id': self.config['controller_id'], 'node': '4', 'secret': 'fixture secret', 'text': 'private fixture', 'audio': b'not-audio'}
        log.append('executing', value); log.append('unknown_event', value)
        snapshot = log.snapshot()
        self.assertEqual(snapshot['events'], [{'event': 'executing', 'prompt_id': self.config['controller_id'], 'node': '4'}])
        for _ in range(260): log.append('execution_error', value)
        snapshot = log.snapshot()
        self.assertEqual(len(snapshot['events']), 256)
        self.assertTrue(snapshot['overflow'])
        encoded = json.dumps(snapshot)
        self.assertNotIn('fixture secret', encoded); self.assertNotIn('private fixture', encoded)
        other = p.EventLog(); other.append('execution_cached', {'nodes': list(range(9))})
        self.assertTrue(other.snapshot()['overflow'])

    def test_observation_wires_auth_and_state_without_constructing_a_real_server(self):
        class Response:
            def __init__(self, status=200, value=None): self.status, self.value = status, value
        class Routes:
            def __init__(self): self.handlers = {}
            def get(self, path):
                def register(handler): self.handlers[path] = handler; return handler
                return register
        class FixtureServer:
            def __init__(self): self.app = SimpleNamespace(middlewares=[]); self.routes = Routes(); self.sent = []
            def send_sync(self, event, data, sid=None): self.sent.append((event, data, sid)); return 42
        class Request:
            method, path, content_length = 'GET', '/validation/state', None
            def __init__(self, headers): self.headers = headers
        def drive(coroutine):
            # These fixture handlers have immediate awaits only: no event loop,
            # actual socket, running server or scheduled background work.
            try: coroutine.send(None)
            except StopIteration as done: return done.value
            raise AssertionError('Fixture unexpectedly attempted asynchronous I/O')
        web = SimpleNamespace(middleware=lambda fn: fn, Response=Response,
            json_response=lambda value: Response(value=value), FileResponse=lambda path: Response(value=str(path)))
        aiohttp = ModuleType('aiohttp'); aiohttp.web = web
        framework = SimpleNamespace(PromptServer=FixtureServer)
        original = FixtureServer.__init__
        with patch.dict(sys.modules, {'aiohttp': aiohttp}), patch.object(p, 'cache_snapshot', return_value={'ready': True, 'cache_entries': []}):
            restore = p.install_observation(framework, self.config, {'paths': {'t8': str(self.work)}})
            try:
                service = FixtureServer()
                self.assertEqual(service.send_sync('executing', {'node': '4', 'prompt_id': self.config['controller_id'], 'text': 'must not appear'}), 42)
                handler = service.routes.handlers['/validation/state']
                middleware = service.app.middlewares[0]
                self.assertEqual(drive(middleware(Request({}), handler)).status, 403)
                state = drive(middleware(Request({'X-Validation-Nonce': self.config['nonce']}), handler)).value
                self.assertEqual(state['controller_id'], self.config['controller_id'])
                self.assertEqual(state['server_pid'], os.getpid())
                self.assertNotIn(self.config['nonce'], json.dumps(state))
                self.assertNotIn('must not appear', json.dumps(state))
                request = Request({}); request.path = '/validation/prompt.wav'
                result = drive(middleware(request, service.routes.handlers[request.path]))
                self.assertEqual(result.value, str(self.prompt))
            finally: restore()
        self.assertIs(FixtureServer.__init__, original)

    def test_network_guard_contract_uses_no_actual_socket_or_dns(self):
        class FixtureSocket:
            def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None): self.family = family
            def connect(self, address): return address
            def connect_ex(self, address): return 0
            def bind(self, address): return address
            def sendto(self, *args, **kwargs): return 1
        with patch.object(socket, 'socket', FixtureSocket), patch.object(socket, 'getaddrinfo', return_value=['local-fixture']) as dns:
            restore = p.install_network_policy(18889)
            try:
                local = socket.socket()
                self.assertEqual(local.bind(('127.0.0.1', 18889)), ('127.0.0.1', 18889))
                self.assertEqual(local.connect_ex(('127.0.0.1', 18889)), 0)
                for address in (('127.0.0.1', 8188), ('0.0.0.0', 18889), ('example.invalid', 18889)):
                    with self.assertRaises(RuntimeError): local.connect(address)
                    with self.assertRaises(RuntimeError): local.bind(address)
                with self.assertRaises(RuntimeError): socket.socket(type=socket.SOCK_DGRAM)
                with self.assertRaises(RuntimeError): socket.getaddrinfo('example.invalid', 18889)
                self.assertEqual(socket.getaddrinfo('127.0.0.1', 18889), ['local-fixture'])
                self.assertEqual(dns.call_count, 1)
            finally: restore()
            self.assertIs(socket.socket, FixtureSocket)

    def test_real_cache_adapter_gets_distinct_directories_for_each_constructor(self):
        class Base:
            def __init__(self): self.zh_normalizer = self.en_normalizer = None; self.load()
        seen = []
        def factory(cache_dir, **kwargs): seen.append(cache_dir); return object()
        original = p.synth.cache_normalizer_class
        def adapted(base, root): return original(base, root, factories=(factory, factory), system='Linux')
        with patch.object(p.synth, 'cache_normalizer_class', side_effect=adapted):
            construct = p.normalizer_factory(Base, self.work / 'cache')
            first, second = construct(), construct()
        self.assertIsNot(first, second)
        self.assertEqual(len(seen), 4)
        self.assertEqual({Path(path).parent.name for path in seen}, {'tn-1', 'tn-2'})
        self.assertTrue(all(Path(path).is_dir() for path in seen))

    def test_requests_environment_is_disabled_without_request_or_credentials(self):
        class Session:
            def __init__(self): self.trust_env = True
        module = SimpleNamespace(sessions=SimpleNamespace(Session=Session))
        original = Session.__init__
        restore = p.disable_request_environment(module)
        try: self.assertFalse(Session().trust_env)
        finally: restore()
        self.assertIs(Session.__init__, original)

    def test_parent_death_binding_checks_both_sides_and_never_calls_real_prctl(self):
        class Function:
            def __call__(self, *args): self.args = args; return 0
        function = Function()
        with patch.object(p.sys, 'platform', 'linux'), patch.object(p.os, 'getppid', side_effect=[123, 123]), patch.object(p.ctypes, 'CDLL', return_value=SimpleNamespace(prctl=function)):
            p.bind_parent_lifetime(123)
        self.assertEqual(function.args, (1, p.signal.SIGKILL, 0, 0, 0))
        with patch.object(p.sys, 'platform', 'linux'), patch.object(p.os, 'getppid', side_effect=[123, 456]), patch.object(p.ctypes, 'CDLL', return_value=SimpleNamespace(prctl=Function())), self.assertRaises(RuntimeError):
            p.bind_parent_lifetime(123)

    def test_cache_metadata_follows_registered_loader_and_whitelists_actual_entry_fields(self):
        t8 = self.work / 't8'
        modules = {}
        def cls(name, relative, module_name):
            path = t8 / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# metadata fixture only\n')
            module = ModuleType(module_name); module.__file__ = str(path); modules[module_name] = module
            return type(name, (), {'__module__': module_name})
        Loader = cls('Loader', 'nodes_v3.py', '_fixture_actual_loader')
        Cache = cls('Cache', 'runtime/model_cache.py', '_fixture_actual_cache')
        Model = cls('Model', 'indextts/infer_v2_5.py', '_fixture_actual_model')
        model = Model(); model.gpt_path = str(self.work / 'model/gpt.pth')
        cache = Cache(); cache._guard = threading.RLock()
        key = (str(self.work / 'model'), 'cuda:0', True, False, 'cuda:0', False, False, False, False, False, 'a' * 40, 'fingerprint', False, '')
        cache._entries = {key: SimpleNamespace(model=model, users=0, completed_runs=1, secret='never serialize')}
        modules[Loader.__module__].MODEL_CACHE = cache
        nodes = ModuleType('nodes'); nodes.NODE_CLASS_MAPPINGS = {'T8_IndexTTS25_ModelLoader': Loader}; modules['nodes'] = nodes
        with patch.dict(sys.modules, modules):
            value = p.cache_snapshot(t8)
            self.assertTrue(value['ready'])
            self.assertEqual(value['cache_entries'][0]['model_id'], id(model))
            self.assertEqual(value['cache_entries'][0]['model_revision'], 'a' * 40)
            self.assertNotIn('secret', json.dumps(value))
            modules[Loader.__module__].__file__ = str(self.work / 'wrong-loader.py')
            with self.assertRaisesRegex(RuntimeError, 'loader source'): p.cache_snapshot(t8)


if __name__ == '__main__':
    unittest.main()
