#!/usr/bin/env python3
"""Owned ComfyUI bootstrap: local-only networking and read-only telemetry.

Called only by run_comfy_validation.py in a fresh subprocess. This is no model
or replacement node. It serves one fixed reviewed public PCM prompt, retains
real ComfyUI queue execution, and never logs nonce, request bodies or secrets.
"""
from __future__ import annotations
from contextlib import ExitStack
import ctypes
import hmac
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import runpy
import signal
import socket
import stat
import sys
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hi_h200_native import synth

PUBLIC_PROMPT_SHA256 = 'fd23ddac68f2db3710481c7cbfc1d500117d9c38bc5c3ee51c33444f0d9b019d'
EVENTS = {'execution_start', 'execution_success', 'execution_error', 'execution_interrupted',
          'execution_cached', 'executing', 'executed'}
NODE_IDS = {str(i) for i in range(1, 9)}
CONFIG_KEYS = {'format', 'stage_root', 'controller_pid', 'controller_id', 'nonce',
               'prompt', 'prompt_sha256', 'graph_hashes'}


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def load_config(path):
    path = Path(path).absolute()
    info = path.lstat()
    if path.resolve(strict=True) != path or not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Expected owned, private, regular bounded bootstrap config')
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError('Bootstrap config grew beyond its bound')
    config = strict_json(raw)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS or config['format'] != 'comfy-owned-validation-bootstrap-v1':
        raise ValueError('Invalid bootstrap schema')
    if type(config['controller_pid']) is not int or config['controller_pid'] <= 1 or config['controller_pid'] != os.getppid():
        raise ValueError('Bootstrap does not belong to its current live controller')
    if not isinstance(config['controller_id'], str) or str(uuid.UUID(config['controller_id'])) != config['controller_id']:
        raise ValueError('Invalid controller identity')
    if not isinstance(config['nonce'], str) or re.fullmatch('[0-9a-f]{64}', config['nonce']) is None:
        raise ValueError('Invalid private nonce shape')
    hashes = config['graph_hashes']
    if not isinstance(hashes, list) or len(hashes) != 3 or any(not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None for h in hashes) or len(set(hashes)) != 3:
        raise ValueError('Exactly three distinct preauthorized graph identities required')
    if config['prompt_sha256'] != PUBLIC_PROMPT_SHA256:
        raise ValueError('Only the reviewed public PCM prompt is served')
    for name in ('stage_root', 'prompt'):
        value = config[name]
        if not isinstance(value, str) or not Path(value).is_absolute() or Path(value).resolve(strict=True) != Path(value):
            raise ValueError('Bootstrap dependencies must be canonical absolute paths')
    prompt_info = synth.wav_info(Path(config['prompt']), prompt=True)
    if prompt_info['file_sha256'] != PUBLIC_PROMPT_SHA256:
        raise ValueError('Reviewed public PCM prompt content changed')
    return config


def bind_parent_lifetime(controller_pid):
    if sys.platform != 'linux' or os.getppid() != controller_pid:
        raise RuntimeError('Owned Linux parent lifetime is required')
    library = ctypes.CDLL(None, use_errno=True)
    function = library.prctl
    function.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    function.restype = ctypes.c_int
    if function(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != controller_pid:
        raise RuntimeError('Cannot bind server lifetime to the controller')


def activate_comfy_arguments(stage):
    """Parse the staged CLI before server/TTS imports can cache default args."""
    if 'comfy.cli_args' in sys.modules:
        raise RuntimeError('Comfy arguments were already imported; use a fresh process')
    core = Path(stage['paths']['core'])
    sys.argv = list(stage['launch_cli'][1:])
    options = importlib.import_module('comfy.options')
    if Path(inspect.getfile(options)).resolve() != core / 'comfy/options.py':
        raise RuntimeError('Comfy options came from another framework tree')
    options.enable_args_parsing()
    cli = importlib.import_module('comfy.cli_args')
    if Path(inspect.getfile(cli)).resolve() != core / 'comfy/cli_args.py':
        raise RuntimeError('Comfy CLI came from another framework tree')
    expected = {'listen': '127.0.0.1', 'port': stage['port'], 'base_directory': stage['paths']['base'],
        'input_directory': stage['paths']['input'], 'output_directory': stage['paths']['output'],
        'temp_directory': stage['paths']['base'], 'user_directory': stage['paths']['user'],
        'database_url': 'sqlite:///' + stage['paths']['database'], 'disable_all_custom_nodes': True,
        'disable_api_nodes': True, 'disable_auto_launch': True, 'cache_none': True,
        'enable_manager': False, 'enable_manager_legacy_ui': False, 'enable_assets': False,
        'enable_asset_hashing': False, 'extra_model_paths_config': None, 'auto_launch': False}
    import stage_comfy_validation as staging
    expected['whitelist_custom_nodes'] = [staging.T8_LINK_NAME, *staging.PLUGIN_NAMES.values()]
    for name, value in expected.items():
        if getattr(cli.args, name, object()) != value:
            raise RuntimeError('Effective Comfy option mismatch: ' + name)
    return cli.args


def install_network_policy(port):
    original_socket, original_dns = socket.socket, socket.getaddrinfo
    def local(address):
        return isinstance(address, tuple) and len(address) >= 2 and address[0] in {'127.0.0.1', '::1'} and type(address[1]) is int and address[1] == port
    class LocalOnlySocket(original_socket):
        def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
            if family not in {socket.AF_INET, socket.AF_INET6, socket.AF_UNIX} or family != socket.AF_UNIX and type & socket.SOCK_DGRAM:
                raise RuntimeError('Only owned loopback TCP and AF_UNIX sockets are allowed')
            super().__init__(family, type, proto, fileno)
        def connect(self, address):
            if self.family != socket.AF_UNIX and not local(address):
                raise RuntimeError('External network connection is forbidden')
            return super().connect(address)
        def connect_ex(self, address):
            if self.family != socket.AF_UNIX and not local(address):
                raise RuntimeError('External network connection is forbidden')
            return super().connect_ex(address)
        def bind(self, address):
            if self.family != socket.AF_UNIX and not local(address):
                raise RuntimeError('Only the owned loopback listener may be bound')
            return super().bind(address)
        def sendto(self, *args, **kwargs):
            if self.family != socket.AF_UNIX:
                raise RuntimeError('IP datagrams are forbidden')
            return super().sendto(*args, **kwargs)
    def dns(host, requested_port, *args, **kwargs):
        if host not in {'127.0.0.1', '::1'} or requested_port != port:
            raise RuntimeError('External DNS is forbidden')
        return original_dns(host, requested_port, *args, **kwargs)
    socket.socket, socket.getaddrinfo = LocalOnlySocket, dns
    def restore():
        socket.socket, socket.getaddrinfo = original_socket, original_dns
    return restore


def disable_request_environment(requests_module):
    original = requests_module.sessions.Session.__init__
    def session(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.trust_env = False
    requests_module.sessions.Session.__init__ = session
    return lambda: setattr(requests_module.sessions.Session, '__init__', original)


def cache_snapshot(t8_root):
    """Follow the actual registered loader, never a private probe package."""
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get('T8_IndexTTS25_ModelLoader')
    if cls is None:
        return {'ready': False, 'cache_entries': []}
    if Path(inspect.getfile(cls)).resolve() != t8_root / 'nodes_v3.py':
        raise RuntimeError('Registered real loader source identity mismatch')
    module = sys.modules[cls.__module__]
    cache = getattr(module, 'MODEL_CACHE')
    if Path(inspect.getfile(type(cache))).resolve() != t8_root / 'runtime/model_cache.py':
        raise RuntimeError('Registered loader does not use the reviewed cache')
    entries = []
    with cache._guard:
        if len(cache._entries) > 2:
            raise RuntimeError('More than the two authorized model cache entries appeared')
        for key, entry in cache._entries.items():
            model = entry.model
            if len(key) != 14 or Path(inspect.getfile(type(model))).resolve() != t8_root / 'indextts/infer_v2_5.py':
                raise RuntimeError('Actual cached model source/key mismatch')
            if type(entry.users) is not int or entry.users < 0 or type(entry.completed_runs) is not int or entry.completed_runs < 0:
                raise RuntimeError('Unexpected actual cache counter types')
            entries.append({'model_dir': key[0], 'device': key[1], 'use_bf16': key[2], 'use_fp16': key[3],
                'reference_device': key[4], 'model_revision': key[10], 'model_fingerprint': key[11],
                'reference_cache_dir': key[13], 'model_id': id(model), 'users': entry.users,
                'completed_runs': entry.completed_runs, 'gpt_path': str(Path(model.gpt_path).resolve())})
    return {'ready': True, 'cache_entries': entries, 'registered_loader_module': cls.__module__}


class EventLog:
    def __init__(self):
        self.events, self.overflow, self.lock = [], False, threading.RLock()
    def append(self, event, data):
        if event not in EVENTS or not isinstance(data, dict):
            return
        prompt = data.get('prompt_id')
        prompt = prompt if isinstance(prompt, str) and re.fullmatch('[A-Za-z0-9-]{16,80}', prompt) else None
        record = {'event': event, 'prompt_id': prompt}
        if event in {'executing', 'executed'}:
            node = str(data.get('node')) if data.get('node') is not None else None
            record['node'] = node if node in NODE_IDS else None
        if event == 'execution_cached':
            raw = data.get('nodes', [])
            if not isinstance(raw, list) or len(raw) > 8:
                with self.lock:
                    self.overflow = True
                return
            record['nodes'] = [str(node) for node in raw if str(node) in NODE_IDS]
        with self.lock:
            if len(self.events) < 256:
                self.events.append(record)
            else:
                self.overflow = True
    def snapshot(self):
        with self.lock:
            return {'events': list(self.events), 'overflow': self.overflow}


class RequestPolicy:
    def __init__(self, config):
        self.config, self.submissions, self.lock = config, 0, threading.RLock()
    def authorized(self, method, path, headers, raw=None):
        if method == 'GET' and path == '/validation/prompt.wav':
            return True
        if not hmac.compare_digest(headers.get('X-Validation-Nonce', ''), self.config['nonce']):
            return False
        if method == 'GET':
            return path in {'/validation/state', '/queue', '/object_info', '/view'} or re.fullmatch('/history/[A-Za-z0-9-]{16,80}', path) is not None
        if method != 'POST' or path != '/prompt' or raw is None or len(raw) > 64 * 1024:
            return False
        try:
            value = strict_json(raw)
            if not isinstance(value, dict) or set(value) != {'prompt', 'client_id'} or value['client_id'] != self.config['controller_id']:
                return False
            from run_comfy_validation import digest_json, validate_graph
            validate_graph(value['prompt'])
            digest = digest_json(value['prompt'])
        except (ValueError, TypeError, KeyError):
            return False
        with self.lock:
            if self.submissions >= 3 or digest != self.config['graph_hashes'][self.submissions]:
                return False
            self.submissions += 1  # Reserve before handler I/O; never retry uncertain execution.
        return True


def install_observation(server_module, config, stage):
    from aiohttp import web
    original_init = server_module.PromptServer.__init__
    t8_root, events, policy = Path(stage['paths']['t8']), EventLog(), RequestPolicy(config)
    @web.middleware
    async def authorized(request, handler):
        raw = None
        if request.method == 'POST':
            if not hmac.compare_digest(request.headers.get('X-Validation-Nonce', ''), config['nonce']) or request.content_length is None or not 0 < request.content_length <= 64 * 1024:
                return web.Response(status=403)
            raw = await request.read()  # aiohttp caches bytes for the actual core handler.
        if not policy.authorized(request.method, request.path, request.headers, raw):
            return web.Response(status=403)
        return await handler(request)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.app.middlewares.append(authorized)
        original_send = self.send_sync
        def send(event, data, sid=None):
            events.append(event, data)
            return original_send(event, data, sid)
        self.send_sync = send
        async def state(request):
            result = cache_snapshot(t8_root)
            return web.json_response(dict(result, **events.snapshot(), controller_id=config['controller_id'], server_pid=os.getpid()))
        async def prompt(request):
            path = Path(config['prompt'])
            if synth.sha256(path) != PUBLIC_PROMPT_SHA256:
                return web.Response(status=409)
            return web.FileResponse(path)
        self.routes.get('/validation/state')(state)
        self.routes.get('/validation/prompt.wav')(prompt)
    server_module.PromptServer.__init__ = initialize
    return lambda: setattr(server_module.PromptServer, '__init__', original_init)


def normalizer_factory(original, root):
    count, lock = [0], threading.Lock()
    def construct(*args, **kwargs):
        with lock:
            count[0] += 1
            cache = Path(root) / ('tn-' + str(count[0]))
        adapted = synth.cache_normalizer_class(original, cache)
        return adapted(*args, **kwargs)
    return construct


def main():
    if len(sys.argv) != 2 or sys.platform != 'linux':
        raise RuntimeError('Only the owned Linux bootstrap config is supported')
    config_path = Path(sys.argv[1]).absolute()
    config = load_config(config_path)
    bind_parent_lifetime(config['controller_pid'])
    import stage_comfy_validation as staging
    stage = staging.verify_stage(Path(config['stage_root']))
    if not config_path.is_relative_to(Path(stage['workspace_root'])) or config_path.is_relative_to(Path(config['stage_root'])):
        raise RuntimeError('Bootstrap config must be a private controller report in WORK outside staged code')
    port = stage['port']
    sys.dont_write_bytecode = True
    sys.path.insert(0, stage['paths']['core'])
    sys.path.insert(1, stage['paths']['t8'])
    activate_comfy_arguments(stage)
    import requests
    with ExitStack() as stack:
        stack.callback(disable_request_environment(requests))
        stack.callback(install_network_policy(port))
        constructor = stack.enter_context(synth._runtime_import(Path(stage['paths']['t8']), stage['cache_environment'], install_cache_adapter=False))
        module = sys.modules[constructor.__module__]
        original_normalizer = module.TextNormalizer
        module.TextNormalizer = normalizer_factory(original_normalizer, Path(stage['cache_environment']['XDG_CACHE_HOME']))
        stack.callback(setattr, module, 'TextNormalizer', original_normalizer)
        import torch
        torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True; torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True)
        import server
        stack.callback(install_observation(server, config, stage))
        runpy.run_path(str(Path(stage['paths']['core']) / 'main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
