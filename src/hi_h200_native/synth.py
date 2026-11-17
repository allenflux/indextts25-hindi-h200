"""Offline, opt-in single-Hindi synthesis acceptance using the real official pipeline.

The default is metadata/preflight only: no Torch import, model constructor,
CUDA access, download, installation, or environment upgrade is performed.
"""
from __future__ import annotations

import argparse
import array
import ast
from contextlib import contextmanager
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import sys
import time
import warnings
import wave

from hi_h200.data import load_json
from hi_h200.text import normalize_hindi
from .assets import MODEL_REPO, MODEL_REVISION, safe_relative, selected_assets

OFFICIAL_COMMIT = "d9e41aac89fd00b3d71497fddb287b7f24613712"
DISABLED_FLAGS = {"use_qwen_emo": False, "use_deepspeed": False, "use_accel": False,
                  "use_cuda_kernel": False, "use_torch_compile": False}
CORE = ("config.yaml", "gpt.pth", "multilingual_zh_ja_yue_char_del.tiktoken", "LICENSE")


class SynthesisError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise SynthesisError("Missing or oversized JSON metadata")
    value = load_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SynthesisError("Expected JSON metadata object")
    return value


def _contained(root, name):
    relative = safe_relative(name)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise SynthesisError("Local synthesis asset escapes its declared root")
    return path


def _git(root, *args):
    command = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
               "-c", "core.untrackedCache=false", "-C", str(root), *args]
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SynthesisError("Cannot inspect the pinned local official source") from exc


def verify_source(root):
    root = Path(root).resolve(strict=True)
    if _git(root, "rev-parse", "--show-toplevel") != str(root) or _git(root, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise SynthesisError("Synthesis requires the exact reviewed official commit")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise SynthesisError("Use a clean isolated official checkout")
    source = _contained(root, "indextts/infer_v2_5.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    constructor = next((item for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "IndexTTS2"
                        for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
    if constructor is None or not set(DISABLED_FLAGS).issubset({item.arg for item in constructor.args.args}):
        raise SynthesisError("Official synthesis constructor no longer matches the audited controls")
    return root


def verify_synth_assets(root, manifest_path, *, assets=None):
    """Hash the real local bytes against both the manifest and immutable public metadata."""
    root = Path(root).resolve(strict=True)
    manifest = _read(manifest_path)
    if set(manifest) != {"repository", "snapshot_revision", "files"} or manifest["repository"] != MODEL_REPO or manifest["snapshot_revision"] != MODEL_REVISION:
        raise SynthesisError("Use the pinned official 2.5 synthesis manifest")
    files = manifest["files"]
    if not isinstance(files, dict):
        raise SynthesisError("Invalid synthesis file inventory")
    expected = {asset.relative_path: asset for asset in (selected_assets("synth") if assets is None else assets)}
    if not set(expected).issubset(files):
        raise SynthesisError("Synthesis assets are incomplete; use the explicit synth stage, never implicit downloads")
    total = 0
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise SynthesisError("Every local synthesis file needs a SHA256")
        path = _contained(root, name)
        asset = expected.get(name)
        size = path.stat().st_size
        if asset is not None and size != asset.size:
            raise SynthesisError("Pinned synthesis asset size mismatch: " + name)
        actual = hashlib.sha256()
        blob = hashlib.sha1(b"blob " + str(size).encode() + b"\x00")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                actual.update(chunk)
                blob.update(chunk)
        if actual.hexdigest() != digest:
            raise SynthesisError("Local synthesis SHA256 mismatch: " + name)
        if asset is not None and (actual.hexdigest() if asset.hash_kind == "sha256" else blob.hexdigest()) != asset.digest:
            raise SynthesisError("Local synthesis bytes differ from the pinned public snapshot: " + name)
        total += size
    import yaml
    config_bytes = _contained(root, "config.yaml").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != files.get("config.yaml"):
        raise SynthesisError("Synthesis config changed while it was checked")
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict) or str(config.get("version")) != "2.5":
        raise SynthesisError("Actual synthesis config must declare version 2.5")
    paths = {"gpt_checkpoint":"gpt.pth", "w2v_stat":"wav2vec2bert_stats.pt",
             "s2mel_checkpoint":"s2mel.pth", "emo_matrix":"feat2.pt", "spk_matrix":"feat1.pt"}
    if any(config.get(key) != value for key, value in paths.items()):
        raise SynthesisError("Official synthesis config paths differ from the reviewed local layout")
    return config, {"repository": MODEL_REPO, "snapshot_revision": MODEL_REVISION,
                    "manifest_sha256": sha256(manifest_path), "files": files, "verified_bytes": total,
                    "pinned_payload_files": len(expected)}


def verify_merged_bundle(root, original_root, asset_identity):
    root, original_root = Path(root).resolve(strict=True), Path(original_root).resolve(strict=True)
    if root == original_root or root.is_relative_to(original_root) or original_root.is_relative_to(root):
        raise SynthesisError("Original assets and merged bundle must be independent directories")
    complete = _read(_contained(root, "COMPLETE.json"))
    allowed = {"gpt.pth", "LICENSE", "LICENSE_ZH.txt", "NON_ENDORSEMENT.txt", "README.md", "provenance.json", "MODEL_LICENSE"}
    files = complete.get("files")
    if complete.get("schema_version") != 1 or not isinstance(files, dict) or not {"gpt.pth", "provenance.json", "LICENSE", "LICENSE_ZH.txt", "NON_ENDORSEMENT.txt", "README.md"}.issubset(files) or set(files) - allowed:
        raise SynthesisError("Incomplete or unsupported merged export bundle")
    if {item.name for item in root.iterdir()} != set(files) | {"COMPLETE.json"}:
        raise SynthesisError("Merged export has unexpected or partial files")
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest) or sha256(_contained(root, name)) != digest:
            raise SynthesisError("Merged bundle hash mismatch")
    provenance = _read(root / "provenance.json")
    base = provenance.get("base_identity", {})
    if provenance.get("kind") != "hi_h200_native_merged_official_gpt" or provenance.get("official_commit") != OFFICIAL_COMMIT or base.get("kind") == "tiny_random_official_cpu_test":
        raise SynthesisError("Synthesis requires a real native merged 2.5 export, not a tiny test artifact")
    if base.get("model_repository") != MODEL_REPO or base.get("model_snapshot_revision") != MODEL_REVISION:
        raise SynthesisError("Merged export was built from another original model snapshot")
    if any(base.get("model_files", {}).get(name) != asset_identity["files"].get(name) for name in CORE):
        raise SynthesisError("Original/merged GPT, config, tokenizer or license identity differs")
    if provenance.get("gpt_sha256") != files["gpt.pth"] or os.path.samefile(root / "gpt.pth", original_root / "gpt.pth"):
        raise SynthesisError("Merged GPT is inconsistent or aliases the original GPT file")
    return {"gpt_sha256": files["gpt.pth"], "complete_sha256": sha256(root / "COMPLETE.json"),
            "provenance_sha256": files["provenance.json"], "optimizer_updates": provenance.get("optimizer_updates")}


def reviewed_hindi_text(source_root, text):
    normalized = normalize_hindi(text, max_chars=160)
    if any(char in normalized for char in "<>|"):
        raise SynthesisError("Pronunciation/special-token annotations are outside this minimal Hindi acceptance route")
    tree = ast.parse(_contained(source_root, "indextts/utils/front.py").read_text(encoding="utf-8"))
    mapping = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "char_rep_map" for target in node.targets):
            mapping = ast.literal_eval(node.value)
            break
    if not isinstance(mapping, dict):
        raise SynthesisError("Cannot verify the official punctuation map")
    pattern = re.compile("|".join(re.escape(key) for key in mapping))
    if pattern.sub(lambda match: mapping[match.group()], normalized) != normalized:
        raise SynthesisError("Stock inference would rewrite this punctuation; review canonical text and training frontend first")
    return normalized


def wav_info(path, *, prompt=False):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise SynthesisError("Use a bounded local PCM WAV")
    with wave.open(str(path), "rb") as audio:
        channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
        if audio.getcomptype() != "NONE" or channels != 1 or width != 2 or not 8000 <= rate <= 96000 or frames <= 0:
            raise SynthesisError("Expected nonempty mono 16-bit PCM WAV")
        seconds = frames / rate
        if not (1 <= seconds <= 15 if prompt else 0 < seconds <= 60):
            raise SynthesisError("Prompt must be 1..15 seconds; output must be at most 60 seconds")
        pcm = audio.readframes(frames)
        if len(pcm) != frames * 2:
            raise SynthesisError("Truncated WAV")
    values = array.array("h", pcm)
    if sys.byteorder != "little":
        values.byteswap()
    peak = max(abs(value) for value in values)
    if peak == 0:
        raise SynthesisError("All-zero reference/output audio is not an acceptance result")
    return {"file_sha256": sha256(path), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "sample_rate": rate, "frames": frames, "seconds": seconds, "peak_pcm16": peak}


def dependency_report(workspace=None, official_repo=None):
    modules = {"torch":"torch", "torchaudio":"torchaudio", "numpy":"numpy", "librosa":"librosa",
               "omegaconf":"omegaconf", "transformers":"transformers", "einops":"einops", "munch":"munch",
               "sentencepiece":"sentencepiece", "modelscope":"modelscope", "huggingface_hub":"huggingface-hub",
               "safetensors":"safetensors", "whisper":"openai-whisper", "scipy":"scipy", "audiotools":"descript-audiotools",
               "tiktoken":"tiktoken", "argbind":"argbind", "yaml":"PyYAML", "tqdm":"tqdm",
               "numba":"numba", "tokenizers":"tokenizers", "requests":"requests", "packaging":"packaging"}
    if platform.system() == "Linux":
        modules.update(tn="WeTextProcessing", pynini="pynini")
        normalizer = "tn"
    else:
        modules.update(wetext="wetext")
        normalizer = "wetext"
    versions, missing = {}, []
    if importlib.util.find_spec("MeCab") is None:
        modules.update(fugashi="fugashi", unidic_lite="unidic-lite")
    for module, distribution in modules.items():
        spec = importlib.util.find_spec(module)
        if spec is None:
            missing.append(distribution)
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    fugashi, mecab = importlib.util.find_spec("fugashi"), importlib.util.find_spec("MeCab")
    if fugashi is None and mecab is None:
        missing.append("fugashi + unidic-lite, or configured MeCab")
    elif fugashi is not None and importlib.util.find_spec("unidic_lite") is None:
        missing.append("unidic-lite (stock Japanese tagger is initialized even for Hindi)")
    # These direct imports are the reviewed stock synthesis route. Transitive
    # requirements without an official pin are reported separately, not invented.
    constraints, mismatches = [], []
    python_constraint = None
    if official_repo is not None:
        import tomllib
        metadata = tomllib.loads((Path(official_repo) / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        python_constraint = metadata["requires-python"]
        from packaging.requirements import Requirement
        wanted = {value.lower().replace("_", "-") for value in modules.values()}
        for value in metadata["dependencies"]:
            requirement = Requirement(value)
            if requirement.name.lower().replace("_", "-") not in wanted or (requirement.marker and not requirement.marker.evaluate()):
                continue
            constraints.append(value)
            installed = versions.get(requirement.name)
            if installed is None:
                installed = next((v for k, v in versions.items() if k.lower().replace("_", "-") == requirement.name.lower().replace("_", "-")), None)
            if installed is not None and requirement.specifier and installed not in requirement.specifier:
                mismatches.append({"distribution": requirement.name, "installed": installed, "official_constraint": str(requirement.specifier)})
        from packaging.specifiers import SpecifierSet
        if platform.python_version() not in SpecifierSet(python_constraint):
            mismatches.append({"distribution": "Python", "installed": platform.python_version(), "official_constraint": python_constraint})
    return {"versions": versions, "missing": sorted(set(missing)), "official_direct_constraints": constraints,
            "official_python_constraint": python_constraint, "version_mismatches": mismatches,
            "module_availability_checked": True, "runtime_import_verified": False,
            "stock_frontend_constructor_verified": False, "frontend_cache_policy": "real normalizers; explicit cache_dir inside this run; unsupported cache_dir fails closed",
            "note": "Module presence and distribution versions only; ABI/transitive imports and FST construction are unverified. Stock imports ModelScope and initializes Japanese plus Chinese/English frontends even for Hindi. No installation or upgrade is performed."}


def _independent_output(path, workspace, protected):
    if path is None:
        return None
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise SynthesisError("Synthesis output must be a new directory")
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    if not target.is_relative_to(workspace):
        raise SynthesisError("Synthesis output must be inside the isolated workspace")
    if any(target == root or target.is_relative_to(root) or root.is_relative_to(target) for root in protected):
        raise SynthesisError("Synthesis output must not overlap original models, merged artifacts or source")
    return target


def metadata_preflight(args):
    workspace = Path(args.workspace_root).resolve(strict=True)
    source = verify_source(args.official_repo)
    original, merged = Path(args.model_dir).resolve(strict=True), Path(args.merged_bundle).resolve(strict=True)
    if any(not path.is_relative_to(workspace) for path in (source, original, merged)):
        raise SynthesisError("Source/original/merged paths must stay in the isolated workspace")
    manifest_path = Path(args.model_manifest).resolve(strict=True)
    if not manifest_path.is_relative_to(original):
        raise SynthesisError("The original synthesis manifest must be inside its isolated asset directory")
    if args.prompt is not None and not Path(args.prompt).resolve(strict=True).is_relative_to(workspace):
        raise SynthesisError("The explicit acceptance prompt must be inside the isolated workspace")
    output = _independent_output(args.output_dir, workspace, (source, original, merged))
    config, assets = verify_synth_assets(original, args.model_manifest)
    merged_identity = verify_merged_bundle(merged, original, assets)
    text = reviewed_hindi_text(source, args.text) if args.text is not None else None
    prompt = wav_info(args.prompt, prompt=True) if args.prompt is not None else None
    dependencies = dependency_report(workspace, source)
    report = {"status": "metadata_preflight_passed" if not dependencies["missing"] else "metadata_verified_runtime_blocked",
              "official_commit": OFFICIAL_COMMIT, "asset_identity": assets, "merged_identity": merged_identity,
              "variant": args.variant, "seed": args.seed, "constructor_flags": dict(DISABLED_FLAGS, use_bf16=True),
              "directories_independent": True, "workspace_isolated": True, "dependencies": dependencies,
              "input": {"text_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
                        "raw_text_sha256": hashlib.sha256(args.text.encode()).hexdigest() if args.text else None,
                        "prompt": prompt},
              "models_loaded": False, "cuda_accessed": False, "synthesis_executed": False,
              "runtime_import_verified": False, "quality_verified": False}
    state = {"workspace": workspace, "source": source, "original": original, "merged": merged,
             "output": output, "config": config, "text": text, "prompt": Path(args.prompt).resolve() if args.prompt else None}
    return report, state


def parse_gpu_sample(value):
    rows = list(csv.reader(io.StringIO(value)))
    if len(rows) != 1 or len(rows[0]) != 6:
        raise SynthesisError("Expected exactly one NVIDIA GPU telemetry row")
    index, uuid, name, total, used, utilization = [field.strip() for field in rows[0]]
    numbers = [float(item) for item in (total, used, utilization)]
    if not index.isdigit() or not uuid.startswith("GPU-") or any(not math.isfinite(item) for item in numbers):
        raise SynthesisError("Invalid GPU telemetry identity or nonfinite measurement")
    total, used, utilization = numbers
    if total <= 0 or not 0 <= used <= total or not 0 <= utilization <= 100:
        raise SynthesisError("GPU telemetry outside valid bounds")
    return {"index": int(index), "uuid": uuid, "name": name, "total_mib": total,
            "used_mib": used, "free_mib": total-used, "utilization_percent": utilization}


def available_ram_bytes():
    candidates = []
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"(?m)^MemAvailable:\s+(\d+)\s+kB", meminfo.read_text())
        if match:
            candidates.append(int(match.group(1)) * 1024)
    for maximum, current in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(maximum).is_file() and Path(current).is_file():
            limit = Path(maximum).read_text().strip()
            if limit != "max":
                candidates.append(max(0, int(limit) - int(Path(current).read_text().strip())))
    return min(candidates) if candidates else None


def resource_admission(args):
    if not args.allow_cuda or any(value is None for value in (args.min_free_gpu_mib, args.min_free_ram_mib, args.max_utilization)):
        raise SynthesisError("Actual synthesis requires --allow-cuda and explicit GPU/RAM/utilization admission thresholds")
    if not re.fullmatch(r"(?:\d+|GPU-[A-Za-z0-9-]+)", args.gpu_id):
        raise SynthesisError("Choose one explicit GPU index or GPU UUID")
    if any(not math.isfinite(value) for value in (args.min_free_gpu_mib, args.min_free_ram_mib, args.max_utilization)) or args.min_free_gpu_mib <= 0 or args.min_free_ram_mib <= 0 or not 0 <= args.max_utilization <= 100:
        raise SynthesisError("Invalid resource admission thresholds")
    snapshots = []
    for index in range(3):
        result = subprocess.run(["nvidia-smi", "--id", args.gpu_id, "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
        sample = parse_gpu_sample(result.stdout)
        ram = available_ram_bytes()
        if ram is None or ram < args.min_free_ram_mib * 1024**2 or sample["free_mib"] < args.min_free_gpu_mib or sample["utilization_percent"] > args.max_utilization:
            raise SynthesisError("Shared GPU/RAM admission failed; no model is loaded")
        if snapshots and sample["uuid"] != snapshots[0]["uuid"]:
            raise SynthesisError("GPU identity changed during admission checks")
        sample["available_ram_bytes"] = ram
        snapshots.append(sample)
        if index < 2:
            time.sleep(1)
    return snapshots


@contextmanager
def deny_network():
    """Fail closed on Python IP network access; model execution is never substituted."""
    original_socket, original_dns = socket.socket, socket.getaddrinfo
    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
            if family in (socket.AF_INET, socket.AF_INET6):
                raise SynthesisError("Offline synthesis blocked IP network socket creation")
            super().__init__(family, type, proto, fileno)
        def connect(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise SynthesisError("Offline synthesis blocked a network connection")
            return super().connect(address)
        def connect_ex(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise SynthesisError("Offline synthesis blocked a network connection")
            return super().connect_ex(address)
    def denied_dns(*args, **kwargs):
        raise SynthesisError("Offline synthesis blocked DNS/network lookup")
    socket.socket, socket.getaddrinfo = OfflineSocket, denied_dns
    try:
        yield
    finally:
        socket.socket, socket.getaddrinfo = original_socket, original_dns


@contextmanager
def isolated_runtime(output, source, gpu_id):
    runtime = output / "runtime"
    runtime.mkdir()
    cache = runtime / "cache"
    cache.mkdir()
    updates = {name: str(cache / leaf) for name, leaf in {
        "XDG_CACHE_HOME":"xdg", "HF_HOME":"huggingface", "HF_HUB_CACHE":"huggingface/hub",
        "HUGGINGFACE_HUB_CACHE":"huggingface/hub", "MODELSCOPE_CACHE":"modelscope", "TORCH_HOME":"torch",
        "TORCH_EXTENSIONS_DIR":"torch_extensions", "NUMBA_CACHE_DIR":"numba", "TIKTOKEN_CACHE_DIR":"tiktoken",
        "TRITON_CACHE_DIR":"triton", "CUDA_CACHE_PATH":"cuda", "MPLCONFIGDIR":"matplotlib", "TMPDIR":"tmp"}.items()}
    updates.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", USE_MODELSCOPE="false",
                   PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", TORCH_COMPILE_DISABLE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    if gpu_id is not None:
        updates["CUDA_VISIBLE_DEVICES"] = gpu_id
    previous = {key: os.environ.get(key) for key in updates}
    before = Path.cwd()
    previous_bytecode = sys.dont_write_bytecode
    for value in (runtime / "tmp", cache / "tmp"):
        value.mkdir()
    sys.path.insert(0, str(source))
    os.environ.update(updates)
    os.chdir(runtime)
    sys.dont_write_bytecode = True
    try:
        with deny_network():
            yield updates
    finally:
        sys.dont_write_bytecode = previous_bytecode
        os.chdir(before)
        if sys.path[0] == str(source):
            sys.path.pop(0)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def cache_normalizer_class(base, cache_root, *, factories=None, system=None):
    """Original cache-only adapter; all text/FST operations remain the real package's.

    The injected factories are only for CPU argument-contract unit tests. Runtime
    always imports the actual platform normalizers and requires explicit cache_dir.
    """
    cache_root = Path(cache_root).resolve()
    system = platform.system() if system is None else system
    if factories is None:
        if system == "Linux":
            from tn.chinese.normalizer import Normalizer as zh
            from tn.english.normalizer import Normalizer as en
            factories = (zh, en)
        else:
            from wetext import Normalizer
            factories = (Normalizer, Normalizer)
    if len(factories) != 2 or any("cache_dir" not in inspect.signature(factory).parameters for factory in factories):
        raise SynthesisError("Installed real normalizers must expose explicit cache_dir; global/default cache writes are disallowed")
    class IsolatedCacheNormalizer(base):
        def load(self):
            if self.zh_normalizer is not None and self.en_normalizer is not None:
                return
            cache_root.mkdir(parents=True, exist_ok=False)
            zh_cache, en_cache = cache_root / "zh", cache_root / "en"
            zh_cache.mkdir()
            en_cache.mkdir()
            if system == "Linux":
                self.zh_normalizer = factories[0](cache_dir=str(zh_cache), remove_interjections=False,
                                                  remove_erhua=False, overwrite_cache=False)
                self.en_normalizer = factories[1](cache_dir=str(en_cache), overwrite_cache=False)
            else:
                self.zh_normalizer = factories[0](cache_dir=str(zh_cache), remove_erhua=False, lang="zh", operator="tn")
                self.en_normalizer = factories[1](cache_dir=str(en_cache), lang="en", operator="tn")
    return IsolatedCacheNormalizer


@contextmanager
def _runtime_import(source, cache_environment):
    for name, module in tuple(sys.modules.items()):
        if name.startswith("indextts") and getattr(module, "__file__", None) and not Path(module.__file__).resolve().is_relative_to(source):
            raise SynthesisError("Another IndexTTS tree is imported; use a fresh isolated synthesis process")
    downloads = importlib.import_module("indextts.utils.model_download")
    def forbidden(*args, **kwargs):
        raise SynthesisError("Implicit auxiliary download/migration is disabled; stage and verify all assets explicitly")
    saved = {}
    module, original_normalizer, constants, saved_hub = None, None, None, None
    try:
        for name in ("ensure_models_available", "ensure_config_available", "snapshot_download", "_download_single_file"):
            if hasattr(downloads, name):
                saved[name] = getattr(downloads, name)
                setattr(downloads, name, forbidden)
        module = importlib.import_module("indextts.infer_v2_5")
        if not Path(inspect.getfile(module.IndexTTS2)).resolve().is_relative_to(source):
            raise SynthesisError("Synthesis runtime is not the verified official implementation")
        # The stock import sets a relative Hub cache. Reassert this process's cache.
        os.environ.update(cache_environment)
        import huggingface_hub.constants as constants
        saved_hub = constants.HF_HUB_CACHE, constants.HF_HUB_OFFLINE
        constants.HF_HUB_CACHE = cache_environment["HF_HUB_CACHE"]
        constants.HF_HUB_OFFLINE = True
        original_normalizer = module.TextNormalizer
        module.TextNormalizer = cache_normalizer_class(original_normalizer, Path(cache_environment["XDG_CACHE_HOME"]) / "tn")
        yield module.IndexTTS2
    finally:
        for name, value in saved.items():
            setattr(downloads, name, value)
        if original_normalizer is not None:
            module.TextNormalizer = original_normalizer
        if constants is not None and saved_hub is not None:
            constants.HF_HUB_CACHE, constants.HF_HUB_OFFLINE = saved_hub


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute_once(args, report, state):
    if not args.allow_cuda:
        raise SynthesisError("No actual generation without explicit --allow-cuda")
    if state["output"] is None or state["text"] is None or state["prompt"] is None:
        raise SynthesisError("Execution needs a fresh --output-dir, reviewed --text/--text-file and --prompt")
    dependencies = report["dependencies"]
    if dependencies["missing"]:
        raise SynthesisError("Stock synthesis dependencies/frontends are missing; no automatic install is attempted")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise SynthesisError("Use a fresh process with no existing CUDA context")
    admission = resource_admission(args)
    output = state["output"]
    output.mkdir(mode=0o700, exist_ok=False)
    stage, tts, runtime_context = "runtime_import", None, None
    try:
        with isolated_runtime(output, state["source"], admission[0]["uuid"]) as environment, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            runtime_context = _runtime_import(state["source"], environment)
            constructor = runtime_context.__enter__()
            report["runtime_import_verified"] = True
            import numpy as np
            import torch
            torch.set_num_threads(2)
            report["cuda_accessed"] = True
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                raise SynthesisError("One explicitly selected BF16-capable CUDA device is required")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            from indextts.utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(multilingual=True, num_languages=99, language="hi", task="transcribe", model_dir=str(state["original"]))
            prefix = "<|hi|> "
            raw = tokenizer.encode(state["text"], allowed_special=set())
            tokens = tokenizer.encode(prefix + state["text"], allowed_special={"<|hi|>"})
            budget = min(args.max_text_tokens, state["config"]["gpt"]["max_text_tokens"]) - len(tokenizer.encode(prefix, allowed_special={"<|hi|>"}))
            if len(raw) > budget or any(token in (state["config"]["gpt"]["start_text_token"], state["config"]["gpt"]["stop_text_token"]) for token in tokens):
                raise SynthesisError("Text would split or overlap stock framing IDs; shorten/review it without silent token deletion")
            selected = state["original"] / "gpt.pth" if args.variant == "original" else state["merged"] / "gpt.pth"
            config = dict(state["config"], gpt_checkpoint=str(selected))
            import yaml
            config_path = output / "runtime" / "selected-config.yaml"
            with config_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
            # Recheck current resources after imports; a snapshot is not a GPU reservation.
            current_admission = resource_admission(args)
            if current_admission[0]["uuid"] != admission[0]["uuid"]:
                raise SynthesisError("Selected physical GPU changed before model loading")
            admission.extend(current_admission)
            selected_expected = report["asset_identity"]["files"]["gpt.pth"] if args.variant == "original" else report["merged_identity"]["gpt_sha256"]
            if sha256(selected) != selected_expected:
                raise SynthesisError("Selected GPT changed after metadata verification")
            stage = "model_construction"
            report["model_construction_started"] = True
            tts = constructor(cfg_path=str(config_path), model_dir=str(state["original"]), use_bf16=True,
                              device="cuda:0", **DISABLED_FLAGS)
            report["models_loaded"] = True
            if Path(tts.gpt_path).resolve() != selected.resolve() or tts.qwen_emo is not None or tts.use_accel or tts.use_cuda_kernel or tts.use_torch_compile:
                raise SynthesisError("Official runtime did not retain the required checkpoint/disabled acceleration settings")
            if len(tts.split_text_by_tokens(state["text"], args.max_text_tokens, prefix)) != 1:
                raise SynthesisError("This acceptance command supports exactly one text segment")
            stage = "generation"
            report["generation_attempted"] = True
            partial = output / ".audio.partial.wav"
            torch.cuda.synchronize()
            started = time.perf_counter()
            generation = {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                          "max_mel_tokens": args.max_mel_tokens}
            tts.infer(spk_audio_prompt=str(state["prompt"]), text=state["text"], output_path=str(partial), lang="hi",
                      use_emo_text=False, use_random=False, text_normalization=False, stream_return=False,
                      max_text_tokens_per_segment=args.max_text_tokens, verbose=False, **generation)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if any("exceeding `max_mel_tokens`" in str(item.message) for item in captured):
                raise SynthesisError("Stock generation hit the maximum-code limit; truncated output is not an acceptance pass")
            stage = "audio_validation"
            audio = wav_info(partial)
            os.link(partial, output / "audio.wav")
            partial.unlink()
            report.update(status="single_audio_generated_quality_unverified", runtime_import_verified=True,
                          models_loaded=True, cuda_accessed=True, synthesis_executed=True,
                          generation={"language":"hi", "text_normalization":False, "seed":args.seed,
                                      "strict_deterministic_algorithms":True, "settings":generation,
                                      "text_tokens":len(tokens), "token_ids_sha256":hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                      "selected_config_sha256":sha256(config_path), "selected_gpt_sha256":sha256(selected)},
                          admission_snapshots=admission, admission_scope="Observed headroom only; no GPU reservation or future ComfyUI peak guarantee.",
                          output=audio, elapsed_seconds=elapsed, old_language_regression_verified=False,
                          stock_frontend_full_parity_verified=False, cache_only_normalizer_adapter=True)
            _write_json(output / "result.json", report)
            return report
    except BaseException as exc:
        failure = dict(report, status="synthesis_blocked_or_failed", failure_stage=stage,
                       failure_type=type(exc).__name__, reason=str(exc)[:4000], synthesis_executed=False,
                       quality_verified=False, admission_snapshots=admission)
        _write_json(output / "failure.json", failure)
        raise
    finally:
        if runtime_context is not None:
            runtime_context.__exit__(None, None, None)
        if tts is not None:
            del tts
        if "torch" in sys.modules:
            import gc
            gc.collect()
            if sys.modules["torch"].cuda.is_initialized():
                sys.modules["torch"].cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace-root", "official-repo", "model-dir", "model-manifest", "merged-bundle"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--check-deps-only", action="store_true", help="Only inspect module presence/versions and the pinned source pyproject; no model import or download")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", choices=("original", "merged"), default="merged")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-text-tokens", type=int, default=120)
    parser.add_argument("--max-mel-tokens", type=int, default=256)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--min-free-gpu-mib", type=float)
    parser.add_argument("--min-free-ram-mib", type=float)
    parser.add_argument("--max-utilization", type=float)
    args = parser.parse_args(argv)
    if args.check_deps_only and args.execute:
        parser.error("Dependency-only checks cannot execute synthesis")
    if not args.check_deps_only and any(getattr(args, name.replace("-", "_")) is None for name in ("workspace-root", "official-repo", "model-dir", "model-manifest", "merged-bundle")):
        parser.error("Metadata verification requires workspace/source/original manifest/merged paths")
    if args.execute and not args.allow_cuda:
        parser.error("--execute requires explicit --allow-cuda")
    if not 0 <= args.seed < 2**32 or not 8 <= args.max_text_tokens <= 120 or not 8 <= args.max_mel_tokens <= 512:
        parser.error("Use a uint32 seed and bounded text/code generation limits")
    try:
        if args.check_deps_only:
            source = verify_source(args.official_repo) if args.official_repo is not None else None
            report = {"status":"dependency_metadata_only", "models_loaded":False, "cuda_accessed":False,
                      "synthesis_executed":False, "dependencies":dependency_report(official_repo=source)}
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
            return 2 if report["dependencies"]["missing"] else 0
        if args.text_file:
            if not args.text_file.resolve(strict=True).is_relative_to(args.workspace_root.resolve(strict=True)):
                raise SynthesisError("Acceptance text file must be inside the isolated workspace")
            if not args.text_file.is_file() or args.text_file.stat().st_size > 16384:
                raise SynthesisError("Text file must be a small local UTF-8 file")
            args.text = args.text_file.read_text(encoding="utf-8")
        report, state = metadata_preflight(args)
        if args.execute:
            report = execute_once(args, report, state)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ImportError) as exc:
        print(json.dumps({"status":"blocked", "reason":str(exc), "synthesis_executed":False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not report["dependencies"]["missing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
