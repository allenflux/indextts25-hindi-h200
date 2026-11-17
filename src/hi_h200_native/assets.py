"""Plan or explicitly stage exact public, pinned native-training assets.

Default execution fetches repository metadata only. Downloads require a byte cap,
free-space reserve, and a new destination; no caller's HF login is used.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import unicodedata
import uuid

PUBLIC_ENDPOINT = "https://huggingface.co"
MODEL_REPO = "IndexTeam/IndexTTS-2.5"
MODEL_REVISION = "c39ce5ba981572cb187443877ff559dfb246ce63"
W2V_REPO = "facebook/w2v-bert-2.0"
W2V_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
CAMP_REPO = "funasr/campplus"
CAMP_REVISION = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
BIGVGAN_REVISION = "633ff708ed5b74903e86ff1298cf4a98e921c513"


class AssetError(ValueError):
    pass


@dataclass(frozen=True)
class Asset:
    repository: str
    revision: str
    filename: str
    relative_path: str
    size: int
    hash_kind: str
    digest: str


# Immutable public metadata independently checked on 2026-11-17. Small-file
# digests are Git blob SHA1 (header + file), not a mislabeled file SHA256.
TRAIN = (
    Asset(MODEL_REPO, MODEL_REVISION, "config.yaml", "config.yaml", 2860, "git_blob_sha1", "b3f15b22f394f2327de574e9b5801c1d9654e203"),
    Asset(MODEL_REPO, MODEL_REVISION, "gpt.pth", "gpt.pth", 3259599833, "sha256", "43a8f4c30eccdf201958d3b9713511482c19d56dc20b0b1c4ee1e6b080b19d85"),
    Asset(MODEL_REPO, MODEL_REVISION, "multilingual_zh_ja_yue_char_del.tiktoken", "multilingual_zh_ja_yue_char_del.tiktoken", 907395, "git_blob_sha1", "ac47fe685d24565030965a94c0d98e00e0331eaa"),
    Asset(MODEL_REPO, MODEL_REVISION, "LICENSE", "LICENSE", 10553, "git_blob_sha1", "b96dffb19fc6e0e048edf010f1cd0d54e8ce103c"),
    Asset(MODEL_REPO, MODEL_REVISION, "README.md", "README.md", 4272, "git_blob_sha1", "7a58eb771ef86135b363c26ace868810ce1a2acc"),
)
EXTRACT = (
    Asset(MODEL_REPO, MODEL_REVISION, "codec.pth", "codec.pth", 607290935, "sha256", "d15cbed16a40f478438c961fb043f68dfa6353bf56c966761315db3433e9722c"),
    Asset(MODEL_REPO, MODEL_REVISION, "wav2vec2bert_stats.pt", "wav2vec2bert_stats.pt", 9343, "sha256", "c9c176c2b8850ab2e3ba828bbfa969deaf4566ce55db5f2687b8430b87526ad2"),
    Asset(W2V_REPO, W2V_REVISION, "config.json", "hf_cache/w2v-bert-2.0/config.json", 1874, "git_blob_sha1", "a383a594dac18459628cd2837168cd276342a31a"),
    Asset(W2V_REPO, W2V_REVISION, "preprocessor_config.json", "hf_cache/w2v-bert-2.0/preprocessor_config.json", 275, "git_blob_sha1", "5db61951cdf5edab6337fd84ee619500c27aaa3d"),
    Asset(W2V_REPO, W2V_REVISION, "model.safetensors", "hf_cache/w2v-bert-2.0/model.safetensors", 2322063736, "sha256", "eb890c9660ed6e3414b6812e27257b8ce5454365d5490d3ad581ea60b93be043"),
    Asset(W2V_REPO, W2V_REVISION, "README.md", "hf_cache/w2v-bert-2.0/README.md", 4792, "git_blob_sha1", "23115b6e9bf0776b4cc3d480fde0a90a2ff33011"),
    Asset(CAMP_REPO, CAMP_REVISION, "campplus_cn_common.bin", "hf_cache/campplus_cn_common.bin", 28036335, "sha256", "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"),
    Asset(CAMP_REPO, CAMP_REVISION, "README.md", "hf_cache/campplus/README.md", 2020, "git_blob_sha1", "10b7ae7912f6c4eae3b67316ed59902963591a58"),
)
SYNTH = (
    Asset(MODEL_REPO, MODEL_REVISION, "s2mel.pth", "s2mel.pth", 414908601, "sha256", "9b1b0003fc189c94cc349758d7ebc25f903b7eb2de4602879959cc64ce816456"),
    Asset(MODEL_REPO, MODEL_REVISION, "feat1.pt", "feat1.pt", 57170, "sha256", "f219cb447d80216ba615666da2ff8d63ac544eee26657f3a7b278692bf7a67c4"),
    Asset(MODEL_REPO, MODEL_REVISION, "feat2.pt", "feat2.pt", 374866, "sha256", "9c4292e96dee535aea9a6206e9a0c856dd578dde9212acdb16dd3ada4d12bf80"),
    Asset(BIGVGAN_REPO, BIGVGAN_REVISION, "config.json", "hf_cache/bigvgan/config.json", 1405, "git_blob_sha1", "635bd8975629bd6d4b51c409986944a281cfe7be"),
    Asset(BIGVGAN_REPO, BIGVGAN_REVISION, "bigvgan_generator.pt", "hf_cache/bigvgan/bigvgan_generator.pt", 449228171, "sha256", "e95ba25972d3de0628d99cd156e9315a9c018899bf739988959ebe3544080ced"),
    Asset(BIGVGAN_REPO, BIGVGAN_REVISION, "LICENSE", "hf_cache/bigvgan/LICENSE", 1076, "git_blob_sha1", "45b7741b4516c22d7314c5f856b17f713bac91af"),
    Asset(BIGVGAN_REPO, BIGVGAN_REVISION, "README.md", "hf_cache/bigvgan/README.md", 8038, "git_blob_sha1", "cb0adf5a6815c78cb7993b3b8083427d20e7357d"),
)

LICENSES = {
    MODEL_REPO: {"name": "bilibili Model Use License Agreement", "reference": f"https://huggingface.co/{MODEL_REPO}/blob/{MODEL_REVISION}/LICENSE"},
    W2V_REPO: {"name": "MIT (public model-card declaration)", "reference": f"https://huggingface.co/{W2V_REPO}/blob/{W2V_REVISION}/README.md"},
    CAMP_REPO: {"name": "Apache-2.0 (public model-card declaration)", "reference": f"https://huggingface.co/{CAMP_REPO}/blob/{CAMP_REVISION}/README.md"},
    BIGVGAN_REPO: {"name": "MIT", "reference": f"https://huggingface.co/{BIGVGAN_REPO}/blob/{BIGVGAN_REVISION}/LICENSE"},
}


def safe_relative(value):
    if not isinstance(value, str) or not value or ":" in value or "\\" in value:
        raise AssetError("Asset paths must be local relative POSIX paths")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
        raise AssetError("Control/format characters in asset path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise AssetError("Unsafe relative asset path")
    return Path(*path.parts)


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def selected_assets(stage):
    if stage not in ("train", "extract", "synth"):
        raise AssetError("Stage must be train, extract or synth")
    assets = TRAIN if stage == "train" else TRAIN + EXTRACT if stage == "extract" else TRAIN + EXTRACT + SYNTH
    names = set()
    for asset in assets:
        safe_relative(asset.filename)
        safe_relative(asset.relative_path)
        if asset.relative_path in names or type(asset.size) is not int or asset.size <= 0:
            raise AssetError("Duplicate asset destination or invalid expected size")
        names.add(asset.relative_path)
    return assets


def plan_assets(stage="train", *, api=None):
    assets = selected_assets(stage)
    if api is None:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from huggingface_hub import HfApi, constants
        constants.HF_HUB_DISABLE_XET = True
        api = HfApi(endpoint=PUBLIC_ENDPOINT, token=False)
    by_repo = {}
    for asset in assets:
        by_repo.setdefault((asset.repository, asset.revision), []).append(asset)
    for (repository, revision), expected in by_repo.items():
        try:
            info = api.model_info(repository, revision=revision, files_metadata=True,
                                  token=False, timeout=30)
        except Exception as exc:
            raise AssetError("Public metadata request failed; no authenticated fallback: " + repository) from exc
        if _field(info, "sha") != revision:
            raise AssetError("Public snapshot revision does not match the immutable lock")
        if _field(info, "gated") is not False and _field(info, "gated") is not None:
            raise AssetError("Gated repository is outside this public asset helper")
        siblings = _field(info, "siblings")
        if not isinstance(siblings, (list, tuple)):
            raise AssetError("Missing public file metadata")
        files = {}
        for sibling in siblings:
            name = _field(sibling, "rfilename")
            if name in files:
                raise AssetError("Duplicate public file metadata")
            files[name] = sibling
        for asset in expected:
            item = files.get(asset.filename)
            if item is None or _field(item, "size") != asset.size:
                raise AssetError("Missing asset or changed pinned size: " + asset.filename)
            if asset.hash_kind == "sha256":
                lfs = _field(item, "lfs")
                if _field(lfs, "sha256") != asset.digest or _field(lfs, "size") != asset.size:
                    raise AssetError("Pinned LFS integrity metadata changed: " + asset.filename)
            elif _field(item, "blob_id") != asset.digest:
                raise AssetError("Pinned Git blob metadata changed: " + asset.filename)
    repositories = sorted({asset.repository for asset in assets})
    return {"schema_version": 1, "stage": stage, "metadata_verified": True,
            "downloaded": False, "network_payload_bytes": sum(asset.size for asset in assets),
            "files": [asdict(asset) for asset in assets],
            "licenses": {name: LICENSES[name] for name in repositories},
            "permission_scope": "Public access is not an independent legal permission grant; review the preserved license/model-card terms before use.",
            "endpoint": PUBLIC_ENDPOINT,
            "auth": "All requests explicitly use token=False; no user login or gated acceptance.",
            "budget_scope": "max-bytes caps the exact selected payload size; protocol overhead/retry traffic is not a metered bandwidth quota."}


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("metadata_verified") is not True or plan.get("downloaded") is not False:
        raise AssetError("Use a freshly verified metadata plan")
    assets = selected_assets(plan.get("stage"))
    if plan.get("files") != [asdict(asset) for asset in assets] or plan.get("network_payload_bytes") != sum(asset.size for asset in assets):
        raise AssetError("Metadata plan changed after selection")
    return assets


def _new_destination(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink() or path.name in ("", ".", ".."):
        raise AssetError("Destination must be fresh; existing models are never overwritten")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise AssetError("Destination parent must exist")
    return parent / path.name


def check_budget(plan, destination, cache_dir, max_bytes, reserve_bytes):
    assets = _validate_plan(plan)
    if type(max_bytes) is not int or max_bytes < 1 or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise AssetError("Explicit positive max-bytes and nonnegative reserve-bytes are required")
    total = sum(asset.size for asset in assets)
    if total > max_bytes:
        raise AssetError("Exact selected payload exceeds max-bytes; no files downloaded")
    destination = _new_destination(destination)
    cache = Path(cache_dir).absolute()
    if cache.is_symlink():
        raise AssetError("Cache directory must not be a symlink")
    cache_parent = cache if cache.exists() else cache.parent
    cache_parent = cache_parent.resolve(strict=True)
    if not cache_parent.is_dir():
        raise AssetError("Cache parent must exist")
    shared_device = os.stat(destination.parent).st_dev == os.stat(cache_parent).st_dev
    required_destination = total * (2 if shared_device else 1) + reserve_bytes
    if shutil.disk_usage(destination.parent).free < required_destination:
        raise AssetError("Insufficient free space for snapshot, retained cache and reserve")
    if not shared_device and shutil.disk_usage(cache_parent).free < total + reserve_bytes:
        raise AssetError("Insufficient cache space plus reserve")
    if cache.exists() and not cache.is_dir():
        raise AssetError("Cache path is not a directory")
    return destination, cache, total


def copy_verified(source, target, asset):
    """Copy one bounded file while hashing the exact bytes that will be published."""
    size = 0
    digest = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(asset.size).encode("ascii") + b"\x00")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            if size > asset.size:
                raise AssetError("Downloaded asset exceeds its exact pinned size")
            outgoing.write(chunk)
            digest.update(chunk)
            blob.update(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    actual = digest.hexdigest() if asset.hash_kind == "sha256" else blob.hexdigest()
    if size != asset.size or actual != asset.digest:
        raise AssetError("Downloaded asset size/hash mismatch: " + asset.filename)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_no_replace(source, destination):
    """Atomic no-clobber rename; fail closed when the platform cannot provide it."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:
            raise AssetError("Atomic no-replace directory publication is unavailable")
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        function = getattr(library, "renamex_np", None)
        if function is None:
            raise AssetError("Atomic no-replace directory publication is unavailable")
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(source), os.fsencode(destination), 4)
    else:
        raise AssetError("This platform has no reviewed no-replace publication path")
    if result != 0:
        value = ctypes.get_errno()
        raise OSError(value, os.strerror(value), str(destination))


def stage_assets(plan, destination, cache_dir, *, max_bytes, reserve_bytes, downloader=None):
    """Download only listed assets into a new snapshot; leave partial cache resumable."""
    assets = _validate_plan(plan)
    destination, cache, total = check_budget(plan, destination, cache_dir, max_bytes, reserve_bytes)
    # Set these before the lazy Hub import; token=False remains mandatory even if
    # the caller already imported the library. Disable Xet's separate global cache.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if downloader is None:
        from huggingface_hub import hf_hub_download, constants
        constants.HF_HUB_DISABLE_XET = True
        downloader = hf_hub_download
    cache.mkdir(exist_ok=True)
    cache = cache.resolve(strict=True)
    stage = destination.parent / ("." + destination.name + ".tmp-" + uuid.uuid4().hex)
    stage.mkdir()
    hashes = {}
    try:
        for asset in assets:
            try:
                source = Path(downloader(repo_id=asset.repository, filename=asset.filename,
                                         revision=asset.revision, repo_type="model", token=False,
                                         cache_dir=str(cache), etag_timeout=30,
                                         endpoint=PUBLIC_ENDPOINT)).resolve(strict=True)
            except Exception as exc:
                raise AssetError("Public file request failed; no authenticated fallback: " + asset.filename) from exc
            if not source.is_relative_to(cache) or not source.is_file():
                raise AssetError("Downloader returned a file outside the isolated cache")
            target = stage / safe_relative(asset.relative_path)
            hashes[asset.relative_path] = copy_verified(source, target, asset)
        try:
            import yaml
            config = yaml.safe_load((stage / "config.yaml").read_text(encoding="utf-8"))
        except (ImportError, ValueError) as exc:
            raise AssetError("PyYAML is required to verify the actual downloaded config") from exc
        if not isinstance(config, dict) or str(config.get("version")) != "2.5":
            raise AssetError("Downloaded config is not version 2.5")
        for repository, filename in ((W2V_REPO, "w2v-bert-2.0.manifest.json"),
                                     (CAMP_REPO, "campplus.manifest.json"),
                                     (BIGVGAN_REPO, "bigvgan.manifest.json")):
            selected = [asset for asset in assets if asset.repository == repository]
            if not selected:
                continue
            record = {"repository": repository, "snapshot_revision": selected[0].revision,
                      "license": LICENSES[repository],
                      "files": {asset.relative_path: hashes[asset.relative_path] for asset in selected}}
            _write_json(stage / filename, record)
            hashes[filename] = hashlib.sha256((stage / filename).read_bytes()).hexdigest()
        _write_json(stage / "asset-plan.json", plan)
        hashes["asset-plan.json"] = hashlib.sha256((stage / "asset-plan.json").read_bytes()).hexdigest()
        _write_json(stage / "model-manifest.json", {"repository": MODEL_REPO,
                    "snapshot_revision": MODEL_REVISION, "files": hashes})
        if destination.exists() or destination.is_symlink():
            raise AssetError("Destination appeared during staging; refusing overwrite")
        # Same-parent stage ensures same-filesystem publication; no existing
        # destination is removed. The model files themselves were fsync'ed.
        publish_no_replace(stage, destination)
        directory_fsync = True
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
                directory_fsync = False
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"status": "assets_verified_not_loaded", "stage": plan["stage"],
            "destination": str(destination), "manifest": str(destination / "model-manifest.json"),
            "payload_bytes": total, "files_downloaded": len(assets),
            "model_loaded": False, "gpu_used": False, "training_started": False,
            "directory_fsync_verified": directory_fsync}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "extract", "synth"), default="train")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--reserve-bytes", type=int)
    args = parser.parse_args(argv)
    try:
        plan = plan_assets(args.stage)
        if not args.download:
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.destination is None or args.cache_dir is None:
            raise AssetError("Download requires explicit destination and isolated cache-dir")
        result = stage_assets(plan, args.destination, args.cache_dir,
                              max_bytes=args.max_bytes, reserve_bytes=args.reserve_bytes)
    except (AssetError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "download_complete": False}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
