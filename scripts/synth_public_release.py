#!/usr/bin/env python3
"""Offline Hindi inference from the reviewed 13-file public best100 release.

Source checkout required: this entry reuses the adjacent prepare_hf_release.py
verifier and the existing synthesis runtime. Use --merged-bundle for the public
release directory, --model-dir for the independent full official 2.5 assets,
and --model-manifest for those original assets. Native private COMPLETE/provenance
are not needed or recreated. Default: metadata only; execution still requires
--execute --allow-cuda and every original source/asset/input/resource guard.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hi_h200_native import synth


def _preparation_verifier():
    path = Path(__file__).with_name("prepare_hf_release.py")
    if path.is_symlink() or not path.is_file():
        raise ImportError("Use the complete reviewed source checkout, including release preparation")
    spec = importlib.util.spec_from_file_location("_hi_h200_reviewed_release_preparation", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import the reviewed local release verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _preparation_verifier()


def verify_public_bundle(root, original_root, asset_identity):
    """Bind the strict public release to the already verified original assets.

    No private export metadata is manufactured. The core runtime needs only the
    verified selected GPT identity, and records these actual public JSON hashes.
    """
    root = Path(root).resolve(strict=True)
    original = Path(original_root).resolve(strict=True)
    if root == original or root.is_relative_to(original) or original.is_relative_to(root):
        raise synth.SynthesisError("Original assets and public release must be independent directories")
    if (not isinstance(asset_identity, dict)
            or asset_identity.get("repository") != release.MODEL_REPO
            or asset_identity.get("snapshot_revision") != release.MODEL_REVISION
            or not isinstance(asset_identity.get("files"), dict)):
        raise synth.SynthesisError("Public inference requires the fixed verified original 2.5 assets")
    try:
        with synth.deny_network():
            verified = release.verify_release(root)
            provenance, provenance_sha = release.bounded_json(root / "release.json", with_digest=True)
            _, manifest_sha = release.bounded_json(root / "manifest.json", with_digest=True)
    except release.ReleaseError as exc:
        raise synth.SynthesisError(str(exc)) from exc
    except Exception as exc:
        raise synth.SynthesisError("Cannot verify the bounded local public release metadata") from exc
    files = asset_identity["files"]
    if (provenance.get("base_model") != asset_identity["repository"]
            or provenance.get("base_revision") != asset_identity["snapshot_revision"]
            or provenance.get("official_source_commit") != synth.OFFICIAL_COMMIT
            or provenance.get("base_gpt_sha256") != files.get("gpt.pth")
            or provenance.get("gpt_sha256") != verified["gpt_sha256"]
            or provenance.get("checkpoint_step") != verified["checkpoint_step"]):
        raise synth.SynthesisError("Public release and actual original GPT/source identities differ")
    for published, original_name in (("config.yaml", "config.yaml"), (release.TOKENIZER, release.TOKENIZER),
                                     ("MODEL_LICENSE", "LICENSE")):
        if synth.sha256(root / published) != files.get(original_name):
            raise synth.SynthesisError("Public config/tokenizer/model license differ from the verified original assets")
    if os.path.samefile(root / "gpt.pth", original / "gpt.pth"):
        raise synth.SynthesisError("Public candidate cannot alias the original GPT")
    return {"format": "reviewed_public_release_v1", "gpt_sha256": verified["gpt_sha256"],
            "manifest_sha256": manifest_sha, "release_sha256": provenance_sha,
            "optimizer_updates": verified["checkpoint_step"], "verified_public_files": verified["files"]}


def main(argv=None):
    return synth.main(argv, bundle_verifier=verify_public_bundle, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
