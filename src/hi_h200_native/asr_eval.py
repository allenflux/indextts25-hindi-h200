"""Small opt-in Whisper Hindi ASR probe; scores are not a TTS quality gate.

Official openai-whisper v20250625 code AND weights are MIT licensed:
https://github.com/openai/whisper/tree/v20250625 . No model is downloaded by
plan/execute. Only --download transfers the one fixed public small checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import ssl
import unicodedata
import urllib.parse
import urllib.request
import wave

from hi_h200.data import read_jsonl
from .synth import deny_network, sha256, wav_info

VERSION = "20250625"
MODEL_BYTES = 483617219
MODEL_SHA = "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
MODEL_URL = "https://openaipublic.azureedge.net/main/whisper/models/" + MODEL_SHA + "/small.pt"
SOURCE = "https://github.com/openai/whisper/tree/v20250625"
LICENSE_URL = "https://github.com/openai/whisper/blob/v20250625/LICENSE"
MAX_CASES = 12
SETTINGS = dict(language="hi", task="transcribe", temperature=0.0, beam_size=5,
                condition_on_previous_text=False, initial_prompt=None, carry_initial_prompt=False,
                fp16=False, word_timestamps=False, verbose=False,
                compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6)


class ASRError(ValueError):
    pass


def normalized(text, punctuation=False):
    if not isinstance(text, str) or len(text) > 4096:
        raise ASRError("Text must be a bounded string")
    if any(unicodedata.category(c).startswith("C") and c not in "\u200c\u200d" and not c.isspace() for c in text):
        raise ASRError("Unsupported control characters in ASR text")
    value = unicodedata.normalize("NFC", text)
    if punctuation:
        value = "".join(" " if unicodedata.category(c).startswith("P") else c for c in value)
    return re.sub(r"\s+", " ", value).strip()


def edit_distance(reference, hypothesis):
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def scores(reference, hypothesis, punctuation=False):
    ref, hyp = normalized(reference, punctuation), normalized(hypothesis, punctuation)
    if not ref:
        raise ASRError("Reference must remain nonempty under the declared metric policy")
    characters, words = edit_distance(ref, hyp), edit_distance(ref.split(), hyp.split())
    return dict(reference=ref, hypothesis=hyp, char_edits=characters, reference_chars=len(ref),
                cer=characters / len(ref), word_edits=words, reference_words=len(ref.split()), wer=words / len(ref.split()))


def workspace(value):
    root = Path(value).resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ASRError("Declare an existing dedicated project workspace")
    return root


def contained(root, value, *, exists=False):
    path = Path(value)
    if any(unicodedata.category(c).startswith("C") for c in str(path)):
        raise ASRError("Control characters are not allowed in paths")
    result = (path if path.is_absolute() else root / path).resolve(strict=exists)
    if not result.is_relative_to(root) or result == root:
        raise ASRError("ASR inputs/models/outputs must remain inside the declared project workspace")
    return result


def package():
    try:
        dist = importlib.metadata.distribution("openai-whisper")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ASRError("Install the separately pinned openai-whisper20250625 environment first") from exc
    if dist.version != VERSION:
        raise ASRError("ASR requires exactly openai-whisper20250625")
    license_file = next((dist.locate_file(name) for name in (dist.files or []) if str(name).endswith(".dist-info/licenses/LICENSE")), None)
    if license_file is None or Path(license_file).stat().st_size > 65536:
        raise ASRError("Installed Whisper MIT license metadata is missing")
    text = Path(license_file).read_text(encoding="utf-8")
    if "MIT License" not in text or "OpenAI" not in text:
        raise ASRError("Unexpected installed Whisper license metadata")
    return text


def verify_model(folder):
    path = folder / "small.pt"
    if path.is_symlink() or not path.is_file() or path.stat().st_size != MODEL_BYTES or sha256(path) != MODEL_SHA:
        raise ASRError("Local small.pt must match the fixed official byte count and full SHA256")
    return path


class FixedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        destination = urllib.parse.urlsplit(new_url)
        if destination.scheme != "https" or destination.hostname != "openaipublic.azureedge.net" or destination.username or destination.password or destination.port not in (None, 443):
            raise ASRError("ASR model download refused a different destination")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def download_model(folder, maximum, *, opener=None):
    if type(maximum) is not int or not MODEL_BYTES <= maximum <= 500000000:
        raise ASRError("The one-checkpoint download budget must cover 483617219 bytes and be at most 500000000")
    license_text = package()
    if folder.exists():
        raise ASRError("Download model directory must be fresh; existing assets are never overwritten")
    folder.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder.parent).free < MODEL_BYTES + 256 * 1024 * 1024:
        raise ASRError("Insufficient disk headroom for the ASR checkpoint")
    folder.mkdir()
    partial = folder / "small.pt.partial"
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), FixedRedirect(),
                                                   urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    digest, count = hashlib.sha256(), 0
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "hi-h200-asr-eval/0.1"})
    with opener.open(request, timeout=30) as response, partial.open("xb") as output:
        if response.headers.get("Content-Length") != str(MODEL_BYTES):
            raise ASRError("Official ASR Content-Length differs from the fixed byte count")
        while chunk := response.read(min(1024 * 1024, MODEL_BYTES - count + 1)):
            count += len(chunk)
            if count > MODEL_BYTES or count > maximum:
                raise ASRError("ASR checkpoint exceeded its byte budget")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if count != MODEL_BYTES or digest.hexdigest() != MODEL_SHA:
        raise ASRError("Official ASR checkpoint SHA256 mismatch")
    os.link(partial, folder / "small.pt")
    partial.unlink()
    (folder / "LICENSE.whisper").write_text(license_text, encoding="utf-8")
    return {"status": "downloaded_verified", "bytes": count, "sha256": digest.hexdigest(),
            "source": MODEL_URL, "license": "MIT", "package_version": VERSION}


def load_cases(root, manifest):
    records, seen, grouped = [], set(), {}
    for row in read_jsonl(manifest):
        uid, condition = row.get("id"), row.get("condition")
        if not isinstance(uid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", uid) or condition not in {"reference", "baseline", "merged"}:
            raise ASRError("Cases require a safe id and reference/baseline/merged condition")
        if (uid, condition) in seen or len(records) >= MAX_CASES:
            raise ASRError("Duplicate case or more than 12 cases")
        text = row.get("text")
        if not normalized(text) or not normalized(text, True):
            raise ASRError("Case reference text is empty")
        path = contained(root, row.get("audio", ""), exists=True)
        if path.suffix.lower() != ".wav" or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ASRError("Cases must use bounded local WAV files")
        stats = wav_info(path)
        if stats["sample_rate"] not in (22050, 24000) or not 1 <= stats["seconds"] <= 20:
            raise ASRError("ASR input must be complete mono PCM16 at 22050/24000 Hz and 1 to 20 seconds")
        prior = grouped.setdefault(uid, {"text": text, "conditions": set()})
        if prior["text"] != text:
            raise ASRError("Paired conditions must use identical target text")
        prior["conditions"].add(condition)
        seen.add((uid, condition))
        records.append(dict(id=uid, condition=condition, text=text, path=path, audio=stats))
    if not records or not any(row["condition"] == "reference" for row in records):
        raise ASRError("Include real held-out reference calibration audio; the caller must verify that role")
    if any(bool(item["conditions"] & {"baseline", "merged"}) and not {"baseline", "merged"}.issubset(item["conditions"]) for item in grouped.values()):
        raise ASRError("Generated probes must pair baseline and merged conditions")
    return records


def waveform(path):
    import numpy as np
    from scipy.signal import resample_poly
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype("float32") / 32768
    factor = math.gcd(rate, 16000)
    return resample_poly(samples, 16000 // factor, rate // factor).astype("float32")


def runtime_guard(torch, whisper, device):
    expected = Path(importlib.metadata.distribution("openai-whisper").locate_file("whisper/__init__.py")).resolve(strict=True)
    if Path(getattr(whisper, "__file__", "")).resolve(strict=True) != expected:
        raise ASRError("Imported Whisper must belong to the verified installed distribution")
    if device == "cuda" and torch.cuda.is_initialized():
        raise ASRError("CUDA execution requires a fresh process without an initialized CUDA context")


def execute(args, root, folder):
    manifest = contained(root, args.input, exists=True)
    records = load_cases(root, manifest)
    package()
    model_path = verify_model(folder)
    output = contained(root, args.output)
    if output.exists() or output.is_symlink():
        raise ASRError("ASR result directory must be fresh")
    if args.device == "cuda" and (not args.allow_cuda or args.gpu_id is None or not 0 <= args.gpu_id <= 255):
        raise ASRError("CUDA needs explicit --allow-cuda and a single --gpu-id after resource review")
    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.device == "cuda" else ""
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    for key, child in (("NUMBA_CACHE_DIR", "numba"), ("TORCH_HOME", "torch"), ("TIKTOKEN_CACHE_DIR", "tiktoken")):
        os.environ[key] = str(output / "cache" / child)
    with deny_network():
        import torch
        import whisper
        runtime_guard(torch, whisper, args.device)
        torch.set_num_threads(2)
        torch.manual_seed(0)
        model = whisper.load_model(str(model_path), device="cuda:0" if args.device == "cuda" else "cpu", download_root=str(folder))
        verify_model(folder)
        results = []
        for row in records:
            if sha256(row["path"]) != row["audio"]["file_sha256"]:
                raise ASRError("WAV changed after preflight")
            result = model.transcribe(waveform(row["path"]), **SETTINGS)
            hypothesis = result.get("text")
            results.append(dict(id=row["id"], condition=row["condition"], reference=row["text"], hypothesis=hypothesis,
                                audio=row["audio"], strict=scores(row["text"], hypothesis),
                                punctuation_only=scores(row["text"], hypothesis, True)))
    totals = {}
    for condition in {row["condition"] for row in results}:
        totals[condition] = {}
        for view in ("strict", "punctuation_only"):
            selected = [row[view] for row in results if row["condition"] == condition]
            ce, nc = sum(x["char_edits"] for x in selected), sum(x["reference_chars"] for x in selected)
            we, nw = sum(x["word_edits"] for x in selected), sum(x["reference_words"] for x in selected)
            totals[condition][view] = dict(char_edits=ce, reference_chars=nc, cer=ce / nc, word_edits=we, reference_words=nw, wer=we / nw)
    report = dict(status="asr_metrics_computed", source=SOURCE, model_url=MODEL_URL, model_sha256=MODEL_SHA, model_bytes=MODEL_BYTES,
                  package_version=VERSION, license="MIT", license_url=LICENSE_URL, input_manifest_sha256=sha256(manifest),
                  device=args.device, seed=0, runtime_versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
                  decode_settings=SETTINGS, cases=results, totals=totals,
                  metrics_policy="NFC+whitespace; CER counts Unicode codepoints including normalized spaces; WER splits whitespace. Second view replaces only Unicode P punctuation; preserves M/ZWJ/ZWNJ; no number rewriting.",
                  reference_role_verified_by_tool=False, manual_listening_completed=False, quality_gate_passed=False,
                  limitation="ASR errors/hallucinations are not necessarily TTS errors; reference calibration and human Hindi listening remain required.")
    (output / "asr.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(status=report["status"], cases=len(results), output=str(output / "asr.json"), quality_gate_passed=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("assets-local/whisper-small"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--max-download-bytes", type=int, default=500000000)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--allow-cuda", action="store_true")
    p.add_argument("--gpu-id", type=int)
    args = p.parse_args(argv)
    try:
        root = workspace(args.project_root)
        folder = contained(root, args.model_dir)
        if args.download:
            result = download_model(folder, args.max_download_bytes)
        elif args.execute:
            if args.input is None or args.output is None:
                raise ASRError("--execute requires --input and a fresh --output directory")
            result = execute(args, root, folder)
        else:
            result = dict(status="plan_only", model="small", bytes=MODEL_BYTES, sha256=MODEL_SHA, url=MODEL_URL,
                          required_package_version=VERSION, license="MIT", source=SOURCE, cases_limit=MAX_CASES, download=False, asr_executed=False)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ASRError) else "Preflight/runtime failed; no ASR quality success is asserted"
        print(json.dumps(dict(status="blocked", reason=reason, error_type=type(exc).__name__, quality_gate_passed=False)))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
