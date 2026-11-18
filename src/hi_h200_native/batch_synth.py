"""Bounded, fixed-manifest offline acceptance with one real stock model per process.

Default: metadata/hash checks only, no Torch/model/CUDA/download. Generation is
explicit, sequential, one attempt per case; every failure/unexecuted case is kept.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from . import synth
from . import stock_segments
from .assets import safe_relative
from hi_h200.data import load_json
from hi_h200.text import normalize_hindi

LANGUAGES = {"en":0,"zh":1,"es":3,"ja":7,"ar":13,"hi":17}
OFFICIAL_LANGUAGES = ("zh","en","ja","es","ar")
SEED, MAX_TEXT_TOKENS = 1234, 120


class FatalBatchError(RuntimeError):
    pass


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_text(source, language, value, segmentation_mode="single"):
    if segmentation_mode not in ("single",stock_segments.MODE):
        raise synth.SynthesisError("Unknown bounded segmentation mode")
    if not isinstance(value, str) or not 1 <= len(value) <= 240 or any(c in value for c in "<>|"):
        raise synth.SynthesisError("Use bounded plain text without pronunciation/special-token annotations")
    for c in value:
        category = unicodedata.category(c)
        if category in {"Cc","Cs","Cn"} or (category == "Cf" and c not in "\u200c\u200d"):
            raise synth.SynthesisError("Unsupported/control text character")
    canonical = re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()
    if canonical != value:
        raise synth.SynthesisError("Manifest text must already be NFC and canonical whitespace")
    if language == "hi":
        normalize_hindi(value, max_chars=240 if segmentation_mode == stock_segments.MODE else 160)  # Preserve marks; keep the established Hindi number policy.
    tree = ast.parse((source / "indextts/utils/front.py").read_text(encoding="utf-8"))
    maps = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=="char_rep_map" for t in n.targets)]
    if len(maps) != 1:
        raise synth.SynthesisError("Cannot audit official punctuation policy")
    pattern = re.compile("|".join(re.escape(key) for key in maps[0]))
    if pattern.sub(lambda match:maps[0][match.group()], value) != value:
        raise synth.SynthesisError("Official punctuation replacement would alter authored text")
    return value


def load_cases(path, workspace, source, segmentation_mode="single"):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(workspace) or not path.is_file() or not 0 < path.stat().st_size <= 128 * 1024:
        raise synth.SynthesisError("Use a small fixed private case manifest inside WORK")
    payload = path.read_bytes()
    rows = load_json(payload.decode("utf-8"))
    if not isinstance(rows,list) or not 1 <= len(rows) <= 16:
        raise synth.SynthesisError("Batch manifest must be a JSON array of 1..16 cases")
    result, seen = [], set()
    required = {"id","language","text","prompt","max_mel_tokens"}
    for row in rows:
        if not isinstance(row,dict) or not required <= set(row) or set(row)-required-{"text_sha256"}:
            raise synth.SynthesisError("Unknown/missing batch case fields")
        uid, language = row["id"], row["language"]
        if not isinstance(uid,str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}",uid) or uid in seen or uid == "runtime" or not isinstance(language,str) or language not in LANGUAGES:
            raise synth.SynthesisError("Unsafe/duplicate case ID or undeclared language")
        value = canonical_text(source,language,row["text"],segmentation_mode)
        if "text_sha256" in row and row["text_sha256"] != text_hash(value):
            raise synth.SynthesisError("Authored text SHA256 mismatch")
        if type(row["max_mel_tokens"]) is not int or not 8 <= row["max_mel_tokens"] <= 512:
            raise synth.SynthesisError("Each case needs a fixed 8..512 mel-code cap")
        prompt = (workspace / safe_relative(row["prompt"])).resolve(strict=True)
        if not prompt.is_relative_to(workspace):
            raise synth.SynthesisError("Prompt escaped WORK")
        audio = synth.wav_info(prompt,prompt=True)
        result.append({**row,"text":value,"text_sha256":text_hash(value),"prompt_sha256":audio["file_sha256"],
            "prompt_metadata":audio,"frontend_verified":False,"status":"not_executed","reason":"not_started",
            "generation_attempted":False,"seed":SEED,"text_normalization":False,
            "settings":{"do_sample":False,"num_beams":1,"repetition_penalty":1.,"max_mel_tokens":row["max_mel_tokens"]}})
        seen.add(uid)
    return result, hashlib.sha256(payload).hexdigest()


def preflight(args):
    expected_merged = getattr(args,"expected_merged_gpt_sha256",None)
    if expected_merged is not None and (not isinstance(expected_merged,str) or not re.fullmatch(r"[0-9a-f]{64}",expected_merged)):
        raise synth.SynthesisError("Expected merged GPT SHA256 must be exactly 64 lowercase hexadecimal characters")
    workspace = args.workspace_root.resolve(strict=True)
    source = synth.verify_source(args.official_repo)
    original, merged = args.model_dir.resolve(strict=True), args.merged_bundle.resolve(strict=True)
    if not source.is_relative_to(workspace) or not merged.is_relative_to(workspace):
        raise synth.SynthesisError("Source and merged artifacts must stay in WORK")
    manifest = args.model_manifest.resolve(strict=True)
    if not manifest.is_relative_to(original):
        raise synth.SynthesisError("Asset manifest must be inside its original model directory")
    cases, cases_sha = load_cases(args.cases,workspace,source,getattr(args,"segmentation_mode","single"))
    protected = (source,original,merged,args.cases.resolve(),Path(__file__).resolve().parents[2],
                 *(workspace / row["prompt"] for row in cases))
    output = synth._independent_output(args.output_dir,workspace,protected)
    config, assets = synth.verify_synth_assets(original,manifest)
    merged_id = synth.verify_merged_bundle(merged,original,assets)
    if expected_merged is not None and merged_id["gpt_sha256"] != expected_merged:
        raise synth.SynthesisError("Verified merged GPT differs from the caller-pinned SHA256")
    selected = original / "gpt.pth" if args.variant == "original" else merged / "gpt.pth"
    selected_sha = assets["files"]["gpt.pth"] if args.variant == "original" else merged_id["gpt_sha256"]
    for row in cases:
        row.update(selected_gpt_sha256=selected_sha,condition="baseline" if args.variant=="original" else "merged")
    report = {"schema_version":1,"kind":"hi_h200_native_batch_synthesis","status":"metadata_preflight_passed",
        "variant":args.variant,"seed":SEED,"cases_manifest_sha256":cases_sha,"cases":cases,
        "official_commit":synth.OFFICIAL_COMMIT,"asset_identity":assets,"merged_identity":merged_id,
        "selected_gpt_sha256":selected_sha,"constructor_flags":dict(synth.DISABLED_FLAGS,use_bf16=True),
        "official_declared_languages":list(OFFICIAL_LANGUAGES),"experimental_language":"hi",
        "dependencies":synth.dependency_report(workspace,source),"one_model_constructions":0,"models_loaded":False,
        "synthesis_executed":False,"quality_verified":False,"old_language_regression_verified":False,
        "inference_cache":{"enabled":not args.no_kv_cache,"experimental":args.no_kv_cache,"actual_wrapper_verified":False}}
    if expected_merged is not None:
        report["expected_merged_gpt_sha256"] = expected_merged
    if getattr(args, "segmentation_mode", "single") == stock_segments.MODE:
        report["segmentation_experiment"] = {"mode":stock_segments.MODE,
            "max_segments":stock_segments.MAX_SEGMENTS,"max_text_tokens":MAX_TEXT_TOKENS,
            "coverage_policy":"exact complete stock-frontend text; per-segment BPE may retokenize",
            "code_completion_required":"nonempty semantic codes followed by one EOS within each fixed cap",
            "wav_seconds_limit":60,"hindi_characters_limit":240,"quality_verified":False}
    if report["dependencies"]["missing"]:
        report["status"]="metadata_verified_runtime_blocked"
    return dict(workspace=workspace,source=source,original=original,merged=merged,output=output,config=config,selected=selected,cases_path=args.cases.resolve(strict=True)), report


def save_ledger(output, report):
    temporary = output / ".batch-results.partial.json"
    synth._write_json(temporary,report)
    os.replace(temporary,output / "batch-results.json")
    descriptor = os.open(output,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def case_resources(args, uuid):
    try:
        sample = synth.parse_gpu_sample(subprocess.run(["nvidia-smi","--id",uuid,
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"],
            check=True,capture_output=True,text=True,timeout=10).stdout)
        ram = synth.available_ram_bytes()
        if sample["uuid"] != uuid or ram is None or ram < args.min_free_ram_mib*1024**2 or sample["free_mib"] < args.min_free_gpu_mib:
            raise FatalBatchError("Resident-model GPU/RAM/UUID gate failed")
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        raise FatalBatchError("Cannot establish resident-model resource admission") from exc
    return dict(sample,available_ram_bytes=ram,utilization_gate_applied=False,
                utilization_note="Recorded only: utilization may include this process's previous case")


def authored_to_stock(tts, language, text):
    module = importlib.import_module("indextts.infer_v2_5")
    actual = tts.text_process.clean_pattern.sub(lambda m:tts.text_process.char_rep_map[m.group()],text)
    if actual != text:
        raise synth.SynthesisError("Stock punctuation unexpectedly changed authored text")
    if language in ("ja","zh","en"):
        actual = actual.lower()
    if language == "es":
        actual = actual.upper()
    actual = module.apply_pronunciation_annotations(actual)
    if language == "ja":
        actual = tts.ja_text_process.process_ja_text(actual)
    return re.sub(r"<\|([^|]+)\|>",lambda m:f"<|{m.group(1).upper()}|>",actual)


@contextmanager
def audit_gpt_inputs(tts, source, tokens, slot):
    module = importlib.import_module("indextts.gpt.model_v2")
    if type(tts.gpt) is not module.UnifiedVoice or not Path(inspect.getfile(type(tts.gpt))).resolve().is_relative_to(source):
        raise FatalBatchError("GPT input auditing requires the real stock official class")
    original = tts.gpt.inference_speech
    observed = {"calls":0}
    def audited(speech, text_inputs, langs=None, *args, **kwargs):
        expected = [tokens + [tts.gpt.stop_text_token]]
        if text_inputs.detach().cpu().tolist() != expected or langs.detach().cpu().tolist() != [slot]:
            raise synth.SynthesisError("Actual stock generation tokens/language differ from audited frontend")
        observed["calls"] += 1
        return original(speech,text_inputs,langs,*args,**kwargs)
    tts.gpt.inference_speech = audited
    try:
        yield observed
    finally:
        tts.gpt.inference_speech = original


def output_audio(path, segmentation_mode="single"):
    if segmentation_mode not in ("single",stock_segments.MODE):
        raise synth.SynthesisError("Unknown bounded segmentation mode")
    info = synth.wav_info(path)
    limit = 60 if segmentation_mode == stock_segments.MODE else 20
    if not 1 <= info["seconds"] <= limit:
        raise synth.SynthesisError(f"Acceptance audio must be 1..{limit} seconds; never truncate for ASR")
    return info


def reset_seed(*, cuda=True):
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed_all(SEED)


def attempt_case(args, plan, tts, row, uuid):
    import torch
    from indextts.utils.tokenizer import lang_to_token, LANGUAGE_DICT
    folder = plan["output"] / row["id"]
    folder.mkdir(mode=0o700)
    observation = case_resources(args,uuid)
    prompt = (plan["workspace"] / row["prompt"]).resolve(strict=True)
    if synth.sha256(prompt) != row["prompt_sha256"]:
        raise FatalBatchError("Fixed prompt changed after preflight")
    actual = authored_to_stock(tts,row["language"],row["text"])
    prefix = f'<|{row["language"]}|> '
    slot = lang_to_token(row["language"])
    if LANGUAGE_DICT.get(row["language"]) != slot or slot != LANGUAGES[row["language"]] or not 0 <= slot < tts.gpt.lang_embedding.num_embeddings:
        raise FatalBatchError("Actual official language slot differs from pinned source")
    control = tts.tokenizer.encode(prefix.strip(),allowed_special={prefix.strip()})
    tokens = tts.tokenizer.encode(prefix+actual,allowed_special={prefix.strip()})
    segmented = getattr(args,"segmentation_mode","single") == stock_segments.MODE
    segment_plans = None
    if segmented:
        segment_plans, segment_report = stock_segments.plan_segments(tts,plan["source"],actual,prefix,MAX_TEXT_TOKENS)
        row["segmentation"] = segment_report
        expected_calls = len(segment_plans)
    else:
        if len(control) != 1 or not tokens or tokens[0] != control[0] or any(t in (tts.gpt.start_text_token,tts.gpt.stop_text_token) for t in tokens) or len(tts.split_text_by_tokens(actual,MAX_TEXT_TOKENS,prefix)) != 1:
            raise synth.SynthesisError("Actual tokenization would split or silently strip framing IDs")
        expected_calls = 1
    reset_seed()
    torch.cuda.synchronize()
    partial = folder / ".audio.partial.wav"
    settings = {"do_sample":False,"num_beams":1,"repetition_penalty":1.,"max_mel_tokens":row["max_mel_tokens"]}
    started = time.perf_counter()
    row.update(actual_stock_text=actual,actual_stock_text_sha256=text_hash(actual),language_slot=slot,
        token_ids_sha256=text_hash(json.dumps(tokens)),resource_observation=observation,
        generation_attempted=True,partial_audio_path=str(partial.relative_to(plan["workspace"])))
    observed, captured = {"calls":0}, []
    try:
        with (folder / "official.log").open("x",encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            audit = (stock_segments.audit_segments(tts,plan["source"],segment_plans,slot,row["max_mel_tokens"])
                if segmented else audit_gpt_inputs(tts,plan["source"],tokens,slot))
            with audit as observed:
                tts.infer(spk_audio_prompt=str(prompt),text=row["text"],output_path=str(partial),lang=row["language"],
                    use_emo_text=False,use_random=False,text_normalization=False,stream_return=False,
                    max_text_tokens_per_segment=MAX_TEXT_TOKENS,verbose=False,**settings)
            torch.cuda.synchronize()
    finally:
        row.update(frontend_verified=observed["calls"]==expected_calls and (not segmented or all(p["record"]["input_verified"] for p in segment_plans)),partial_audio_exists=partial.is_file(),
            elapsed_seconds=time.perf_counter()-started,
            warnings=[{"category":item.category.__name__,"message":str(item.message)[:1000]} for item in captured[:50]])
        if segmented:
            row["segmentation"].update(actual_generation_calls=observed["calls"],
                completed_code_segments=observed.get("completed",0),
                all_expected_inputs_observed=row["frontend_verified"],
                all_returned_codes_ended_with_eos=observed.get("completed")==expected_calls,
                acoustic_output_verified=False)
    if observed["calls"] != expected_calls or (segmented and observed.get("completed") != expected_calls):
        raise synth.SynthesisError("Expected every planned stock segment to complete its audited generation call" if segmented else "Expected exactly one audited stock generation call")
    if any("exceeding `max_mel_tokens`" in str(item.message) for item in captured):
        raise synth.SynthesisError("Maximum-code limit hit; partial audio is not success")
    info = output_audio(partial,getattr(args,"segmentation_mode","single"))
    if segmented:
        row["segmentation"]["acoustic_output_verified"] = True
    os.link(partial,folder / "audio.wav")
    partial.unlink()
    return dict(row,status="generated",reason=None,frontend_verified=True,actual_stock_text=actual,
        actual_stock_text_sha256=text_hash(actual),language_slot=slot,token_ids_sha256=text_hash(json.dumps(tokens)),
        audio=str((folder / "audio.wav").relative_to(plan["workspace"])),output=info,seed=SEED,partial_audio_exists=False,
        settings=settings,text_normalization=False,resource_observation=observation,
        elapsed_seconds=time.perf_counter()-started,quality_verified=False)


def run_cases(cases, attempt, save):
    """Ledger orchestration; injected tests exercise only failures, never fake audio."""
    for index, row in enumerate(cases):
        row.update(status="attempt_started",reason="no_completed_result_yet")
        save()
        try:
            cases[index] = attempt(row)
        except BaseException as exc:
            fatal = isinstance(exc,(RuntimeError,subprocess.SubprocessError,KeyboardInterrupt,SystemExit))
            cases[index] = dict(row,status="failed",reason=str(exc)[:4000],failure_type=type(exc).__name__,fatal=fatal)
            if fatal:
                for following in range(index+1,len(cases)):
                    cases[following] = dict(cases[following],status="not_executed",reason="stopped_after_fatal_case",fatal_case_id=row["id"])
                save()
                break
        save()
    return cases


def execute(args, plan, report):
    if not args.allow_cuda:
        raise synth.SynthesisError("Batch execution requires explicit --allow-cuda")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise synth.SynthesisError("Use a fresh process without an existing CUDA context")
    output = plan["output"]
    output.mkdir(mode=0o700)
    save = lambda:save_ledger(output,report)
    save()
    try:
        if report["dependencies"]["missing"]:
            raise FatalBatchError("Required real stock synthesis dependencies are missing")
        admission = synth.resource_admission(args)
        uuid = admission[0]["uuid"]
        report["initial_resource_admission"] = admission
        with synth.isolated_runtime(output,plan["source"],uuid) as environment, synth._runtime_import(plan["source"],environment) as constructor:
            import torch
            torch.set_num_threads(2)
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
                raise FatalBatchError("One explicit BF16-capable CUDA device is required")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            if synth.sha256(plan["cases_path"]) != report["cases_manifest_sha256"] or synth.sha256(plan["selected"]) != report["selected_gpt_sha256"]:
                raise FatalBatchError("Fixed manifest or model changed after preflight")
            config = dict(plan["config"],gpt_checkpoint=str(plan["selected"]))
            import yaml
            cfg = output / "runtime/selected-config.yaml"
            with cfg.open("x",encoding="utf-8") as stream:
                yaml.safe_dump(config,stream,allow_unicode=True,sort_keys=False)
            current_admission=synth.resource_admission(args)
            if current_admission[0]["uuid"] != uuid:
                raise FatalBatchError("Physical GPU changed before model construction")
            report["preconstruction_resource_admission"]=current_admission
            reset_seed()
            report["one_model_constructions"] = 1
            save()
            with (output / "constructor.log").open("x",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
                tts=constructor(cfg_path=str(cfg),model_dir=str(plan["original"]),use_bf16=True,device="cuda:0",**synth.DISABLED_FLAGS)
            report["models_loaded"]=True
            report["selected_config_sha256"]=synth.sha256(cfg)
            report["inference_cache"]=synth.configure_kv_cache_experiment(tts,plan["source"],no_kv_cache=args.no_kv_cache)
            if Path(tts.gpt_path).resolve()!=plan["selected"] or tts.qwen_emo is not None or tts.use_accel or tts.use_cuda_kernel or tts.use_torch_compile:
                raise FatalBatchError("Stock constructor did not retain audited flags/weights")
            run_cases(report["cases"],lambda row:attempt_case(args,plan,tts,row,uuid),save)
    except BaseException as exc:
        report.update(status="batch_stopped",failure_type=type(exc).__name__,reason=str(exc)[:4000])
        for index,row in enumerate(report["cases"]):
            if row["status"]=="attempt_started":
                report["cases"][index]=dict(row,status="failed",reason="batch_aborted_during_case",fatal=True)
            elif row["status"]=="not_executed":
                report["cases"][index]=dict(row,reason="batch_initialization_or_resource_failure")
    else:
        report["status"]="batch_stopped_after_fatal_case" if any(row["status"]=="not_executed" for row in report["cases"]) else "batch_attempts_completed"
    report["synthesis_executed"]=any(row["status"]=="generated" for row in report["cases"])
    report["coverage"]={name:sum(row["status"]==name for row in report["cases"]) for name in ("generated","failed","not_executed")}
    save()
    for row in report["cases"]:
        folder=output / row["id"]
        folder.mkdir(mode=0o700,exist_ok=True)
        synth._write_json(folder / "result.json",row)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("workspace-root","official-repo","model-dir","model-manifest","merged-bundle","cases","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--variant",choices=("original","merged"),required=True)
    parser.add_argument("--expected-merged-gpt-sha256",
        help="Optional caller pin: reject any different verified merged GPT before model/CUDA execution")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--allow-cuda",action="store_true")
    parser.add_argument("--no-kv-cache",action="store_true")
    parser.add_argument("--segmentation-mode",choices=("single",stock_segments.MODE),default="single",
        help="Explicit experiment: official token-based segmentation with complete-text and per-segment EOS auditing")
    parser.add_argument("--gpu-id",default="0")
    parser.add_argument("--min-free-gpu-mib",type=float)
    parser.add_argument("--min-free-ram-mib",type=float)
    parser.add_argument("--max-utilization",type=float)
    args=parser.parse_args(argv)
    if args.execute and not args.allow_cuda:
        parser.error("Execution requires explicit --allow-cuda")
    sys.dont_write_bytecode=True
    try:
        plan,report=preflight(args)
        if args.execute:
            report=execute(args,plan,report)
    except (OSError,ValueError,RuntimeError,ImportError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        print(json.dumps({"status":"blocked","failure_type":type(exc).__name__,"reason":str(exc),"quality_verified":False}),file=sys.stderr)
        return 2
    print(json.dumps({key:report[key] for key in ("status","variant","cases_manifest_sha256","one_model_constructions","models_loaded","quality_verified")}|{"cases":len(report["cases"]),"coverage":report.get("coverage"),"dependency_missing":report["dependencies"]["missing"]},ensure_ascii=False))
    if not args.execute:
        return 2 if report["dependencies"]["missing"] else 0
    return 0 if report.get("coverage",{}).get("generated")==len(report["cases"]) else 2


if __name__=="__main__":
    raise SystemExit(main())
