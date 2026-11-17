#!/usr/bin/env python3
"""One-shot anonymous fixed public release download; no GPU/training/installation."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
sys.dont_write_bytecode = True
REPO = "fluxa0223/IndexTTS-2.5-Hindi-best100-experimental"
REVISION = "117c7922e4dcedc3f9885f3f6efa5be8f54c7919"
ENDPOINT = "https://huggingface.co"
GPT_SIZE, GPT_SHA = 3259538991, "423a8491b8856c7c3075254b12defc3ae47dbd9e0de285772523ddd4e9a97021"
FILES = {"gpt.pth","NON_ENDORSEMENT.txt","DATA_ATTRIBUTION.md","LICENSE","LICENSE_ZH.txt","MODEL_LICENSE","config.yaml","README.md","release.json","manifest.json","SHA256SUMS","evaluation.json","multilingual_zh_ja_yue_char_del.tiktoken"}
PINS = {"manifest.json": (1587,"99bc0bf9506baf0052573062bff3a4e76c13f6fa56f4bc68a73394d68c4c9b43"), "SHA256SUMS": (983,"ab6b2cc5865ca78a54e53dfa2c1c1dc7c3bf232db11f99a95b3083dfc4ea6447")}
class Blocked(ValueError): pass
class Quiet:
    def write(self,text): return len(text)
    def flush(self): pass
    def isatty(self): return False

def state(work, phase, **extra):
    value = dict(phase=phase,repository=REPO,revision=REVISION,training_started=False,gpu_touched=False,**extra)
    temporary = work/".download-status.tmp"
    with temporary.open("x",encoding="utf-8") as out:
        json.dump(value,out,sort_keys=True);out.flush();os.fsync(out.fileno())
    os.replace(temporary,work/"download-status.json")
    return value

def environment(cache):
    paths = {"HF_HOME":"home","HF_HUB_CACHE":"hub","HUGGINGFACE_HUB_CACHE":"hub","HF_XET_CACHE":"xet","HF_ASSETS_CACHE":"assets","XDG_CACHE_HOME":"xdg","TMPDIR":"tmp","TMP":"tmp","TEMP":"tmp"}
    for key, child in paths.items():
        folder=cache/child;folder.mkdir(exist_ok=True);os.environ[key]=str(folder)
    for key in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN","HF_TOKEN_PATH"):
        if key in os.environ: del os.environ[key]  # Inspect names only, never credential values.
    os.environ.update(HF_ENDPOINT=ENDPOINT,HF_HUB_OFFLINE="0",HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",HF_HUB_DISABLE_XET="1",HF_HUB_ENABLE_HF_TRANSFER="0",HF_HUB_DISABLE_PROGRESS_BARS="1",HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    tempfile.tempdir=str(cache/"tmp")

def copy_verified(source, destination, size, digest):
    if not source.is_file() or source.stat().st_size != size: raise Blocked("Downloaded size differs")
    actual,count=hashlib.sha256(),0
    with source.open("rb") as src,destination.open("xb") as dst:
        for block in iter(lambda:src.read(8*1024*1024),b""):
            count+=len(block)
            if count>size: raise Blocked("Downloaded source grew")
            actual.update(block);dst.write(block)
        dst.flush();os.fsync(dst.fileno())
    if count!=size or actual.hexdigest()!=digest: raise Blocked("Downloaded SHA256 differs")
    return actual.hexdigest()

def bounded(path):
    with path.open("rb") as stream: data=stream.read(2*1024*1024+1)
    if len(data)>2*1024*1024: raise Blocked("Small metadata exceeded bound")
    return data

def run(work, downloader=None):
    work=Path(work).resolve(strict=True)
    if not work.is_dir() or work==Path(work.anchor): raise Blocked("Use an existing dedicated test directory")
    cache,public=work/"cache",work/"public13"
    if any(path.exists() or path.is_symlink() for path in (cache,public,work/"download-status.json",work/".download-status.tmp")): raise Blocked("Download directories/status must be fresh")
    if shutil.disk_usage(work).free<10*1024**3: raise Blocked("At least 10GiB free disk is required")
    cache.mkdir(mode=0o700);public.mkdir(mode=0o700);environment(cache)
    state(work,"starting",complete=False)
    try:
        with redirect_stdout(Quiet()),redirect_stderr(Quiet()):
            if downloader is None:
                if "huggingface_hub" in sys.modules: raise Blocked("Use a fresh Python process")
                import requests
                from huggingface_hub import hf_hub_download,configure_http_backend
                def backend():
                    session=requests.Session();session.trust_env=False;return session  # No netrc, proxy or ambient auth.
                configure_http_backend(backend_factory=backend)
                downloader=hf_hub_download
            hashes={}
            def obtain(name,size,digest):
                if name not in FILES or type(size) is not int or not 0<size<=4*1024**3 or not re.fullmatch(r"[0-9a-f]{64}",digest): raise Blocked("Unreviewed file metadata")
                if shutil.disk_usage(work).free<2*size+1024**3: raise Blocked("Disk headroom fell below cache/copy reserve")
                state(work,"downloading",file=name,complete=False)
                source=Path(downloader(repo_id=REPO,filename=name,revision=REVISION,repo_type="model",token=False,endpoint=ENDPOINT,cache_dir=str(cache/"hub"),local_files_only=False)).resolve(strict=True)
                if not source.is_relative_to(cache): raise Blocked("SDK cache path escaped test directory")
                hashes[name]=copy_verified(source,public/name,size,digest)
            for name,(size,digest) in PINS.items(): obtain(name,size,digest)
            record=json.loads(bounded(public/"manifest.json"))
            payload=record.get("files")
            if record.get("schema_version")!=1 or record.get("kind")!="public_release_whitelist" or not isinstance(payload,dict) or set(payload)!=FILES-set(PINS): raise Blocked("Release manifest inventory mismatch")
            if payload.get("gpt.pth")!={"bytes":GPT_SIZE,"sha256":GPT_SHA}: raise Blocked("Frozen GPT identity differs")
            for name in sorted(payload,key=lambda value:(value=="gpt.pth",value)):
                entry=payload[name]
                if not isinstance(entry,dict) or set(entry)!={"bytes","sha256"}: raise Blocked("Unexpected manifest entry")
                if name!="gpt.pth" and not 0<entry["bytes"]<=2*1024*1024: raise Blocked("Small file exceeds bound")
                obtain(name,entry["bytes"],entry["sha256"])
            expected="".join(hashes[name]+"  "+name+"\n" for name in sorted(FILES-{"SHA256SUMS"})).encode()
            if bounded(public/"SHA256SUMS")!=expected or {path.name for path in public.iterdir()}!=FILES or any(path.is_symlink() or not path.is_file() for path in public.iterdir()): raise Blocked("Final checksum/file inventory mismatch")
        return state(work,"complete",complete=True,files=13,gpt_bytes=GPT_SIZE,gpt_sha256=GPT_SHA,token_used=False)
    except Exception as exc:
        state(work,"blocked",complete=False,error_type=type(exc).__name__)
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--work-root",type=Path,required=True)
    args=parser.parse_args(argv)
    try: result=run(args.work_root)
    except Exception as exc:
        print(json.dumps(dict(phase="blocked",error_type=type(exc).__name__,complete=False,training_started=False,gpu_touched=False)));return 2
    print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
