#!/usr/bin/env python3
"""Offline recovery of the fixed public release ZIP's missing tail; no transfer.

Only explicit prepare/combine actions. CLI pins cannot be changed by metadata,
flags or environment. Private function parameters exist for tiny CPU tests only.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import zipfile

SOURCE_BYTES = 3260503973
SOURCE_SHA = "b0df1c597222b78a4aa87cc6ee9a4d167ad12aa91706cff651bd403ca64618db"
PREFIX_BYTES = 2277806075
PREFIX_SHA = "4c782e8835f7da49ee85e61f3af86820aec190938cd7181973bf190d3c7e37fe"
CHUNK_BYTES = 100 * 1024 * 1024
FORMAT = "indextts25_hindi_best100_tail_v1"
META_KEYS = {"format","index","offset","bytes","payload_sha256","source_sha256","source_bytes","prefix_bytes"}
BUFFER = 1024 * 1024


class RecoveryError(ValueError):
    pass


@dataclass(frozen=True)
class _Pins:
    source_bytes: int = SOURCE_BYTES
    source_sha: str = SOURCE_SHA
    prefix_bytes: int = PREFIX_BYTES
    prefix_sha: str = PREFIX_SHA
    chunk_bytes: int = CHUNK_BYTES

    def validate(self):
        if any(type(v) is not int or v <= 0 for v in (self.source_bytes,self.prefix_bytes,self.chunk_bytes)) or self.prefix_bytes >= self.source_bytes:
            raise RecoveryError("Invalid fixed geometry")
        if any(not isinstance(v,str) or not re.fullmatch(r"[0-9a-f]{64}",v) for v in (self.source_sha,self.prefix_sha)) or (self.source_bytes-self.prefix_bytes+self.chunk_bytes-1)//self.chunk_bytes != 10:
            raise RecoveryError("Expected fixed hashes and exactly ten parts")
        return self

    def geometry(self,index):
        offset=self.prefix_bytes+index*self.chunk_bytes
        return offset,min(self.chunk_bytes,self.source_bytes-offset)


PINS = _Pins()


def _name(index):
    return f"indextts25-hindi-best100-tail-{index:02d}.zip"


def _stamp(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _path(value,kind):
    path=Path(value).absolute()
    info=path.lstat()
    if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) if kind=="file" else stat.S_ISDIR(info.st_mode)):
        raise RecoveryError("Expected a non-symlink regular file/directory")
    return path.parent.resolve(strict=True)/path.name


def _fresh(value):
    path=Path(value).absolute()
    if path.exists() or path.is_symlink():
        raise RecoveryError("Output must be fresh; existing paths are never overwritten")
    parent=_path(path.parent,"dir")
    return parent/path.name


def _open(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
    stream=os.fdopen(fd,"rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode) or _stamp(os.fstat(fd)) != _stamp(path.lstat()):
        stream.close()
        raise RecoveryError("Input changed or is not regular")
    return stream


def _copy(reader,count,digest,writer=None,extra=None):
    while count:
        block=reader.read(min(BUFFER,count))
        if not block:
            raise RecoveryError("Truncated stream")
        digest.update(block)
        if extra is not None:
            extra.update(block)
        if writer is not None:
            writer.write(block)
        count-=len(block)


def _fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _entry(name):
    value=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
    value.create_system=3
    value.external_attr=(stat.S_IFREG|0o600)<<16
    value.compress_type=zipfile.ZIP_STORED
    return value


def prepare(source,fresh_dir,*,_pins=PINS):
    pins=_pins.validate()
    source,output=_path(source,"file"),_fresh(fresh_dir)
    before=_stamp(source.stat())
    if before[3] != pins.source_bytes or shutil.disk_usage(output.parent).free < 3*1024**3:
        raise RecoveryError("Source byte count or 3 GiB free-space gate failed")
    full,prefix,part_hashes=hashlib.sha256(),hashlib.sha256(),[]
    with _open(source) as stream:
        _copy(stream,pins.prefix_bytes,full,extra=prefix)
        for index in range(10):
            digest=hashlib.sha256()
            _copy(stream,pins.geometry(index)[1],full,extra=digest)
            part_hashes.append(digest.hexdigest())
        if stream.read(1) or _stamp(os.fstat(stream.fileno())) != before:
            raise RecoveryError("Source changed during verification")
    if full.hexdigest()!=pins.source_sha or prefix.hexdigest()!=pins.prefix_sha or _stamp(source.stat())!=before:
        raise RecoveryError("Trusted source/prefix SHA256 mismatch")
    output.mkdir(mode=0o700,exist_ok=False)
    with _open(source) as stream:
        stream.seek(pins.prefix_bytes)
        for index in range(10):
            offset,count=pins.geometry(index)
            path=output/_name(index)
            with path.open("xb") as target:
                with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED,allowZip64=False) as archive:
                    payload_hash=hashlib.sha256()
                    with archive.open(_entry("payload.bin"),"w") as payload:
                        _copy(stream,count,payload_hash,payload)
                    if payload_hash.hexdigest()!=part_hashes[index]:
                        raise RecoveryError("Source tail changed after trusted verification")
                    metadata=dict(format=FORMAT,index=index,offset=offset,bytes=count,payload_sha256=part_hashes[index],
                        source_sha256=pins.source_sha,source_bytes=pins.source_bytes,prefix_bytes=pins.prefix_bytes)
                    encoded=json.dumps(metadata,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
                    if len(encoded)>2048:
                        raise RecoveryError("Metadata exceeds bound")
                    archive.writestr(_entry("part.json"),encoded)
                target.flush();os.fsync(target.fileno())
        if stream.read(1) or _stamp(os.fstat(stream.fileno()))!=before or _stamp(source.stat())!=before:
            raise RecoveryError("Source changed during preparation")
    _fsync_dir(output)
    return {"status":"prepared_verified_tail_parts","parts":10,"source_bytes":pins.source_bytes,"source_sha256":pins.source_sha,"tail_bytes":pins.source_bytes-pins.prefix_bytes}


def _zip_geometry(stream,size,count):
    if not 0<size<=count+4096:
        raise RecoveryError("ZIP size exceeds the exact payload/metadata bound")
    stream.seek(size-22)
    footer=stream.read(22)
    if len(footer)!=22:
        raise RecoveryError("Missing classic ZIP footer")
    magic,disk,central_disk,local_entries,entries,central_bytes,central_offset,comment=struct.unpack("<4s4H2LH",footer)
    if magic!=b"PK\x05\x06" or (disk,central_disk,local_entries,entries,comment)!=(0,0,2,2,0) or central_bytes!=112 or central_offset+central_bytes!=size-22:
        raise RecoveryError("Only a classic two-entry unprefixed ZIP is accepted")
    return central_offset


def _metadata(archive,infos,index,pins):
    if [info.filename for info in infos]!=["payload.bin","part.json"]:
        raise RecoveryError("ZIP entries/order differ from the exact whitelist")
    offset,count=pins.geometry(index)
    for info in infos:
        mode=info.external_attr>>16
        if info.compress_type!=zipfile.ZIP_STORED or info.compress_size!=info.file_size or info.flag_bits!=0 or info.extra or info.comment or info.is_dir() or info.extract_version!=20 or stat.S_IFMT(mode)!=stat.S_IFREG:
            raise RecoveryError("Compressed/encrypted/extra/special ZIP entry rejected")
    if infos[0].file_size!=count or not 0<infos[1].file_size<=2048:
        raise RecoveryError("ZIP entry byte counts differ from fixed geometry")
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:
                raise RecoveryError("Duplicate metadata field")
            value[key]=item
        return value
    value=json.loads(archive.read(infos[1]).decode("utf-8"),object_pairs_hook=unique,
        parse_constant=lambda _value:(_ for _ in ()).throw(RecoveryError("Nonfinite metadata")))
    if not isinstance(value,dict) or set(value)!=META_KEYS or value["format"]!=FORMAT:
        raise RecoveryError("Unknown/partial metadata schema")
    expected={"index":index,"offset":offset,"bytes":count,"source_bytes":pins.source_bytes,"prefix_bytes":pins.prefix_bytes}
    if any(type(value[key]) is not int or value[key]!=number for key,number in expected.items()) or value["source_sha256"]!=pins.source_sha or not isinstance(value["payload_sha256"],str) or not re.fullmatch(r"[0-9a-f]{64}",value["payload_sha256"]):
        raise RecoveryError("Part metadata differs from trusted fixed pins")
    return value


def _payload(path,index,pins,writer,full):
    before=_stamp(path.stat())
    with _open(path) as stream:
        central=_zip_geometry(stream,before[3],pins.geometry(index)[1])
        with zipfile.ZipFile(stream,"r",allowZip64=False) as archive:
            infos=archive.infolist()
            if len(infos)!=2:
                raise RecoveryError("Expected exactly two ZIP members")
            metadata=_metadata(archive,infos,index,pins)
            end=0
            for info in infos:
                if info.header_offset!=end:
                    raise RecoveryError("Hidden or overlapping ZIP member data")
                stream.seek(end)
                header=stream.read(30)
                if len(header)!=30 or header[:4]!=b"PK\x03\x04":
                    raise RecoveryError("Invalid local ZIP header")
                fields=struct.unpack("<4s5H3L2H",header)
                if fields[1]!=20 or fields[2]!=0 or fields[3]!=0 or fields[6]!=info.CRC or fields[7]!=info.compress_size or fields[8]!=info.file_size or fields[9]!=len(info.filename.encode()) or fields[10]!=0 or stream.read(fields[9])!=info.filename.encode():
                    raise RecoveryError("Local/central ZIP header mismatch")
                end+=30+fields[9]+info.compress_size
            if end!=central:
                raise RecoveryError("Hidden ZIP trailer or payload gap")
            digest=hashlib.sha256()
            with archive.open(infos[0],"r") as payload:
                _copy(payload,pins.geometry(index)[1],digest,writer,full)
                if payload.read(1):
                    raise RecoveryError("Payload exceeds trusted geometry")
            if digest.hexdigest()!=metadata["payload_sha256"]:
                raise RecoveryError("Payload SHA256 mismatch")
        if _stamp(os.fstat(stream.fileno()))!=before or _stamp(path.stat())!=before:
            raise RecoveryError("Part changed during combination")


def combine(prefix,chunk_dir,new_output,*,_pins=PINS):
    pins=_pins.validate()
    prefix,chunks,output=_path(prefix,"file"),_path(chunk_dir,"dir"),_fresh(new_output)
    if output.is_relative_to(chunks) or {p.name for p in chunks.iterdir()}!={_name(i) for i in range(10)}:
        raise RecoveryError("Chunk directory must contain exactly the ten named ZIPs; output must be separate")
    parts=[_path(chunks/_name(index),"file") for index in range(10)]
    if shutil.disk_usage(output.parent).free < pins.source_bytes + 64*1024**2:
        raise RecoveryError("Insufficient free space for source bytes plus 64 MiB reserve")
    before=_stamp(prefix.stat())
    if before[3]!=pins.prefix_bytes:
        raise RecoveryError("Prefix byte count mismatch")
    verified=hashlib.sha256()
    with _open(prefix) as stream:
        _copy(stream,pins.prefix_bytes,verified)
        if stream.read(1) or _stamp(os.fstat(stream.fileno()))!=before:
            raise RecoveryError("Prefix changed during verification")
    if verified.hexdigest()!=pins.prefix_sha or _stamp(prefix.stat())!=before:
        raise RecoveryError("Trusted prefix SHA256 mismatch")
    full=hashlib.sha256()
    with output.open("xb") as target:
        with _open(prefix) as stream:
            _copy(stream,pins.prefix_bytes,full,target)
            if stream.read(1) or _stamp(os.fstat(stream.fileno()))!=before:
                raise RecoveryError("Prefix changed during copy")
        for index,path in enumerate(parts):
            _payload(path,index,pins,target,full)
        target.flush();os.fsync(target.fileno())
    if output.stat().st_size!=pins.source_bytes or full.hexdigest()!=pins.source_sha or _stamp(prefix.stat())!=before:
        raise RecoveryError("Final source size/SHA256 mismatch; output remains unverified partial")
    _fsync_dir(output.parent)
    return {"status":"combined_verified_source","source_bytes":pins.source_bytes,"source_sha256":pins.source_sha,"prefix_modified":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    create=actions.add_parser("prepare");create.add_argument("source",type=Path);create.add_argument("fresh_dir",type=Path)
    join=actions.add_parser("combine");join.add_argument("prefix",type=Path);join.add_argument("chunk_dir",type=Path);join.add_argument("new_output",type=Path)
    args=parser.parse_args(argv)
    try:
        result=prepare(args.source,args.fresh_dir) if args.action=="prepare" else combine(args.prefix,args.chunk_dir,args.new_output)
    except Exception as exc:
        print(json.dumps({"status":"failed_not_verified","error_type":type(exc).__name__,"reason":str(exc),"partial_may_exist":True}),file=sys.stderr)
        return 2
    print(json.dumps(result,sort_keys=True));return 0


if __name__=="__main__":
    raise SystemExit(main())
