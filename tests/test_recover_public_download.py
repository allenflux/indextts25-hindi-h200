"""Tiny real ZIP recovery and adversarial inputs; never process production bytes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
import recover_public_download as recovery


def sha(value):
    return hashlib.sha256(value).hexdigest()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.root=Path(self.tmp.name)
        self.prefix_bytes=bytes(range(51))  # Raw, deliberately not an independently valid ZIP.
        self.source_bytes=self.prefix_bytes+bytes((index*7)%256 for index in range(9*31+7))
        self.pins=recovery._Pins(len(self.source_bytes),sha(self.source_bytes),len(self.prefix_bytes),sha(self.prefix_bytes),31)
        self.source=self.root / "source.zip"
        self.prefix=self.root / "prefix.part"
        self.parts=self.root / "parts"
        self.output=self.root / "restored.zip"
        self.source.write_bytes(self.source_bytes)
        self.prefix.write_bytes(self.prefix_bytes)

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self):
        return recovery.prepare(self.source,self.parts,_pins=self.pins)

    def rewrite(self,index,*,payload=None,metadata=None,names=None,extra=False,special=False,compression=None):
        path=self.parts / recovery._name(index)
        with zipfile.ZipFile(path) as archive:
            old_payload=archive.read("payload.bin")
            old_meta=json.loads(archive.read("part.json"))
        data=old_payload if payload is None else payload
        meta=old_meta if metadata is None else metadata
        with zipfile.ZipFile(path,"w",allowZip64=False) as archive:
            first=recovery._entry("payload.bin" if names is None else names[0])
            second=recovery._entry("part.json" if names is None else names[1])
            if special:
                first.external_attr=(stat.S_IFLNK|0o600)<<16
            if compression is not None:
                first.compress_type=compression
            archive.writestr(first,data)
            archive.writestr(second,json.dumps(meta,separators=(",",":")))
            if extra:
                archive.writestr("extra",b"not permitted")

    def test_actual_tiny_roundtrip_preserves_prefix_and_exact_ten_geometry(self):
        result=self.prepare()
        self.assertEqual(result["parts"],10)
        self.assertEqual(result["tail_bytes"],len(self.source_bytes)-len(self.prefix_bytes))
        self.assertEqual({p.name for p in self.parts.iterdir()},{recovery._name(i) for i in range(10)})
        for index in range(10):
            with zipfile.ZipFile(self.parts/recovery._name(index)) as archive:
                self.assertEqual(archive.namelist(),["payload.bin","part.json"])
                meta=json.loads(archive.read("part.json"))
                offset,count=self.pins.geometry(index)
                self.assertEqual((meta["offset"],meta["bytes"]),(offset,count))
                self.assertEqual(archive.read("payload.bin"),self.source_bytes[offset:offset+count])
                self.assertLessEqual(len(archive.read("part.json")),2048)
        result=recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        self.assertEqual(result["status"],"combined_verified_source")
        self.assertEqual(self.output.read_bytes(),self.source_bytes)
        self.assertEqual(self.prefix.read_bytes(),self.prefix_bytes)
        with self.assertRaises(ValueError):
            recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        self.assertEqual(self.output.read_bytes(),self.source_bytes)

    def test_bad_source_prefix_and_space_fail_without_new_artifact(self):
        self.source.write_bytes(self.source_bytes[:-1]+b"x")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.parts.exists())
        self.source.write_bytes(self.source_bytes)
        with patch.object(recovery.shutil,"disk_usage",return_value=shutil._ntuple_diskusage(4*1024**3,3*1024**3,1024**3)):
            with self.assertRaises(ValueError):
                self.prepare()
        self.prepare()
        prefix_before=self.prefix.read_bytes()
        required=self.pins.source_bytes + 64*1024**2
        with patch.object(recovery.shutil,"disk_usage",return_value=shutil._ntuple_diskusage(required,1,required-1)):
            with self.assertRaisesRegex(ValueError,"Insufficient free space"):
                recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.prefix.read_bytes(),prefix_before)
        self.prefix.write_bytes(b"x"*len(self.prefix_bytes))
        before=self.prefix.read_bytes()
        with self.assertRaises(ValueError):
            recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.prefix.read_bytes(),before)

    def test_symlinks_and_directory_inventory_are_rejected(self):
        alias=self.root / "alias"
        alias.symlink_to(self.source)
        with self.assertRaises(ValueError):
            recovery.prepare(alias,self.parts,_pins=self.pins)
        self.prepare()
        (self.parts / "extra.zip").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        (self.parts / "extra.zip").unlink()
        chunk=self.parts/recovery._name(0)
        saved=self.root / "saved.zip"
        chunk.rename(saved);chunk.symlink_to(saved)
        with self.assertRaises(ValueError):
            recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)

    def test_unknown_members_compression_special_and_schema_rejected(self):
        variants=({"extra":True},{"names":["../payload.bin","part.json"]},
                  {"special":True},{"compression":zipfile.ZIP_DEFLATED},
                  {"metadata":{"format":"unknown"}})
        for changes in variants:
            self.prepare()
            self.rewrite(0,**changes)
            with self.assertRaises((ValueError,zipfile.BadZipFile)):
                recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
            if self.output.exists():
                self.output.unlink()
            shutil.rmtree(self.parts)

    def test_raw_crc_corruption_rejected(self):
        self.prepare()
        path=self.parts/recovery._name(0)
        raw=bytearray(path.read_bytes())
        raw[30+len("payload.bin")+2]^=1
        path.write_bytes(raw)
        with self.assertRaises(zipfile.BadZipFile):
            recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        self.assertEqual(self.prefix.read_bytes(),self.prefix_bytes)

    def test_locally_consistent_tampered_payload_still_fails_final_trusted_sha(self):
        self.prepare()
        path=self.parts/recovery._name(0)
        with zipfile.ZipFile(path) as archive:
            payload=bytearray(archive.read("payload.bin"))
            meta=json.loads(archive.read("part.json"))
        payload[0]^=1
        meta["payload_sha256"]=sha(payload)
        self.rewrite(0,payload=bytes(payload),metadata=meta)
        with self.assertRaisesRegex(ValueError,"Final source size/SHA256"):
            recovery.combine(self.prefix,self.parts,self.output,_pins=self.pins)
        self.assertTrue(self.output.exists())  # Explicitly unverified partial; prefix remains untouched.
        self.assertNotEqual(sha(self.output.read_bytes()),self.pins.source_sha)
        self.assertEqual(self.prefix.read_bytes(),self.prefix_bytes)

    def test_fixed_cli_has_no_pin_override_and_never_accepts_tiny_fixture(self):
        result=subprocess.run([sys.executable,str(ROOT / "scripts/recover_public_download.py"),"prepare",str(self.source),str(self.parts)],
            text=True,capture_output=True,timeout=10,env=dict(os.environ,SOURCE_BYTES=str(len(self.source_bytes)),SOURCE_SHA=self.pins.source_sha,PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(result.returncode,2)
        self.assertEqual(json.loads(result.stderr)["status"],"failed_not_verified")
        self.assertFalse(self.parts.exists())
        result=subprocess.run([sys.executable,str(ROOT / "scripts/recover_public_download.py"),"prepare",str(self.source),str(self.parts),"--source-sha",self.pins.source_sha],text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,2)
        self.assertIn("unrecognized arguments",result.stderr)


if __name__=="__main__":
    unittest.main()
