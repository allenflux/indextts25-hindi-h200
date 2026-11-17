"""Tiny actual Torch checkpoints only; no full model, production weights or GPU."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_merged_gpt as audit


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional Torch CPU audit tests')
class WeightAuditTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        self.temp=tempfile.TemporaryDirectory(prefix='gpt-byte-audit-',dir=ROOT/'.runtime')
        self.root=Path(self.temp.name)
        self.base={'gpt.h.0.attn.c_attn.weight':torch.arange(20,dtype=torch.float32).reshape(4,5),
            'gpt.h.0.attn.c_attn.bias':torch.zeros(5), 'gpt.h.0.mlp.c_fc.weight':torch.ones(4,5),
            'lang_embedding.weight':torch.zeros(20,4), 'text_embedding.weight':torch.zeros(30,4),
            'mel_head.weight':torch.zeros(5,4), 'text_head.bias':torch.zeros(30)}
        self.candidate={key:value.clone() for key,value in self.base.items()}
        self.candidate['gpt.h.0.attn.c_attn.weight'][1,2]+=1
        self.candidate['lang_embedding.weight'][17,0]=2
        self.candidate['text_embedding.weight'][2,0]=1
        self.candidate['text_embedding.weight'][12,2]=3
        self.candidate['mel_head.weight'][0,0]=4
        self.candidate['text_head.bias'][0]=5

    def tearDown(self):self.temp.cleanup()

    def checkpoints(self):
        paths=[];pins=[]
        for name,state in [('base',self.base),('candidate',self.candidate)]:
            path=self.root/(name+'.pth');self.torch.save({'model':state},path)
            paths.append(path);pins.append(audit.WeightPin(path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()))
        return paths,pins

    def test_actual_weights_only_mmap_roundtrip_groups_and_embedding_rows(self):
        paths,pins=self.checkpoints();before=[p.read_bytes() for p in paths]
        with patch.object(self.torch.cuda,'is_initialized',return_value=False):
            report=audit._execute(*paths,time.monotonic()+30,64,_pins=tuple(pins))
        self.assertTrue(report['key_shape_dtype_equal'])
        self.assertEqual(report['base_tensor_count'],7)
        self.assertEqual(report['changed_tensors'],5)
        self.assertEqual(report['bitwise_identical_tensors'],2)
        lang=report['embedding_rows']['lang_embedding.weight']
        self.assertTrue(lang['hindi_row_changed']);self.assertTrue(lang['non_hindi_all_bitwise_equal'])
        self.assertEqual(lang['non_hindi_changed_rows'],0)
        self.assertEqual(report['embedding_rows']['text_embedding.weight']['changed_rows'],2)
        self.assertEqual(report['groups']['shared_attention_projection_weights']['changed'],1)
        self.assertEqual(report['groups']['transformer_mlp']['bitwise_identical'],1)
        self.assertEqual(before,[p.read_bytes() for p in paths])
        self.assertNotIn('tensor(',json.dumps(report));self.assertNotIn(str(self.root),json.dumps(report))

    def test_byte_exact_signed_zero_nan_and_non_hindi_row_detection(self):
        t=self.torch
        left=t.tensor([0.,float('nan')]);right=left.clone();right[0]=-0.
        self.assertTrue(t.equal(left.view(t.uint8)[4:],right.view(t.uint8)[4:]))
        self.assertFalse(audit._identical(left,right,t,4,time.monotonic()+10))
        self.assertTrue(audit._identical(left,left.clone(),t,4,time.monotonic()+10))
        self.candidate['lang_embedding.weight'][1,1]=1
        value=audit._compare_states(self.base,self.candidate,t,64,time.monotonic()+10)
        self.assertFalse(value['embedding_rows']['lang_embedding.weight']['non_hindi_all_bitwise_equal'])
        self.assertEqual(value['embedding_rows']['lang_embedding.weight']['non_hindi_changed_rows'],1)

    def test_shape_dtype_and_key_changes_are_separate_evidence(self):
        t=self.torch
        self.candidate.pop('text_head.bias');self.candidate['new_tensor']=t.ones(1)
        self.candidate['mel_head.weight']=t.ones(6,4,dtype=t.float64)
        self.candidate['gpt.h.0.attn.c_attn.bias']=t.zeros(5,dtype=t.float64)
        r=audit._compare_states(self.base,self.candidate,t,64,time.monotonic()+10)
        self.assertFalse(r['key_shape_dtype_equal']);self.assertEqual(r['missing_keys'],['text_head.bias'])
        self.assertEqual(r['unexpected_keys'],['new_tensor']);self.assertEqual(r['shape_mismatches'],['mel_head.weight'])
        self.assertEqual(r['dtype_mismatches'],['gpt.h.0.attn.c_attn.bias','mel_head.weight'])

    def test_bad_sha_deadline_layout_and_input_mutation_fail(self):
        paths,pins=self.checkpoints()
        with self.assertRaisesRegex(audit.AuditError,'SHA256'):
            audit._execute(*paths,time.monotonic()+20,64,_pins=(pins[0],audit.WeightPin(pins[1].size,'0'*64)))
        with self.assertRaisesRegex(audit.AuditError,'deadline'):
            audit._execute(*paths,time.monotonic()-1,64,_pins=tuple(pins))
        state=dict(self.candidate);state['bad']=self.torch.ones(2,3).T
        base=dict(self.base,bad=self.torch.ones(3,2))
        with self.assertRaisesRegex(audit.AuditError,'layout'):
            audit._compare_states(base,state,self.torch,64,time.monotonic()+20)
        original=audit._compare_states
        def mutate(*args):
            result=original(*args)
            with paths[1].open('ab') as stream:stream.write(b'changed after mapping')
            return result
        with patch.object(audit,'_compare_states',side_effect=mutate),patch.object(self.torch.cuda,'is_initialized',return_value=False):
            with self.assertRaisesRegex(audit.AuditError,'changed during'):
                audit._execute(*paths,time.monotonic()+20,64,_pins=tuple(pins))

    def test_output_and_input_paths_are_protected_and_publish_is_exclusive(self):
        paths,_=self.checkpoints()
        link=self.root/'link.pth';link.symlink_to(paths[0])
        with self.assertRaises(audit.AuditError):audit.source_path(link)
        with self.assertRaises(audit.AuditError):audit.destination(self.root/'report.json',self.root,paths)
        code=self.root/'code';code.mkdir()
        model_a=self.root/'models/base';model_b=self.root/'models/candidate'
        model_a.mkdir(parents=True);model_b.mkdir()
        inputs=[model_a/'gpt.pth',model_b/'gpt.pth']
        with patch.object(audit,'ROOT',code):
            self.assertEqual(audit.destination(self.root/'allowed.json',self.root,inputs),self.root/'allowed.json')
            for forbidden in (code/'report.json',model_a/'report.json',model_b/'report.json',self.root.parent/'outside.json'):
                with self.subTest(path=forbidden.name),self.assertRaises(audit.AuditError):
                    audit.destination(forbidden,self.root,inputs)
        target=self.root/'exclusive.json'
        audit.publish(target,{'status':'fixture'})
        with self.assertRaises(FileExistsError):audit.publish(target,{'status':'must not overwrite'})
        self.assertEqual(json.loads(target.read_text()),{'status':'fixture'})

    def test_cli_fixed_production_pins_cannot_accept_fixture_or_override(self):
        paths,_=self.checkpoints();output=self.root/'never.json'
        stream=io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(audit.main(['--base',str(paths[0]),'--candidate',str(paths[1]),
                '--workspace-root',str(self.root),'--output',str(output),'--execute-cpu']),2)
        self.assertFalse(output.exists());self.assertIn('byte count mismatch',stream.getvalue())
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            audit.main(['--base-sha256','0'*64])


class MetadataImportTests(unittest.TestCase):
    def test_import_and_help_load_no_torch_or_model(self):
        result=subprocess.run([sys.executable,'-B','-c',
            "import sys;sys.path.insert(0,'scripts');import audit_merged_gpt;assert 'torch' not in sys.modules;print('cpu-audit-import-only')"],
            cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'cpu-audit-import-only')


if __name__=='__main__':unittest.main()
