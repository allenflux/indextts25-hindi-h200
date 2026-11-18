"""Actual pinned tokenizer/splitter and tiny GPT CPU contracts; no TTS/WAV claims."""
import ast
import array
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import wave
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hi_h200_native import batch_synth as batch
from hi_h200_native import stock_segments as segments
from hi_h200_native import synth

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get('HI_H200_OFFICIAL_REPO', ROOT / '.runtime/external-official'))
TOKEN_DIR = ROOT / '.runtime/pure-lora-20261118/release-transfer-001/pure-lora-pilot48-step100-001/base-metadata'
TOKEN_FILE = TOKEN_DIR / 'multilingual_zh_ja_yue_char_del.tiktoken'
AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


def real_splitter():
    """Compile unchanged methods from the verified local file, without TTS imports."""
    source = synth.verify_source(OFFICIAL)
    path = source / 'indextts/infer_v2_5.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndexTTS2')
    methods = {'_token_len', '_split_atomic_pieces', 'split_text_by_tokens'}
    body = [copy.deepcopy(n) for n in cls.body if
        (isinstance(n, ast.FunctionDef) and n.name in methods) or
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPLIT_PROTECTED_PATTERN' for t in n.targets))]
    namespace = {'re':re, '__name__':'_cpu_pinned_split_methods'}
    new_cls = ast.ClassDef(name='PinnedSplitter', bases=[], keywords=[], body=body, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new_cls], type_ignores=[])), str(path), 'exec'), namespace)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from indextts.utils.tokenizer import get_tokenizer
    blob = TOKEN_FILE.read_bytes()
    identity = __import__('hashlib').sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
    if identity != 'ac47fe685d24565030965a94c0d98e00e0331eaa':
        raise AssertionError('Test must use the real pinned tokenizer, never an invented tokenizer')
    tts = namespace['PinnedSplitter']()
    tts.tokenizer = get_tokenizer(True, model_dir=str(TOKEN_DIR))
    tts.low_vram = False
    tts.gpt = SimpleNamespace(start_text_token=0, stop_text_token=1,
        text_pos_embedding=SimpleNamespace(emb=SimpleNamespace(num_embeddings=122)))
    return tts


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir() and TOKEN_FILE.is_file(), 'Pinned local source/tokenizer required; never download')
class RealStockSegmentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts = real_splitter()

    def test_actual_hindi_tokenizer_split_preserves_all_unicode_and_framing(self):
        text = 'यह एक छोटा परीक्षण है। सभी वाक्यों का पूरा उच्चारण होना चाहिए। भाषा और शब्दों को बदले बिना आवाज़ बननी चाहिए।'
        plans, report = segments.plan_segments(self.tts, OFFICIAL, text, '<|hi|> ', 120)
        self.assertGreater(len(plans), 1)
        self.assertEqual(report['source_text_sha256'], report['reconstructed_text_sha256'])
        self.assertTrue(report['complete_character_coverage'])
        self.assertFalse(report['semantic_or_quality_completeness_verified'])
        self.assertTrue(report['bpe_boundary_retokenization'])
        position = 0
        for plan in plans:
            row = plan['record']
            self.assertEqual(row['char_start'], position)
            piece = text[position:row['char_end']]
            self.assertEqual(row['text_sha256'], segments.digest(piece))
            self.assertEqual(plan['tokens'], self.tts.tokenizer.encode('<|hi|> ' + piece, allowed_special='all'))
            self.assertLessEqual(row['body_token_count'], report['body_token_budget'])
            self.assertLessEqual(row['input_token_count'] + 1, report['text_position_capacity'])
            position = row['char_end']
        self.assertEqual(position, len(text))
        for text in ('क़िला और शक्ति की भाषा है।', 'The first sentence. The next sentence.', '中文测试。日语测试！'):
            plans, report = segments.plan_segments(self.tts, OFFICIAL, text, '<|hi|> ', 24)
            self.assertTrue(report['complete_character_coverage'])
            self.assertEqual(''.join(text[p['record']['char_start']:p['record']['char_end']] for p in plans), text)
            if text.startswith('क़िला'):
                self.assertGreater(report['unicode_mark_boundary_count'],0)

    def test_opt_in_60_uses_same_audited_real_split_and_infer_budget(self):
        import torch
        text = 'यह एक छोटा परीक्षण है। सभी वाक्यों का पूरा उच्चारण होना चाहिए। भाषा और शब्दों को बदले बिना आवाज़ बननी चाहिए।'
        tts = copy.copy(self.tts)
        tts.gpt = copy.copy(tts.gpt)
        tts.gpt.lang_embedding = SimpleNamespace(num_embeddings=18)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            root = Path(tmp)
            prompt = root/'prompt-fixture'
            prompt.write_bytes(b'Identity fixture only; no decoder or generated output')
            segment_counts = {}
            for budget in (120, 60):
                with self.subTest(budget=budget):
                    output = root/f'budget-{budget}'
                    output.mkdir()
                    row = dict(id='hi', language='hi', text=text, prompt=prompt.name,
                        prompt_sha256=synth.sha256(prompt), max_mel_tokens=512)
                    args = SimpleNamespace(segmentation_mode=segments.MODE)
                    tts.infer = lambda **_: None
                    with patch.object(tts,'infer',side_effect=synth.SynthesisError('fixture stop before audio')) as infer, \
                         patch.object(batch,'case_resources',return_value={}), \
                         patch.object(batch,'authored_to_stock',return_value=text), \
                         patch.object(batch,'reset_seed'), patch.object(torch.cuda,'synchronize'), \
                         patch.object(segments,'audit_segments',return_value=nullcontext(dict(calls=0,completed=0))):
                        with self.assertRaisesRegex(ValueError,'fixture stop before audio'):
                            batch.attempt_case(args,dict(output=output,workspace=root,source=OFFICIAL,text_budget=budget),tts,row,'fixture')
                    infer.assert_called_once()
                    self.assertEqual(infer.call_args.kwargs['max_text_tokens_per_segment'],budget)
                    self.assertEqual(infer.call_args.kwargs['max_mel_tokens'],512)
                    audit = row['segmentation']
                    self.assertEqual(audit['max_text_tokens'],budget)
                    self.assertEqual(audit['body_token_budget'],budget-2)
                    self.assertTrue(audit['complete_character_coverage'])
                    self.assertFalse(audit['all_returned_codes_ended_with_eos'])
                    self.assertFalse(audit['acoustic_output_verified'])
                    self.assertFalse(any(output.rglob('*.wav')))
                    segment_counts[budget] = audit['segments']
            self.assertGreater(segment_counts[60],segment_counts[120])

    def test_real_official_punctuation_and_unpunctuated_token_fallback(self):
        for text in ('one two three four five. Next clause. Final clause?', 'क़िला शक्ति भाषा स्वर अक्षर उच्चारण परीक्षण सरल भाषा'):
            plans, report = segments.plan_segments(self.tts, OFFICIAL, text, '<|hi|> ', 24)
            expected = self.tts.split_text_by_tokens(text, 24, '<|hi|> ')
            self.assertEqual([text[p['record']['char_start']:p['record']['char_end']] for p in plans], expected)
            self.assertLessEqual(len(plans), segments.MAX_SEGMENTS)

    def test_rejects_unbounded_cap_overflow_alternate_frontend_and_wrong_origin(self):
        for cap in (0, 7, 121, True):
            with self.assertRaises(ValueError):
                segments.plan_segments(self.tts, OFFICIAL, 'छोटा परीक्षण', '<|hi|> ', cap)
        with self.assertRaisesRegex(ValueError, 'eight'):
            segments.plan_segments(self.tts, OFFICIAL, 'यह भाषा परीक्षण है। ' * 30, '<|hi|> ', 8)
        with patch.object(self.tts, 'low_vram', True):
            with self.assertRaisesRegex(ValueError, 'Low-VRAM'):
                segments.plan_segments(self.tts, OFFICIAL, 'भाषा', '<|hi|> ', 120)
        with patch.object(self.tts, 'split_text_by_tokens', lambda *a:['lost']):
            with self.assertRaisesRegex(ValueError, 'pinned official'):
                segments.plan_segments(self.tts, OFFICIAL, 'भाषा', '<|hi|> ', 120)

    def test_lossy_official_result_and_reserved_control_are_fail_closed(self):
        original = self.tts.split_text_by_tokens
        def lossy(*args):
            return ['शक्ति']
        # Only the origin check is a fixture; the independent no-loss check is real.
        with patch.object(self.tts, 'split_text_by_tokens', lossy), patch.object(segments.inspect, 'getfile', return_value=str(OFFICIAL / 'indextts/infer_v2_5.py')):
            with self.assertRaisesRegex(ValueError, 'cover the complete text'):
                segments.plan_segments(self.tts, OFFICIAL, 'शक्ति भाषा', '<|hi|> ', 120)
        with self.assertRaisesRegex(ValueError, 'framing'):
            segments.plan_segments(self.tts, OFFICIAL, 'This sentence!', '<|hi|> ', 120)
        self.assertEqual(self.tts.split_text_by_tokens, original)


@unittest.skipUnless(AVAILABLE and OFFICIAL.is_dir(), 'Actual tiny official GPT required')
class RealSegmentCodeAuditTests(unittest.TestCase):
    def setUp(self):
        import torch
        from hi_h200_native.model import make_tiny_official_model, import_official_model
        torch.set_num_threads(1)
        native = make_tiny_official_model(OFFICIAL, seed=12)
        self.gpt = import_official_model(OFFICIAL)(**native.native_config['gpt']).eval()
        del native
        with torch.no_grad():
            self.gpt.mel_head.weight.zero_()
            self.gpt.mel_head.bias.zero_()
            self.gpt.mel_head.bias[self.gpt.stop_mel_token] = 100
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
        self.tts = SimpleNamespace(gpt=self.gpt)
        self.plans = [dict(tokens=[3,4,5], record=dict(status='not_executed', input_verified=False, semantic_eos_verified=False)) for _ in range(2)]

    def call(self, tokens=None, cap=3, minimum=2, **updates):
        import torch
        kwargs = dict(emo_vec=torch.zeros(1,16), campplus_embedding=torch.zeros(1,192),
            max_generate_length=cap, min_new_tokens=minimum, do_sample=False, num_beams=1, num_return_sequences=1)
        kwargs.update(updates)
        return self.gpt.inference_speech(torch.zeros(1,4,1024), torch.tensor([(tokens or [3,4,5])+[1]]), torch.tensor([17]), **kwargs)

    def test_actual_two_tiny_greedy_generations_are_observed_and_method_restored(self):
        import torch
        original = self.gpt.inference_speech
        with torch.inference_mode(), segments.audit_segments(self.tts, OFFICIAL, self.plans, 17, 3) as observed:
            for _ in self.plans:
                codes, _ = self.call()
                self.assertEqual(codes.tolist(), [[0,0,33]])
        self.assertEqual(observed, {'calls':2, 'completed':2})
        self.assertEqual(self.gpt.inference_speech, original)
        self.assertNotIn('inference_speech',vars(self.gpt))
        self.assertTrue(all(p['record']['semantic_eos_verified'] for p in self.plans))
        self.assertTrue(all(p['record']['returned_length_reached_cap'] for p in self.plans))
        self.assertTrue(all(p['record']['first_eos_index']==2 for p in self.plans))
        self.assertFalse(any(p['record']['decoder_stopping_reason_verified'] for p in self.plans))

    def test_actual_no_eos_limit_is_failure_before_any_decoder_and_keeps_unreached_segment(self):
        import torch
        original = self.gpt.inference_speech
        with torch.inference_mode(), segments.audit_segments(self.tts, OFFICIAL, self.plans, 17, 3) as observed:
            with self.assertRaisesRegex(ValueError, 'terminate'):
                self.call(minimum=3)
        self.assertEqual(observed, {'calls':1, 'completed':0})
        self.assertEqual(self.plans[0]['record']['status'], 'failed')
        self.assertFalse(self.plans[0]['record']['returned_eos_observed'])
        self.assertEqual(self.plans[1]['record']['status'], 'not_executed')
        self.assertEqual(self.plans[1]['record']['reason'], 'not_reached_before_inference_exit')
        self.assertEqual(self.gpt.inference_speech, original)
        self.assertNotIn('inference_speech',vars(self.gpt))

    def test_input_cap_sampling_mismatch_and_extra_calls_reject_without_synthesis(self):
        import torch
        for kwargs in ({'tokens':[3,4,6]}, {'cap':4}, {'do_sample':True}, {'num_beams':2}):
            plans=copy.deepcopy(self.plans)
            with segments.audit_segments(self.tts, OFFICIAL, plans, 17, 3):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    self.call(**kwargs)
            self.assertFalse(plans[0]['record']['input_verified'])
        plans=copy.deepcopy(self.plans[:1])
        with torch.inference_mode(), segments.audit_segments(self.tts, OFFICIAL, plans, 17, 3):
            self.call()
            with self.assertRaisesRegex(ValueError, 'additional'):
                self.call()

    def test_invalid_code_geometry_reserved_and_early_eos_never_pass(self):
        import torch
        for codes in (torch.tensor([[33]]), torch.tensor([[1.,33.]]), torch.tensor([[1,33],[1,33]]),
                      torch.tensor([[1,33,33]]), torch.tensor([[32,33]]), torch.tensor([[-1,33]]),
                      torch.tensor([[1,2,3,33]])):
            record={}
            with self.assertRaises(ValueError):
                segments.validate_codes(codes, self.gpt, 3, record)
            self.assertFalse(record.get('semantic_eos_verified',False))


@unittest.skipUnless(OFFICIAL.is_dir(), 'Pinned source required; no model assets')
class SegmentedMetadataBoundsTests(unittest.TestCase):
    def test_hindi_240_only_explicit_and_load_cases_keeps_exact_manifest_hash(self):
        text = ('यह सरल भाषा परीक्षण है। ' * 9).strip()
        self.assertTrue(160 < len(text) <= 240)
        with self.assertRaises(ValueError):
            batch.canonical_text(OFFICIAL,'hi',text)
        self.assertEqual(batch.canonical_text(OFFICIAL,'hi',text,segments.MODE),text)
        for value in (text+'  ', text+' १२', text+'<|hi|>', text+'\u202e'):
            with self.assertRaises(ValueError):
                batch.canonical_text(OFFICIAL,'hi',value,segments.MODE)
        for mode in ('unsupported',True):
            with self.assertRaises(ValueError):
                batch.canonical_text(OFFICIAL,'hi',text,mode)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            root=Path(tmp)
            with wave.open(str(root/'prompt.wav'),'wb') as audio:
                audio.setparams((1,2,16000,0,'NONE','not compressed'))
                audio.writeframes(array.array('h',[1000]*16000).tobytes())
            path=root/'cases.json'
            payload=json.dumps([dict(id='val-fixed',language='hi',text=text,prompt='prompt.wav',max_mel_tokens=256)],ensure_ascii=False).encode()
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                batch.load_cases(path,root,OFFICIAL)
            rows, identity=batch.load_cases(path,root,OFFICIAL,segmentation_mode=segments.MODE)
            self.assertEqual(identity,__import__('hashlib').sha256(payload).hexdigest())
            self.assertEqual(rows[0]['text'],text)
            self.assertEqual(path.read_bytes(),payload)
            self.assertFalse(rows[0]['generation_attempted'])
            for budget in (60,120):
                self.assertEqual(batch.load_cases(path,root,OFFICIAL,segments.MODE,
                    segmented_text_budget=budget),(rows,identity))
                with self.assertRaisesRegex(ValueError,'stock-token-v1'):
                    batch.load_cases(path,root,OFFICIAL,segmented_text_budget=budget)
            for budget in ('60',True,61):
                with self.assertRaisesRegex(ValueError,'60 or 120'):
                    batch.load_cases(path,root,OFFICIAL,segments.MODE,segmented_text_budget=budget)
            en=dict(id='fixed-en',language='en',text='This is a test.',prompt='prompt.wav',max_mel_tokens=256)
            path.write_text(json.dumps([json.loads(payload)[0],en]))
            self.assertEqual(len(batch.load_cases(path,root,OFFICIAL,segments.MODE,
                segmented_text_budget=120)[0]),2)
            with self.assertRaisesRegex(ValueError,'all-Hindi'):
                batch.load_cases(path,root,OFFICIAL,segments.MODE,segmented_text_budget=60)

    def test_optional_caller_pin_rejects_before_runtime_without_changing_unpinned_route(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            root=Path(tmp)
            original,merged=root/'original',root/'merged'
            original.mkdir();merged.mkdir()
            (original/'model-manifest.json').write_text('{}')
            with wave.open(str(root/'prompt.wav'),'wb') as audio:
                audio.setparams((1,2,16000,0,'NONE','not compressed'))
                audio.writeframes(array.array('h',[1000]*16000).tobytes())
            cases=root/'cases.json'
            cases.write_text(json.dumps([dict(id='pin-check',language='hi',text='यह परीक्षण है।',
                prompt=str((root/'prompt.wav').relative_to(ROOT.parent)),max_mel_tokens=256)],ensure_ascii=False))
            script="""import sys
from pathlib import Path
from types import SimpleNamespace
from hi_h200_native import batch_synth as b
args=SimpleNamespace(workspace_root=Path(WORK),official_repo=Path(SOURCE),model_dir=Path(ORIGINAL),
 model_manifest=Path(ORIGINAL)/'model-manifest.json',merged_bundle=Path(MERGED),cases=Path(CASES),
 output_dir=Path(OUTPUT),variant='original',no_kv_cache=False,segmentation_mode='single')
b.synth.verify_synth_assets=lambda *a: ({},{'files':{'gpt.pth':'a'*64}})
b.synth.verify_merged_bundle=lambda *a: {'gpt_sha256':'b'*64}
b.synth.dependency_report=lambda *a: {'missing':[]}
plan,report=b.preflight(args)
assert 'expected_merged_gpt_sha256' not in report
assert report['selected_gpt_sha256']=='a'*64
for pin in ('c'*64,'B'*64,'b'*63,'b'*65,True):
 args.expected_merged_gpt_sha256=pin
 try: b.preflight(args)
 except ValueError: pass
 else: raise AssertionError('wrong/invalid caller pin accepted')
args.expected_merged_gpt_sha256='b'*64
plan,report=b.preflight(args)
assert report['expected_merged_gpt_sha256']=='b'*64
assert report['selected_gpt_sha256']=='a'*64
args.segmentation_mode='stock-token-v1'
default_plan, default_report=b.preflight(args)
assert default_plan['text_budget']==120
assert default_report['segmentation_experiment']['max_text_tokens']==120
args.segmented_text_budget=120
explicit_plan, explicit_report=b.preflight(args)
assert (explicit_plan, explicit_report)==(default_plan, default_report)
args.segmented_text_budget=60
short_plan, short_report=b.preflight(args)
assert short_plan['text_budget']==60
assert short_report['segmentation_experiment']['max_text_tokens']==60
assert short_report['cases']==default_report['cases']
assert not Path(OUTPUT).exists()
assert 'torch' not in sys.modules
assert 'indextts.infer_v2_5' not in sys.modules
"""
            for name,value in dict(WORK=str(ROOT.parent),SOURCE=str(OFFICIAL),ORIGINAL=str(original),
                                   MERGED=str(merged),CASES=str(cases),OUTPUT=str(ROOT.parent/('pin-preflight-'+root.name))).items():
                script=script.replace(name,repr(value))
            result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=30,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')))
            self.assertEqual(result.returncode,0,result.stderr)

    def test_complete_pcm_60_seconds_only_explicit_never_truncated_for_asr(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            path=Path(tmp)/'complete.wav'
            for seconds in (20.01,60,60.01):
                with wave.open(str(path),'wb') as audio:
                    audio.setparams((1,2,8000,0,'NONE','not compressed'))
                    audio.writeframes(array.array('h',[1000]*int(seconds*8000)).tobytes())
                original=path.read_bytes()
                with self.assertRaises(ValueError):
                    batch.output_audio(path)
                if seconds<=60:
                    self.assertEqual(batch.output_audio(path,segments.MODE)['seconds'],seconds)
                else:
                    with self.assertRaises(ValueError):
                        batch.output_audio(path,segments.MODE)
                self.assertEqual(path.read_bytes(),original)


class SegmentationCLIContractTests(unittest.TestCase):
    def test_default_single_and_new_explicit_cli_without_runtime_imports(self):
        tree=ast.parse(Path(batch.__file__).read_text())
        parser_call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and any(isinstance(arg,ast.Constant) and arg.value=='--segmentation-mode' for arg in node.args))
        self.assertEqual(next(k.value.value for k in parser_call.keywords if k.arg=='default'),'single')
        script="from hi_h200_native import batch_synth; import sys; assert 'torch' not in sys.modules; assert 'indextts.infer_v2_5' not in sys.modules; raise SystemExit(batch_synth.main(['--help']))"
        result=subprocess.run([sys.executable,'-c',script], cwd=ROOT, capture_output=True, text=True, timeout=30,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--segmentation-mode {single,stock-token-v1}',result.stdout)
        self.assertIn('--segmented-text-budget {60,120}',result.stdout)


if __name__ == '__main__':
    unittest.main()
