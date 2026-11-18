"""Audited use of the pinned official token splitter; no alternative text frontend.

Private reports bind every character span and every actual per-segment GPT input.
BPE may retokenize at segment boundaries: character coverage, not concatenated
BPE IDs, is the lossless-coverage invariant. EOS verifies code termination only.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import unicodedata
from . import synth

MODE = 'stock-token-v1'
MAX_SEGMENTS = 8


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def token_digest(tokens):
    return digest(json.dumps(tokens))


def plan_segments(tts, source, text, prefix, max_text_tokens):
    """Call real source methods, then independently check their bounded coverage."""
    expected = (Path(source) / 'indextts/infer_v2_5.py').resolve(strict=True)
    for name in ('split_text_by_tokens', '_token_len', '_split_atomic_pieces'):
        method = getattr(tts, name)
        if Path(inspect.getfile(method)).resolve() != expected:
            raise synth.SynthesisError('Segment splitter must come from the pinned official source')
    if getattr(tts, 'low_vram', False):
        raise synth.SynthesisError('Low-VRAM alternate frontend is outside the segmented experiment')
    capacity = tts.gpt.text_pos_embedding.emb.num_embeddings
    if type(capacity) is not int or capacity < 4 or type(max_text_tokens) is not int or not 8 <= max_text_tokens <= 120:
        raise synth.SynthesisError('Invalid bounded text position/token budget')
    control = tts.tokenizer.encode(prefix.strip(), allowed_special='all')
    prefix_tokens = tts.tokenizer.encode(prefix, allowed_special='all')
    if len(control) != 1:
        raise synth.SynthesisError('Expected one real language control token')
    budget = min(max_text_tokens, capacity - 2) - len(prefix_tokens)
    if budget < 1:
        raise synth.SynthesisError('No usable official segment token budget')
    pieces = tts.split_text_by_tokens(text, max_text_tokens, prefix)
    if (not isinstance(pieces, list) or not 1 <= len(pieces) <= MAX_SEGMENTS
            or any(not isinstance(piece, str) or not piece.strip() for piece in pieces)
            or ''.join(pieces) != text):
        raise synth.SynthesisError('Official segments must cover the complete text exactly, in at most eight nonempty pieces')
    plans, position = [], 0
    for index, piece in enumerate(pieces):
        tokens = tts.tokenizer.encode(prefix + piece, allowed_special='all')
        body = tts.tokenizer.encode(piece, allowed_special='all')
        if (not tokens or tokens[0] != control[0] or len(body) > budget
                or len(tokens) > min(max_text_tokens, capacity - 2)
                or any(token in (tts.gpt.start_text_token, tts.gpt.stop_text_token) for token in tokens)):
            raise synth.SynthesisError('Official segment exceeds its actual token/position budget or contains framing IDs')
        record = dict(index=index, char_start=position, char_end=position + len(piece),
            text_sha256=digest(piece), body_token_count=len(body), input_token_count=len(tokens) + 1,
            token_ids_sha256=token_digest(tokens), status='not_executed', reason='not_started',
            input_verified=False, semantic_eos_verified=False,
            starts_inside_unicode_mark=position > 0 and unicodedata.category(text[position]).startswith('M'))
        plans.append(dict(tokens=tokens, record=record))
        position += len(piece)
    return plans, dict(mode=MODE, source_text_sha256=digest(text), source_characters=len(text),
        reconstructed_text_sha256=digest(''.join(pieces)), complete_character_coverage=True,
        unsegmented_token_ids_sha256=token_digest(tts.tokenizer.encode(prefix + text, allowed_special='all')),
        max_text_tokens=max_text_tokens, text_position_capacity=capacity, body_token_budget=budget,
        segments=len(plans), max_segments=MAX_SEGMENTS, interval_silence_ms=200,
        bpe_boundary_retokenization=True,
        unicode_mark_boundary_count=sum(p['record']['starts_inside_unicode_mark'] for p in plans),
        boundary_policy='unchanged official splitter; character coverage is not a word/grapheme guarantee',
        semantic_or_quality_completeness_verified=False,
        segments_audit=[plan['record'] for plan in plans])


def validate_codes(codes, gpt, cap, record):
    """Inspect the real generated codes before the official decoder can consume them."""
    import torch
    if (not isinstance(codes, torch.Tensor) or codes.ndim != 2 or codes.shape[0] != 1
            or codes.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)):
        raise synth.SynthesisError('Expected one integer semantic-code sequence per segment')
    if type(cap) is not int or not 2 <= codes.shape[1] <= cap <= 512:
        record.update(code_count_including_eos=codes.shape[1], max_mel_tokens=cap)
        raise synth.SynthesisError('Empty or over-cap segment semantic-code sequence')
    values = codes.detach().cpu().tolist()[0]
    record.update(code_count_including_eos=len(values), max_mel_tokens=cap,
                  semantic_code_ids_sha256=token_digest(values), stop_mel_token=gpt.stop_mel_token,
                  returned_eos_observed=bool(values and values[-1] == gpt.stop_mel_token),
                  first_eos_index=values.index(gpt.stop_mel_token) if gpt.stop_mel_token in values else None,
                  returned_length_reached_cap=len(values) == cap,
                  decoder_stopping_reason_verified=False)
    if values[-1] != gpt.stop_mel_token or gpt.stop_mel_token in values[:-1]:
        raise synth.SynthesisError('Segment did not terminate exactly once with the real semantic EOS')
    if any(value < 0 or value >= gpt.start_mel_token for value in values[:-1]):
        raise synth.SynthesisError('Segment contains non-code/reserved semantic IDs')
    record.update(status='semantic_eos_verified', reason=None, semantic_eos_verified=True,
                  semantic_code_count=len(values) - 1)


@contextmanager
def audit_segments(tts, source, plans, slot, cap):
    module = importlib.import_module('indextts.gpt.model_v2')
    if type(tts.gpt) is not module.UnifiedVoice or not Path(inspect.getfile(type(tts.gpt))).resolve().is_relative_to(source):
        raise synth.SynthesisError('Segment input audit requires the real stock official GPT class')
    original = tts.gpt.inference_speech
    had_override = 'inference_speech' in vars(tts.gpt)
    original_override = vars(tts.gpt).get('inference_speech')
    observed = {'calls': 0, 'completed': 0}
    def audited(speech, text_inputs, langs=None, *args, **kwargs):
        index = observed['calls']
        if index >= len(plans):
            raise synth.SynthesisError('Unexpected additional stock segment generation call')
        observed['calls'] += 1
        record = plans[index]['record']
        record.update(status='attempt_started', reason='no_completed_code_result')
        try:
            expected = [plans[index]['tokens'] + [tts.gpt.stop_text_token]]
            if (text_inputs.detach().cpu().tolist() != expected or langs.detach().cpu().tolist() != [slot]
                    or kwargs.get('max_generate_length') != cap or kwargs.get('num_return_sequences') != 1
                    or kwargs.get('do_sample') is not False or kwargs.get('num_beams') != 1):
                raise synth.SynthesisError('Actual segment GPT input/language/code cap differs from the audited plan')
            record['input_verified'] = True
            codes, latent = original(speech, text_inputs, langs, *args, **kwargs)
            validate_codes(codes, tts.gpt, cap, record)
            observed['completed'] += 1
            return codes, latent
        except BaseException as exc:
            record.update(status='failed', reason=str(exc)[:2000], failure_type=type(exc).__name__)
            raise
    tts.gpt.inference_speech = audited
    try:
        yield observed
    finally:
        if had_override:
            tts.gpt.inference_speech = original_override
        else:
            del tts.gpt.inference_speech
        for plan in plans[observed['calls']:]:
            plan['record']['reason'] = 'not_reached_before_inference_exit'
