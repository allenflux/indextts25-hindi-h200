#!/usr/bin/env python3
"""Prepare a frozen private Hindi evaluation sample from reviewed local shards.

Uses only given files. Corpus provenance is operator-supplied, and disjointness
covers the supplied prior inventories' publisher speaker IDs and clip identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hi_h200.data import load_json
from hi_h200.text import DataError, normalize_hindi
from hi_h200_native import batch_synth, hf_data, stock_segments, synth

MAX_SPEAKERS = 10
MAX_SECONDS = 15
MAX_INVENTORY_BYTES = 16 * 1024**2
MEL_CAP = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = path.absolute()
    require(path.resolve(strict=True) == path, "Use existing input paths without symlinks")
    return path


def contained(path, work):
    path = canonical(path)
    require(path.is_relative_to(work), "Input escaped its declared root")
    return path


def read_input(path, work, limit=MAX_INVENTORY_BYTES):
    path = contained(path, work)
    require(path.is_file() and 0 < path.stat().st_size <= limit, "Input exceeds its declared bound")
    return path.read_bytes()


def corpus_audits(root):
    direct = any((root / name).exists() for name in ("plan.json", "downloaded_files.json", "CORPUS_COMPLETE.json"))
    nested = any((root / "audit" / name).exists() for name in ("plan.json", "downloaded_files.json"))
    require(direct != nested, "The authorized corpus needs a single unambiguous audit layout")
    folder = root if direct else root / "audit"
    plan = load_json(read_input(folder / "plan.json", root).decode())
    audit = load_json(read_input(folder / "downloaded_files.json", root).decode())
    if direct:
        completion = load_json(read_input(root / "CORPUS_COMPLETE.json", root).decode())
        rights_path = contained(root / "operator-rights.json", root)
        recorded = hf_data.rights_record(rights_path)
        require(completion.get("status") == "previously_acquired_corpus_staged" and
                completion.get("authentication_copied") is False and completion.get("network_downloaded") is False and
                completion.get("bytes") == sum(size for _, size in hf_data.SHARDS) and
                plan.get("rights", {}).get("sha256") == recorded["sha256"],
                "Shared corpus completion or operator rights record differs")
        complete_files = completion.get("files")
        audit_files = {item["filename"]: item for item in audit["files"]}
        fields = {"filename", "size", "sha256", "actual_sha256", "git_blob_sha1"}
        require(isinstance(complete_files, list) and
                len(complete_files) == len(hf_data.SHARDS) and
                {item["filename"] for item in complete_files} == {name for name, _ in hf_data.SHARDS} and
                all(set(item) == fields and all(item[key] == audit_files[item["filename"]].get(key) for key in fields)
                    for item in complete_files),
                "Shared corpus completion inventory differs from the transfer audit")
    return plan, audit


def audited_shards(roots):
    found = {}
    for root in roots:
        plan, audit = corpus_audits(root)
        rights = plan.get("rights")
        require(plan.get("repository") == hf_data.REPOSITORY and plan.get("revision") == hf_data.REVISION and
                plan.get("card_license") == "CC-BY-4.0" and plan.get("official_test_used", False) is False and
                isinstance(rights, dict) and rights == audit.get("rights") and
                isinstance(rights.get("sha256"), str) and hf_data.SHA256.fullmatch(rights["sha256"]),
                "Pinned source and matching rights provenance are required")
        for name, size in hf_data.SHARDS:
            receipts = [item for item in audit["files"] if item["filename"] == name]
            if not receipts:
                continue
            pins = [item for item in plan["shards"] if item["filename"] == name]
            require(len(pins) == len(receipts) == 1, "Each reviewed shard requires a unique audit record")
            pin, receipt = pins[0], receipts[0]
            expected = pin.get("sha256")
            require(pin["size"] == receipt["size"] == size and
                    isinstance(expected, str) and hf_data.SHA256.fullmatch(expected) and
                    expected == receipt["sha256"] == receipt["actual_sha256"],
                    "Download pin differs from the transfer audit")
            if name in found:
                require(found[name][1]["sha256"] == expected, "Conflicting audit hashes for the same shard")
                continue
            path = contained(root / "downloads" / name, root)
            require(path.is_file() and path.stat().st_size == size and hf_data.digest_file(path) == expected,
                    "Local fixed shard bytes differ from the recorded download")
            found[name] = ((name, path), {"filename": name, "bytes": size, "sha256": expected})
    require(len(found) == len(hf_data.SHARDS), "Both fixed downloaded shard records are required")
    ordered = [found[name] for name, _ in hf_data.SHARDS]
    return [item[0] for item in ordered], [item[1] for item in ordered]


def text_keys(texts):
    # Preserve letters and combining marks while matching punctuation/space variants.
    keys = set()
    for text in texts:
        if isinstance(text, str):
            value = "".join(c for c in unicodedata.normalize("NFC", text.casefold())
                            if unicodedata.category(c)[0] in "LMN")
            if value:
                keys.add(("text", value))
    return keys


def published_key(value):
    if isinstance(value, (int, str)) and not isinstance(value, bool) and str(value):
        return {("published_id", str(value))}
    return set()


def prior_exclusions(inventories, work):
    speakers, identities, receipts = set(), set(), []
    for path in inventories:
        path = contained(path, work)
        data = read_input(path, work)
        receipt = load_json(read_input(path.with_name("import.json"), work).decode())
        digest = hashlib.sha256(data).hexdigest()
        require(receipt.get("kind") == "ivr_hindi_import" and receipt.get("status") == "ready" and
                receipt.get("repository") == hf_data.REPOSITORY and receipt.get("revision") == hf_data.REVISION and
                receipt.get("input_sha256") == digest, "Prior inventory must match a ready import receipt")
        rows = [load_json(line) for line in data.decode().splitlines() if line.strip()]
        require(0 < len(rows) <= hf_data.MAX_ROWS and receipt["clips"] == len(rows),
                "Prior inventory count differs from the completed import")
        local_speakers = set()
        for row in rows:
            speaker = hf_data._speaker(row["source_speaker_id"])
            require(row.get("dataset_revision") == hf_data.REVISION and row.get("language") == "hi" and
                    isinstance(row.get("source"), str) and isinstance(row.get("text"), str) and
                    isinstance(row.get("source_text_fields"), dict) and
                    all(isinstance(row.get(key), str) and hf_data.SHA256.fullmatch(row[key])
                        for key in ("source_audio_sha256", "pcm_sha256")),
                    "Each prior clip requires reviewed publisher, text and audio identities")
            local_speakers.add(speaker)
            identities.update({("source", row["source"]), ("audio", row["source_audio_sha256"]),
                               ("pcm", row["pcm_sha256"])})
            identities.update(published_key(row.get("source_published_id")))
            identities.update(text_keys([row["text"], *row["source_text_fields"].values()]))
        require(receipt["speakers"] == len(local_speakers), "Prior speaker count differs from its receipt")
        speakers.update(local_speakers)
        receipts.append({"input_sha256": digest, "clips": len(rows), "speakers": len(local_speakers)})
    return speakers, identities, receipts


def select_pairs(shards, source, excluded_speakers, excluded_keys):
    counts, candidates, pending, completed = Counter(), Counter(), {}, set()
    used_keys = set(excluded_keys)
    pairs = []
    for name, path in shards:
        rows = {}
        for index, row in hf_data._metadata_rows(path, hf_data.MAX_ROWS - counts["scanned_rows"]):
            counts["scanned_rows"] += 1
            try:
                speaker = hf_data._speaker(row.get("speaker_id"))
                if speaker in excluded_speakers:
                    counts["prior_speaker_rows"] += 1
                    continue
                if speaker not in candidates and len(candidates) >= hf_data.MAX_CANDIDATE_SPEAKERS:
                    counts["candidate_cap_rows"] += 1
                    continue
                if candidates[speaker] >= hf_data.MAX_CANDIDATES_PER_SPEAKER:
                    counts["speaker_cap_rows"] += 1
                    continue
                duration = row.get("duration")
                if (type(duration) not in (int, float) or not math.isfinite(duration) or
                        not 1 <= duration <= MAX_SECONDS or
                        ("lang" in row and str(row["lang"]).strip().casefold() not in {"hi", "hindi"})):
                    raise DataError("Invalid voice row metadata")
                texts = [row.get(field) for field in ("normalized", "text", "verbatim")]
                first = next((value for value in texts if isinstance(value, str) and value.strip()), "")
                text = normalize_hindi(first, max_chars=240)
                batch_synth.canonical_text(source, "hi", text, stock_segments.MODE)
                locator = f"{hf_data.REPOSITORY}@{hf_data.REVISION}:{name}:{index}"
                keys = {("source", locator)} | published_key(row.get("id")) | text_keys([text, *texts])
            except (DataError, hf_data.ImportBlocked, synth.SynthesisError):
                counts["metadata_policy_rows"] += 1
                continue
            if keys & used_keys:
                counts["prior_or_duplicate_text_or_source_rows"] += 1
                continue
            if speaker in completed:
                continue
            candidates[speaker] += 1
            rows[index] = (speaker, text, locator, keys)
        for index, cell in hf_data._audio_cells(path, rows):
            speaker, text, locator, keys = rows[index]
            if speaker in completed or keys & used_keys:
                continue
            try:
                wav, info = hf_data.decode_audio(cell, max_duration_seconds=MAX_SECONDS)
            except hf_data.ImportBlocked:
                counts["audio_policy_rows"] += 1
                continue
            keys |= {("audio", info["source_audio_sha256"]), ("pcm", info["pcm_sha256"])}
            if keys & used_keys:
                counts["prior_or_duplicate_audio_rows"] += 1
                continue
            used_keys.update(keys)
            clip = (text, locator, wav, info)
            if speaker not in pending:
                pending[speaker] = clip
                continue
            pairs.append((speaker, pending.pop(speaker), clip))
            completed.add(speaker)
            if len(pairs) == MAX_SPEAKERS:
                break
        if len(pairs) == MAX_SPEAKERS or counts["scanned_rows"] == hf_data.MAX_ROWS:
            break
    return pairs, dict(counts)


def write_batch(output, work, source, pairs, receipt):
    output.mkdir(mode=0o700)
    marker = output / "INCOMPLETE"
    marker.touch(exist_ok=False)
    cases, references, mappings = [], [], []
    for number, (speaker, prompt, target) in enumerate(pairs, 1):
        uid = f"hi-new-{number:02d}"
        row = {"id": uid, "source_speaker_id": speaker}
        for role, (text, locator, wav, info) in (("prompt", prompt), ("target", target)):
            path = output / f"{uid}-{role}.wav"
            with path.open("xb") as stream:
                stream.write(wav)
            row[role] = {"source": locator, "source_audio_sha256": info["source_audio_sha256"],
                         "pcm_sha256": info["pcm_sha256"], "wav_sha256": hashlib.sha256(wav).hexdigest()}
        text = target[0]
        prompt_path = str((output / f"{uid}-prompt.wav").relative_to(work))
        target_path = str((output / f"{uid}-target.wav").relative_to(work))
        shared = {"id": uid, "language": "hi", "text": text, "text_sha256": batch_synth.text_hash(text)}
        cases.append({**shared, "prompt": prompt_path, "max_mel_tokens": MEL_CAP})
        references.append({**shared, "condition": "reference", "audio": target_path})
        mappings.append(row)
    for name, data in (("cases.json", cases), ("references.json", references)):
        path = output / name
        hf_data.write_json(path, data)
        receipt[name] = hf_data.digest_file(path)
    batch_synth.load_cases(output / "cases.json", work, source, stock_segments.MODE)
    hf_data.write_json(output / "receipt.json", {**receipt, "private_pairs": mappings})
    marker.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, action="append", default=[],
                        help="Existing authorized corpus import root inside WORK; repeat for distinct audit roots")
    parser.add_argument("--readonly-corpus-root", type=Path,
                        help="Explicit existing authorized shared corpus root, read separately from WORK")
    parser.add_argument("--prior-inventory", type=Path, action="append", required=True,
                        help="Repeat for each previously exposed input.jsonl paired with its ready import.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    stage = "path_preflight"
    try:
        work = args.workspace_root.resolve(strict=True)
        require(work.is_dir() and work.parent != work, "A dedicated WORK directory is required")
        output = args.output_dir.absolute()
        require(output.is_relative_to(work) and output != work and
                output.parent.resolve(strict=True) == output.parent and not output.exists() and not output.is_symlink(),
                "Output must be a fresh canonical directory inside WORK")
        source = synth.verify_source(contained(args.official_repo, work))
        require(not output.is_relative_to(source), "Output cannot be in the official source checkout")
        downloads = [contained(path, work) for path in args.download_root]
        if args.readonly_corpus_root is not None:
            shared = canonical(args.readonly_corpus_root)
            require(shared.is_dir() and shared.parent != shared and
                    not work.is_relative_to(shared) and not shared.is_relative_to(work),
                    "Shared corpus and WORK must be disjoint canonical roots")
            downloads.append(shared)
        inventories = [contained(path, work) for path in args.prior_inventory]
        require(1 <= len(set(inventories)) == len(inventories) <= 16 and
                1 <= len(set(downloads)) == len(downloads) <= 4 and
                not any(output.is_relative_to(path) for path in downloads) and
                not any(output.is_relative_to(path.parent) for path in inventories),
                "Explicit distinct prior inventories and independent output are required")
        stage = "prior_inventories"
        speakers, keys, priors = prior_exclusions(inventories, work)
        stage = "shard_audit"
        shards, shard_receipts = audited_shards(downloads)
        stage = "bounded_selection"
        pairs, stats = select_pairs(shards, source, speakers, keys)
        require(pairs, "No complete speaker-disjoint pair in the fixed scan")
        os.umask(0o077)
        receipt = {"schema_version": 1, "kind": "private_ivr_hindi_holdout", "repository": hf_data.REPOSITORY,
                   "revision": hf_data.REVISION, "shards": shard_receipts,
                   "prior_inventories": priors, "excluded_publisher_speakers": len(speakers), "scan": stats,
                   "selected_cases": len(pairs), "selected_publisher_speakers": len(pairs),
                   "speaker_scope": "Zero publisher ID overlap with the supplied prior inventories",
                   "selection": "First complete pair per eligible speaker in fixed shard/row order, earlier clip as prompt",
                   "official_test_used": False, "max_audio_seconds": MAX_SECONDS,
                   "transcript_alignment_manually_verified": False, "quality_gate_passed": False}
        stage = "private_output_validation"
        write_batch(output, work, source, pairs, receipt)
        print(json.dumps({"status": "ready_private", "cases": len(pairs), "speakers": len(pairs),
                          "excluded_prior_speakers": len(speakers), "scan": stats}))
        return 0
    except (ValueError, OSError, KeyError, AttributeError, TypeError, ImportError) as exc:
        # File/parser exceptions may include sensitive paths or corpus text.
        print(json.dumps({"status": "blocked_or_incomplete", "stage": stage,
                          "failure_type": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
