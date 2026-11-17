"""CPU-only data CLI. Error returns nonzero; no model import, download, or shell."""
import argparse
import json
import sys

from .data import DEFAULT_SETTINGS, build_pairs, prepare, validate
from .text import DataError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare", help="validate raw JSONL, then create split manifests in a fresh directory")
    build.add_argument("input")
    build.add_argument("--audio-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--val-ratio", type=float, default=.1)
    build.add_argument("--test-ratio", type=float, default=.1)
    build.add_argument("--sample-rate", type=int, default=DEFAULT_SETTINGS["sample_rate"])
    build.add_argument("--min-seconds", type=float, default=DEFAULT_SETTINGS["min_seconds"])
    build.add_argument("--max-seconds", type=float, default=DEFAULT_SETTINGS["max_seconds"])
    build.add_argument("--min-chars", type=int, default=DEFAULT_SETTINGS["min_chars"])
    build.add_argument("--max-chars", type=int, default=DEFAULT_SETTINGS["max_chars"])
    build.add_argument("--allow-as-spoken-numbers", action="store_true",
                       help="also requires numbers_as_spoken=true on every row containing digits/currency")
    check = sub.add_parser("validate", help="recompute all prepared data checks and split/prompt isolation")
    check.add_argument("directory")
    check.add_argument("--audio-root", required=True)
    pairs = sub.add_parser("pairs", help="pair complete backend features with distinct same-speaker prompts")
    pairs.add_argument("feature_manifest")
    pairs.add_argument("--prepared", required=True)
    pairs.add_argument("--audio-root", required=True)
    pairs.add_argument("--split", required=True, choices=("train", "val", "test"))
    pairs.add_argument("--feature-root", required=True)
    pairs.add_argument("--output", required=True)
    pairs.add_argument("--max-text-tokens", type=int, required=True)
    pairs.add_argument("--max-code-tokens", type=int, required=True)
    pairs.add_argument("--max-prompt-code-tokens", type=int, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            result = prepare(args.pop("input"), **args)
            result = {"status": "prepared", "schema_version": result["schema_version"], "counts": result["counts"]}
        elif command == "validate":
            result = validate(**args)
        else:
            result = build_pairs(**args)
    except (DataError, OSError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
