"""Build the persistent UE lexical/symbol index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ue_rag.retrieval import LexicalIndex, load_lexical_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the UE5.8 lexical and symbol index.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    parser.add_argument("--input", type=Path, help="Override C++ chunk JSONL input.")
    parser.add_argument("--index", type=Path, help="Override SQLite index path.")
    parser.add_argument("--batch-size", type=int, help="Override indexing batch size.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Build a fresh index beside the target and replace it atomically on success.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print a progress line after approximately this many chunks (0 disables progress).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_lexical_config(args.ue_config, args.lexical_config)
        if args.input:
            config.input_path = args.input
        if args.index:
            config.index_path = args.index
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise ValueError("batch-size must be greater than zero")
            config.batch_size = args.batch_size
        if args.progress_every < 0:
            raise ValueError("progress-every must not be negative")
        target = config.index_path
        build_path = target.with_name(target.name + ".building") if args.rebuild else target
        if args.rebuild and build_path.exists():
            build_path.unlink()
        with LexicalIndex(build_path, bulk_build=args.rebuild) as index:
            summary = index.index_jsonl(
                config.input_path,
                batch_size=config.batch_size,
                bulk_build=args.rebuild,
                progress_every=args.progress_every,
            )
        if args.rebuild:
            os.replace(build_path, target)
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Index: {config.index_path}")
    print(f"Total: {summary.total}")
    print(f"Added: {summary.added}")
    print(f"Updated: {summary.updated}")
    print(f"Skipped: {summary.skipped}")
    print(f"Failed: {summary.failed}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
