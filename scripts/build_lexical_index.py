"""Build the persistent UE lexical/symbol index."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.retrieval import LexicalIndex, load_lexical_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the UE5.8 lexical and symbol index.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    parser.add_argument("--input", type=Path, help="Override C++ chunk JSONL input.")
    parser.add_argument("--index", type=Path, help="Override SQLite index path.")
    parser.add_argument("--batch-size", type=int, help="Override indexing batch size.")
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
        with LexicalIndex(config.index_path) as index:
            summary = index.index_jsonl(config.input_path, batch_size=config.batch_size)
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
