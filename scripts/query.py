"""Query the UE lexical/symbol index."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from ue_rag.retrieval import LexicalIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the Unreal Engine developer RAG.")
    parser.add_argument("query", help="Symbol or natural-language query.")
    parser.add_argument("--index", type=Path, default=Path("data/index/lexical.sqlite3"))
    parser.add_argument("--mode", choices=("symbol", "lexical", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--engine-version")
    parser.add_argument("--source-type")
    parser.add_argument("--module")
    parser.add_argument("--plugin")
    parser.add_argument("--class", dest="class_name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit <= 0:
        print("Error: limit must be greater than zero")
        return 2
    filters = {
        key: value
        for key, value in {
            "engine_version": args.engine_version,
            "source_type": args.source_type,
            "module": args.module,
            "plugin": args.plugin,
            "class_name": args.class_name,
        }.items()
        if value is not None
    }
    try:
        with LexicalIndex(args.index) as index:
            if args.mode == "symbol":
                results = index.search_symbol(args.query, limit=args.limit, filters=filters)
            elif args.mode == "lexical":
                results = index.search_lexical(args.query, limit=args.limit, filters=filters)
            else:
                results = index.search(args.query, limit=args.limit, filters=filters)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Error: {error}")
        return 2
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.metadata.get('symbol') or result.chunk_id}")
        print(f"   {result.metadata.get('file_path') or ''}")
        print(f"   score: {result.score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
