"""Create an auditable changed-file plan for Project RAG incremental indexing."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.crawler import diff_inventories, load_inventory, write_diff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diff Project RAG inventories by SHA-256.")
    parser.add_argument("--previous", type=Path, required=True, help="Previous project files.jsonl.")
    parser.add_argument("--current", type=Path, required=True, help="Current project files.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="Changed-file plan JSONL output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        diff = diff_inventories(load_inventory(args.previous), load_inventory(args.current))
        write_diff(args.output, diff)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Plan: {args.output}")
    print(f"Added: {len(diff.added)}")
    print(f"Modified: {len(diff.modified)}")
    print(f"Deleted: {len(diff.deleted)}")
    print(f"Unchanged: {diff.unchanged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
