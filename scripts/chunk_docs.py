"""Create heading-aware semantic chunks from parsed UE documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.chunker import DocumentationChunker, load_docs_chunker_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk parsed UE documentation by heading hierarchy."
    )
    parser.add_argument(
        "--ue-config",
        type=Path,
        default=Path("config/ue58.yaml"),
        help="UE configuration containing version and data paths.",
    )
    parser.add_argument(
        "--chunker-config",
        type=Path,
        default=Path("config/docs_chunker.yaml"),
        help="Documentation chunk-size and overlap configuration.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Override data/parsed/docs/documents.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override data/chunks/docs/chunks.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_docs_chunker_config(args.ue_config, args.chunker_config)
        if args.input:
            config.input_path = args.input
        if args.output:
            config.output_path = args.output
        summary = DocumentationChunker(config).chunk_corpus()
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Output: {summary.output_path}")
    print(f"Documents: {summary.documents}")
    print(f"Chunks: {summary.chunks}")
    print(f"Oversized atomic code chunks: {summary.oversized_atomic_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
