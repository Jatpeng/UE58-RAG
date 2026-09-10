"""Create semantic chunks from parsed Unreal Engine C++ symbols."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.chunker import CPPSemanticChunker, load_cpp_chunker_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk parsed UE C++ symbols semantically.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--chunker-config", type=Path, default=Path("config/cpp_chunker.yaml"))
    parser.add_argument("--input", type=Path, help="Override parsed C++ documents JSONL.")
    parser.add_argument("--output", type=Path, help="Override C++ chunks JSONL.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_cpp_chunker_config(args.ue_config, args.chunker_config)
        if args.input:
            config.input_path = args.input
        if args.output:
            config.output_path = args.output
        summary = CPPSemanticChunker(config).chunk_corpus()
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Output: {summary.output_path}")
    print(f"Documents: {summary.documents}")
    print(f"Chunks: {summary.chunks}")
    print(f"Property groups: {summary.property_groups}")
    print(f"Oversized chunks: {summary.oversized_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
