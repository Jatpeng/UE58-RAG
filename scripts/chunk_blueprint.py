"""Create semantic chunks from normalized Blueprint documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.chunker import BlueprintChunker, load_blueprint_chunker_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk Blueprint summaries, graphs, functions, variables, and components.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--chunker-config", type=Path, default=Path("config/blueprint_chunker.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_blueprint_chunker_config(args.ue_config, args.chunker_config)
        if args.input:
            config.input_path = args.input
        if args.output:
            config.output_path = args.output
        documents, chunks, output = BlueprintChunker(config).chunk_corpus()
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Output: {output}")
    print(f"Documents: {documents}")
    print(f"Chunks: {chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
