"""Embed JSONL documents or chunks into a NumPy artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.embedding import QwenEmbeddingProvider, embed_jsonl, load_embedding_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create embeddings for UE JSONL records.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--embedding-config", type=Path, default=Path("config/embedding.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Input UEDocument or UEChunk JSONL.")
    parser.add_argument("--output", type=Path, help="Output .npy path; defaults to embedding config.")
    parser.add_argument("--batch-size", type=int, help="Override model batch size.")
    parser.add_argument("--max-length", type=int, help="Override model token limit.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress after approximately this many records.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provider = None
    try:
        config = load_embedding_config(args.ue_config, args.embedding_config)
        if args.output:
            config.output_path = args.output
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise ValueError("batch-size must be greater than zero")
            config.batch_size = args.batch_size
        if args.max_length is not None:
            if args.max_length <= 0:
                raise ValueError("max-length must be greater than zero")
            config.max_length = args.max_length
        if args.progress_every < 0:
            raise ValueError("progress-every must not be negative")
        provider = QwenEmbeddingProvider(config)
        summary = embed_jsonl(args.input, config.output_path, provider, progress_every=args.progress_every)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2
    finally:
        if provider is not None:
            provider.close()
    print(f"Output: {summary.output_path}")
    print(f"IDs: {summary.ids_path}")
    print(f"Manifest: {summary.manifest_path}")
    print(f"Records: {summary.records}")
    print(f"Dimension: {summary.dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
