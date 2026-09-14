"""Build or update the UE5.8 Qdrant vector collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.index import QdrantVectorStore, ingest_jsonl, load_qdrant_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the UE5.8 Qdrant vector index.")
    parser.add_argument("--input", type=Path, required=True, help="UEChunk JSONL input.")
    parser.add_argument("--vectors", type=Path, required=True, help="Embedding .npy matrix.")
    parser.add_argument("--ids", type=Path, help="Embedding ID sidecar JSONL.")
    parser.add_argument("--qdrant-config", type=Path, default=Path("config/qdrant.yaml"))
    parser.add_argument("--collection", help="Override collection name.")
    parser.add_argument("--batch-size", type=int, help="Override ingest batch size.")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection before ingest.")
    parser.add_argument("--in-memory", action="store_true", help="Use an in-memory store for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress after approximately this many chunks.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_qdrant_config(args.qdrant_config)
        if args.collection:
            config.collection = args.collection
        if args.batch_size is not None:
            if args.batch_size <= 0:
                raise ValueError("batch-size must be greater than zero")
            config.batch_size = args.batch_size
        if args.progress_every < 0:
            raise ValueError("progress-every must not be negative")
        store = QdrantVectorStore(config) if args.in_memory else QdrantVectorStore.from_config(config)
        if args.recreate:
            import numpy as np
            size = int(np.load(args.vectors, mmap_mode="r").shape[1])
            store.create_collection(size, recreate=True)
        summary = ingest_jsonl(
            store,
            args.input,
            args.vectors,
            ids_path=args.ids,
            batch_size=args.batch_size,
            fast=args.recreate,
            progress_every=args.progress_every,
        )
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Collection: {config.collection}")
    print(f"Total: {summary.total}")
    print(f"Added: {summary.added}")
    print(f"Updated: {summary.updated}")
    print(f"Skipped: {summary.skipped}")
    print(f"Failed: {summary.failed}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
