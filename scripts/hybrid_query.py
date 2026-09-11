"""Run dense plus lexical hybrid retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ue_rag.embedding import QwenEmbeddingProvider, load_embedding_config
from ue_rag.index import QdrantVectorStore, load_qdrant_config
from ue_rag.retrieval import HybridRetriever, LexicalIndex, load_lexical_config, load_retrieval_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the UE5.8 hybrid retriever.")
    parser.add_argument("query", help="Natural-language or symbol query.")
    parser.add_argument("--query-vector", help="Optional JSON vector; skips embedding model loading.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("config/retrieval.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    parser.add_argument("--embedding-config", type=Path, default=Path("config/embedding.yaml"))
    parser.add_argument("--qdrant-config", type=Path, default=Path("config/qdrant.yaml"))
    parser.add_argument("--limit", type=int, help="Override fusion_top_k.")
    parser.add_argument("--engine-version")
    parser.add_argument("--source-type")
    parser.add_argument("--module")
    parser.add_argument("--plugin")
    parser.add_argument("--class", dest="class_name")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provider = None
    lexical = None
    store = None
    try:
        retrieval_config = load_retrieval_config(args.ue_config, args.retrieval_config)
        lexical_config = load_lexical_config(args.ue_config, args.lexical_config)
        qdrant_config = load_qdrant_config(args.qdrant_config)
        lexical = LexicalIndex(lexical_config.index_path)
        store = QdrantVectorStore.from_config(qdrant_config)
        query_vector = json.loads(args.query_vector) if args.query_vector else None
        if query_vector is None:
            provider = QwenEmbeddingProvider(load_embedding_config(args.ue_config, args.embedding_config))
        retriever = HybridRetriever(store, lexical, embedding_provider=provider, config=retrieval_config)
        results = retriever.retrieve(args.query, query_vector=query_vector, fusion_top_k=args.limit)
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 2
    finally:
        if provider is not None:
            provider.close()
        if lexical is not None:
            lexical.close()
        if store is not None and getattr(store, "client", None) is not None:
            store.client.close()
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.metadata.get('symbol') or result.chunk_id}")
        print(f"   {result.metadata.get('file_path') or ''}")
        print(f"   score: {result.score:.6f}")
        print(f"   dense_rank: {result.dense_rank}, sparse_rank: {result.sparse_rank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
