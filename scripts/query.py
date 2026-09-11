"""Unified UE query CLI for symbol, lexical, hybrid, and reranked search."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ue_rag.embedding import QwenEmbeddingProvider, load_embedding_config
from ue_rag.index import QdrantVectorStore, load_qdrant_config
from ue_rag.retrieval import (
    HybridRetriever,
    LexicalIndex,
    UnifiedQueryService,
    load_lexical_config,
    load_retrieval_config,
)
from ue_rag.reranker import QwenReranker, load_rerank_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the Unreal Engine developer RAG.")
    parser.add_argument("query", help="Symbol or natural-language query.")
    parser.add_argument("--index", type=Path, help="SQLite lexical index path.")
    parser.add_argument("--mode", choices=("symbol", "lexical", "hybrid", "rerank"), default="lexical")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--query-vector", help="Optional JSON vector for hybrid/rerank modes.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--qdrant-config", type=Path, default=Path("config/qdrant.yaml"))
    parser.add_argument("--embedding-config", type=Path, default=Path("config/embedding.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("config/retrieval.yaml"))
    parser.add_argument("--reranker-config", type=Path, default=Path("config/reranker.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON results.")
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
            "class": args.class_name,
        }.items()
        if value is not None
    }
    lexical = None
    provider = None
    store = None
    try:
        lexical_config = load_lexical_config(args.ue_config, args.lexical_config)
        lexical = LexicalIndex(args.index or lexical_config.index_path)
        hybrid = None
        reranker = None
        query_vector = json.loads(args.query_vector) if args.query_vector else None
        if args.mode in {"hybrid", "rerank"}:
            retrieval_config = load_retrieval_config(args.ue_config, args.retrieval_config)
            qdrant_config = load_qdrant_config(args.qdrant_config)
            store = QdrantVectorStore.from_config(qdrant_config)
            if query_vector is None:
                provider = QwenEmbeddingProvider(load_embedding_config(args.ue_config, args.embedding_config))
            hybrid = HybridRetriever(store, lexical, embedding_provider=provider, config=retrieval_config)
            if args.mode == "rerank":
                reranker = QwenReranker(load_rerank_config(args.ue_config, args.reranker_config))
        service = UnifiedQueryService(
            lexical_index=lexical,
            hybrid_retriever=hybrid,
            reranker=reranker,
            rerank_top_k=args.limit,
        )
        results = service.search(args.query, mode=args.mode, limit=args.limit, query_vector=query_vector, filters=filters)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 2
    finally:
        if provider is not None:
            provider.close()
        if lexical is not None:
            lexical.close()
        if store is not None and getattr(store, "client", None) is not None:
            store.client.close()
    if args.as_json:
        print(json.dumps([result.model_dump(mode="json") for result in results], ensure_ascii=False, indent=2))
        return 0
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.metadata.get('symbol') or result.chunk_id}")
        print(f"   {result.metadata.get('file_path') or ''}")
        print(f"   score: {result.score:.6f}")
        if result.dense_rank is not None or result.sparse_rank is not None:
            print(f"   dense_rank: {result.dense_rank}, sparse_rank: {result.sparse_rank}")
        preview = " ".join(result.content.split())
        if len(preview) > 240:
            preview = preview[:237] + "..."
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
