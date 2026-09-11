"""Run the UE retrieval tools as a Model Context Protocol server."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.embedding import QwenEmbeddingProvider, load_embedding_config
from ue_rag.index import QdrantVectorStore, load_qdrant_config
from ue_rag.mcp import MCPTools, create_mcp_server
from ue_rag.retrieval import (
    HybridRetriever,
    LexicalIndex,
    UnifiedQueryService,
    load_lexical_config,
    load_retrieval_config,
)
from ue_rag.reranker import QwenReranker, load_rerank_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve UE RAG search tools over MCP.")
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("config/retrieval.yaml"))
    parser.add_argument("--qdrant-config", type=Path, default=Path("config/qdrant.yaml"))
    parser.add_argument("--embedding-config", type=Path, default=Path("config/embedding.yaml"))
    parser.add_argument("--reranker-config", type=Path, default=Path("config/reranker.yaml"))
    parser.add_argument("--enable-dense", action="store_true", help="Enable hybrid mode through Qdrant.")
    parser.add_argument("--enable-rerank", action="store_true", help="Enable rerank mode; implies dense mode.")
    parser.add_argument("--max-limit", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lexical = None
    embedding = None
    store = None
    try:
        lexical_config = load_lexical_config(args.ue_config, args.lexical_config)
        lexical = LexicalIndex(lexical_config.index_path)
        hybrid = None
        reranker = None
        if args.enable_dense or args.enable_rerank:
            retrieval_config = load_retrieval_config(args.ue_config, args.retrieval_config)
            store = QdrantVectorStore.from_config(load_qdrant_config(args.qdrant_config))
            embedding = QwenEmbeddingProvider(load_embedding_config(args.ue_config, args.embedding_config))
            hybrid = HybridRetriever(store, lexical, embedding_provider=embedding, config=retrieval_config)
        if args.enable_rerank:
            reranker = QwenReranker(load_rerank_config(args.ue_config, args.reranker_config))
        service = UnifiedQueryService(
            lexical_index=lexical,
            hybrid_retriever=hybrid,
            reranker=reranker,
            rerank_top_k=load_rerank_config(args.ue_config, args.reranker_config).top_k if args.enable_rerank else 8,
        )
        server = create_mcp_server(MCPTools(service, max_limit=args.max_limit))
        server.run(transport=args.transport)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2
    finally:
        if embedding is not None:
            embedding.close()
        if lexical is not None:
            lexical.close()
        if store is not None and getattr(store, "client", None) is not None:
            store.client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
