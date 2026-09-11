"""Unified query service spanning symbol, lexical, hybrid, and reranked modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ue_rag.retrieval.hybrid import HybridRetriever
from ue_rag.reranker import RAGQueryPipeline, Reranker
from ue_rag.schema import RetrievalResult


class UnifiedQueryService:
    """Dispatch one query API to the available retrieval stages."""

    def __init__(
        self,
        *,
        lexical_index: Any,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        rerank_top_k: int = 8,
    ) -> None:
        self.lexical_index = lexical_index
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def search(
        self,
        query: str,
        *,
        mode: str = "lexical",
        limit: int = 10,
        query_vector: Sequence[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run a unified query in symbol, lexical, hybrid, or rerank mode."""

        if mode not in {"symbol", "lexical", "hybrid", "rerank"}:
            raise ValueError("mode must be symbol, lexical, hybrid, or rerank")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if mode == "symbol":
            return self.lexical_index.search_symbol(query, limit=limit, filters=filters)
        if mode == "lexical":
            return self.lexical_index.search(query, limit=limit, filters=filters)
        if self.hybrid_retriever is None:
            raise ValueError("hybrid_retriever is required for hybrid/rerank mode")
        if mode == "hybrid":
            return self.hybrid_retriever.retrieve(
                query, query_vector=query_vector, fusion_top_k=limit, filters=filters
            )
        if self.reranker is None:
            raise ValueError("reranker is required for rerank mode")
        pipeline = RAGQueryPipeline(
            self.hybrid_retriever,
            self.reranker,
            rerank_top_k=limit if limit else self.rerank_top_k,
        )
        return pipeline.query(query, query_vector=query_vector, filters=filters)
