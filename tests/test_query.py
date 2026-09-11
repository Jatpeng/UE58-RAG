"""Tests for the unified query service."""

from __future__ import annotations

import pytest

from ue_rag.retrieval.query import UnifiedQueryService
from ue_rag.schema import RetrievalResult, SourceType


def result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        engine_version="5.8",
        source_type=SourceType.ENGINE_SOURCE,
        content=chunk_id,
        score=1.0,
        metadata={"symbol": chunk_id},
    )


class FakeLexical:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def search_symbol(self, query: str, *, limit: int, filters: dict | None = None):
        self.calls.append(("symbol", query, limit))
        return [result("symbol")]

    def search(self, query: str, *, limit: int, filters: dict | None = None):
        self.calls.append(("lexical", query, limit))
        return [result("lexical")]


class FakeHybrid:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, int]] = []

    def retrieve(self, query: str, **kwargs):
        self.calls.append((query, kwargs.get("query_vector"), kwargs.get("fusion_top_k")))
        return [result("hybrid")]


class FakeReranker:
    def rerank(self, query: str, documents, *, top_k: int):
        return [documents[0].model_copy(update={"score": 9.0})][:top_k]


def test_symbol_and_lexical_modes_dispatch() -> None:
    lexical = FakeLexical()
    service = UnifiedQueryService(lexical_index=lexical)

    assert service.search("Tick", mode="symbol", limit=3)[0].chunk_id == "symbol"
    assert service.search("movement", mode="lexical", limit=4)[0].chunk_id == "lexical"
    assert lexical.calls == [("symbol", "Tick", 3), ("lexical", "movement", 4)]


def test_hybrid_mode_passes_vector_and_limit() -> None:
    hybrid = FakeHybrid()
    service = UnifiedQueryService(lexical_index=FakeLexical(), hybrid_retriever=hybrid)

    results = service.search("movement", mode="hybrid", limit=5, query_vector=[1, 0])

    assert results[0].chunk_id == "hybrid"
    assert hybrid.calls == [("movement", [1, 0], 5)]


def test_rerank_mode_composes_pipeline() -> None:
    hybrid = FakeHybrid()
    service = UnifiedQueryService(
        lexical_index=FakeLexical(), hybrid_retriever=hybrid, reranker=FakeReranker()
    )

    results = service.search("movement", mode="rerank", limit=1, query_vector=[1, 0])

    assert results[0].score == 9.0


def test_hybrid_and_rerank_require_dependencies() -> None:
    service = UnifiedQueryService(lexical_index=FakeLexical())

    with pytest.raises(ValueError, match="hybrid_retriever"):
        service.search("movement", mode="hybrid", query_vector=[1, 0])
    with pytest.raises(ValueError, match="hybrid_retriever"):
        service.search("movement", mode="rerank", query_vector=[1, 0])


def test_rerank_requires_reranker() -> None:
    service = UnifiedQueryService(lexical_index=FakeLexical(), hybrid_retriever=FakeHybrid())

    with pytest.raises(ValueError, match="reranker"):
        service.search("movement", mode="rerank", query_vector=[1, 0])


def test_service_validates_mode_and_limit() -> None:
    service = UnifiedQueryService(lexical_index=FakeLexical())

    with pytest.raises(ValueError, match="mode must be"):
        service.search("query", mode="bad")
    with pytest.raises(ValueError, match="greater than zero"):
        service.search("query", limit=0)
