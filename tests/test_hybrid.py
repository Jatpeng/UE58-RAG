"""Tests for dense/lexical Reciprocal Rank Fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ue_rag.index import QdrantConfig, QdrantVectorStore
from ue_rag.retrieval import HybridRetriever, LexicalIndex, RetrievalConfig, load_retrieval_config
from ue_rag.schema import RetrievalResult, SourceScope, SourceType, UEChunk


def make_chunk(chunk_id: str, symbol: str, content: str, *, module: str = "Engine") -> UEChunk:
    return UEChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=0,
        engine_version="5.8",
        source_scope=SourceScope.GLOBAL,
        source_type=SourceType.ENGINE_SOURCE,
        content=content,
        title=symbol,
        module=module,
        plugin="Engine",
        file_path="Engine/Source/Runtime/Engine.cpp",
        symbol=symbol,
        symbol_type="method",
        class_name="UCharacterMovementComponent",
        function_name=symbol.rsplit("::", 1)[-1],
        metadata={"chunk_kind": "function"},
    )


def result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        engine_version="5.8",
        source_type=SourceType.ENGINE_SOURCE,
        content=chunk_id,
        score=score,
        metadata={"symbol": chunk_id},
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray([1.0, 0.0], dtype=np.float32)


def make_config() -> RetrievalConfig:
    return RetrievalConfig(
        engine_version="5.8",
        dense_top_k=2,
        sparse_top_k=2,
        fusion_top_k=3,
        rerank_top_k=2,
        rrf_k=60,
    )


def test_retrieval_config_loads_rrf_parameter() -> None:
    config = load_retrieval_config()

    assert config.engine_version == "5.8"
    assert config.rrf_k == 60
    assert config.dense_top_k == 30


def test_retrieval_config_rejects_impossible_fusion_limit() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        RetrievalConfig(
            engine_version="5.8",
            dense_top_k=1,
            sparse_top_k=1,
            fusion_top_k=3,
            rerank_top_k=1,
            rrf_k=60,
        )


def test_rrf_fuse_combines_duplicate_results_and_preserves_ranks() -> None:
    fused = HybridRetriever.fuse(
        [result("dense-first", 0.9), result("shared", 0.8)],
        [result("shared", 0.99), result("sparse-second", 0.7)],
        limit=3,
        rrf_k=60,
    )

    assert fused[0].chunk_id == "shared"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    assert fused[0].fusion_score == fused[0].score
    assert fused[0].metadata["retrieval"] == "hybrid"
    assert [item.chunk_id for item in fused] == ["shared", "dense-first", "sparse-second"]


def test_rrf_fuse_validates_limits() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        HybridRetriever.fuse([], [], limit=0)
    with pytest.raises(ValueError, match="greater than zero"):
        HybridRetriever.fuse([], [], rrf_k=0)


def test_hybrid_retrieve_runs_qdrant_and_lexical_paths(tmp_path: Path) -> None:
    chunks = [
        make_chunk("chunk-1", "UCharacterMovementComponent::PerformMovement", "perform movement"),
        make_chunk("chunk-2", "UCharacterMovementComponent::PhysWalking", "walking movement"),
    ]
    dense = QdrantVectorStore(QdrantConfig(collection="hybrid", path=tmp_path / "qdrant", batch_size=2))
    dense.create_collection(2)
    dense.upsert(chunks, [[1, 0], [0.8, 0.2]])
    lexical = LexicalIndex(tmp_path / "lexical.sqlite3")
    lexical.upsert(chunks)
    embedder = FakeEmbedder()
    retriever = HybridRetriever(dense, lexical, embedding_provider=embedder, config=make_config())

    results = retriever.retrieve("movement", filters={"module": "Engine"})

    assert embedder.queries == ["movement"]
    assert len(results) == 2
    assert all(item.dense_rank is not None for item in results)
    assert all(item.sparse_rank is not None for item in results)
    assert all(item.fusion_score == item.score for item in results)
    lexical.close()


def test_hybrid_accepts_explicit_query_vector_without_embedder(tmp_path: Path) -> None:
    chunk = make_chunk("chunk-1", "AHero::Tick", "tick")
    dense = QdrantVectorStore(QdrantConfig(collection="hybrid", path=tmp_path / "qdrant", batch_size=2))
    dense.create_collection(2)
    dense.upsert([chunk], [[1, 0]])
    lexical = LexicalIndex(tmp_path / "lexical.sqlite3")
    lexical.upsert([chunk])
    retriever = HybridRetriever(dense, lexical)

    results = retriever.retrieve("tick", query_vector=[1, 0], fusion_top_k=1)

    assert len(results) == 1
    lexical.close()


def test_hybrid_requires_query_vector_or_embedder(tmp_path: Path) -> None:
    lexical = LexicalIndex(tmp_path / "lexical.sqlite3")
    dense = QdrantVectorStore(QdrantConfig(collection="hybrid", path=tmp_path / "qdrant", batch_size=2))
    retriever = HybridRetriever(dense, lexical)

    with pytest.raises(ValueError, match="query_vector or embedding_provider"):
        retriever.retrieve("tick")
    lexical.close()
