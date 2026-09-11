"""Tests for reranker abstraction and retrieval-to-rerank pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ue_rag.reranker import QwenReranker, RAGQueryPipeline, RerankConfig, load_rerank_config
from ue_rag.schema import RetrievalResult, SourceType


def make_config(tmp_path: Path, **overrides: object) -> RerankConfig:
    values: dict[str, object] = {
        "engine_version": "5.8",
        "model": "fake-reranker",
        "device": "cpu",
        "batch_size": 2,
        "top_k": 2,
        "max_length": 128,
    }
    values.update(overrides)
    return RerankConfig(**values)


def make_result(chunk_id: str, content: str, score: float = 0.1) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        engine_version="5.8",
        source_type=SourceType.ENGINE_SOURCE,
        content=content,
        score=score,
        dense_rank=1,
        sparse_rank=2,
        metadata={"dense_rank": 1, "sparse_rank": 2},
    )


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]], **_: object) -> np.ndarray:
        self.calls.append(pairs)
        return np.asarray([float(len(pair[1])) for pair in pairs], dtype=np.float32)


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query: str, **_: object) -> list[RetrievalResult]:
        self.queries.append(query)
        return self.results


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, documents: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        self.calls.append((query, len(documents)))
        return list(documents)[:top_k]


def test_config_loads_qwen_model_and_version() -> None:
    config = load_rerank_config()

    assert config.engine_version == "5.8"
    assert config.model == "Qwen/Qwen3-Reranker-0.6B"
    assert config.top_k == 8


def test_config_rejects_invalid_batch_or_device(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_config(tmp_path, batch_size=0)
    with pytest.raises(ValidationError, match="device must be"):
        make_config(tmp_path, device="metal")


def test_reranker_scores_in_batches_and_returns_top_k(tmp_path: Path) -> None:
    model = FakeCrossEncoder()
    reranker = QwenReranker(make_config(tmp_path), model=model)
    documents = [make_result(f"chunk-{i}", "x" * i) for i in range(1, 6)]

    results = reranker.rerank("movement", documents, top_k=3)

    assert [result.chunk_id for result in results] == ["chunk-5", "chunk-4", "chunk-3"]
    assert len(model.calls) == 3
    assert results[0].score == 5.0
    assert results[0].metadata["rerank_rank"] == 1
    assert results[0].metadata["reranker"] == "fake-reranker"


def test_reranker_preserves_hybrid_rank_metadata(tmp_path: Path) -> None:
    reranker = QwenReranker(make_config(tmp_path), model=FakeCrossEncoder())
    result = reranker.rerank("query", [make_result("one", "content")])[0]

    assert result.dense_rank == 1
    assert result.sparse_rank == 2
    assert result.metadata["retrieval"] == "reranked"


def test_reranker_validates_query_and_top_k(tmp_path: Path) -> None:
    reranker = QwenReranker(make_config(tmp_path), model=FakeCrossEncoder())
    document = make_result("one", "content")

    with pytest.raises(ValueError, match="non-empty"):
        reranker.rerank(" ", [document])
    with pytest.raises(ValueError, match="greater than zero"):
        reranker.rerank("query", [document], top_k=0)
    assert reranker.rerank("query", []) == []


def test_reranker_rejects_incomplete_model_scores(tmp_path: Path) -> None:
    class BadModel:
        def predict(self, pairs: list[list[str]], **_: object) -> list[float]:
            return [1.0] * max(0, len(pairs) - 1)

    reranker = QwenReranker(make_config(tmp_path), model=BadModel())
    with pytest.raises(RuntimeError, match="incomplete"):
        reranker.rerank("query", [make_result("one", "content"), make_result("two", "content")])


def test_query_pipeline_retrieves_then_reranks_to_configured_limit() -> None:
    retriever = FakeRetriever([make_result("one", "one"), make_result("two", "two")])
    reranker = FakeReranker()
    pipeline = RAGQueryPipeline(retriever, reranker, rerank_top_k=1)

    results = pipeline.query("movement", filters={"module": "Engine"})

    assert [item.chunk_id for item in results] == ["one"]
    assert retriever.queries == ["movement"]
    assert reranker.calls == [("movement", 2)]


def test_query_pipeline_validates_top_k() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        RAGQueryPipeline(FakeRetriever([]), FakeReranker(), rerank_top_k=0)
