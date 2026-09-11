"""Dense plus lexical hybrid retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ue_rag.embedding import EmbeddingProvider
from ue_rag.retrieval.lexical import LexicalIndex
from ue_rag.schema import RetrievalResult


class DenseRetriever(Protocol):
    """Minimal dense-search contract implemented by QdrantVectorStore."""

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return dense results ranked by descending vector score."""


class RetrievalConfig(BaseModel):
    """Validated Top-K and RRF settings."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    dense_top_k: int = Field(gt=0)
    sparse_top_k: int = Field(gt=0)
    fusion_top_k: int = Field(gt=0)
    rerank_top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_top_k(self) -> "RetrievalConfig":
        if self.fusion_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("fusion_top_k cannot exceed dense_top_k + sparse_top_k")
        return self


def load_retrieval_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    retrieval_config_path: str | Path = "config/retrieval.yaml",
) -> RetrievalConfig:
    """Load the UE version from the central config and retrieval limits from YAML."""

    ue_path = Path(ue_config_path).resolve()
    retrieval_path = Path(retrieval_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with retrieval_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    values.setdefault("rrf_k", 60)
    values["engine_version"] = str(ue_config["engine"]["version"])
    return RetrievalConfig(**values)


class HybridRetriever:
    """Fuse dense Qdrant and lexical SQLite results with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_index: LexicalIndex,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.lexical_index = lexical_index
        self.embedding_provider = embedding_provider
        self.config = config

    def retrieve(
        self,
        query: str,
        *,
        query_vector: Sequence[float] | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        fusion_top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run both retrievers and return RRF-ranked unified results."""

        dense_limit = dense_top_k if dense_top_k is not None else (self.config.dense_top_k if self.config else 30)
        sparse_limit = sparse_top_k if sparse_top_k is not None else (self.config.sparse_top_k if self.config else 30)
        fusion_limit = fusion_top_k if fusion_top_k is not None else (self.config.fusion_top_k if self.config else 30)
        for name, value in (("dense_top_k", dense_limit), ("sparse_top_k", sparse_limit), ("fusion_top_k", fusion_limit)):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if query_vector is None:
            if self.embedding_provider is None:
                raise ValueError("query_vector or embedding_provider is required")
            query_vector = self.embedding_provider.embed_query(query)

        effective_filters = dict(filters or {})
        if self.config is not None:
            effective_filters.setdefault("engine_version", self.config.engine_version)

        dense_results = self.dense_retriever.search(
            query_vector, limit=dense_limit, filters=effective_filters
        )
        sparse_results = self.lexical_index.search(
            query, limit=sparse_limit, filters=effective_filters
        )
        return self.fuse(
            dense_results,
            sparse_results,
            limit=fusion_limit,
            rrf_k=self.config.rrf_k if self.config else 60,
        )

    @staticmethod
    def fuse(
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
        *,
        limit: int = 30,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Fuse pre-ranked result lists; duplicate chunks receive both ranks."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        entries: dict[str, dict[str, Any]] = {}
        for rank, result in enumerate(dense_results, start=1):
            key = result.chunk_id or result.document_id
            entry = entries.setdefault(key, {"result": result, "dense_rank": None, "sparse_rank": None, "score": 0.0})
            entry["dense_rank"] = rank
            entry["score"] += 1.0 / (rrf_k + rank)
        for rank, result in enumerate(sparse_results, start=1):
            key = result.chunk_id or result.document_id
            entry = entries.setdefault(key, {"result": result, "dense_rank": None, "sparse_rank": None, "score": 0.0})
            if entry["dense_rank"] is None:
                entry["result"] = result
            entry["sparse_rank"] = rank
            entry["score"] += 1.0 / (rrf_k + rank)
        ordered = sorted(
            entries.values(),
            key=lambda entry: (-entry["score"], entry["result"].chunk_id or entry["result"].document_id),
        )
        fused: list[RetrievalResult] = []
        for entry in ordered[:limit]:
            result = entry["result"]
            score = float(entry["score"])
            metadata = {
                **result.metadata,
                "retrieval": "hybrid",
                "dense_rank": entry["dense_rank"],
                "sparse_rank": entry["sparse_rank"],
                "fusion_score": score,
            }
            fused.append(
                result.model_copy(
                    update={
                        "score": score,
                        "dense_rank": entry["dense_rank"],
                        "sparse_rank": entry["sparse_rank"],
                        "fusion_score": score,
                        "metadata": metadata,
                    }
                )
            )
        return fused
