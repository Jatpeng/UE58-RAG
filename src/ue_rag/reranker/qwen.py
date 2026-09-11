"""Reranker abstraction and Qwen3 CrossEncoder adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ue_rag.schema import RetrievalResult


class RerankConfig(BaseModel):
    """Validated reranker model and execution settings."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    device: str = "auto"
    batch_size: int = Field(gt=0)
    top_k: int = Field(gt=0)
    max_length: int = Field(gt=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value != "auto" and value != "cpu" and value != "cuda" and not value.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        return value


def load_rerank_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    reranker_config_path: str | Path = "config/reranker.yaml",
) -> RerankConfig:
    """Load the UE version from the central config and reranker settings."""

    ue_path = Path(ue_config_path).resolve()
    rerank_path = Path(reranker_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with rerank_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    values["engine_version"] = str(ue_config["engine"]["version"])
    return RerankConfig(**values)


class Reranker(ABC):
    """Stable interface independent of a particular reranking model."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return the highest-scoring documents for a query."""


class QwenReranker(Reranker):
    """Qwen3-Reranker adapter using the Sentence Transformers CrossEncoder API."""

    def __init__(self, config: RerankConfig, *, model: Any | None = None) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for Qwen reranking; "
                    "install the project dependencies first"
                ) from error
            self._model = CrossEncoder(
                self.config.model,
                device=self.device,
                max_length=self.config.max_length,
            )
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        requested = self.config.top_k if top_k is None else top_k
        if requested <= 0:
            raise ValueError("top_k must be greater than zero")
        values = list(documents)
        if not values:
            return []
        scores: list[float] = []
        for start in range(0, len(values), self.config.batch_size):
            batch = values[start : start + self.config.batch_size]
            pairs = [[query, document.content] for document in batch]
            scores.extend(self._predict(pairs))
        if len(scores) != len(values):
            raise RuntimeError("reranker returned an incomplete score list")
        ranked = sorted(
            enumerate(values),
            key=lambda item: (-scores[item[0]], item[1].chunk_id or item[1].document_id, item[0]),
        )[:requested]
        output: list[RetrievalResult] = []
        for rank, (index, document) in enumerate(ranked, start=1):
            score = float(scores[index])
            metadata = {
                **document.metadata,
                "retrieval": "reranked",
                "reranker": self.config.model,
                "rerank_score": score,
                "rerank_rank": rank,
            }
            output.append(document.model_copy(update={"score": score, "metadata": metadata}))
        return output

    def _predict(self, pairs: list[list[str]]) -> list[float]:
        kwargs = {"batch_size": self.config.batch_size, "show_progress_bar": False}
        try:
            values = self.model.predict(pairs, **kwargs)
        except TypeError:
            values = self.model.predict(pairs, batch_size=self.config.batch_size)
        scores = np.asarray(values, dtype=np.float32).reshape(-1)
        return [float(value) for value in scores]


class Retriever(Protocol):
    def retrieve(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        """Return initial retrieval candidates."""


class RAGQueryPipeline:
    """Compose candidate retrieval with reranking without coupling either side."""

    def __init__(self, retriever: Retriever, reranker: Reranker, *, rerank_top_k: int = 8) -> None:
        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than zero")
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def query(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        candidates = self.retriever.retrieve(query, **kwargs)
        return self.reranker.rerank(query, candidates, top_k=self.rerank_top_k)


class RerankSummary:
    """Small result summary for callers that need candidate/final counts."""

    def __init__(self, candidates: int, results: int) -> None:
        self.candidates = candidates
        self.results = results


def _resolve_device(device: str) -> str:
    if device != "auto":
        if device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")
            except ImportError as error:
                raise RuntimeError("PyTorch is required to use a CUDA reranker device") from error
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
