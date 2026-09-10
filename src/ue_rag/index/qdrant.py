"""Qdrant vector-store integration with an in-memory test backend."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ue_rag.schema import RetrievalResult, SourceType, UEChunk


class QdrantConfig(BaseModel):
    """Connection and collection settings for the vector store."""

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(min_length=1)
    path: Path | None = None
    url: str | None = None
    api_key_env: str | None = None
    batch_size: int = Field(gt=0)
    distance: str = "cosine"

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, value: str) -> str:
        if value not in {"cosine", "dot", "euclid"}:
            raise ValueError("distance must be cosine, dot, or euclid")
        return value


def load_qdrant_config(path: str | Path = "config/qdrant.yaml") -> QdrantConfig:
    """Load project-relative Qdrant settings."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    for key in ("path",):
        if values.get(key) is not None:
            value = Path(values[key])
            values[key] = value if value.is_absolute() else config_path.parent.parent / value
    return QdrantConfig(**values)


@dataclass(frozen=True)
class IngestSummary:
    """Counters returned by an idempotent ingest operation."""

    total: int
    added: int
    updated: int
    skipped: int
    failed: int

    def __add__(self, other: "IngestSummary") -> "IngestSummary":
        return IngestSummary(
            self.total + other.total,
            self.added + other.added,
            self.updated + other.updated,
            self.skipped + other.skipped,
            self.failed + other.failed,
        )


class QdrantVectorStore:
    """Qdrant-backed vector store; ``client=None`` enables deterministic memory mode."""

    FILTER_FIELDS = ("engine_version", "source_type", "module", "plugin", "class", "symbol")

    def __init__(self, config: QdrantConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.client = client
        self._memory: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        self._dimension: int | None = None
        self._created = False

    @classmethod
    def from_config(cls, config: QdrantConfig) -> "QdrantVectorStore":
        """Create a store using a remote URL or local Qdrant path."""

        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError("qdrant-client is required for QdrantVectorStore") from error
        kwargs: dict[str, Any] = {}
        if config.url:
            kwargs["url"] = config.url
            if config.api_key_env:
                key = os.getenv(config.api_key_env)
                if key:
                    kwargs["api_key"] = key
        elif config.path:
            kwargs["path"] = str(config.path)
        else:
            raise ValueError("Qdrant config requires either url or path")
        return cls(config, client=QdrantClient(**kwargs))

    def create_collection(self, vector_size: int, *, recreate: bool = False) -> bool:
        """Create the configured collection and searchable metadata indexes."""

        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")
        if self._dimension is not None and self._dimension != vector_size:
            raise ValueError("vector dimension does not match the existing collection")
        self._dimension = vector_size
        if self.client is None:
            if recreate:
                self._memory.clear()
            was_created = self._created
            self._created = True
            return not was_created or recreate

        models = _qdrant_models()
        exists = self.client.collection_exists(collection_name=self.config.collection)
        if exists and recreate:
            self.client.delete_collection(collection_name=self.config.collection)
            exists = False
        if not exists:
            distance = {
                "cosine": models.Distance.COSINE,
                "dot": models.Distance.DOT,
                "euclid": models.Distance.EUCLID,
            }[self.config.distance]
            self.client.create_collection(
                collection_name=self.config.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=distance),
            )
        for field in self.FILTER_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.config.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                if not exists:
                    raise
        return not exists or recreate

    def upsert(
        self,
        chunks: Sequence[UEChunk],
        vectors: Sequence[Sequence[float]] | np.ndarray,
    ) -> IngestSummary:
        """Idempotently insert or update chunks and their vectors."""

        values = np.asarray(vectors, dtype=np.float32)
        if not chunks:
            return IngestSummary(0, 0, 0, 0, 0)
        if values.ndim != 2 or values.shape[0] != len(chunks):
            raise ValueError("vectors must be a 2D matrix with one row per chunk")
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique within an ingest batch")
        if self._dimension is None:
            self.create_collection(int(values.shape[1]))
        if values.shape[1] != self._dimension:
            raise ValueError("vector dimension does not match the collection")
        records: list[tuple[UEChunk, np.ndarray, dict[str, Any], str, str]] = []
        summary = IngestSummary(len(chunks), 0, 0, 0, 0)
        existing = self._existing(chunks)
        for chunk, vector in zip(chunks, values, strict=True):
            payload = _payload(chunk)
            point_id = _point_id(chunk.id)
            content_hash = payload["content_sha256"]
            previous = existing.get(point_id)
            if previous is not None and previous.get("content_sha256") == content_hash:
                summary += IngestSummary(0, 0, 0, 1, 0)
                continue
            kind = "updated" if previous is not None else "added"
            records.append((chunk, vector, payload, point_id, kind))
        if records:
            try:
                self._write_records(records)
            except Exception:
                return summary + IngestSummary(0, 0, 0, 0, len(records))
            summary += IngestSummary(
                0,
                sum(kind == "added" for *_, kind in records),
                sum(kind == "updated" for *_, kind in records),
                0,
                0,
            )
        return summary

    def ingest(
        self,
        chunks: Sequence[UEChunk],
        vectors: Sequence[Sequence[float]] | np.ndarray,
    ) -> IngestSummary:
        """Alias for batch upsert used by ingestion callers."""

        return self.upsert(chunks, vectors)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 10,
        filters: dict[str, str | int | float] | None = None,
    ) -> list[RetrievalResult]:
        """Search by vector and optional exact metadata filters."""

        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        _validate_filters(filters)
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._dimension is None or query.size != self._dimension:
            raise ValueError("query vector dimension does not match the collection")
        if self.client is None:
            query_norm = np.linalg.norm(query)
            scored: list[tuple[float, dict[str, Any]]] = []
            for vector, payload in self._memory.values():
                if not _matches(payload, filters):
                    continue
                score = float(np.dot(query, vector) / (query_norm * np.linalg.norm(vector))) if query_norm and np.linalg.norm(vector) else 0.0
                scored.append((score, payload))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [_result(payload, score) for score, payload in scored[:limit]]

        query_filter = _build_filter(filters)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.config.collection,
                query=query.tolist(),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        else:
            points = self.client.search(
                collection_name=self.config.collection,
                query_vector=query.tolist(),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        return [_result(point.payload, float(point.score)) for point in points]

    def delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        filters: dict[str, str | int | float] | None = None,
    ) -> int:
        """Delete points by deterministic chunk IDs or exact metadata filters."""

        if chunk_ids and filters:
            raise ValueError("provide chunk_ids or filters, not both")
        _validate_filters(filters)
        if self.client is None:
            targets = set(_point_id(value) for value in chunk_ids) if chunk_ids else {
                point_id for point_id, (_, payload) in self._memory.items() if _matches(payload, filters)
            }
            for point_id in targets:
                self._memory.pop(point_id, None)
            return len(targets)
        models = _qdrant_models()
        selector: Any
        if chunk_ids:
            selector = models.PointIdsList(points=[_point_id(value) for value in chunk_ids])
        elif filters:
            selector = _build_filter(filters)
        else:
            raise ValueError("chunk_ids or filters is required")
        self.client.delete(collection_name=self.config.collection, points_selector=selector)
        return len(chunk_ids) if chunk_ids else 0

    def _existing(self, chunks: Sequence[UEChunk]) -> dict[str, dict[str, Any]]:
        ids = [_point_id(chunk.id) for chunk in chunks]
        if self.client is None:
            return {point_id: payload for point_id, (_, payload) in self._memory.items() if point_id in ids}
        points = self.client.retrieve(
            collection_name=self.config.collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return {str(point.id): point.payload or {} for point in points}

    def _write_records(self, records: Sequence[tuple[UEChunk, np.ndarray, dict[str, Any], str, str]]) -> None:
        if self.client is None:
            for _, vector, payload, point_id, _ in records:
                self._memory[point_id] = (vector.copy(), payload)
            return
        models = _qdrant_models()
        points = [
            models.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            for _, vector, payload, point_id, _ in records
        ]
        self.client.upsert(collection_name=self.config.collection, points=points, wait=True)


def ingest_jsonl(
    store: QdrantVectorStore,
    chunks_path: str | Path,
    vectors_path: str | Path,
    *,
    ids_path: str | Path | None = None,
    batch_size: int | None = None,
) -> IngestSummary:
    """Stream UEChunk JSONL and a NumPy embedding matrix into the store."""

    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError("embedding artifact must be a 2D NumPy matrix")
    ids_stream = Path(ids_path).open(encoding="utf-8") if ids_path else None
    summary = IngestSummary(0, 0, 0, 0, 0)
    chunks: list[UEChunk] = []
    rows: list[int] = []
    next_row = 0
    size = batch_size or store.config.batch_size
    try:
        with Path(chunks_path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = UEChunk.model_validate_json(line)
                except Exception as error:
                    raise ValueError(f"Invalid UEChunk at line {line_number}: {chunks_path}") from error
                row = next_row if ids_stream is None else _read_artifact_row(ids_stream, chunk.id, line_number)
                next_row += 1
                if row < 0 or row >= vectors.shape[0]:
                    raise ValueError(f"Embedding row {row} is outside artifact at line {line_number}")
                chunks.append(chunk)
                rows.append(row)
                if len(chunks) == size:
                    summary += store.upsert(chunks, vectors[rows])
                    chunks, rows = [], []
            if chunks:
                summary += store.upsert(chunks, vectors[rows])
    finally:
        if ids_stream:
            ids_stream.close()
    if summary.total != vectors.shape[0]:
        raise ValueError(f"chunk count {summary.total} does not match embedding rows {vectors.shape[0]}")
    return summary


def _read_artifact_row(stream, expected_id: str, line_number: int) -> int:
    line = stream.readline()
    if not line:
        raise ValueError(f"Missing embedding ID at input line {line_number}")
    try:
        payload = json.loads(line)
        if payload["id"] != expected_id:
            raise ValueError(f"Embedding ID does not match chunk at input line {line_number}")
        return int(payload["row"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid embedding ID record at input line {line_number}") from error


def _payload(chunk: UEChunk) -> dict[str, Any]:
    payload = {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "content": chunk.content,
        "engine_version": chunk.engine_version,
        "source_type": chunk.source_type.value,
        "module": chunk.module,
        "plugin": chunk.plugin,
        "class": chunk.class_name,
        "class_name": chunk.class_name,
        "symbol": chunk.symbol,
        "symbol_type": chunk.symbol_type,
        "file_path": chunk.file_path,
        "metadata": chunk.metadata,
        "content_sha256": hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
    }
    return payload


def _result(payload: dict[str, Any], score: float) -> RetrievalResult:
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"content", "content_sha256", "metadata"}
    }
    metadata.update(payload.get("metadata") or {})
    return RetrievalResult(
        document_id=payload["document_id"],
        chunk_id=payload.get("chunk_id"),
        engine_version=payload["engine_version"],
        source_type=SourceType(payload["source_type"]),
        content=payload["content"],
        score=score,
        metadata=metadata,
    )


def _matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return all(payload.get(key) == value for key, value in filters.items())


def _validate_filters(filters: dict[str, Any] | None) -> None:
    if not filters:
        return
    unknown = set(filters) - set(QdrantVectorStore.FILTER_FIELDS)
    if unknown:
        raise ValueError(f"unsupported Qdrant filter fields: {sorted(unknown)}")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ue58-rag:{chunk_id}"))


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as error:
        raise RuntimeError("qdrant-client is required for QdrantVectorStore") from error
    return models


def _build_filter(filters: dict[str, Any] | None) -> Any:
    if not filters:
        return None
    _validate_filters(filters)
    models = _qdrant_models()
    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value)) for key, value in filters.items()]
    )
