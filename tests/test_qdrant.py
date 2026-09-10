"""Tests for Qdrant vector-store behavior using its dependency-free memory backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ue_rag.index import QdrantConfig, QdrantVectorStore, ingest_jsonl, load_qdrant_config
from ue_rag.jsonl import save_jsonl
from ue_rag.schema import SourceScope, SourceType, UEChunk


def make_config(tmp_path: Path, **overrides: object) -> QdrantConfig:
    values: dict[str, object] = {
        "collection": "test_collection",
        "path": tmp_path / "qdrant",
        "url": None,
        "batch_size": 2,
        "distance": "cosine",
    }
    values.update(overrides)
    return QdrantConfig(**values)


def make_chunk(
    chunk_id: str,
    *,
    symbol: str = "AHero::Tick",
    module: str = "Hero",
    class_name: str = "AHero",
    content: str = "movement code",
) -> UEChunk:
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
        plugin="Gameplay",
        file_path="Engine/Source/Hero/Hero.h",
        symbol=symbol,
        symbol_type="method",
        class_name=class_name,
        function_name="Tick",
        metadata={"chunk_kind": "function"},
    )


def test_config_loads_collection_and_project_relative_path() -> None:
    config = load_qdrant_config()

    assert config.collection == "ue58_global"
    assert config.path == Path.cwd() / "data/qdrant"
    assert config.batch_size == 64


def test_config_rejects_unknown_distance(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="distance must be"):
        make_config(tmp_path, distance="manhattan")


def test_memory_store_creates_collection_and_is_idempotent(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    assert store.create_collection(3) is True
    chunks = [make_chunk("chunk-1"), make_chunk("chunk-2", symbol="AHero::Move")]
    vectors = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    first = store.upsert(chunks, vectors)
    second = store.upsert(chunks, vectors)

    assert (first.total, first.added, first.updated, first.skipped, first.failed) == (2, 2, 0, 0, 0)
    assert (second.total, second.added, second.updated, second.skipped, second.failed) == (2, 0, 0, 2, 0)


def test_changed_content_is_updated(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    store.create_collection(2)
    original = make_chunk("chunk-1", content="old")
    changed = make_chunk("chunk-1", content="new")

    assert store.upsert([original], [[1, 0]]).updated == 0
    assert store.upsert([changed], [[0, 1]]).updated == 1


def test_search_returns_ranked_results_and_metadata(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    store.create_collection(3)
    chunks = [
        make_chunk("chunk-1", symbol="AHero::Tick"),
        make_chunk("chunk-2", symbol="BOther::Tick", module="Other", class_name="BOther"),
    ]
    store.upsert(chunks, [[1, 0, 0], [0.8, 0.2, 0]])

    results = store.search([1, 0, 0], limit=2)

    assert [result.chunk_id for result in results] == ["chunk-1", "chunk-2"]
    assert results[0].score > results[1].score
    assert results[0].metadata["chunk_kind"] == "function"
    assert results[0].metadata["module"] == "Hero"
    assert results[0].metadata["class"] == "AHero"


def test_search_supports_required_metadata_filters(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    store.create_collection(2)
    store.upsert(
        [make_chunk("chunk-1"), make_chunk("chunk-2", module="Other", class_name="BOther")],
        [[1, 0], [1, 0]],
    )

    results = store.search([1, 0], filters={"module": "Other", "class": "BOther"})

    assert [result.chunk_id for result in results] == ["chunk-2"]


def test_search_rejects_bad_limit_or_dimension(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    store.create_collection(2)
    with pytest.raises(ValueError, match="greater than zero"):
        store.search([1, 0], limit=0)
    with pytest.raises(ValueError, match="dimension"):
        store.search([1, 0, 0])
    with pytest.raises(ValueError, match="unsupported Qdrant filter"):
        store.search([1, 0], filters={"unknown": "value"})


def test_delete_by_ids_and_filter(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    store.create_collection(2)
    store.upsert([make_chunk("chunk-1"), make_chunk("chunk-2", module="Other")], [[1, 0], [0, 1]])

    assert store.delete(chunk_ids=["chunk-1"]) == 1
    assert store.delete(filters={"module": "Other"}) == 1
    assert store.search([1, 0]) == []


def test_delete_rejects_ambiguous_selector(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    with pytest.raises(ValueError, match="not both"):
        store.delete(chunk_ids=["chunk-1"], filters={"module": "Hero"})


def test_ingest_jsonl_streams_artifact_and_matches_ids(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = QdrantVectorStore(config)
    chunks = [make_chunk("chunk-1"), make_chunk("chunk-2", symbol="AHero::Move"), make_chunk("chunk-3", symbol="AHero::Jump")]
    chunks_path = tmp_path / "chunks.jsonl"
    save_jsonl(chunks_path, chunks)
    vectors_path = tmp_path / "vectors.npy"
    np.save(vectors_path, np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    ids_path = tmp_path / "vectors.npy.ids.jsonl"
    ids_path.write_text("\n".join(json.dumps({"row": i, "id": chunk.id}) for i, chunk in enumerate(chunks)) + "\n", encoding="utf-8")

    summary = ingest_jsonl(store, chunks_path, vectors_path, ids_path=ids_path)

    assert summary.total == summary.added == 3
    assert len(store.search([1, 0], limit=3)) == 3


def test_ingest_without_id_sidecar_keeps_global_row_offsets(tmp_path: Path) -> None:
    config = make_config(tmp_path, batch_size=2)
    store = QdrantVectorStore(config)
    chunks = [make_chunk(f"chunk-{i}") for i in range(5)]
    chunks_path = tmp_path / "chunks.jsonl"
    save_jsonl(chunks_path, chunks)
    vectors_path = tmp_path / "vectors.npy"
    np.save(vectors_path, np.eye(5, 2, dtype=np.float32))

    summary = ingest_jsonl(store, chunks_path, vectors_path)

    assert summary.total == 5
    assert summary.added == 5


def test_ingest_rejects_vector_row_mismatch(tmp_path: Path) -> None:
    store = QdrantVectorStore(make_config(tmp_path))
    chunks_path = tmp_path / "chunks.jsonl"
    save_jsonl(chunks_path, [make_chunk("chunk-1"), make_chunk("chunk-2")])
    vectors_path = tmp_path / "vectors.npy"
    np.save(vectors_path, np.asarray([[1, 0]], dtype=np.float32))

    with pytest.raises(ValueError, match="outside artifact|does not match"):
        ingest_jsonl(store, chunks_path, vectors_path)
