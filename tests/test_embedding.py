"""Tests for the replaceable embedding provider and artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ue_rag.embedding import (
    EmbeddingConfig,
    QwenEmbeddingProvider,
    embed_jsonl,
    load_embedding_config,
)


class FakeModel:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.document_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode_document(self, texts: list[str], **_: object) -> np.ndarray:
        self.document_calls.append(texts)
        return np.asarray([[len(text), 1.0, 2.0][: self.dimension] for text in texts], dtype=np.float32)

    def encode_query(self, texts: list[str], **_: object) -> np.ndarray:
        self.query_calls.append(texts)
        return np.asarray([[1.0, len(text), 2.0][: self.dimension] for text in texts], dtype=np.float32)


def make_config(tmp_path: Path, **overrides: object) -> EmbeddingConfig:
    values: dict[str, object] = {
        "engine_version": "5.8",
        "provider": "qwen",
        "model": "fake-qwen",
        "device": "cpu",
        "batch_size": 2,
        "max_length": 128,
        "normalize": True,
        "cache_path": tmp_path / "cache.sqlite3",
        "output_path": tmp_path / "embeddings.npy",
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


def test_config_loads_project_relative_paths() -> None:
    config = load_embedding_config()

    assert config.engine_version == "5.8"
    assert config.provider == "qwen"
    assert config.model == "Qwen/Qwen3-Embedding-0.6B"
    assert config.cache_path == Path.cwd() / "data/embeddings/cache.sqlite3"
    assert config.output_path == Path.cwd() / "data/embeddings/engine/embeddings.npy"


def test_config_rejects_unknown_provider_and_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="provider must be 'qwen'"):
        make_config(tmp_path, provider="other")
    with pytest.raises(ValidationError):
        make_config(tmp_path, batch_size=0)


def test_document_embedding_is_batched_and_normalized(tmp_path: Path) -> None:
    model = FakeModel()
    provider = QwenEmbeddingProvider(make_config(tmp_path), model=model)

    vectors = provider.embed_documents(["one", "two", "three", "four", "five"])

    assert vectors.shape == (5, 3)
    assert len(model.document_calls) == 3
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert provider.dimension == 3
    provider.close()


def test_query_uses_query_encoder_and_returns_vector(tmp_path: Path) -> None:
    model = FakeModel()
    provider = QwenEmbeddingProvider(make_config(tmp_path), model=model)

    vector = provider.embed_query("movement")

    assert vector.shape == (3,)
    assert len(model.query_calls) == 1
    assert model.document_calls == []
    provider.close()


def test_cache_avoids_reencoding_duplicate_text(tmp_path: Path) -> None:
    model = FakeModel()
    config = make_config(tmp_path)
    provider = QwenEmbeddingProvider(config, model=model)

    first = provider.embed_documents(["same", "other"])
    second = provider.embed_documents(["other", "same"])

    assert len(model.document_calls) == 1
    assert np.allclose(first, second[[1, 0]])
    provider.close()


def test_cache_persists_across_provider_instances(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    first_model = FakeModel()
    first = QwenEmbeddingProvider(config, model=first_model)
    expected = first.embed_documents(["persisted"])
    first.close()

    second_model = FakeModel()
    second = QwenEmbeddingProvider(config, model=second_model)
    actual = second.embed_documents(["persisted"])

    assert second_model.document_calls == []
    assert np.allclose(expected, actual)
    second.close()


def test_empty_and_invalid_texts_are_handled(tmp_path: Path) -> None:
    provider = QwenEmbeddingProvider(make_config(tmp_path), model=FakeModel())

    assert provider.embed_documents([]).shape == (0, 3)
    with pytest.raises(ValueError, match="non-empty"):
        provider.embed_documents(["ok", ""])
    with pytest.raises(ValueError, match="non-empty"):
        provider.embed_query(" ")
    provider.close()


def test_artifact_writer_streams_ids_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    with input_path.open("w", encoding="utf-8") as stream:
        for index in range(5):
            stream.write(json.dumps({"id": f"chunk-{index}", "content": f"text {index}"}) + "\n")
    output_path = tmp_path / "vectors.npy"
    provider = QwenEmbeddingProvider(make_config(tmp_path), model=FakeModel())

    summary = embed_jsonl(input_path, output_path, provider)
    vectors = np.load(output_path, mmap_mode="r")
    ids = [json.loads(line)["id"] for line in summary.ids_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert summary.records == 5
    assert summary.dimension == 3
    assert vectors.shape == (5, 3)
    assert ids == [f"chunk-{index}" for index in range(5)]
    assert manifest["records"] == 5
    assert manifest["dimension"] == 3
    provider.close()


def test_artifact_writer_rejects_invalid_input(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(json.dumps({"id": "only-id"}) + "\n", encoding="utf-8")
    provider = QwenEmbeddingProvider(make_config(tmp_path), model=FakeModel())

    with pytest.raises(ValueError, match="Invalid embedding input"):
        embed_jsonl(input_path, tmp_path / "vectors.npy", provider)
    provider.close()


def test_artifact_writer_supports_empty_input(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")
    output_path = tmp_path / "empty.npy"
    provider = QwenEmbeddingProvider(make_config(tmp_path), model=FakeModel())

    summary = embed_jsonl(input_path, output_path, provider)

    assert summary.records == 0
    assert np.load(output_path).shape == (0, 0)
    provider.close()
