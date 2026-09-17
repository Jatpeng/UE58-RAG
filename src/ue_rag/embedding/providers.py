"""Embedding provider abstractions with a lazy Sentence Transformers backend."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Validated model, device, batching, cache, and artifact settings."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    provider: str = "qwen"
    model: str = Field(min_length=1)
    device: str = "auto"
    batch_size: int = Field(gt=0)
    max_length: int = Field(default=2048, gt=0)
    normalize: bool = True
    cache_path: Path | None = None
    output_path: Path

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        if value != "qwen":
            raise ValueError("provider must be 'qwen'")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value != "auto" and value != "cpu" and value != "cuda" and not value.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        return value


def load_embedding_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    embedding_config_path: str | Path = "config/embedding.yaml",
) -> EmbeddingConfig:
    """Load UE version and project-relative embedding paths from YAML."""

    ue_path = Path(ue_config_path).resolve()
    embedding_path = Path(embedding_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with embedding_path.open(encoding="utf-8") as stream:
        embedding_config = dict(yaml.safe_load(stream) or {})

    project_root = ue_path.parent.parent
    for key in ("cache_path", "output_path"):
        value = embedding_config.get(key)
        if value is not None:
            path = Path(value)
            embedding_config[key] = path if path.is_absolute() else project_root / path
    embedding_config["engine_version"] = str(ue_config["engine"]["version"])
    return EmbeddingConfig(**embedding_config)


class EmbeddingProvider(ABC):
    """Stable interface used by indexing and retrieval layers."""

    @property
    @abstractmethod
    def dimension(self) -> int | None:
        """Return vector dimension when known without an encode call."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed document/passages in input order as a float32 matrix."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query as a one-dimensional float32 vector."""


class EmbeddingCache:
    """Small SQLite cache keyed by model/options and text hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self._cache_lock = threading.Lock()
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self.connection.commit()

    def get(self, key: str) -> np.ndarray | None:
        with self._cache_lock:
            row = self.connection.execute(
                "SELECT dimension, vector FROM embeddings WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        dimension, blob = row
        return np.frombuffer(blob, dtype=np.float32).copy().reshape((dimension,))

    def put(self, key: str, vector: np.ndarray) -> None:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        with self._cache_lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
                (key, value.size, value.tobytes()),
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers provider with lazy model loading and persistent cache."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        model: Any | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self._model = model
        self._cache = cache or (EmbeddingCache(config.cache_path) if config.cache_path else None)
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for Qwen embeddings; "
                    "install the project dependencies first"
                ) from error
            self._model = SentenceTransformer(self.config.model, device=self.device)
            # UE source chunks can contain very large class/type bodies.  Cap
            # tokenization before batching so one oversized chunk cannot cause
            # an unbounded CPU allocation during full-corpus embedding.
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self.config.max_length
        return self._model

    @property
    def dimension(self) -> int | None:
        if self._dimension is not None:
            return self._dimension
        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if getter is not None:
            value = getter()
            self._dimension = int(value) if value is not None else None
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, mode="document")

    def embed_query(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        return self._embed([text], mode="query")[0]

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def _embed(self, texts: Sequence[str], *, mode: str) -> np.ndarray:
        values = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("all texts must be non-empty strings")
        if not values:
            return np.empty((0, self.dimension or 0), dtype=np.float32)

        vectors: list[np.ndarray | None] = [None] * len(values)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(values):
            key = self._cache_key(mode, text)
            cached = self._cache.get(key) if self._cache else None
            if cached is None:
                missing.append((index, text, key))
            else:
                vectors[index] = cached

        for start in range(0, len(missing), self.config.batch_size):
            batch = missing[start : start + self.config.batch_size]
            encoded = self._encode_batch([item[1] for item in batch], mode=mode)
            for (index, _, key), vector in zip(batch, encoded, strict=True):
                vectors[index] = vector
                if self._cache:
                    self._cache.put(key, vector)

        result = np.stack([vector for vector in vectors if vector is not None]).astype(
            np.float32, copy=False
        )
        if result.shape[0] != len(values):
            raise RuntimeError("embedding provider returned an incomplete batch")
        self._dimension = int(result.shape[1])
        return result

    def _encode_batch(self, texts: list[str], *, mode: str) -> np.ndarray:
        method_name = "encode_document" if mode == "document" else "encode_query"
        method = getattr(self.model, method_name, None) or getattr(self.model, "encode")
        kwargs = {
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": False,
        }
        try:
            raw = method(texts, **kwargs)
        except TypeError:
            kwargs.pop("normalize_embeddings")
            raw = method(texts, **kwargs)
        result = _as_float_matrix(raw)
        if self.config.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = np.divide(result, norms, out=np.zeros_like(result), where=norms != 0)
        return result

    def _cache_key(self, mode: str, text: str) -> str:
        raw = f"{self.config.model}|{self.config.normalize}|{mode}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class QwenEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    """Qwen3 embedding provider using the configured Sentence Transformers model."""


@dataclass(frozen=True)
class EmbeddingRunSummary:
    """Summary of a streaming embedding artifact run."""

    records: int
    dimension: int
    output_path: Path
    ids_path: Path
    manifest_path: Path


def embed_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    provider: EmbeddingProvider,
    *,
    progress_every: int = 0,
    progress: Any | None = None,
) -> EmbeddingRunSummary:
    """Embed JSONL records into a memory-mapped ``.npy`` plus ID/manifest sidecars."""

    if progress_every < 0:
        raise ValueError("progress_every must not be negative")
    input_file = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ids_path = output.with_suffix(output.suffix + ".ids.jsonl")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    temp_output = output.with_suffix(output.suffix + ".tmp.npy")
    temp_ids = ids_path.with_suffix(ids_path.suffix + ".tmp")
    temp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    total = sum(1 for _ in _iter_input_records(input_file))
    dimension = provider.dimension
    matrix: np.memmap | None = None
    try:
        if total == 0:
            matrix = np.lib.format.open_memmap(temp_output, mode="w+", dtype="float32", shape=(0, 0))
            temp_ids.write_text("", encoding="utf-8")
        else:
            with temp_ids.open("w", encoding="utf-8", newline="\n") as ids_file:
                batch_size = _provider_batch_size(provider)
                row_start = 0
                for batch in _iter_batches(_iter_input_records(input_file), batch_size):
                    vectors = provider.embed_documents([record[1] for record in batch])
                    if vectors.shape[0] != len(batch):
                        raise ValueError("provider returned a vector count different from input batch")
                    if matrix is None:
                        dimension = int(vectors.shape[1])
                        matrix = np.lib.format.open_memmap(
                            temp_output, mode="w+", dtype="float32", shape=(total, dimension)
                        )
                    if vectors.shape[1] != dimension:
                        raise ValueError("embedding dimension changed during run")
                    matrix[row_start : row_start + len(batch)] = vectors
                    for index, (record_id, _) in enumerate(batch, start=row_start):
                        ids_file.write(json.dumps({"row": index, "id": record_id}, ensure_ascii=False) + "\n")
                    row_start += len(batch)
                    if progress is not None:
                        progress(row_start, total)
                    elif progress_every and row_start and row_start % progress_every < batch_size:
                        print(f"Embedded: {row_start}/{total}", flush=True)
            assert matrix is not None
            matrix.flush()
        if dimension is None:
            dimension = 0
        manifest = {
            "records": total,
            "dimension": dimension,
            "model": getattr(getattr(provider, "config", None), "model", None),
            "normalize": getattr(getattr(provider, "config", None), "normalize", None),
            "format": "numpy-memmap-plus-jsonl-ids",
        }
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _close_memmap(matrix)
        matrix = None
        os.replace(temp_output, output)
        os.replace(temp_ids, ids_path)
        os.replace(temp_manifest, manifest_path)
    except BaseException:
        _close_memmap(matrix)
        for path in (temp_output, temp_ids, temp_manifest):
            path.unlink(missing_ok=True)
        raise
    return EmbeddingRunSummary(total, dimension, output, ids_path, manifest_path)


def _iter_input_records(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record_id = payload["id"]
                content = payload["content"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid embedding input at line {line_number}: {path}") from error
            if not isinstance(record_id, str) or not record_id.strip() or not isinstance(content, str) or not content.strip():
                raise ValueError(f"Embedding input requires non-empty id/content at line {line_number}: {path}")
            yield record_id, content


def _provider_batch_size(provider: EmbeddingProvider) -> int:
    return int(getattr(getattr(provider, "config", None), "batch_size", 32))


def _iter_batches(
    records: Iterator[tuple[str, str]], batch_size: int
) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _resolve_device(device: str) -> str:
    if device != "auto":
        if device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError(f"CUDA device {device!r} requested but CUDA is unavailable")
            except ImportError as error:
                raise RuntimeError("PyTorch is required to use a CUDA embedding device") from error
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _as_float_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif isinstance(value, list) and value and hasattr(value[0], "detach"):
        value = np.stack([item.detach().cpu().numpy() for item in value])
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"embedding model must return a 2D matrix, received shape {matrix.shape}")
    return matrix


def _close_memmap(matrix: np.memmap | None) -> None:
    if matrix is None:
        return
    matrix.flush()
    handle = getattr(matrix, "_mmap", None)
    if handle is not None:
        handle.close()
