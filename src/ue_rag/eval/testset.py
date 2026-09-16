"""RAGAS testset preparation and benchmark export helpers."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ue_rag.schema import UEChunk


CHUNK_MARKER = "UE_RAG_CHUNK_ID"
SYMBOL_MARKER = "UE_RAG_SYMBOL"
_CHUNK_PATTERN = re.compile(r"UE_RAG_CHUNK_ID:\s*([^\s]+)")
_SYMBOL_PATTERN = re.compile(r"UE_RAG_SYMBOL:\s*([^\r\n]+)")


class LLMTestsetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    context: str | None = None
    base_url: str | None = None
    api_key_env: str = Field(default="OPENAI_API_KEY", min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_seconds: int = Field(default=180, gt=0)
    max_retries: int = Field(default=3, ge=0)
    max_workers: int = Field(default=4, gt=0)


class EmbeddingTestsetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    device: str = Field(default="cuda", min_length=1)
    batch_size: int = Field(default=8, gt=0)


class TestsetConfig(BaseModel):
    """Configuration for bounded RAGAS testset generation."""

    model_config = ConfigDict(extra="forbid")

    input: Path
    sources_output: Path
    testset_output: Path
    benchmark_output: Path
    documents: int = Field(default=48, gt=0)
    testset_size: int = Field(default=30, gt=0)
    seed: int = 58
    min_chars: int = Field(default=400, ge=1)
    max_chars: int = Field(default=12000, gt=0)
    max_per_stratum: int = Field(default=4, gt=0)
    llm: LLMTestsetConfig
    embedding: EmbeddingTestsetConfig


@dataclass(frozen=True)
class PreparedSource:
    """A sampled source document with traceable retrieval labels."""

    page_content: str
    metadata: dict[str, Any]


def load_testset_config(path: str | Path = "config/testset.yaml") -> TestsetConfig:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    project_root = config_path.parent.parent
    for key in ("input", "sources_output", "testset_output", "benchmark_output"):
        value = Path(values[key])
        values[key] = value if value.is_absolute() else project_root / value
    config = TestsetConfig(**values)
    if config.max_chars < config.min_chars:
        raise ValueError("max_chars must be greater than or equal to min_chars")
    if config.documents < config.testset_size:
        raise ValueError("documents must be greater than or equal to testset_size")
    return config


def sample_sources(config: TestsetConfig) -> list[PreparedSource]:
    """Stream and deterministically sample a bounded, module-balanced corpus."""

    rng = random.Random(config.seed)
    reservoirs: dict[str, list[UEChunk]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    with config.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunk = UEChunk.model_validate_json(line)
            except Exception as error:
                raise ValueError(f"Invalid UEChunk at {config.input}:{line_number}") from error
            length = len(chunk.content.strip())
            if length < config.min_chars or length > config.max_chars:
                continue
            stratum = ":".join(
                (
                    chunk.source_type.value,
                    chunk.module or chunk.plugin or "unknown",
                    chunk.symbol_type or "content",
                )
            )
            seen[stratum] += 1
            bucket = reservoirs[stratum]
            if len(bucket) < config.max_per_stratum:
                bucket.append(chunk)
            else:
                position = rng.randrange(seen[stratum])
                if position < config.max_per_stratum:
                    bucket[position] = chunk

    candidates = [chunk for bucket in reservoirs.values() for chunk in bucket]
    if len(candidates) < config.documents:
        raise ValueError(
            f"only {len(candidates)} eligible chunks remain; requested {config.documents} documents"
        )
    selected = rng.sample(candidates, config.documents)
    return [_prepared_source(chunk) for chunk in selected]


def write_sources(sources: Sequence[PreparedSource], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for source in sources:
            stream.write(json.dumps({"page_content": source.page_content, "metadata": source.metadata}, ensure_ascii=False))
            stream.write("\n")


def write_generated_testset(
    rows: Iterable[dict[str, Any]],
    *,
    testset_output: str | Path,
    benchmark_output: str | Path,
) -> tuple[int, int]:
    """Write native RAGAS rows plus retrieval-compatible benchmark cases."""

    native_path = Path(testset_output)
    benchmark_path = Path(benchmark_output)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    native_count = 0
    benchmark_count = 0
    with native_path.open("w", encoding="utf-8") as native, benchmark_path.open("w", encoding="utf-8") as benchmark:
        for index, row in enumerate(rows, start=1):
            native.write(json.dumps(row, ensure_ascii=False))
            native.write("\n")
            native_count += 1
            contexts = row.get("reference_contexts") or []
            if isinstance(contexts, str):
                contexts = [contexts]
            context_text = "\n".join(str(value) for value in contexts)
            chunk_ids = list(dict.fromkeys(_CHUNK_PATTERN.findall(context_text)))
            symbols = [value.strip() for value in _SYMBOL_PATTERN.findall(context_text) if value.strip() and value.strip() != "-"]
            symbols = list(dict.fromkeys(symbols))
            query = row.get("user_input") or row.get("question")
            if not isinstance(query, str) or not query.strip() or not (chunk_ids or symbols):
                continue
            case = {
                "id": f"ragas_{index:04d}",
                "query": query.strip(),
                "category": str(row.get("synthesizer_name") or "synthetic"),
                "expected_symbols": symbols,
                "expected_chunk_ids": chunk_ids,
                "expected_document_ids": [],
                "engine_version": None,
                "filters": {},
            }
            benchmark.write(json.dumps(case, ensure_ascii=False))
            benchmark.write("\n")
            benchmark_count += 1
    return native_count, benchmark_count


def _prepared_source(chunk: UEChunk) -> PreparedSource:
    symbol = chunk.symbol or chunk.class_name or chunk.function_name or "-"
    header = f"{CHUNK_MARKER}: {chunk.id}\n{SYMBOL_MARKER}: {symbol}\n"
    metadata = {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "engine_version": chunk.engine_version,
        "source_type": chunk.source_type.value,
        "module": chunk.module,
        "plugin": chunk.plugin,
        "symbol": chunk.symbol,
        "symbol_type": chunk.symbol_type,
        "file_path": chunk.file_path,
    }
    return PreparedSource(page_content=header + chunk.content, metadata=metadata)
