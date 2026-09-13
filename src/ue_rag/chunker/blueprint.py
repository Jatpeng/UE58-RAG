"""Semantic chunking for normalized Blueprint documents."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ue_rag.chunker.docs import HeuristicTokenCounter, TokenCounter
from ue_rag.jsonl import load_jsonl
from ue_rag.schema import SourceType, UEChunk, UEDocument


class BlueprintChunkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    target_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_limits(self) -> "BlueprintChunkerConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        return self


def load_blueprint_chunker_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    chunker_config_path: str | Path = "config/blueprint_chunker.yaml",
) -> BlueprintChunkerConfig:
    ue_path = Path(ue_config_path).resolve()
    chunker_path = Path(chunker_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with chunker_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    project_root = ue_path.parent.parent
    parsed_dir = Path(ue_config["data"]["parsed"])
    chunks_dir = Path(ue_config["data"]["chunks"])
    if not parsed_dir.is_absolute():
        parsed_dir = project_root / parsed_dir
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir
    values.update(
        {
            "engine_version": str(ue_config["engine"]["version"]),
            "input_path": parsed_dir / "project" / "blueprints.jsonl",
            "output_path": chunks_dir / "project" / "blueprints.jsonl",
        }
    )
    return BlueprintChunkerConfig(**values)


class BlueprintChunker:
    """Build one context-rich chunk per Blueprint semantic view."""

    def __init__(self, config: BlueprintChunkerConfig, *, token_counter: TokenCounter | None = None) -> None:
        self.config = config
        self.token_counter = token_counter or HeuristicTokenCounter()

    def chunk_document(self, document: UEDocument) -> list[UEChunk]:
        if document.source_type is not SourceType.BLUEPRINT:
            raise ValueError(f"Expected blueprint source, received {document.source_type.value}")
        if document.engine_version != self.config.engine_version:
            raise ValueError(f"Document engine version does not match configured version: {document.id}")
        header = self._header(document)
        body_limit = max(1, self.config.max_tokens - self.token_counter.count(header))
        if self.token_counter.count(document.content) <= body_limit or document.metadata.get("export_kind") != "graph":
            return [self._chunk(document, document.content, part_index=0, part_count=1)]
        parts = _split_graph(document.content, self.token_counter, body_limit, self.config.target_tokens, self.config.overlap_tokens)
        return [
            self._chunk(document, part, part_index=index, part_count=len(parts))
            for index, part in enumerate(parts)
        ]

    def chunk_corpus(self) -> tuple[int, int, Path]:
        documents = load_jsonl(self.config.input_path, UEDocument)
        chunks = [chunk for document in documents for chunk in self.chunk_document(document)]
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.output_path.with_suffix(self.config.output_path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                for chunk in chunks:
                    stream.write(chunk.model_dump_json() + "\n")
            os.replace(temporary, self.config.output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return len(documents), len(chunks), self.config.output_path

    def _header(self, document: UEDocument) -> str:
        return "\n".join(
            [
                f"UE Version: {document.engine_version}",
                f"Asset: {document.asset_name or '<unknown>'}",
                f"Graph: {document.graph_name or '<none>'}",
                f"Symbol: {document.symbol or '<unknown>'}",
                f"Symbol Type: {document.symbol_type or '<unknown>'}",
                f"Package: {document.file_path or '<unknown>'}",
                f"Parent Class: {document.metadata.get('parent_class') or '<unknown>'}",
            ]
        )

    def _chunk(self, document: UEDocument, body: str, *, part_index: int, part_count: int) -> UEChunk:
        header = self._header(document)
        content = f"{header}\n\nSource:\n{body.strip()}".strip()
        return UEChunk(
            id=_chunk_id(document.id, part_index, content),
            document_id=document.id,
            chunk_index=part_index,
            engine_version=document.engine_version,
            source_scope=document.source_scope,
            source_type=document.source_type,
            content=content,
            title=document.title,
            file_path=document.file_path,
            symbol=document.symbol,
            symbol_type=document.symbol_type,
            class_name=document.class_name,
            function_name=document.function_name,
            asset_name=document.asset_name,
            graph_name=document.graph_name,
            metadata={
                **document.metadata,
                "chunker_version": "1",
                "chunk_kind": "blueprint_graph_part" if document.metadata.get("export_kind") == "graph" else "blueprint_view",
                "part_index": part_index,
                "part_count": part_count,
                "oversized": self.token_counter.count(content) > self.config.max_tokens,
            },
        )


def _split_graph(text: str, counter: TokenCounter, max_tokens: int, target_tokens: int, overlap_tokens: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = counter.count(line)
        if current and (current_tokens + line_tokens > max_tokens or current_tokens >= target_tokens):
            parts.append("\n".join(current).strip())
            overlap = counter.tail("\n".join(current), overlap_tokens)
            current = [overlap] if overlap else []
            current_tokens = counter.count(overlap)
        current.append(line)
        current_tokens += line_tokens
    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part]


def _chunk_id(document_id: str, part_index: int, content: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{part_index}:{content}".encode("utf-8")).hexdigest()
    return f"bpchunk-{digest}"
