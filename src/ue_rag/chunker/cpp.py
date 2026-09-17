"""Semantic chunking for Unreal Engine C++ symbol documents."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ue_rag.chunker.docs import HeuristicTokenCounter, TokenCounter
from ue_rag.schema import SourceType, UEChunk, UEDocument


CPP_CHUNKER_VERSION = "1"
_FUNCTION_TYPES = {"function", "method", "constructor"}
_TYPE_TYPES = {"class", "struct", "enum"}


class CPPChunkerConfig(BaseModel):
    """Validated token and path configuration for C++ chunking."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    token_counter: str = "heuristic"
    target_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    property_grouping: str = "class"

    @model_validator(mode="after")
    def validate_limits(self) -> "CPPChunkerConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        if self.property_grouping not in {"class", "none"}:
            raise ValueError("property_grouping must be 'class' or 'none'")
        return self


def load_cpp_chunker_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    chunker_config_path: str | Path = "config/cpp_chunker.yaml",
) -> CPPChunkerConfig:
    """Load UE version/data paths and C++ chunking limits from YAML."""

    ue_path = Path(ue_config_path).resolve()
    chunker_path = Path(chunker_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with chunker_path.open(encoding="utf-8") as stream:
        chunker_config = yaml.safe_load(stream)

    project_root = ue_path.parent.parent
    parsed_dir = Path(ue_config["data"]["parsed"])
    chunks_dir = Path(ue_config["data"]["chunks"])
    if not parsed_dir.is_absolute():
        parsed_dir = project_root / parsed_dir
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir
    return CPPChunkerConfig(
        engine_version=str(ue_config["engine"]["version"]),
        input_path=parsed_dir / "engine" / "documents.jsonl",
        output_path=chunks_dir / "engine" / "chunks.jsonl",
        **chunker_config,
    )


@dataclass(frozen=True)
class CPPChunkSummary:
    """Counters produced while chunking a C++ corpus."""

    documents: int
    chunks: int
    property_groups: int
    oversized_chunks: int
    output_path: Path


class CPPSemanticChunker:
    """Turn parsed symbols into context-rich, retrieval-ready C++ chunks."""

    def __init__(
        self,
        config: CPPChunkerConfig,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if config.token_counter != "heuristic" and token_counter is None:
            raise ValueError(f"Unsupported token counter: {config.token_counter}")
        self.config = config
        self.token_counter = token_counter or HeuristicTokenCounter()

    def chunk_document(self, document: UEDocument) -> list[UEChunk]:
        """Create chunks for one symbol document."""

        self._validate_document(document)
        if document.symbol_type in _FUNCTION_TYPES:
            return self._chunk_function(document)
        return [self._make_chunk(document, document.content, kind=self._kind(document))]

    def chunk_property_group(self, documents: list[UEDocument]) -> UEChunk:
        """Merge fields belonging to one class into one properties chunk."""

        if not documents:
            raise ValueError("property group cannot be empty")
        for document in documents:
            self._validate_document(document)
            if document.symbol_type != "field":
                raise ValueError("property groups can contain only field documents")
        first = documents[0]
        class_name = first.class_name or "<global>"
        source_lines = [f"{document.symbol}:\n{document.content}" for document in documents]
        content = "\n\n".join(source_lines)
        synthetic = first.model_copy(
            update={
                "id": _synthetic_group_id(documents),
                "symbol": f"{class_name}::Properties",
                "symbol_type": "property_group",
                "class_name": first.class_name,
                "function_name": None,
                "content": content,
            }
        )
        chunk = self._make_chunk(synthetic, content, kind="property_group")
        chunk.metadata.update(
            {
                "field_symbols": [document.symbol for document in documents],
                "field_document_ids": [document.id for document in documents],
                "property_grouping": "class",
            }
        )
        return chunk

    def chunk_corpus(self) -> CPPChunkSummary:
        """Stream parsed JSONL grouped by source file and atomically replace output."""

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.config.output_path.with_suffix(self.config.output_path.suffix + ".tmp")
        documents_count = chunks_count = property_groups = oversized = 0
        try:
            with self.config.input_path.open(encoding="utf-8") as input_file, temp_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as output_file:
                current_path: str | None = None
                current_documents: list[UEDocument] = []
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        document = UEDocument.model_validate_json(line)
                    except Exception as error:
                        raise ValueError(
                            f"Invalid C++ document at line {line_number}: {self.config.input_path}"
                        ) from error
                    documents_count += 1
                    if document.file_path != current_path and current_documents:
                        emitted = self._emit_file(current_documents)
                        chunks_count += len(emitted)
                        property_groups += sum(
                            chunk.metadata["chunk_kind"] == "property_group" for chunk in emitted
                        )
                        oversized += sum(bool(chunk.metadata["oversized"]) for chunk in emitted)
                        for chunk in emitted:
                            output_file.write(chunk.model_dump_json() + "\n")
                        current_documents = []
                    current_path = document.file_path
                    current_documents.append(document)
                if current_documents:
                    emitted = self._emit_file(current_documents)
                    chunks_count += len(emitted)
                    property_groups += sum(
                        chunk.metadata["chunk_kind"] == "property_group" for chunk in emitted
                    )
                    oversized += sum(bool(chunk.metadata["oversized"]) for chunk in emitted)
                    for chunk in emitted:
                        output_file.write(chunk.model_dump_json() + "\n")
            os.replace(temp_path, self.config.output_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return CPPChunkSummary(
            documents=documents_count,
            chunks=chunks_count,
            property_groups=property_groups,
            oversized_chunks=oversized,
            output_path=self.config.output_path,
        )

    def _emit_file(self, documents: list[UEDocument]) -> list[UEChunk]:
        chunks: list[UEChunk] = []
        fields: dict[tuple[str | None, str | None, str | None], list[UEDocument]] = {}
        for document in documents:
            if self.config.property_grouping == "class" and document.symbol_type == "field":
                key = (document.class_name, document.module, document.plugin)
                fields.setdefault(key, []).append(document)
                continue
            chunks.extend(self.chunk_document(document))
        if self.config.property_grouping == "class":
            for group in fields.values():
                chunks.append(self.chunk_property_group(group))
        return chunks

    def _chunk_function(self, document: UEDocument) -> list[UEChunk]:
        header = self._semantic_header(document)
        body = document.content.strip()
        body_limit = max(1, self.config.max_tokens - self.token_counter.count(header))
        if self.token_counter.count(body) <= body_limit:
            return [self._make_chunk(document, body, kind="function")]
        parts = _split_code(body, self.token_counter, body_limit, self.config.target_tokens, self.config.overlap_tokens)
        return [
            self._make_chunk(
                document,
                part,
                kind="function",
                part_index=index,
                part_count=len(parts),
            )
            for index, part in enumerate(parts)
        ]

    def _make_chunk(
        self,
        document: UEDocument,
        body: str,
        *,
        kind: str,
        part_index: int = 0,
        part_count: int = 1,
    ) -> UEChunk:
        header = self._semantic_header(document)
        content = f"{header}\n\nSource:\n{body.strip()}".strip()
        token_count = self.token_counter.count(content)
        return UEChunk(
            id=_chunk_id(document.id, kind, part_index, content),
            document_id=document.id,
            chunk_index=part_index,
            engine_version=document.engine_version,
            source_scope=document.source_scope,
            source_type=document.source_type,
            content=content,
            title=document.title,
            module=document.module,
            plugin=document.plugin,
            file_path=document.file_path,
            symbol=document.symbol,
            symbol_type=document.symbol_type,
            class_name=document.class_name,
            function_name=document.function_name,
            asset_name=document.asset_name,
            graph_name=document.graph_name,
            section_path=list(
                dict.fromkeys(value for value in (document.class_name, document.symbol) if value)
            ),
            token_count=token_count,
            metadata={
                **document.metadata,
                "chunker_version": CPP_CHUNKER_VERSION,
                "chunk_kind": kind,
                "part_index": part_index,
                "part_count": part_count,
                "oversized": token_count > self.config.max_tokens,
                "source_token_count": self.token_counter.count(body),
            },
        )

    def _semantic_header(self, document: UEDocument) -> str:
        return "\n".join(
            [
                f"UE Version: {document.engine_version}",
                f"Module: {document.module or '<global>'}",
                f"Class: {document.class_name or '<global>'}",
                f"Symbol: {document.symbol or '<anonymous>'}",
                f"Symbol Type: {document.symbol_type or '<unknown>'}",
                f"File: {document.file_path or '<unknown>'}",
            ]
        )

    def _validate_document(self, document: UEDocument) -> None:
        if document.source_type not in {SourceType.ENGINE_SOURCE, SourceType.PROJECT_SOURCE}:
            raise ValueError(
                "Expected engine_source or project_source, "
                f"received {document.source_type.value}"
            )
        if document.engine_version != self.config.engine_version:
            raise ValueError(
                f"Document engine version {document.engine_version!r} does not match "
                f"configured version {self.config.engine_version!r}: {document.id}"
            )

    @staticmethod
    def _kind(document: UEDocument) -> str:
        if document.symbol_type in _TYPE_TYPES:
            return "type"
        return document.symbol_type or "symbol"


def _split_code(
    text: str,
    counter: TokenCounter,
    max_tokens: int,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split oversized source on lines, with token overlap and no empty pieces."""

    lines = text.splitlines() or [text]
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in lines:
        line_tokens = counter.count(line)
        if line_tokens > max_tokens:
            if current:
                parts.append("\n".join(current).strip())
                current, current_tokens = [], 0
            pieces = _split_tokens(line, counter, max_tokens)
            parts.extend(pieces[:-1])
            current = [pieces[-1]]
            current_tokens = counter.count(pieces[-1])
            continue
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


def _split_tokens(text: str, counter: TokenCounter, limit: int) -> list[str]:
    remaining = text.strip()
    pieces: list[str] = []
    while counter.count(remaining) > limit:
        head = counter.head(remaining, limit)
        if not head:
            break
        pieces.append(head)
        remaining = remaining[len(head) :].strip()
    if remaining:
        pieces.append(remaining)
    return pieces or [text.strip()]


def _synthetic_group_id(documents: Iterable[UEDocument]) -> str:
    raw = "|".join(document.id for document in documents)
    return "cpp-properties-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chunk_id(document_id: str, kind: str, part_index: int, content: str) -> str:
    raw = f"{document_id}:{kind}:{part_index}:{content}".encode("utf-8")
    return "chunk-" + hashlib.sha256(raw).hexdigest()
