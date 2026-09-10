"""Heading-aware semantic chunking for parsed UE documentation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ue_rag.jsonl import load_jsonl, save_jsonl
from ue_rag.schema import SourceType, UEChunk, UEDocument


CHUNKER_VERSION = "1"
TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]"
)
HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
FENCE_PATTERN = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\n]*)$")


class TokenCounter(Protocol):
    """Replaceable token-count interface independent of embedding models."""

    def count(self, text: str) -> int:
        """Return a deterministic token estimate."""

    def head(self, text: str, limit: int) -> str:
        """Return at most ``limit`` leading estimated tokens."""

    def tail(self, text: str, limit: int) -> str:
        """Return at most ``limit`` trailing estimated tokens."""


class HeuristicTokenCounter:
    """Lightweight counter that handles CJK, identifiers, numbers, and punctuation."""

    def count(self, text: str) -> int:
        return sum(1 for _ in TOKEN_PATTERN.finditer(text))

    def head(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        matches = list(TOKEN_PATTERN.finditer(text))
        if len(matches) <= limit:
            return text.strip()
        return text[: matches[limit - 1].end()].strip()

    def tail(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        matches = list(TOKEN_PATTERN.finditer(text))
        if len(matches) <= limit:
            return text.strip()
        return text[matches[-limit].start() :].strip()


class DocumentationChunkerConfig(BaseModel):
    """Validated token and path configuration for documentation chunking."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    token_counter: str = "heuristic"
    target_tokens: int = Field(gt=0)
    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_token_limits(self) -> "DocumentationChunkerConfig":
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("token limits must satisfy min <= target <= max")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        return self


def load_docs_chunker_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    chunker_config_path: str | Path = "config/docs_chunker.yaml",
) -> DocumentationChunkerConfig:
    """Load data paths/version from UE config and chunk sizes from chunker config."""

    ue_path = Path(ue_config_path).resolve()
    chunker_path = Path(chunker_config_path).resolve()
    with ue_path.open(encoding="utf-8") as config_file:
        ue_config = yaml.safe_load(config_file)
    with chunker_path.open(encoding="utf-8") as config_file:
        chunker_config = yaml.safe_load(config_file)

    project_root = ue_path.parent.parent
    parsed_dir = Path(ue_config["data"]["parsed"])
    chunks_dir = Path(ue_config["data"]["chunks"])
    if not parsed_dir.is_absolute():
        parsed_dir = project_root / parsed_dir
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir

    return DocumentationChunkerConfig(
        engine_version=str(ue_config["engine"]["version"]),
        input_path=parsed_dir / "docs" / "documents.jsonl",
        output_path=chunks_dir / "docs" / "chunks.jsonl",
        **chunker_config,
    )


@dataclass(frozen=True)
class ChunkSummary:
    """Counters produced while chunking a documentation corpus."""

    documents: int
    chunks: int
    oversized_atomic_chunks: int
    output_path: Path


@dataclass
class _Section:
    path: list[tuple[int, str]]
    body_lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Atom:
    text: str
    is_code: bool = False


@dataclass(frozen=True)
class _ChunkBody:
    text: str
    oversized_atomic_block: bool = False


class DocumentationChunker:
    """Split Markdown by headings, then split only oversized semantic sections."""

    def __init__(
        self,
        config: DocumentationChunkerConfig,
        *,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if config.token_counter != "heuristic" and token_counter is None:
            raise ValueError(f"Unsupported token counter: {config.token_counter}")
        self.config = config
        self.token_counter = token_counter or HeuristicTokenCounter()

    def chunk_document(self, document: UEDocument) -> list[UEChunk]:
        """Create deterministic semantic chunks for one documentation page."""

        if document.source_type is not SourceType.DOCS:
            raise ValueError(f"Expected docs source, received {document.source_type.value}")
        if document.engine_version != self.config.engine_version:
            raise ValueError(
                f"Document engine version {document.engine_version!r} does not match "
                f"configured version {self.config.engine_version!r}: {document.id}"
            )

        sections = _split_sections(document.content, document.title or "Untitled")
        drafts: list[tuple[int, _Section, _ChunkBody]] = []
        for section_index, section in enumerate(sections):
            prefix = _section_prefix(section.path)
            body = "\n".join(section.body_lines).strip()
            if not body:
                continue
            full_content = _join_prefix_and_body(prefix, body)
            if self.token_counter.count(full_content) <= self.config.max_tokens:
                drafts.append((section_index, section, _ChunkBody(full_content)))
                continue
            for chunk_body in self._split_oversized_section(prefix, body):
                drafts.append((section_index, section, chunk_body))

        section_part_counts: dict[int, int] = {}
        for section_index, _, _ in drafts:
            section_part_counts[section_index] = section_part_counts.get(section_index, 0) + 1
        section_part_indices: dict[int, int] = {}
        chunks: list[UEChunk] = []
        for global_index, (section_index, section, draft) in enumerate(drafts):
            part_index = section_part_indices.get(section_index, 0)
            section_part_indices[section_index] = part_index + 1
            token_count = self.token_counter.count(draft.text)
            path_titles = [title for _, title in section.path]
            chunk_id = _chunk_id(document.id, section_index, part_index, draft.text)
            chunks.append(
                UEChunk(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_index=global_index,
                    engine_version=document.engine_version,
                    source_scope=document.source_scope,
                    source_type=document.source_type,
                    content=draft.text,
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
                    section_path=path_titles,
                    token_count=token_count,
                    metadata={
                        "page_title": document.title,
                        "section_path": path_titles,
                        "url": document.metadata.get("url"),
                        "topic": document.metadata.get("topic"),
                        "fetched_at": document.metadata.get("fetched_at"),
                        "parser_version": document.metadata.get("parser_version"),
                        "chunker_version": CHUNKER_VERSION,
                        "section_index": section_index,
                        "part_index": part_index,
                        "part_count": section_part_counts[section_index],
                        "estimated_tokens": token_count,
                        "oversized_atomic_block": draft.oversized_atomic_block,
                    },
                )
            )
        return chunks

    def chunk_corpus(self) -> ChunkSummary:
        """Chunk configured input JSONL and replace configured output deterministically."""

        documents = load_jsonl(self.config.input_path, UEDocument)
        chunks: list[UEChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        save_jsonl(self.config.output_path, chunks)
        return ChunkSummary(
            documents=len(documents),
            chunks=len(chunks),
            oversized_atomic_chunks=sum(
                bool(chunk.metadata["oversized_atomic_block"]) for chunk in chunks
            ),
            output_path=self.config.output_path,
        )

    def _split_oversized_section(self, prefix: str, body: str) -> list[_ChunkBody]:
        prefix_tokens = self.token_counter.count(prefix)
        max_body_tokens = max(1, self.config.max_tokens - prefix_tokens)
        target_body_tokens = max(1, self.config.target_tokens - prefix_tokens)
        min_body_tokens = max(1, self.config.min_tokens - prefix_tokens)
        atoms = _markdown_atoms(body)
        expanded: list[_Atom] = []
        for atom in atoms:
            if atom.is_code or self.token_counter.count(atom.text) <= max_body_tokens:
                expanded.append(atom)
            else:
                expanded.extend(self._split_plain_atom(atom, max_body_tokens))

        results: list[_ChunkBody] = []
        current: list[_Atom] = []
        current_tokens = 0
        for atom in expanded:
            atom_tokens = self.token_counter.count(atom.text)
            if atom.is_code and atom_tokens > max_body_tokens:
                if current:
                    results.append(_ChunkBody(_join_prefix_and_body(prefix, _join_atoms(current))))
                    current = []
                    current_tokens = 0
                results.append(
                    _ChunkBody(
                        _join_prefix_and_body(prefix, atom.text),
                        oversized_atomic_block=True,
                    )
                )
                continue

            exceeds_max = current and current_tokens + atom_tokens > max_body_tokens
            reached_target = (
                current
                and current_tokens >= min_body_tokens
                and current_tokens + atom_tokens > target_body_tokens
            )
            if exceeds_max or reached_target:
                results.append(_ChunkBody(_join_prefix_and_body(prefix, _join_atoms(current))))
                overlap_budget = min(
                    self.config.overlap_tokens,
                    max(0, max_body_tokens - atom_tokens),
                )
                current = self._overlap_tail(current, overlap_budget)
                current_tokens = sum(self.token_counter.count(item.text) for item in current)
            current.append(atom)
            current_tokens += atom_tokens

        if current:
            results.append(_ChunkBody(_join_prefix_and_body(prefix, _join_atoms(current))))
        return results

    def _split_plain_atom(self, atom: _Atom, limit: int) -> list[_Atom]:
        remaining = atom.text.strip()
        pieces: list[_Atom] = []
        while self.token_counter.count(remaining) > limit:
            head = self.token_counter.head(remaining, limit)
            if not head:
                break
            pieces.append(_Atom(head))
            remaining = remaining[len(head) :].strip()
        if remaining:
            pieces.append(_Atom(remaining))
        return pieces

    def _overlap_tail(self, atoms: list[_Atom], budget: int) -> list[_Atom]:
        if budget <= 0:
            return []
        selected: list[_Atom] = []
        remaining = budget
        for atom in reversed(atoms):
            atom_tokens = self.token_counter.count(atom.text)
            if atom.is_code:
                if atom_tokens <= remaining:
                    selected.append(atom)
                    remaining -= atom_tokens
                continue
            if atom_tokens <= remaining:
                selected.append(atom)
                remaining -= atom_tokens
            elif remaining:
                tail = self.token_counter.tail(atom.text, remaining)
                if tail:
                    selected.append(_Atom(tail))
                remaining = 0
            if remaining <= 0:
                break
        return list(reversed(selected))


def _split_sections(markdown: str, fallback_title: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = [(1, fallback_title)]
    current = _Section(path=[(1, fallback_title)])
    fence_marker: str | None = None

    for line in markdown.replace("\r\n", "\n").split("\n"):
        fence_match = FENCE_PATTERN.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_marker is None:
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                fence_marker = None
            current.body_lines.append(line)
            continue

        heading_match = HEADING_PATTERN.match(line) if fence_marker is None else None
        if not heading_match:
            current.body_lines.append(line)
            continue

        if any(part.strip() for part in current.body_lines):
            sections.append(current)
        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current = _Section(path=list(stack))

    if any(part.strip() for part in current.body_lines):
        sections.append(current)
    return sections


def _markdown_atoms(markdown: str) -> list[_Atom]:
    atoms: list[_Atom] = []
    current: list[str] = []
    fence_marker: str | None = None

    def flush(is_code: bool = False) -> None:
        text = "\n".join(current).strip()
        if text:
            atoms.append(_Atom(text=text, is_code=is_code))
        current.clear()

    for line in markdown.split("\n"):
        fence_match = FENCE_PATTERN.match(line)
        if fence_marker is not None:
            current.append(line)
            if fence_match:
                marker = fence_match.group(1)
                if marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                    flush(is_code=True)
                    fence_marker = None
            continue
        if fence_match:
            flush()
            fence_marker = fence_match.group(1)
            current.append(line)
            continue
        if not line.strip():
            flush()
            continue
        current.append(line)
    flush(is_code=fence_marker is not None)
    return atoms


def _section_prefix(path: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"{'#' * level} {title}" for level, title in path)


def _join_prefix_and_body(prefix: str, body: str) -> str:
    return f"{prefix.strip()}\n\n{body.strip()}".strip()


def _join_atoms(atoms: list[_Atom]) -> str:
    return "\n\n".join(atom.text.strip() for atom in atoms if atom.text.strip())


def _chunk_id(document_id: str, section_index: int, part_index: int, content: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}:{section_index}:{part_index}:{content}".encode("utf-8")
    ).hexdigest()
    return f"chunk-{digest}"
