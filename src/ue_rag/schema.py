"""Shared data contracts for every Unreal Engine RAG source."""

from __future__ import annotations

from enum import Enum
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SourceScope(str, Enum):
    """Whether a record belongs to the shared UE corpus or a local project."""

    GLOBAL = "global"
    PROJECT = "project"


class SourceType(str, Enum):
    """Supported sources that can contribute records to the RAG corpus."""

    DOCS = "docs"
    API = "api"
    ENGINE_SOURCE = "engine_source"
    LYRA = "lyra"
    PROJECT_SOURCE = "project_source"
    BLUEPRINT = "blueprint"


class _SourceRecord(BaseModel):
    """Metadata shared by documents and their derived chunks."""

    model_config = ConfigDict(extra="forbid")

    engine_version: NonEmptyString
    source_scope: SourceScope
    source_type: SourceType
    content: str
    title: str | None = None
    module: str | None = None
    plugin: str | None = None
    file_path: str | None = None
    symbol: str | None = None
    symbol_type: str | None = None
    class_name: str | None = None
    function_name: str | None = None
    asset_name: str | None = None
    graph_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UEDocument(_SourceRecord):
    """A parsed source unit before semantic chunking."""

    id: NonEmptyString


class UEChunk(_SourceRecord):
    """A retrieval-ready semantic unit derived from a :class:`UEDocument`."""

    id: NonEmptyString
    document_id: NonEmptyString
    chunk_index: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    token_count: int | None = Field(default=None, ge=0)


class RetrievalResult(BaseModel):
    """A common result shape for dense, lexical, hybrid, and reranked search."""

    model_config = ConfigDict(extra="forbid")

    document_id: NonEmptyString
    chunk_id: str | None = None
    engine_version: NonEmptyString
    source_type: SourceType
    content: str
    score: float
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    fusion_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
