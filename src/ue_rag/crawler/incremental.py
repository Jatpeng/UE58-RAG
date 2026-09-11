"""Hash-based incremental Project RAG planning and pipeline orchestration."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ue_rag.crawler.project import ProjectFileRecord
from ue_rag.jsonl import load_jsonl


class ChangeKind(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class ProjectChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relative_path: str = Field(min_length=1)
    kind: ChangeKind
    previous_sha256: str | None = None
    current_sha256: str | None = None
    current: ProjectFileRecord | None = None


@dataclass(frozen=True)
class ProjectDiff:
    added: tuple[ProjectChange, ...]
    modified: tuple[ProjectChange, ...]
    deleted: tuple[ProjectChange, ...]
    unchanged: int

    @property
    def changed(self) -> tuple[ProjectChange, ...]:
        return tuple(sorted((*self.added, *self.modified), key=lambda item: item.relative_path.casefold()))

    @property
    def total(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted) + self.unchanged

    def to_jsonl(self) -> str:
        records = sorted(
            (*self.added, *self.modified, *self.deleted),
            key=lambda item: item.relative_path.casefold(),
        )
        return "".join(record.model_dump_json() + "\n" for record in records)


@dataclass(frozen=True)
class IncrementalRunSummary:
    changed_files: int
    deleted_files: int
    parsed_documents: int
    chunks: int
    embedded_chunks: int
    upserted_chunks: int
    deleted_vectors: int


class IncrementalProjectIndexer:
    """Run only changed files through caller-provided pipeline stages."""

    def __init__(
        self,
        *,
        parse: Callable[[ProjectFileRecord], Sequence[Any]],
        chunk: Callable[[Sequence[Any]], Sequence[Any]],
        embed: Callable[[Sequence[Any]], Any],
        upsert: Callable[[Sequence[Any], Any], int | None],
        delete: Callable[[str], int | None] | None = None,
    ) -> None:
        self.parse = parse
        self.chunk = chunk
        self.embed = embed
        self.upsert = upsert
        self.delete = delete

    def run(self, diff: ProjectDiff) -> IncrementalRunSummary:
        parsed_documents = chunks_count = embedded_count = upserted_count = 0
        deleted_vectors = 0
        for change in diff.deleted:
            if self.delete is not None:
                deleted_vectors += self.delete(change.relative_path) or 0
        for change in diff.changed:
            if change.current is None:
                continue
            documents = list(self.parse(change.current))
            parsed_documents += len(documents)
            chunks = list(self.chunk(documents))
            chunks_count += len(chunks)
            if not chunks:
                continue
            vectors = self.embed(chunks)
            embedded_count += len(chunks)
            upserted_count += self.upsert(chunks, vectors) or len(chunks)
        return IncrementalRunSummary(
            changed_files=len(diff.changed),
            deleted_files=len(diff.deleted),
            parsed_documents=parsed_documents,
            chunks=chunks_count,
            embedded_chunks=embedded_count,
            upserted_chunks=upserted_count,
            deleted_vectors=deleted_vectors,
        )


def diff_inventories(
    previous: Iterable[ProjectFileRecord], current: Iterable[ProjectFileRecord]
) -> ProjectDiff:
    """Compare inventories by relative path and SHA-256."""

    previous_by_path = {record.relative_path: record for record in previous}
    current_by_path = {record.relative_path: record for record in current}
    added: list[ProjectChange] = []
    modified: list[ProjectChange] = []
    deleted: list[ProjectChange] = []
    unchanged = 0
    for path in sorted(current_by_path):
        record = current_by_path[path]
        old = previous_by_path.get(path)
        if old is None:
            added.append(ProjectChange(relative_path=path, kind=ChangeKind.ADDED, current_sha256=record.sha256, current=record))
        elif old.sha256 != record.sha256:
            modified.append(ProjectChange(relative_path=path, kind=ChangeKind.MODIFIED, previous_sha256=old.sha256, current_sha256=record.sha256, current=record))
        else:
            unchanged += 1
    for path in sorted(set(previous_by_path) - set(current_by_path)):
        old = previous_by_path[path]
        deleted.append(ProjectChange(relative_path=path, kind=ChangeKind.DELETED, previous_sha256=old.sha256))
    return ProjectDiff(tuple(added), tuple(modified), tuple(deleted), unchanged)


def load_inventory(path: str | Path) -> list[ProjectFileRecord]:
    return load_jsonl(path, ProjectFileRecord)


def write_diff(path: str | Path, diff: ProjectDiff) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(diff.to_jsonl(), encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
