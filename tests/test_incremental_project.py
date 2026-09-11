"""Tests for hash-based Project RAG incremental plans."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ue_rag.crawler import (
    ChangeKind,
    IncrementalProjectIndexer,
    ProjectFileRecord,
    diff_inventories,
    write_diff,
)
from ue_rag.schema import SourceType


def record(path: str, sha: str, *, source_type: SourceType = SourceType.PROJECT_SOURCE) -> ProjectFileRecord:
    return ProjectFileRecord(
        relative_path=path,
        root_kind="source",
        module="Game",
        plugin=None,
        file_type="cpp",
        source_type=source_type,
        engine_version="5.8",
        size=10,
        sha256=sha * 64,
    )


def test_diff_classifies_added_modified_deleted_and_unchanged() -> None:
    previous = [record("Source/A.cpp", "a"), record("Source/B.cpp", "b"), record("Source/Deleted.cpp", "c")]
    current = [record("Source/A.cpp", "a"), record("Source/B.cpp", "d"), record("Source/New.cpp", "e")]

    diff = diff_inventories(previous, current)

    assert [item.relative_path for item in diff.added] == ["Source/New.cpp"]
    assert [item.relative_path for item in diff.modified] == ["Source/B.cpp"]
    assert [item.relative_path for item in diff.deleted] == ["Source/Deleted.cpp"]
    assert diff.unchanged == 1
    assert diff.total == 4
    assert diff.modified[0].previous_sha256 == "b" * 64


def test_diff_is_deterministic_and_jsonl_excludes_unchanged(tmp_path: Path) -> None:
    diff = diff_inventories([record("A.cpp", "a")], [record("A.cpp", "b"), record("B.cpp", "c")])
    path = tmp_path / "plan.jsonl"
    write_diff(path, diff)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["relative_path"] for line in lines] == ["A.cpp", "B.cpp"]
    assert all('"unchanged"' not in line for line in lines)


def test_incremental_indexer_runs_only_changed_and_deletes() -> None:
    diff = diff_inventories(
        [record("A.cpp", "a"), record("Deleted.cpp", "b")],
        [record("A.cpp", "c"), record("New.cpp", "d")],
    )
    calls: list[str] = []

    indexer = IncrementalProjectIndexer(
        parse=lambda item: calls.append("parse:" + item.relative_path) or [item],
        chunk=lambda docs: calls.append("chunk:" + str(len(docs))) or docs,
        embed=lambda chunks: calls.append("embed:" + str(len(chunks))) or np.ones((len(chunks), 2)),
        upsert=lambda chunks, vectors: calls.append("upsert:" + str(len(chunks))) or len(chunks),
        delete=lambda path: calls.append("delete:" + path) or 2,
    )

    summary = indexer.run(diff)

    assert summary.changed_files == 2
    assert summary.deleted_files == 1
    assert summary.parsed_documents == summary.chunks == summary.embedded_chunks == summary.upserted_chunks == 2
    assert summary.deleted_vectors == 2
    assert calls == [
        "delete:Deleted.cpp",
        "parse:A.cpp",
        "chunk:1",
        "embed:1",
        "upsert:1",
        "parse:New.cpp",
        "chunk:1",
        "embed:1",
        "upsert:1",
    ]


def test_incremental_indexer_skips_empty_parse_results() -> None:
    diff = diff_inventories([], [record("A.cpp", "a")])
    embedded = []
    indexer = IncrementalProjectIndexer(
        parse=lambda _: [],
        chunk=lambda _: [],
        embed=lambda chunks: embedded.append(chunks),
        upsert=lambda chunks, vectors: 0,
    )

    summary = indexer.run(diff)

    assert summary.parsed_documents == summary.chunks == 0
    assert embedded == []


def test_diff_preserves_current_record_for_added_and_modified() -> None:
    diff = diff_inventories([record("A.cpp", "a")], [record("A.cpp", "b"), record("B.cpp", "c")])

    assert diff.modified[0].current is not None
    assert diff.modified[0].current.sha256 == "b" * 64
    assert diff.added[0].kind is ChangeKind.ADDED
