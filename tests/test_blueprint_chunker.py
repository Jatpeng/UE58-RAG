"""Tests for Blueprint RAG semantic chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ue_rag.blueprint import JsonBlueprintExporter
from ue_rag.chunker import BlueprintChunker, BlueprintChunkerConfig, load_blueprint_chunker_config
from ue_rag.jsonl import load_jsonl, save_jsonl
from ue_rag.schema import SourceScope, SourceType, UEChunk


def asset_payload() -> dict:
    return {
        "asset_name": "BP_Player",
        "package_path": "/Game/BP_Player",
        "parent_class": "Character",
        "graphs": [{"name": "EventGraph", "nodes": [{"id": str(i), "name": f"Node{i}", "type": "Call"} for i in range(30)], "links": []}],
        "functions": [{"name": "Jump", "graph_name": "EventGraph"}],
        "variables": [{"name": "Health", "type": "float"}],
        "components": [{"name": "Camera", "class": "CameraComponent"}],
    }


def make_config(tmp_path: Path, **overrides: int) -> BlueprintChunkerConfig:
    values = {"target_tokens": 40, "max_tokens": 60, "overlap_tokens": 5}
    values.update(overrides)
    return BlueprintChunkerConfig(
        engine_version="5.8",
        input_path=tmp_path / "documents.jsonl",
        output_path=tmp_path / "chunks.jsonl",
        **values,
    )


def blueprint_documents():
    from ue_rag.blueprint import BlueprintAsset

    return JsonBlueprintExporter("5.8").export_asset(BlueprintAsset.model_validate(asset_payload()))


def test_chunker_adds_blueprint_semantic_header(tmp_path: Path) -> None:
    document = blueprint_documents()[0]

    chunk = BlueprintChunker(make_config(tmp_path)).chunk_document(document)[0]

    assert "UE Version: 5.8" in chunk.content
    assert "Asset: BP_Player" in chunk.content
    assert "Symbol Type: blueprint_summary" in chunk.content
    assert "Parent Class: Character" in chunk.content
    assert chunk.source_scope is SourceScope.PROJECT
    assert chunk.source_type is SourceType.BLUEPRINT


def test_graph_is_split_by_node_lines_and_repeats_context(tmp_path: Path) -> None:
    graph = next(document for document in blueprint_documents() if document.metadata["export_kind"] == "graph")
    chunks = BlueprintChunker(make_config(tmp_path, target_tokens=30, max_tokens=45)).chunk_document(graph)

    assert len(chunks) > 1
    assert all("Asset: BP_Player" in chunk.content for chunk in chunks)
    assert all("Graph: EventGraph" in chunk.content for chunk in chunks)
    assert all(chunk.metadata["part_count"] == len(chunks) for chunk in chunks)
    assert [chunk.metadata["part_index"] for chunk in chunks] == list(range(len(chunks)))


def test_non_graph_views_remain_single_chunks(tmp_path: Path) -> None:
    documents = blueprint_documents()
    chunks = [chunk for document in documents if document.metadata["export_kind"] != "graph" for chunk in BlueprintChunker(make_config(tmp_path)).chunk_document(document)]

    assert len(chunks) == 4
    assert all(chunk.metadata["chunk_kind"] == "blueprint_view" for chunk in chunks)


def test_wrong_source_and_version_are_rejected(tmp_path: Path) -> None:
    chunker = BlueprintChunker(make_config(tmp_path))
    document = blueprint_documents()[0]
    with pytest.raises(ValueError, match="Expected blueprint"):
        chunker.chunk_document(document.model_copy(update={"source_type": SourceType.DOCS}))
    with pytest.raises(ValueError, match="does not match"):
        chunker.chunk_document(document.model_copy(update={"engine_version": "5.7"}))


def test_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    chunker = BlueprintChunker(make_config(tmp_path))
    document = blueprint_documents()[0]

    assert [chunk.id for chunk in chunker.chunk_document(document)] == [chunk.id for chunk in chunker.chunk_document(document)]


def test_corpus_output_is_deterministic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    save_jsonl(config.input_path, blueprint_documents())
    chunker = BlueprintChunker(config)

    first = chunker.chunk_corpus()
    bytes_first = config.output_path.read_bytes()
    second = chunker.chunk_corpus()

    assert first[:2] == second[:2]
    assert config.output_path.read_bytes() == bytes_first
    assert len(load_jsonl(config.output_path, UEChunk)) == first[1]


def test_config_uses_project_paths() -> None:
    config = load_blueprint_chunker_config()

    assert config.input_path == Path.cwd() / "data/parsed/project/blueprints.jsonl"
    assert config.output_path == Path.cwd() / "data/chunks/project/blueprints.jsonl"
