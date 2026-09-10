"""Tests for Unreal C++ semantic chunking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ue_rag.chunker.cpp import (
    CPPSemanticChunker,
    CPPChunkerConfig,
    load_cpp_chunker_config,
)
from ue_rag.jsonl import load_jsonl, save_jsonl
from ue_rag.schema import SourceScope, SourceType, UEChunk, UEDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, **overrides: int | str) -> CPPChunkerConfig:
    values: dict[str, int | str] = {
        "target_tokens": 30,
        "max_tokens": 45,
        "overlap_tokens": 5,
        "property_grouping": "class",
    }
    values.update(overrides)
    return CPPChunkerConfig(
        engine_version="5.8",
        input_path=tmp_path / "documents.jsonl",
        output_path=tmp_path / "chunks.jsonl",
        token_counter="heuristic",
        **values,
    )


def make_document(
    symbol: str,
    symbol_type: str,
    content: str,
    *,
    document_id: str | None = None,
    class_name: str | None = "AHero",
    file_path: str = "Engine/Source/Runtime/Hero/Hero.h",
    metadata: dict | None = None,
) -> UEDocument:
    return UEDocument(
        id=document_id or f"cpp-{symbol.replace(':', '-')}",
        engine_version="5.8",
        source_scope=SourceScope.GLOBAL,
        source_type=SourceType.ENGINE_SOURCE,
        content=content,
        title=symbol,
        module="Hero",
        plugin=None,
        file_path=file_path,
        symbol=symbol,
        symbol_type=symbol_type,
        class_name=class_name,
        function_name=symbol.rsplit("::", 1)[-1] if symbol_type in {"function", "method", "constructor"} else None,
        metadata={"line_start": 10, "line_end": 20, "inheritance": ["UObject"], **(metadata or {})},
    )


def test_function_chunk_contains_required_semantic_header(tmp_path: Path) -> None:
    document = make_document("AHero::PerformMovement", "method", "void AHero::PerformMovement() {}")

    chunk = CPPSemanticChunker(make_config(tmp_path)).chunk_document(document)[0]

    assert "UE Version: 5.8" in chunk.content
    assert "Module: Hero" in chunk.content
    assert "Class: AHero" in chunk.content
    assert "Symbol: AHero::PerformMovement" in chunk.content
    assert "Symbol Type: method" in chunk.content
    assert "File: Engine/Source/Runtime/Hero/Hero.h" in chunk.content
    assert "Source:\nvoid AHero::PerformMovement() {}" in chunk.content


def test_function_is_one_chunk_when_under_limit(tmp_path: Path) -> None:
    document = make_document("AHero::Tick", "method", "void AHero::Tick() { Value += 1; }")

    chunks = CPPSemanticChunker(
        make_config(tmp_path, target_tokens=60, max_tokens=80)
    ).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_kind"] == "function"
    assert chunks[0].metadata["part_count"] == 1


@pytest.mark.parametrize("symbol_type", ["class", "struct", "enum"])
def test_type_symbols_are_single_standalone_chunks(tmp_path: Path, symbol_type: str) -> None:
    document = make_document("FThing", symbol_type, f"{symbol_type} FThing {{ int Value; }};", class_name=None)

    chunks = CPPSemanticChunker(make_config(tmp_path)).chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_kind"] == "type"
    assert chunks[0].symbol_type == symbol_type


def test_properties_merge_by_class(tmp_path: Path) -> None:
    fields = [
        make_document("AHero::MaxSpeed", "field", "UPROPERTY() float MaxSpeed;", document_id="field-1"),
        make_document("AHero::Acceleration", "field", "UPROPERTY() float Acceleration;", document_id="field-2"),
    ]

    chunk = CPPSemanticChunker(make_config(tmp_path)).chunk_property_group(fields)

    assert chunk.symbol == "AHero::Properties"
    assert chunk.symbol_type == "property_group"
    assert "AHero::MaxSpeed" in chunk.content
    assert "AHero::Acceleration" in chunk.content
    assert chunk.metadata["field_symbols"] == ["AHero::MaxSpeed", "AHero::Acceleration"]
    assert chunk.metadata["field_document_ids"] == ["field-1", "field-2"]


def test_property_group_uses_class_document_as_context_id(tmp_path: Path) -> None:
    field = make_document("AHero::Health", "field", "int Health;", document_id="field-1")
    chunk = CPPSemanticChunker(make_config(tmp_path)).chunk_property_group([field])

    assert chunk.document_id.startswith("cpp-properties-")
    assert chunk.class_name == "AHero"
    assert "Class: AHero" in chunk.content


def test_property_grouping_none_keeps_each_field(tmp_path: Path) -> None:
    config = make_config(tmp_path, property_grouping="none")
    fields = [make_document("AHero::A", "field", "int A;"), make_document("AHero::B", "field", "int B;")]

    chunks = []
    for field in fields:
        chunks.extend(CPPSemanticChunker(config).chunk_document(field))

    assert len(chunks) == 2
    assert all(chunk.symbol_type == "field" for chunk in chunks)


def test_oversized_function_splits_and_repeats_context(tmp_path: Path) -> None:
    body = "void AHero::PerformMovement() {\n" + "\n".join(
        f"    Position += Step_{index};" for index in range(40)
    ) + "\n}"
    document = make_document("AHero::PerformMovement", "method", body)
    config = make_config(tmp_path, target_tokens=25, max_tokens=35, overlap_tokens=4)

    chunks = CPPSemanticChunker(config).chunk_document(document)

    assert len(chunks) > 1
    assert all("Class: AHero" in chunk.content for chunk in chunks)
    assert all("Function" not in chunk.content or "PerformMovement" in chunk.content for chunk in chunks)
    assert [chunk.metadata["part_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["part_count"] == len(chunks) for chunk in chunks)


def test_large_single_line_function_is_split_without_empty_chunks(tmp_path: Path) -> None:
    body = "void AHero::Move() { " + " ".join(f"Step({index});" for index in range(200)) + " }"
    chunks = CPPSemanticChunker(make_config(tmp_path, target_tokens=20, max_tokens=30)).chunk_document(
        make_document("AHero::Move", "method", body)
    )

    assert len(chunks) > 1
    assert all(chunk.content.strip() for chunk in chunks)


def test_metadata_and_provenance_are_carried_to_chunks(tmp_path: Path) -> None:
    document = make_document("AHero", "class", "class AHero : public UObject {};", metadata={"api_macro": "HERO_API"})

    chunk = CPPSemanticChunker(make_config(tmp_path)).chunk_document(document)[0]

    assert chunk.metadata["api_macro"] == "HERO_API"
    assert chunk.metadata["inheritance"] == ["UObject"]
    assert chunk.section_path == ["AHero"]


def test_wrong_source_type_is_rejected(tmp_path: Path) -> None:
    document = make_document("Guide", "class", "class Guide {}; ").model_copy(
        update={"source_type": SourceType.DOCS}
    )

    with pytest.raises(ValueError, match="Expected engine_source"):
        CPPSemanticChunker(make_config(tmp_path)).chunk_document(document)


def test_wrong_engine_version_is_rejected(tmp_path: Path) -> None:
    document = make_document("AHero", "class", "class AHero {}; ").model_copy(
        update={"engine_version": "5.7"}
    )

    with pytest.raises(ValueError, match="does not match configured version"):
        CPPSemanticChunker(make_config(tmp_path)).chunk_document(document)


def test_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    document = make_document("AHero::Tick", "method", "void AHero::Tick() {}")
    chunker = CPPSemanticChunker(make_config(tmp_path))

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


def test_corpus_streams_files_and_merges_only_same_file_class(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    documents = [
        make_document("AHero", "class", "class AHero {};", document_id="class-a"),
        make_document("AHero::A", "field", "int A;", document_id="field-a"),
        make_document("AHero::Tick", "method", "void Tick() {}", document_id="method-a"),
        make_document("BOther::B", "field", "int B;", document_id="field-b", file_path="Engine/Source/Runtime/Other/Other.h", class_name="BOther"),
    ]
    save_jsonl(config.input_path, documents)

    summary = CPPSemanticChunker(config).chunk_corpus()
    chunks = load_jsonl(config.output_path, UEChunk)

    assert summary.documents == 4
    assert summary.chunks == len(chunks) == 4
    assert summary.property_groups == 2
    assert len({chunk.id for chunk in chunks}) == 4


def test_corpus_output_is_deterministic(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    save_jsonl(config.input_path, [make_document("AHero", "class", "class AHero {};", document_id="class-a")])
    chunker = CPPSemanticChunker(config)

    first = chunker.chunk_corpus()
    first_bytes = config.output_path.read_bytes()
    second = chunker.chunk_corpus()

    assert first.documents == second.documents == 1
    assert first.chunks == second.chunks == 1
    assert config.output_path.read_bytes() == first_bytes


def test_config_uses_engine_paths_and_version() -> None:
    config = load_cpp_chunker_config()

    assert config.engine_version == "5.8"
    assert config.input_path == PROJECT_ROOT / "data" / "parsed" / "engine" / "documents.jsonl"
    assert config.output_path == PROJECT_ROOT / "data" / "chunks" / "engine" / "chunks.jsonl"


def test_config_rejects_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="target_tokens must not exceed max_tokens"):
        make_config(tmp_path, target_tokens=50, max_tokens=40)
