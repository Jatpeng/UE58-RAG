"""Tests for the shared UE data contracts and JSONL persistence."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ue_rag.jsonl import load_jsonl, save_jsonl
from ue_rag.schema import RetrievalResult, SourceScope, SourceType, UEChunk, UEDocument


@pytest.mark.parametrize(
    ("source_type", "source_fields"),
    [
        (
            SourceType.DOCS,
            {
                "title": "Character Movement Component",
                "metadata": {"url": "https://dev.epicgames.com/documentation/"},
            },
        ),
        (
            SourceType.ENGINE_SOURCE,
            {
                "module": "Engine",
                "file_path": "Runtime/Engine/CharacterMovementComponent.cpp",
                "class_name": "UCharacterMovementComponent",
                "function_name": "PerformMovement",
                "symbol": "UCharacterMovementComponent::PerformMovement",
                "symbol_type": "method",
            },
        ),
        (
            SourceType.BLUEPRINT,
            {
                "source_scope": SourceScope.PROJECT,
                "asset_name": "BP_PlayerCharacter",
                "graph_name": "EventGraph",
            },
        ),
    ],
)
def test_document_represents_supported_source_kinds(
    source_type: SourceType, source_fields: dict[str, object]
) -> None:
    """Documentation, C++, and Blueprint records share one validated shape."""

    scope = source_fields.pop("source_scope", SourceScope.GLOBAL)
    document = UEDocument(
        id=f"record-{source_type.value}",
        engine_version="5.8",
        source_scope=scope,
        source_type=source_type,
        content="Example Unreal Engine content",
        **source_fields,
    )

    assert document.engine_version == "5.8"
    assert document.source_type is source_type


def test_engine_version_is_required_and_non_empty() -> None:
    """Records cannot silently lose their UE version provenance."""

    common = {
        "id": "docs-1",
        "source_scope": SourceScope.GLOBAL,
        "source_type": SourceType.DOCS,
        "content": "Movement documentation",
    }

    with pytest.raises(ValidationError):
        UEDocument(**common)

    with pytest.raises(ValidationError):
        UEDocument(engine_version="   ", **common)


def test_source_type_rejects_unknown_values() -> None:
    """Source names remain consistent across future pipeline stages."""

    with pytest.raises(ValidationError):
        UEDocument(
            id="unknown-1",
            engine_version="5.8",
            source_scope=SourceScope.GLOBAL,
            source_type="web_page",
            content="Unknown source",
        )


def test_chunk_and_retrieval_result_share_provenance() -> None:
    """Chunks and results retain the fields needed to trace their source."""

    chunk = UEChunk(
        id="chunk-1",
        document_id="cpp-1",
        chunk_index=0,
        engine_version="5.8",
        source_scope=SourceScope.GLOBAL,
        source_type=SourceType.ENGINE_SOURCE,
        content="void PerformMovement(float DeltaSeconds);",
        module="Engine",
        class_name="UCharacterMovementComponent",
        symbol="UCharacterMovementComponent::PerformMovement",
        section_path=["UCharacterMovementComponent", "PerformMovement"],
    )
    result = RetrievalResult(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        engine_version=chunk.engine_version,
        source_type=chunk.source_type,
        content=chunk.content,
        score=0.92,
        dense_rank=2,
        sparse_rank=1,
        fusion_score=0.0325,
        metadata={"symbol": chunk.symbol},
    )

    assert result.chunk_id == "chunk-1"
    assert result.metadata["symbol"] == "UCharacterMovementComponent::PerformMovement"


def test_jsonl_round_trip_preserves_unicode_and_enum_values(tmp_path: Path) -> None:
    """JSONL persistence round-trips validated models without losing Unicode."""

    path = tmp_path / "nested" / "documents.jsonl"
    documents = [
        UEDocument(
            id="docs-zh-1",
            engine_version="5.8",
            source_scope=SourceScope.GLOBAL,
            source_type=SourceType.DOCS,
            content="角色最大移动速度在哪里控制？",
            title="角色移动",
        ),
        UEDocument(
            id="bp-1",
            engine_version="5.8",
            source_scope=SourceScope.PROJECT,
            source_type=SourceType.BLUEPRINT,
            content="Blueprint graph summary",
            asset_name="BP_PlayerCharacter",
            graph_name="EventGraph",
        ),
    ]

    save_jsonl(path, documents)
    loaded = load_jsonl(path, UEDocument)

    assert loaded == documents
    assert "角色最大移动速度" in path.read_text(encoding="utf-8")


def test_load_jsonl_reports_the_invalid_line(tmp_path: Path) -> None:
    """Malformed corpus data identifies the failing line for quick diagnosis."""

    path = tmp_path / "broken.jsonl"
    path.write_text("\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 2"):
        load_jsonl(path, UEDocument)
