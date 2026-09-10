"""Tests for exact symbol and SQLite FTS lexical retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ue_rag.index import QdrantConfig
from ue_rag.retrieval import LexicalIndex, load_lexical_config
from ue_rag.schema import SourceScope, SourceType, UEChunk


def make_chunk(
    chunk_id: str,
    symbol: str,
    content: str,
    *,
    class_name: str = "UCharacterMovementComponent",
    module: str = "Engine",
) -> UEChunk:
    return UEChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=0,
        engine_version="5.8",
        source_scope=SourceScope.GLOBAL,
        source_type=SourceType.ENGINE_SOURCE,
        content=content,
        title=symbol,
        module=module,
        plugin="Engine",
        file_path="Engine/Source/Runtime/Engine/CharacterMovementComponent.cpp",
        symbol=symbol,
        symbol_type="method",
        class_name=class_name,
        function_name=symbol.rsplit("::", 1)[-1],
        metadata={"ue_macros": [{"name": "UFUNCTION"}], "chunk_kind": "function"},
    )


def test_config_uses_chunk_and_index_paths() -> None:
    config = load_lexical_config()

    assert config.engine_version == "5.8"
    assert config.input_path == Path.cwd() / "data/chunks/engine/chunks.jsonl"
    assert config.index_path == Path.cwd() / "data/index/lexical.sqlite3"


def test_upsert_is_idempotent_and_updates_changed_content(tmp_path: Path) -> None:
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        chunk = make_chunk("chunk-1", "UCharacterMovementComponent::PerformMovement", "perform movement")

        first = index.upsert([chunk])
        repeat = index.upsert([chunk])
        changed = index.upsert([chunk.model_copy(update={"content": "new movement"})])

    assert (first.total, first.added, first.updated, first.skipped) == (1, 1, 0, 0)
    assert (repeat.total, repeat.added, repeat.updated, repeat.skipped) == (1, 0, 0, 1)
    assert (changed.total, changed.added, changed.updated, changed.skipped) == (1, 0, 1, 0)


def test_exact_symbol_search_prioritizes_symbol_over_class_and_function(tmp_path: Path) -> None:
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        index.upsert(
            [
                make_chunk("exact", "FNetworkPredictionData_Client_Character", "exact symbol"),
                make_chunk("class", "AHero::Tick", "class reference", class_name="FNetworkPredictionData_Client_Character"),
                make_chunk("other", "FOther::Tick", "other"),
            ]
        )
        results = index.search_symbol("FNetworkPredictionData_Client_Character", limit=5)

    assert [result.chunk_id for result in results[:2]] == ["exact", "class"]
    assert results[0].metadata["retrieval"] == "symbol"


def test_exact_symbol_search_includes_merged_property_aliases(tmp_path: Path) -> None:
    chunk = make_chunk("properties", "AHero::Properties", "AHero::MaxWalkSpeed: float MaxWalkSpeed;")
    chunk.metadata["field_symbols"] = ["AHero::MaxWalkSpeed"]
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        index.upsert([chunk])
        results = index.search_symbol("AHero::MaxWalkSpeed")

    assert [result.chunk_id for result in results] == ["properties"]


def test_lexical_search_matches_symbols_paths_macros_and_content(tmp_path: Path) -> None:
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        index.upsert(
            [
                make_chunk("movement", "UCharacterMovementComponent::PhysWalking", "walking prediction"),
                make_chunk("render", "FRenderer::Draw", "rendering pipeline", class_name="FRenderer", module="RenderCore"),
            ]
        )
        results = index.search_lexical("PhysWalking", limit=5)

    assert results[0].chunk_id == "movement"
    assert results[0].metadata["lexical_rank"] == 1


def test_hybrid_search_fills_exact_results_with_lexical_matches(tmp_path: Path) -> None:
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        index.upsert(
            [
                make_chunk("exact", "MaxWalkSpeed", "exact speed"),
                make_chunk("related", "GetMaxSpeed", "calls MaxWalkSpeed for the maximum walk speed"),
            ]
        )
        results = index.search("MaxWalkSpeed", limit=2)

    assert [result.chunk_id for result in results] == ["exact", "related"]


def test_filters_support_module_class_and_engine_version(tmp_path: Path) -> None:
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        index.upsert(
            [
                make_chunk("engine", "Engine::Tick", "tick", module="Engine"),
                make_chunk("other", "Other::Tick", "tick", module="Other", class_name="Other"),
            ]
        )
        results = index.search_lexical(
            "tick", filters={"engine_version": "5.8", "module": "Other", "class": "Other"}
        )

    assert [result.chunk_id for result in results] == ["other"]


def test_invalid_query_and_filter_are_rejected(tmp_path: Path) -> None:
    with LexicalIndex(tmp_path / "index.sqlite3") as index:
        with pytest.raises(ValueError, match="non-empty"):
            index.search_lexical(" ")
        with pytest.raises(ValueError, match="unsupported lexical filter"):
            index.search_symbol("Tick", filters={"unknown": "value"})


def test_index_jsonl_streams_batches_and_reloads(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    chunks = [make_chunk(f"chunk-{i}", f"AHero::Method{i}", f"method {i}") for i in range(5)]
    input_path.write_text("\n".join(chunk.model_dump_json() for chunk in chunks) + "\n", encoding="utf-8")
    index_path = tmp_path / "index.sqlite3"

    with LexicalIndex(index_path) as index:
        summary = index.index_jsonl(input_path, batch_size=2)
    with LexicalIndex(index_path) as index:
        results = index.search_symbol("AHero::Method3")

    assert summary.total == summary.added == 5
    assert results[0].chunk_id == "chunk-3"
