"""Tests for heading-aware documentation semantic chunking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ue_rag.chunker.docs import (
    DocumentationChunker,
    DocumentationChunkerConfig,
    HeuristicTokenCounter,
    load_docs_chunker_config,
)
from ue_rag.jsonl import load_jsonl, save_jsonl
from ue_rag.schema import SourceScope, SourceType, UEChunk, UEDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path: Path, **overrides: int) -> DocumentationChunkerConfig:
    values = {
        "target_tokens": 80,
        "min_tokens": 50,
        "max_tokens": 120,
        "overlap_tokens": 12,
    }
    values.update(overrides)
    return DocumentationChunkerConfig(
        engine_version="5.8",
        input_path=tmp_path / "documents.jsonl",
        output_path=tmp_path / "chunks.jsonl",
        token_counter="heuristic",
        **values,
    )


def make_document(content: str, *, document_id: str = "docs-1") -> UEDocument:
    return UEDocument(
        id=document_id,
        engine_version="5.8",
        source_scope=SourceScope.GLOBAL,
        source_type=SourceType.DOCS,
        content=content,
        title="Movement Guide",
        file_path="pages/example.html",
        metadata={
            "url": "https://dev.epicgames.com/documentation/example",
            "topic": "gameplay",
            "fetched_at": "2026-09-10T03:00:00+00:00",
            "parser_version": "1",
        },
    )


def test_heading_hierarchy_creates_semantic_sections(tmp_path: Path) -> None:
    document = make_document(
        """# Movement Guide

Page introduction.

## Character Movement

Movement component overview.

### Walking

Walking behavior.

## Networking

Prediction overview.
"""
    )

    chunks = DocumentationChunker(make_config(tmp_path)).chunk_document(document)

    assert [chunk.section_path for chunk in chunks] == [
        ["Movement Guide"],
        ["Movement Guide", "Character Movement"],
        ["Movement Guide", "Character Movement", "Walking"],
        ["Movement Guide", "Networking"],
    ]
    assert chunks[2].content.startswith(
        "# Movement Guide\n\n## Character Movement\n\n### Walking"
    )
    assert all(chunk.metadata["page_title"] == "Movement Guide" for chunk in chunks)
    assert all(chunk.metadata["url"] for chunk in chunks)


def test_section_under_max_is_not_split_even_when_over_target(tmp_path: Path) -> None:
    content = "# Movement Guide\n\n## Details\n\n" + " ".join(
        f"token_{index}" for index in range(35)
    )
    config = make_config(
        tmp_path,
        target_tokens=20,
        min_tokens=10,
        max_tokens=50,
        overlap_tokens=5,
    )

    chunks = DocumentationChunker(config).chunk_document(make_document(content))

    assert len(chunks) == 1
    assert chunks[0].section_path == ["Movement Guide", "Details"]
    assert chunks[0].token_count <= 50


def test_oversized_section_splits_near_target_with_overlap(tmp_path: Path) -> None:
    paragraphs = [
        f"Paragraph_{index} alpha beta gamma delta epsilon zeta eta theta."
        for index in range(30)
    ]
    content = "# Movement Guide\n\n## Large Section\n\n" + "\n\n".join(paragraphs)
    config = make_config(
        tmp_path,
        target_tokens=45,
        min_tokens=30,
        max_tokens=65,
        overlap_tokens=10,
    )
    counter = HeuristicTokenCounter()

    chunks = DocumentationChunker(config, token_counter=counter).chunk_document(
        make_document(content)
    )

    assert len(chunks) > 3
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    assert all(
        chunk.content.startswith("# Movement Guide\n\n## Large Section")
        for chunk in chunks
    )
    assert all(chunk.metadata["part_count"] == len(chunks) for chunk in chunks)
    first_body = chunks[0].content.split("## Large Section", maxsplit=1)[1]
    expected_overlap = counter.tail(first_body, config.overlap_tokens)
    assert expected_overlap in chunks[1].content


def test_fenced_code_is_never_split_mid_block(tmp_path: Path) -> None:
    code_lines = [f"int value_{index} = {index};" for index in range(80)]
    code = "```cpp\n" + "\n".join(code_lines) + "\n```"
    content = "# Movement Guide\n\n## Source\n\nBefore code.\n\n" + code
    config = make_config(
        tmp_path,
        target_tokens=35,
        min_tokens=20,
        max_tokens=50,
        overlap_tokens=8,
    )

    chunks = DocumentationChunker(config).chunk_document(make_document(content))
    code_chunks = [chunk for chunk in chunks if "value_79" in chunk.content]

    assert len(code_chunks) == 1
    assert all(line in code_chunks[0].content for line in code_lines)
    assert code_chunks[0].content.count("```") == 2
    assert code_chunks[0].metadata["oversized_atomic_block"] is True
    assert code_chunks[0].token_count > config.max_tokens


def test_corpus_output_is_deterministic_jsonl(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    documents = [
        make_document("# Movement Guide\n\nIntroduction."),
        make_document(
            "# Other Guide\n\n## Rendering\n\nRendering details.",
            document_id="docs-2",
        ),
    ]
    documents[1].title = "Other Guide"
    save_jsonl(config.input_path, documents)
    chunker = DocumentationChunker(config)

    first = chunker.chunk_corpus()
    first_bytes = config.output_path.read_bytes()
    second = chunker.chunk_corpus()
    chunks = load_jsonl(config.output_path, UEChunk)

    assert first.documents == second.documents == 2
    assert first.chunks == second.chunks == len(chunks)
    assert config.output_path.read_bytes() == first_bytes
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_heuristic_counter_handles_chinese_and_cpp_identifiers() -> None:
    counter = HeuristicTokenCounter()

    assert counter.count("角色移动") == 4
    assert counter.count("UCharacterMovementComponent::MaxWalkSpeed") == 4


def test_chunker_config_enforces_ordered_limits(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="min <= target <= max"):
        make_config(
            tmp_path,
            min_tokens=100,
            target_tokens=80,
            max_tokens=90,
        )


def test_chunker_config_uses_ue_paths_and_version() -> None:
    config = load_docs_chunker_config()

    assert config.engine_version == "5.8"
    assert config.input_path == (
        PROJECT_ROOT / "data" / "parsed" / "docs" / "documents.jsonl"
    )
    assert config.output_path == (
        PROJECT_ROOT / "data" / "chunks" / "docs" / "chunks.jsonl"
    )
    assert 100 <= config.overlap_tokens <= 150
