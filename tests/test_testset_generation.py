"""Tests for bounded RAGAS source preparation and benchmark export."""

from __future__ import annotations

import json
from pathlib import Path

from ue_rag.eval import TestsetConfig as GenerationConfig
from ue_rag.eval import load_cases, sample_sources, write_generated_testset
from scripts.generate_testset import _is_local_base_url


def _chunk(index: int, *, module: str) -> dict[str, object]:
    return {
        "id": f"chunk-{index}",
        "document_id": f"doc-{index}",
        "chunk_index": 0,
        "engine_version": "5.8",
        "source_scope": "global",
        "source_type": "engine_source",
        "content": "Source code for movement speed. " * 20,
        "module": module,
        "symbol": f"AThing::{module}{index}",
        "symbol_type": "method",
    }


def _config(tmp_path: Path, input_path: Path) -> GenerationConfig:
    return GenerationConfig.model_validate(
        {
            "input": input_path,
            "sources_output": tmp_path / "sources.jsonl",
            "testset_output": tmp_path / "testset.jsonl",
            "benchmark_output": tmp_path / "cases.jsonl",
            "documents": 4,
            "testset_size": 2,
            "seed": 58,
            "min_chars": 20,
            "max_chars": 2000,
            "max_per_stratum": 2,
            "llm": {"model": "fake"},
            "embedding": {"model": "fake", "device": "cpu"},
        }
    )


def test_sampling_is_bounded_balanced_and_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "chunks.jsonl"
    rows = [_chunk(index, module="Engine" if index < 6 else "Niagara") for index in range(12)]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = _config(tmp_path, input_path)

    first = sample_sources(config)
    second = sample_sources(config)

    assert [item.metadata["chunk_id"] for item in first] == [item.metadata["chunk_id"] for item in second]
    assert len(first) == 4
    assert {item.metadata["module"] for item in first} == {"Engine", "Niagara"}
    assert all("UE_RAG_CHUNK_ID:" in item.page_content for item in first)


def test_generated_rows_export_to_existing_benchmark_format(tmp_path: Path) -> None:
    raw_path = tmp_path / "ragas.jsonl"
    cases_path = tmp_path / "cases.jsonl"
    rows = [
        {
            "user_input": "How is max speed selected?",
            "reference_contexts": [
                "UE_RAG_CHUNK_ID: chunk-1\nUE_RAG_SYMBOL: UCharacterMovementComponent::GetMaxSpeed\nsource"
            ],
            "reference": "It depends on the movement mode.",
            "synthesizer_name": "single_hop_specific_query_synthesizer",
        }
    ]

    generated, benchmark = write_generated_testset(
        rows, testset_output=raw_path, benchmark_output=cases_path
    )
    cases = load_cases([cases_path])

    assert generated == benchmark == 1
    assert cases[0].expected_chunk_ids == ["chunk-1"]
    assert cases[0].expected_symbols == ["UCharacterMovementComponent::GetMaxSpeed"]


def test_only_loopback_endpoints_may_use_placeholder_api_keys() -> None:
    assert _is_local_base_url("http://127.0.0.1:11434/v1")
    assert _is_local_base_url("http://localhost:8000/v1")
    assert not _is_local_base_url("https://api.deepseek.com")
    assert not _is_local_base_url(None)
