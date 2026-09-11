"""Tests for retrieval benchmark metrics and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ue_rag.eval import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkEvaluator,
    MetricScores,
    load_benchmark_config,
    load_cases,
)
from ue_rag.schema import RetrievalResult, SourceType


def make_case(case_id: str = "case-1", *, category: str = "symbol") -> BenchmarkCase:
    return BenchmarkCase(
        id=case_id,
        query="movement speed",
        category=category,
        expected_symbols=["UCharacterMovementComponent::MaxWalkSpeed"],
    )


def make_result(chunk_id: str, symbol: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        engine_version="5.8",
        source_type=SourceType.ENGINE_SOURCE,
        content=symbol,
        score=1.0,
        metadata={"symbol": symbol},
    )


def make_config(tmp_path: Path, **overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "engine_version": "5.8",
        "output_json": tmp_path / "results.json",
        "output_markdown": tmp_path / "results.md",
        "baseline": {},
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_case_accepts_symbol_chunk_or_document_expectations() -> None:
    assert BenchmarkCase(id="a", query="x", expected_symbols=["Symbol"]).expected_symbols == ["Symbol"]
    assert BenchmarkCase(id="b", query="x", expected_chunk_ids=["chunk"]).expected_chunk_ids == ["chunk"]
    assert BenchmarkCase(id="c", query="x", expected_document_ids=["doc"]).expected_document_ids == ["doc"]


def test_case_requires_at_least_one_expected_value() -> None:
    with pytest.raises(ValidationError, match="at least one expected"):
        BenchmarkCase(id="a", query="x")


def test_metrics_hit_recall_mrr_and_ndcg() -> None:
    results = [
        make_result("wrong", "Other::Thing"),
        make_result("right", "UCharacterMovementComponent::MaxWalkSpeed"),
    ]
    config = make_config(Path("."))
    report = BenchmarkEvaluator(lambda *_args, **_kwargs: results, config).evaluate([make_case()])

    assert report.overall.hit_at_1 == 0
    assert report.overall.hit_at_3 == 1
    assert report.overall.recall_at_5 == 1
    assert report.overall.recall_at_10 == 1
    assert report.overall.mrr == 0.5
    assert report.overall.ndcg > 0


def test_field_aliases_count_as_relevant() -> None:
    case = BenchmarkCase(id="a", query="speed", expected_symbols=["AHero::MaxWalkSpeed"])
    result = make_result("properties", "AHero::Properties").model_copy(
        update={"metadata": {"field_symbols": ["AHero::MaxWalkSpeed"]}}
    )
    report = BenchmarkEvaluator(lambda *_args, **_kwargs: [result], make_config(Path("."))).evaluate([case])

    assert report.overall.hit_at_1 == 1


def test_category_breakdown_is_macro_averaged() -> None:
    cases = [make_case("a", category="symbol"), make_case("b", category="semantic")]
    responses = {
        "movement speed": [make_result("right", "UCharacterMovementComponent::MaxWalkSpeed")]
    }
    report = BenchmarkEvaluator(lambda *_args, **_kwargs: responses["movement speed"], make_config(Path("."))).evaluate(cases)

    assert set(report.by_category) == {"symbol", "semantic"}
    assert report.cases == 2


def test_baseline_passes_or_fails_report() -> None:
    search = lambda *_args, **_kwargs: [make_result("wrong", "Other::Thing")]
    passing = BenchmarkEvaluator(search, make_config(Path("."), baseline={"mrr": 0.0})).evaluate([make_case()])
    failing = BenchmarkEvaluator(search, make_config(Path("."), baseline={"mrr": 0.1})).evaluate([make_case()])

    assert passing.baseline_passed is True
    assert failing.baseline_passed is False
    assert "mrr" in failing.baseline_failures


def test_unknown_baseline_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown baseline metric"):
        BenchmarkEvaluator(lambda *_args, **_kwargs: [], make_config(Path("."), baseline={"bad": 0.1})).evaluate([make_case()])


def test_report_writes_json_and_markdown(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    evaluator = BenchmarkEvaluator(
        lambda *_args, **_kwargs: [make_result("right", "UCharacterMovementComponent::MaxWalkSpeed")],
        config,
    )
    report = evaluator.evaluate([make_case()])
    evaluator.write(report)

    payload = json.loads(config.output_json.read_text(encoding="utf-8"))
    markdown = config.output_markdown.read_text(encoding="utf-8")
    assert payload["cases"] == 1
    assert "Hit@1" in markdown
    assert "symbol" in markdown


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case = make_case()
    path.write_text(case.model_dump_json() + "\n" + case.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate benchmark case id"):
        load_cases([path])


def test_load_benchmark_config_uses_project_paths() -> None:
    config = load_benchmark_config()

    assert config.engine_version == "5.8"
    assert config.output_json == Path.cwd() / "data/benchmark/benchmark_results.json"
