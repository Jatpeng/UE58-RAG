"""Benchmark framework for retrieval ranking quality."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ue_rag.schema import RetrievalResult


class BenchmarkCase(BaseModel):
    """One query and its expected symbols/documents."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    category: str = Field(default="default", min_length=1)
    expected_symbols: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    engine_version: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_expected_values(self) -> "BenchmarkCase":
        for field_name in ("expected_symbols", "expected_chunk_ids", "expected_document_ids"):
            values = getattr(self, field_name)
            setattr(self, field_name, list(dict.fromkeys(item.strip() for item in values if item.strip())))
        if not (self.expected_symbols or self.expected_chunk_ids or self.expected_document_ids):
            raise ValueError("at least one expected symbol, chunk, or document is required")
        return self


class MetricScores(BaseModel):
    """Per-category or overall benchmark metrics."""

    hit_at_1: float = Field(ge=0, le=1)
    hit_at_3: float = Field(ge=0, le=1)
    hit_at_5: float = Field(ge=0, le=1)
    hit_at_10: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)


class BenchmarkConfig(BaseModel):
    """Benchmark input/output paths and optional minimum baselines."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    output_json: Path
    output_markdown: Path
    baseline: dict[str, float] = Field(default_factory=dict)


def load_benchmark_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    benchmark_config_path: str | Path = "config/benchmark.yaml",
) -> BenchmarkConfig:
    """Load UE version and project-relative benchmark output paths."""

    ue_path = Path(ue_config_path).resolve()
    benchmark_path = Path(benchmark_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with benchmark_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    project_root = ue_path.parent.parent
    for key in ("output_json", "output_markdown"):
        path = Path(values[key])
        values[key] = path if path.is_absolute() else project_root / path
    values["engine_version"] = str(ue_config["engine"]["version"])
    return BenchmarkConfig(**values)


@dataclass(frozen=True)
class CaseScore:
    case: BenchmarkCase
    metrics: MetricScores


class BenchmarkReport(BaseModel):
    """Serializable benchmark report with category breakdown and baseline status."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str
    cases: int
    overall: MetricScores
    by_category: dict[str, MetricScores]
    baseline: dict[str, float]
    baseline_passed: bool
    baseline_failures: dict[str, dict[str, float]] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"# Retrieval Benchmark (UE {self.engine_version})",
            "",
            f"Cases: {self.cases}",
            f"Baseline passed: {'yes' if self.baseline_passed else 'no'}",
            "",
            "| Category | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR | nDCG |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            self._markdown_row("overall", self.overall),
        ]
        lines.extend(self._markdown_row(category, scores) for category, scores in sorted(self.by_category.items()))
        if self.baseline_failures:
            lines.extend(["", "## Baseline failures", ""])
            for metric, values in sorted(self.baseline_failures.items()):
                lines.append(f"- `{metric}`: {values['actual']:.4f} < {values['minimum']:.4f}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _markdown_row(category: str, scores: MetricScores) -> str:
        values = [
            scores.hit_at_1,
            scores.hit_at_3,
            scores.hit_at_5,
            scores.hit_at_10,
            scores.recall_at_5,
            scores.recall_at_10,
            scores.mrr,
            scores.ndcg,
        ]
        return "| " + category + " | " + " | ".join(f"{value:.4f}" for value in values) + " |"


class BenchmarkEvaluator:
    """Evaluate any callable that returns ranked ``RetrievalResult`` objects."""

    def __init__(self, search: Callable[..., Sequence[RetrievalResult]], config: BenchmarkConfig) -> None:
        self.search = search
        self.config = config

    def evaluate(self, cases: Iterable[BenchmarkCase]) -> BenchmarkReport:
        scores: list[CaseScore] = []
        for case in cases:
            filters = dict(case.filters)
            filters.setdefault("engine_version", case.engine_version or self.config.engine_version)
            results = self.search(case.query, limit=10, filters=filters)
            scores.append(CaseScore(case, _score_case(case, results)))
        if not scores:
            raise ValueError("benchmark requires at least one case")
        overall = _average(score.metrics for score in scores)
        category_scores: dict[str, list[MetricScores]] = defaultdict(list)
        for score in scores:
            category_scores[score.case.category].append(score.metrics)
        by_category = {category: _average(values) for category, values in category_scores.items()}
        failures: dict[str, dict[str, float]] = {}
        metric_values = overall.model_dump()
        for metric, minimum in self.config.baseline.items():
            if metric not in metric_values:
                raise ValueError(f"unknown baseline metric: {metric}")
            if metric_values[metric] < minimum:
                failures[metric] = {"actual": metric_values[metric], "minimum": minimum}
        return BenchmarkReport(
            engine_version=self.config.engine_version,
            cases=len(scores),
            overall=overall,
            by_category=by_category,
            baseline=self.config.baseline,
            baseline_passed=not failures,
            baseline_failures=failures,
        )

    def write(self, report: BenchmarkReport) -> None:
        self.config.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        temporary_json = self.config.output_json.with_suffix(self.config.output_json.suffix + ".tmp")
        temporary_markdown = self.config.output_markdown.with_suffix(self.config.output_markdown.suffix + ".tmp")
        try:
            temporary_json.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
            temporary_markdown.write_text(report.to_markdown(), encoding="utf-8")
            os.replace(temporary_json, self.config.output_json)
            os.replace(temporary_markdown, self.config.output_markdown)
        except BaseException:
            temporary_json.unlink(missing_ok=True)
            temporary_markdown.unlink(missing_ok=True)
            raise


def load_cases(paths: Iterable[str | Path]) -> list[BenchmarkCase]:
    """Load benchmark cases from one or more JSONL files."""

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    case = BenchmarkCase.model_validate_json(line)
                except Exception as error:
                    raise ValueError(f"Invalid benchmark case at {path}:{line_number}") from error
                if case.id in seen:
                    raise ValueError(f"duplicate benchmark case id: {case.id}")
                seen.add(case.id)
                cases.append(case)
    return cases


def _score_case(case: BenchmarkCase, results: Sequence[RetrievalResult]) -> MetricScores:
    expected = set(case.expected_symbols) | set(case.expected_chunk_ids) | set(case.expected_document_ids)
    relevant: list[bool] = []
    matched_expected: set[str] = set()
    for result in results[:10]:
        candidates = _result_candidates(result)
        matches = expected & candidates
        newly_matched = matches - matched_expected
        relevant.append(bool(newly_matched))
        matched_expected.update(matches)
    first_rank = next((index + 1 for index, value in enumerate(relevant) if value), None)
    return MetricScores(
        hit_at_1=float(any(relevant[:1])),
        hit_at_3=float(any(relevant[:3])),
        hit_at_5=float(any(relevant[:5])),
        hit_at_10=float(any(relevant[:10])),
        recall_at_5=len(_matched_expected(results[:5], expected)) / len(expected),
        recall_at_10=len(matched_expected) / len(expected),
        mrr=1.0 / first_rank if first_rank else 0.0,
        ndcg=_ndcg(relevant, min(len(expected), 10)),
    )


def _result_candidates(result: RetrievalResult) -> set[str]:
    candidates = {value for value in (result.chunk_id, result.document_id) if value}
    metadata = result.metadata
    for key in ("symbol", "class", "class_name", "function_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            candidates.add(value)
    for key in ("field_symbols", "expected_symbols"):
        values = metadata.get(key)
        if isinstance(values, list):
            candidates.update(value for value in values if isinstance(value, str))
    return candidates


def _matched_expected(results: Sequence[RetrievalResult], expected: set[str]) -> set[str]:
    matched: set[str] = set()
    for result in results:
        matched.update(expected & _result_candidates(result))
    return matched


def _ndcg(relevant: Sequence[bool], ideal_relevant: int) -> float:
    dcg = sum((1.0 / math.log2(index + 2)) for index, value in enumerate(relevant) if value)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_relevant))
    return dcg / ideal if ideal else 0.0


def _average(metrics: Iterable[MetricScores]) -> MetricScores:
    values = list(metrics)
    if not values:
        raise ValueError("cannot average empty metrics")
    fields = MetricScores.model_fields
    return MetricScores(**{field: sum(getattr(item, field) for item in values) / len(values) for field in fields})
