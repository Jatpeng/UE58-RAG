from __future__ import annotations

from pathlib import Path
import json
import time

import pytest

from ue_rag.dashboard import DashboardBenchmarkService, DashboardSearchService
from ue_rag.retrieval import LexicalIndex
from ue_rag.schema import SourceScope, SourceType, UEChunk


def _chunk() -> UEChunk:
    return UEChunk(
        id="movement-speed",
        document_id="movement-document",
        chunk_index=0,
        engine_version="5.8",
        source_scope=SourceScope.GLOBAL,
        source_type=SourceType.ENGINE_SOURCE,
        content="MaxWalkSpeed controls the maximum ground movement speed.",
        title="UCharacterMovementComponent::MaxWalkSpeed",
        module="Engine",
        file_path="Engine/Source/Runtime/Engine/CharacterMovementComponent.h",
        symbol="UCharacterMovementComponent::MaxWalkSpeed",
        symbol_type="field",
        class_name="UCharacterMovementComponent",
        metadata={"chunk_kind": "property_group"},
    )


def test_dashboard_lexical_query_returns_visual_result_shape(tmp_path: Path) -> None:
    index_path = tmp_path / "data" / "index" / "lexical.sqlite3"
    with LexicalIndex(index_path) as index:
        index.upsert([_chunk()])
    search = DashboardSearchService(tmp_path)

    response = search.query(
        question="UCharacterMovementComponent::MaxWalkSpeed",
        mode="lexical",
        source_type="engine_source",
    )

    assert response["count"] == 1
    assert response["results"][0]["symbol"] == "UCharacterMovementComponent::MaxWalkSpeed"
    assert response["results"][0]["module"] == "Engine"
    assert response["results"][0]["content"].startswith("MaxWalkSpeed")
    search.close()


def test_dashboard_query_rejects_empty_question(tmp_path: Path) -> None:
    search = DashboardSearchService(tmp_path)
    with pytest.raises(ValueError, match="请输入"):
        search.query(question="  ")


def test_dashboard_benchmark_service_runs_and_syncs_report(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "data" / "benchmark"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "ragas_cases.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    class FakeSearch:
        def evaluate_benchmark(self, **kwargs: object) -> None:
            progress = kwargs["progress"]
            progress(1, 2)  # type: ignore[operator]
            report = {
                "cases": 2,
                "overall": {
                    "hit_at_1": 0.5,
                    "hit_at_5": 1.0,
                    "hit_at_10": 1.0,
                    "recall_at_10": 1.0,
                    "mrr": 0.75,
                    "ndcg": 0.8,
                },
                "by_category": {},
                "baseline_passed": True,
            }
            Path(kwargs["output_json"]).write_text(json.dumps(report), encoding="utf-8")  # type: ignore[arg-type]
            Path(kwargs["output_markdown"]).write_text("report", encoding="utf-8")  # type: ignore[arg-type]
            progress(2, 2)  # type: ignore[operator]

    service = DashboardBenchmarkService(tmp_path, FakeSearch())  # type: ignore[arg-type]
    started = service.start("hybrid")
    assert started["job"]["status"] in {"running", "succeeded"}
    for _ in range(100):
        status = service.status()
        if status["job"]["status"] != "running":
            break
        time.sleep(0.01)

    assert status["testset"]["cases"] == 2
    assert status["job"]["status"] == "succeeded"
    assert status["job"]["completed"] == 2
    assert status["reports"][0]["name"] == "Hybrid"
