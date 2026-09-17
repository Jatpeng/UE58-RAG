from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ue_rag.workflow import DashboardWorkflowService


def _write_minimal_workspace(root: Path) -> None:
    (root / "config").mkdir(parents=True)
    (root / "config" / "ue58.yaml").write_text(
        yaml.safe_dump(
            {
                "engine": {"version": "5.8", "root": ""},
                "paths": {"engine_source": "Engine/Source", "engine_plugins": "Engine/Plugins"},
                "data": {
                    "raw": "data/raw",
                    "parsed": "data/parsed",
                    "chunks": "data/chunks",
                    "benchmark": "data/benchmark",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "config" / "embedding.yaml").write_text(
        "provider: qwen\nmodel: test\ndevice: cpu\nbatch_size: 2\nmax_length: 128\n"
        "normalize: true\ncache_path: data/embeddings/cache.sqlite3\n"
        "output_path: data/embeddings/engine/embeddings.npy\n",
        encoding="utf-8",
    )
    (root / "config" / "qdrant_server.yaml").write_text(
        'collection: "ue58_global"\npath: null\nurl: "http://127.0.0.1:6333"\n'
        "batch_size: 128\ndistance: cosine\ntimeout_seconds: 60\n",
        encoding="utf-8",
    )
    (root / "config" / "blueprint.yaml").write_text(
        'input_path: "data/raw/project/blueprints.jsonl"\n'
        'output_path: "data/parsed/project/blueprints.jsonl"\n',
        encoding="utf-8",
    )


def test_workflow_catalog_lists_pipelines(tmp_path: Path) -> None:
    _write_minimal_workspace(tmp_path)
    service = DashboardWorkflowService(tmp_path)
    catalog = service.catalog()
    step_ids = {step["id"] for pipeline in catalog["pipelines"] for step in pipeline["steps"]}
    assert "engine_ingest" in step_ids
    assert "docs_crawl" in step_ids
    assert "embed" in step_ids
    assert "lexical_index" in step_ids
    assert "qdrant_index" in step_ids
    assert "mcp_start" in step_ids
    assert catalog["job"]["status"] == "idle"
    assert "gpu" in catalog["system"]


def test_workflow_rejects_unknown_step(tmp_path: Path) -> None:
    _write_minimal_workspace(tmp_path)
    service = DashboardWorkflowService(tmp_path)
    with pytest.raises(ValueError, match="未知流水线步骤"):
        service.start("not_a_real_step")


def test_workflow_runs_engine_chunk_with_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_minimal_workspace(tmp_path)
    service = DashboardWorkflowService(tmp_path)

    def fake_runner(params: dict) -> dict:
        return {"documents": 3, "chunks": 9, "output": "fake.jsonl"}

    monkeypatch.setitem(service._runners, "engine_chunk", fake_runner)
    # Exercise the job runner synchronously to avoid background-thread flakiness in CI.
    service._job = {
        "status": "running",
        "step": "engine_chunk",
        "phase": "启动中",
        "completed": 0,
        "total": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": None,
        "error": None,
        "summary": None,
        "params": {},
    }
    service._run("engine_chunk", {})
    status = service.status()
    assert status["job"]["status"] == "succeeded"
    assert status["job"]["summary"]["chunks"] == 9
    assert any("engine_chunk" in line for line in status["log"])


def test_workflow_busy_rejects_second_job(tmp_path: Path) -> None:
    _write_minimal_workspace(tmp_path)
    service = DashboardWorkflowService(tmp_path)
    with service._lock:
        service._job = {
            "status": "running",
            "step": "engine_chunk",
            "phase": "busy",
            "completed": 0,
            "total": 1,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": None,
            "error": None,
            "summary": None,
            "params": {},
        }
    with pytest.raises(ValueError, match="已有流水线任务"):
        service.start("engine_chunk", {})



def test_dashboard_page_includes_pipeline_console(tmp_path: Path) -> None:
    from ue_rag.visualization import render_dashboard

    page = render_dashboard(
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "engine_version": "5.8",
            "corpus": {
                "origin": "test",
                "chunks": 0,
                "documents": 0,
                "characters": 0,
                "symbols": 0,
                "index_bytes": 0,
                "sources": [],
                "modules": [],
                "plugins": [],
                "symbol_types": [],
                "lengths": [],
            },
            "embedding": {"records": 0, "dimensions": [], "models": [], "manifests": []},
            "benchmarks": [],
            "data_sources": [],
            "health": {"lexical": False, "vectors": False, "benchmark": False},
        }
    )
    assert "pipeline-console" in page
    assert "/api/workflow/catalog" in page
    assert "RAG 全流程控制台" in page
