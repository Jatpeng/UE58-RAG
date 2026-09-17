from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ue_rag.visualization import cli, collect_dashboard_data, render_dashboard


def _make_index(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, engine_version TEXT NOT NULL,
            source_type TEXT NOT NULL, module TEXT, plugin TEXT, class_name TEXT,
            function_name TEXT, symbol TEXT, symbol_type TEXT, file_path TEXT,
            content TEXT NOT NULL, metadata_json TEXT NOT NULL, content_sha256 TEXT NOT NULL
        )"""
    )
    rows = [
        ("a", "doc-a", "docs", None, None, None, None, "private docs"),
        ("b", "doc-b", "engine_source", "Core", None, "FVector", "struct", "private source"),
        ("c", "doc-b", "engine_source", "Core", None, "Tick", "method", "x" * 1700),
    ]
    connection.executemany(
        """INSERT INTO chunks
        (chunk_id, document_id, engine_version, source_type, module, plugin, class_name,
         function_name, symbol, symbol_type, file_path, content, metadata_json, content_sha256)
        VALUES (?, ?, '5.8', ?, ?, ?, NULL, NULL, ?, ?, NULL, ?, '{}', 'hash')""",
        rows,
    )
    connection.commit()
    connection.close()


def test_collect_and_render_dashboard_does_not_embed_content(tmp_path: Path) -> None:
    index = tmp_path / "index.sqlite3"
    _make_index(index)
    benchmark = tmp_path / "hybrid_benchmark_results.json"
    benchmark.write_text(
        json.dumps(
            {
                "cases": 2,
                "overall": {"hit_at_1": 0.5, "hit_at_5": 1, "recall_at_10": 1, "mrr": 0.75},
                "by_category": {},
                "baseline_passed": True,
            }
        ),
        encoding="utf-8",
    )
    data = collect_dashboard_data(
        index_path=index,
        chunks_dir=tmp_path / "missing",
        benchmark_paths=[benchmark],
        embeddings_dir=tmp_path / "embeddings",
    )
    page = render_dashboard(data)
    assert data["corpus"]["chunks"] == 3
    assert data["corpus"]["documents"] == 2
    assert data["corpus"]["symbols"] == 2
    assert data["corpus"]["sources"][0] == {"name": "engine_source", "count": 2}
    assert data["data_sources"][0]["name"] == "UE 5.8 引擎 C++ 源码"
    assert data["data_sources"][0]["chunks"] == 2
    assert data["benchmarks"][0]["name"] == "Hybrid"
    assert data["benchmarks"][0]["updated_at"]
    assert data["benchmarks"][0]["source_file"] == str(benchmark.resolve())
    assert "private source" not in page
    assert "RAG 数据观测台" in page
    assert "增量添加项目数据" in page
    assert "当前数据来源" in page
    assert "RAG 检索测试" in page
    assert "Enhanced Input 如何绑定输入动作" in page
    assert "Hit@10" in page
    assert "nDCG" in page
    assert "检索评测中心" in page
    assert "测试 Lexical" in page
    assert "测试 Hybrid" in page
    assert "/api/benchmark/status" in page
    assert "pipeline-console" in page
    assert "RAG 全流程控制台" in page
    assert "/api/workflow/catalog" in page


def test_cli_uses_jsonl_fallback(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "sample.jsonl").write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "source_type": "blueprint",
                "module": "Game",
                "symbol": "BP_Player",
                "symbol_type": "asset",
                "content": "secret blueprint body",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "dashboard.html"
    result = cli(
        [
            "--index",
            str(tmp_path / "missing.sqlite3"),
            "--chunks-dir",
            str(chunks),
            "--benchmark-dir",
            str(tmp_path / "missing-benchmarks"),
            "--embeddings-dir",
            str(tmp_path / "missing-embeddings"),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    assert output.is_file()
    assert "secret blueprint body" not in output.read_text(encoding="utf-8")


def test_dashboard_keeps_only_latest_report_for_each_mode(tmp_path: Path) -> None:
    old = tmp_path / "benchmark_results.json"
    latest = tmp_path / "lexical_benchmark_results.json"
    payload = {
        "cases": 1,
        "overall": {"hit_at_1": 0.0},
        "by_category": {},
        "baseline_passed": True,
    }
    old.write_text(json.dumps(payload), encoding="utf-8")
    latest.write_text(json.dumps({**payload, "cases": 2}), encoding="utf-8")
    old.touch()
    latest.touch()

    data = collect_dashboard_data(
        index_path=tmp_path / "missing.sqlite3",
        chunks_dir=tmp_path / "missing",
        benchmark_paths=[old, latest],
        embeddings_dir=tmp_path / "embeddings",
    )

    assert len(data["benchmarks"]) == 1
    assert data["benchmarks"][0]["name"] == "Lexical"
    assert data["benchmarks"][0]["cases"] == 2
