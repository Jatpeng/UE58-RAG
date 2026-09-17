"""Background RAG pipeline jobs for the local data management dashboard."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[int, int], None]


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "records": 0, "updated_at": None}
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    records = _count_jsonl(path) if path.suffix == ".jsonl" and path.is_file() else 0
    return {"path": str(path), "exists": True, "records": records, "updated_at": updated}


class DashboardWorkflowService:
    """Start and observe full RAG pipeline steps from the browser console."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.ue_config = self.workspace_root / "config" / "ue58.yaml"
        self._lock = threading.Lock()
        self._mcp_process: subprocess.Popen[Any] | None = None
        self._mcp_meta: dict[str, Any] = {"transport": "stdio", "host": "127.0.0.1", "port": 8000}
        self._log: deque[str] = deque(maxlen=200)
        self._job: dict[str, Any] = self._idle_job()
        self._runners: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "engine_ingest": self._run_engine_ingest,
            "engine_parse": self._run_engine_parse,
            "engine_chunk": self._run_engine_chunk,
            "docs_crawl": self._run_docs_crawl,
            "docs_parse": self._run_docs_parse,
            "docs_chunk": self._run_docs_chunk,
            "blueprint_export": self._run_blueprint_export,
            "blueprint_chunk": self._run_blueprint_chunk,
            "embed": self._run_embed,
            "lexical_index": self._run_lexical_index,
            "qdrant_index": self._run_qdrant_index,
            "generate_testset": self._run_generate_testset,
            "mcp_start": self._run_mcp_start,
            "mcp_stop": self._run_mcp_stop,
        }

    def catalog(self) -> dict[str, Any]:
        """Describe pipelines, steps, artifact readiness, and current job."""

        artifacts = self._artifacts()
        pipelines = [
            {
                "id": "engine",
                "name": "引擎源码流水线",
                "description": "扫描 Engine/Source → 解析 C++ 符号 → 语义切块",
                "steps": [
                    self._step(
                        "engine_ingest",
                        "扫描引擎源码",
                        "生成文件清单 inventory JSONL",
                        ready=True,
                        artifact=artifacts["engine_inventory"],
                        params=[],
                    ),
                    self._step(
                        "engine_parse",
                        "解析 C++ 符号",
                        "从 inventory 提取类 / 函数 / 属性文档",
                        ready=artifacts["engine_inventory"]["exists"],
                        artifact=artifacts["engine_documents"],
                        params=[
                            {
                                "key": "limit_files",
                                "label": "限制文件数（可选，冒烟用）",
                                "type": "number",
                                "placeholder": "留空=全量",
                            }
                        ],
                    ),
                    self._step(
                        "engine_chunk",
                        "引擎语义切块",
                        "按符号与属性组生成检索切块",
                        ready=artifacts["engine_documents"]["exists"],
                        artifact=artifacts["engine_chunks"],
                        params=[],
                    ),
                ],
            },
            {
                "id": "docs",
                "name": "官方文档流水线",
                "description": "爬取 Epic 文档 → HTML 解析 → 标题层级切块",
                "steps": [
                    self._step(
                        "docs_crawl",
                        "爬取官方文档",
                        "按主题下载并写入 raw 缓存（需网络）",
                        ready=True,
                        artifact=artifacts["docs_manifest"],
                        params=[
                            {
                                "key": "topic",
                                "label": "主题（可选）",
                                "type": "text",
                                "placeholder": "如 gameplay；留空=全部",
                            },
                            {
                                "key": "max_pages",
                                "label": "最大页数（可选）",
                                "type": "number",
                                "placeholder": "留空=配置默认",
                            },
                        ],
                    ),
                    self._step(
                        "docs_parse",
                        "解析文档 HTML",
                        "清洗为 UEDocument JSONL",
                        ready=artifacts["docs_manifest"]["exists"],
                        artifact=artifacts["docs_documents"],
                        params=[],
                    ),
                    self._step(
                        "docs_chunk",
                        "文档语义切块",
                        "按标题层级切分检索单元",
                        ready=artifacts["docs_documents"]["exists"],
                        artifact=artifacts["docs_chunks"],
                        params=[],
                    ),
                ],
            },
            {
                "id": "blueprint",
                "name": "Blueprint 流水线",
                "description": "导出资产摘要 → Blueprint 切块",
                "steps": [
                    self._step(
                        "blueprint_export",
                        "导出 Blueprint",
                        "connector-neutral JSON → UEDocument",
                        ready=artifacts["blueprint_raw"]["exists"],
                        artifact=artifacts["blueprint_documents"],
                        params=[],
                    ),
                    self._step(
                        "blueprint_chunk",
                        "Blueprint 切块",
                        "图 / 函数 / 变量 / 组件视图切块",
                        ready=artifacts["blueprint_documents"]["exists"],
                        artifact=artifacts["blueprint_chunks"],
                        params=[],
                    ),
                ],
            },
            {
                "id": "index",
                "name": "索引构建",
                "description": "GPU Embedding → Lexical SQLite → Qdrant 向量库",
                "steps": [
                    self._step(
                        "embed",
                        "生成 Embedding",
                        "对切块做 CUDA 向量化并写 .npy",
                        ready=True,
                        artifact=artifacts["embeddings_engine"],
                        params=[
                            {
                                "key": "source",
                                "label": "语料来源",
                                "type": "select",
                                "options": [
                                    {"value": "engine", "label": "引擎"},
                                    {"value": "docs", "label": "文档"},
                                    {"value": "blueprint", "label": "Blueprint"},
                                ],
                                "default": "engine",
                            }
                        ],
                    ),
                    self._step(
                        "lexical_index",
                        "构建 Lexical 索引",
                        "写入 / 重建 SQLite FTS5 + 符号索引",
                        ready=True,
                        artifact=artifacts["lexical_index"],
                        params=[
                            {
                                "key": "source",
                                "label": "语料来源",
                                "type": "select",
                                "options": [
                                    {"value": "engine", "label": "引擎"},
                                    {"value": "docs", "label": "文档"},
                                    {"value": "blueprint", "label": "Blueprint"},
                                    {"value": "all", "label": "全部已有切块"},
                                ],
                                "default": "engine",
                            },
                            {
                                "key": "rebuild",
                                "label": "全量重建（原子替换）",
                                "type": "checkbox",
                                "default": False,
                            },
                        ],
                    ),
                    self._step(
                        "qdrant_index",
                        "导入 Qdrant",
                        "将向量写入集合（可选 recreate）",
                        ready=True,
                        artifact=artifacts["embeddings_engine"],
                        params=[
                            {
                                "key": "source",
                                "label": "语料来源",
                                "type": "select",
                                "options": [
                                    {"value": "engine", "label": "引擎"},
                                    {"value": "docs", "label": "文档"},
                                    {"value": "blueprint", "label": "Blueprint"},
                                ],
                                "default": "engine",
                            },
                            {
                                "key": "recreate",
                                "label": "重建集合",
                                "type": "checkbox",
                                "default": False,
                            },
                        ],
                    ),
                ],
            },
            {
                "id": "eval",
                "name": "评测与测试集",
                "description": "RAGAS 采样 / 生成测试集；完整评测见下方评测中心",
                "steps": [
                    self._step(
                        "generate_testset",
                        "生成 / 采样测试集",
                        "默认仅采样可追溯源（prepare-only）；勾选后调用 LLM 生成",
                        ready=artifacts["engine_chunks"]["exists"] or artifacts["docs_chunks"]["exists"],
                        artifact=artifacts["ragas_cases"],
                        params=[
                            {
                                "key": "prepare_only",
                                "label": "仅采样源（不调用 LLM）",
                                "type": "checkbox",
                                "default": True,
                            },
                            {
                                "key": "documents",
                                "label": "采样文档数（可选）",
                                "type": "number",
                                "placeholder": "配置默认",
                            },
                        ],
                    ),
                ],
            },
            {
                "id": "serve",
                "name": "服务运维",
                "description": "MCP 检索服务启停（HTTP 传输）",
                "steps": [
                    self._step(
                        "mcp_start",
                        "启动 MCP",
                        "后台启动 scripts/mcp_server.py（streamable-http）",
                        ready=artifacts["lexical_index"]["exists"],
                        artifact=None,
                        params=[
                            {
                                "key": "port",
                                "label": "端口",
                                "type": "number",
                                "default": 8000,
                            },
                            {
                                "key": "enable_dense",
                                "label": "启用 Dense / Hybrid",
                                "type": "checkbox",
                                "default": False,
                            },
                        ],
                    ),
                    self._step(
                        "mcp_stop",
                        "停止 MCP",
                        "结束本管理台拉起的 MCP 进程",
                        ready=True,
                        artifact=None,
                        params=[],
                    ),
                ],
            },
        ]
        return {
            "pipelines": pipelines,
            "artifacts": artifacts,
            "job": self._job_snapshot(),
            "mcp": self._mcp_status(),
            "system": self.system_status(),
        }

    def status(self) -> dict[str, Any]:
        return {
            "job": self._job_snapshot(),
            "log": list(self._log),
            "mcp": self._mcp_status(),
            "artifacts": self._artifacts(),
        }

    def start(self, step_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        step_id = str(step_id or "").strip()
        if step_id not in self._runners:
            raise ValueError(f"未知流水线步骤：{step_id}")
        payload = dict(params or {})
        with self._lock:
            if self._job["status"] == "running":
                raise ValueError("已有流水线任务正在运行，请等待完成")
            self._log.clear()
            self._job = {
                "status": "running",
                "step": step_id,
                "phase": "启动中",
                "completed": 0,
                "total": 0,
                "started_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
                "error": None,
                "summary": None,
                "params": payload,
            }
            self._append_log(f"开始步骤 {step_id}")
        threading.Thread(target=self._run, args=(step_id, payload), daemon=True).start()
        return self.status()

    def system_status(self) -> dict[str, Any]:
        """Report GPU / CUDA / Qdrant / path readiness for the console."""

        gpu: dict[str, Any] = {
            "torch_available": False,
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
            "cuda_version": None,
        }
        try:
            import torch

            gpu["torch_available"] = True
            gpu["cuda_available"] = bool(torch.cuda.is_available())
            gpu["cuda_version"] = getattr(torch.version, "cuda", None)
            if gpu["cuda_available"]:
                gpu["device_count"] = int(torch.cuda.device_count())
                for index in range(gpu["device_count"]):
                    props = torch.cuda.get_device_properties(index)
                    total_gb = round(props.total_memory / (1024**3), 1)
                    gpu["devices"].append(
                        {
                            "index": index,
                            "name": props.name,
                            "total_memory_gb": total_gb,
                        }
                    )
        except Exception as error:  # noqa: BLE001 - surface soft failure
            gpu["error"] = f"{type(error).__name__}: {error}"

        embedding = self._read_yaml(self.workspace_root / "config" / "embedding.yaml")
        qdrant = self._probe_qdrant()
        ue_root = os.environ.get("UE_ROOT", "").strip()
        if not ue_root:
            ue_payload = self._read_yaml(self.ue_config)
            ue_root = str(((ue_payload or {}).get("engine") or {}).get("root") or "").strip()

        return {
            "gpu": gpu,
            "embedding": {
                "model": embedding.get("model"),
                "device": embedding.get("device"),
                "batch_size": embedding.get("batch_size"),
                "cache_path": embedding.get("cache_path"),
            },
            "qdrant": qdrant,
            "paths": {
                "workspace": str(self.workspace_root),
                "ue_root": ue_root or None,
                "lexical_index": str(self.workspace_root / "data" / "index" / "lexical.sqlite3"),
                "chunks": str(self.workspace_root / "data" / "chunks"),
                "embeddings": str(self.workspace_root / "data" / "embeddings"),
                "raw": str(self.workspace_root / "data" / "raw"),
            },
            "python": sys.executable,
        }

    def invalidate_stats_cache(self) -> None:
        stats_path = self.workspace_root / "outputs" / "rag_dashboard.stats.json"
        stats_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ runners

    def _run(self, step_id: str, params: dict[str, Any]) -> None:
        try:
            summary = self._runners[step_id](params)
            if step_id not in {"mcp_start", "mcp_stop"}:
                self.invalidate_stats_cache()
            with self._lock:
                self._job["status"] = "succeeded"
                self._job["phase"] = "完成"
                self._job["summary"] = summary
                if self._job["total"]:
                    self._job["completed"] = self._job["total"]
                else:
                    self._job["completed"] = 1
                    self._job["total"] = 1
                self._job["finished_at"] = datetime.now(UTC).isoformat()
            self._append_log(f"步骤 {step_id} 完成")
        except Exception as error:  # noqa: BLE001 - report to UI
            with self._lock:
                self._job["status"] = "failed"
                self._job["phase"] = "失败"
                self._job["error"] = f"{type(error).__name__}: {error}"
                self._job["finished_at"] = datetime.now(UTC).isoformat()
            self._append_log(f"失败：{type(error).__name__}: {error}")

    def _set_phase(self, phase: str, *, completed: int | None = None, total: int | None = None) -> None:
        with self._lock:
            self._job["phase"] = phase
            if completed is not None:
                self._job["completed"] = completed
            if total is not None:
                self._job["total"] = total
        self._append_log(phase)

    def _progress(self, completed: int, total: int) -> None:
        with self._lock:
            self._job["completed"] = completed
            self._job["total"] = total

    def _run_engine_ingest(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.crawler import EngineSourceScanner, load_engine_scanner_config

        self._set_phase("扫描引擎源码…", completed=0, total=1)
        config = load_engine_scanner_config(
            self.ue_config, self.workspace_root / "config" / "engine_scanner.yaml"
        )
        summary = EngineSourceScanner(config).scan()
        return {
            "files": summary.files,
            "modules": summary.modules,
            "plugins": summary.plugins,
            "output": str(summary.output_path),
            "issues": summary.issues,
        }

    def _run_engine_parse(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.parser import UnrealCPPParser, load_cpp_parser_config

        limit = params.get("limit_files")
        limit_files = int(limit) if limit not in (None, "", False) else None
        if limit_files is not None and limit_files <= 0:
            raise ValueError("limit_files 必须为正整数")
        self._set_phase("解析 C++ inventory…", completed=0, total=1)
        config = load_cpp_parser_config(
            self.ue_config, self.workspace_root / "config" / "cpp_parser.yaml"
        )
        output = None
        if limit_files is not None:
            output = self.workspace_root / "data" / "parsed" / "engine" / "documents.smoke.jsonl"
            self._append_log(f"冒烟模式：限制 {limit_files} 个文件 → {output.name}")
        summary = UnrealCPPParser(config).parse_inventory(
            limit_files=limit_files,
            output_path=output,
        )
        return {
            "selected_files": summary.selected_files,
            "parsed_files": summary.parsed_files,
            "documents": summary.documents,
            "output": str(summary.output_path),
            "issues": summary.issues,
        }

    def _run_engine_chunk(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.chunker import CPPSemanticChunker, load_cpp_chunker_config

        self._set_phase("引擎语义切块…", completed=0, total=1)
        config = load_cpp_chunker_config(
            self.ue_config, self.workspace_root / "config" / "cpp_chunker.yaml"
        )
        summary = CPPSemanticChunker(config).chunk_corpus()
        return {
            "documents": summary.documents,
            "chunks": summary.chunks,
            "output": str(summary.output_path),
        }

    def _run_docs_crawl(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.crawler import DocumentationCrawler, load_docs_crawler_config

        topic = str(params.get("topic") or "").strip() or None
        max_pages_raw = params.get("max_pages")
        max_pages = int(max_pages_raw) if max_pages_raw not in (None, "", False) else None
        self._set_phase("爬取官方文档…", completed=0, total=1)
        config = load_docs_crawler_config(
            self.ue_config, self.workspace_root / "config" / "docs_crawler.yaml"
        )
        topics = [topic] if topic else None
        crawler = DocumentationCrawler(config)
        summary = crawler.crawl(topics, max_pages=max_pages)
        return {
            "downloaded": summary.downloaded,
            "cached": summary.cached,
            "failed": summary.failed,
            "blocked": summary.blocked,
            "processed": summary.processed,
            "manifest": str(crawler.manifest_path),
        }

    def _run_docs_parse(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.parser import DocumentationParser, load_docs_parser_config

        self._set_phase("解析文档 HTML…", completed=0, total=1)
        config = load_docs_parser_config(self.ue_config)
        summary = DocumentationParser().parse_manifest(config)
        return {
            "documents": summary.documents,
            "headings": summary.headings,
            "output": str(summary.output_path),
        }

    def _run_docs_chunk(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.chunker import DocumentationChunker, load_docs_chunker_config

        self._set_phase("文档语义切块…", completed=0, total=1)
        config = load_docs_chunker_config(
            self.ue_config, self.workspace_root / "config" / "docs_chunker.yaml"
        )
        summary = DocumentationChunker(config).chunk_corpus()
        return {
            "documents": summary.documents,
            "chunks": summary.chunks,
            "output": str(summary.output_path),
        }

    def _run_blueprint_export(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.blueprint import JsonBlueprintExporter, load_blueprint_config

        self._set_phase("导出 Blueprint…", completed=0, total=1)
        config = load_blueprint_config(
            self.ue_config, self.workspace_root / "config" / "blueprint.yaml"
        )
        if not config.input_path.is_file():
            raise ValueError(f"未找到 Blueprint 输入：{config.input_path}")
        summary = JsonBlueprintExporter(config.engine_version).export_jsonl(
            config.input_path, config.output_path
        )
        return {
            "assets": summary.assets,
            "documents": summary.documents,
            "output": str(summary.output_path),
        }

    def _run_blueprint_chunk(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.chunker import BlueprintChunker, load_blueprint_chunker_config

        self._set_phase("Blueprint 切块…", completed=0, total=1)
        config = load_blueprint_chunker_config(
            self.ue_config, self.workspace_root / "config" / "blueprint_chunker.yaml"
        )
        documents, chunks, output = BlueprintChunker(config).chunk_corpus()
        return {"documents": documents, "chunks": chunks, "output": str(output)}

    def _run_embed(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.embedding import QwenEmbeddingProvider, embed_jsonl, load_embedding_config

        source = str(params.get("source") or "engine").strip()
        chunks_path, vectors_path = self._source_paths(source)
        if not chunks_path.is_file():
            raise ValueError(f"切块文件不存在：{chunks_path}")
        total = _count_jsonl(chunks_path)
        self._set_phase(f"Embedding {source}（{total} 条）…", completed=0, total=total or 1)
        config = load_embedding_config(
            self.ue_config, self.workspace_root / "config" / "embedding.yaml"
        )
        config.output_path = vectors_path
        provider = QwenEmbeddingProvider(config)
        try:
            summary = embed_jsonl(
                chunks_path,
                vectors_path,
                provider,
                progress=self._progress if total else None,
            )
        finally:
            provider.close()
        return {
            "source": source,
            "records": summary.records,
            "dimension": summary.dimension,
            "output": str(summary.output_path),
        }

    def _run_lexical_index(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.retrieval import LexicalIndex, load_lexical_config

        source = str(params.get("source") or "engine").strip()
        rebuild = bool(params.get("rebuild", False))
        sources = ["engine", "docs", "blueprint"] if source == "all" else [source]
        inputs: list[Path] = []
        for name in sources:
            path, _ = self._source_paths(name)
            if path.is_file():
                inputs.append(path)
        if not inputs:
            raise ValueError("没有可用的切块 JSONL")
        total = sum(_count_jsonl(path) for path in inputs)
        self._set_phase(
            f"Lexical 索引（{'重建' if rebuild else '增量'} · {total} 条）…",
            completed=0,
            total=total or 1,
        )
        config = load_lexical_config(
            self.ue_config, self.workspace_root / "config" / "lexical.yaml"
        )
        target = config.index_path
        build_path = target.with_name(target.name + ".building") if rebuild else target
        if rebuild and build_path.exists():
            build_path.unlink()
        completed = 0

        def progress(done: int, _total: int) -> None:
            nonlocal completed
            # done is per-file; accumulate across sources for rebuild first file only resets
            self._progress(completed + done, total or 1)

        with LexicalIndex(build_path, bulk_build=rebuild) as index:
            for index_number, path in enumerate(inputs):
                if index_number > 0 and rebuild:
                    # After first file on rebuild path, continue upserting into same build DB.
                    pass
                file_total = _count_jsonl(path)
                self._append_log(f"索引 {path.name}（{file_total}）")
                before = completed
                summary = index.index_jsonl(
                    path,
                    batch_size=config.batch_size,
                    bulk_build=rebuild and index_number == 0,
                    progress=progress,
                    total_hint=total,
                )
                completed = before + summary.total
                self._progress(completed, total or 1)
        if rebuild:
            os.replace(build_path, target)
        return {
            "source": source,
            "rebuild": rebuild,
            "indexed": completed,
            "index": str(target),
            "inputs": [str(path) for path in inputs],
        }

    def _run_qdrant_index(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.index import QdrantVectorStore, ingest_jsonl, load_qdrant_config

        source = str(params.get("source") or "engine").strip()
        recreate = bool(params.get("recreate", False))
        chunks_path, vectors_path = self._source_paths(source)
        ids_path = Path(str(vectors_path) + ".ids.jsonl")
        if not chunks_path.is_file():
            raise ValueError(f"切块文件不存在：{chunks_path}")
        if not vectors_path.is_file():
            raise ValueError(f"向量文件不存在：{vectors_path}，请先运行 Embedding")
        total = _count_jsonl(chunks_path)
        self._set_phase(f"导入 Qdrant（{source} · {total}）…", completed=0, total=total or 1)
        config = load_qdrant_config(self.workspace_root / "config" / "qdrant_server.yaml")
        store = QdrantVectorStore.from_config(config)
        try:
            if recreate:
                import numpy as np

                size = int(np.load(vectors_path, mmap_mode="r").shape[1])
                store.create_collection(size, recreate=True)
                self._append_log(f"已重建集合 {config.collection}（dim={size}）")
            summary = ingest_jsonl(
                store,
                chunks_path,
                vectors_path,
                ids_path=ids_path if ids_path.is_file() else None,
                fast=recreate,
                progress=self._progress,
            )
        finally:
            if getattr(store, "client", None) is not None:
                store.client.close()
        return {
            "source": source,
            "recreate": recreate,
            "collection": config.collection,
            "total": summary.total,
            "added": summary.added,
            "updated": summary.updated,
            "failed": summary.failed,
        }

    def _run_generate_testset(self, params: dict[str, Any]) -> dict[str, Any]:
        from ue_rag.eval import load_testset_config, sample_sources, write_sources

        prepare_only = bool(params.get("prepare_only", True))
        documents_raw = params.get("documents")
        documents = int(documents_raw) if documents_raw not in (None, "", False) else None
        self._set_phase(
            "采样测试集源…" if prepare_only else "调用 RAGAS 生成测试集…",
            completed=0,
            total=1,
        )
        config_path = self.workspace_root / "config" / "testset.yaml"
        if prepare_only:
            config = load_testset_config(config_path)
            if documents is not None:
                if documents <= 0:
                    raise ValueError("documents 必须为正整数")
                config = config.model_copy(update={"documents": documents})
            sources = sample_sources(config)
            write_sources(sources, config.sources_output)
            return {
                "prepare_only": True,
                "sources": len(sources),
                "sources_path": str(config.sources_output),
                "cases_path": str(config.benchmark_output),
                "note": "已采样可追溯源；取消「仅采样」将调用 LLM 生成完整用例",
            }

        command = [
            sys.executable,
            str(self.workspace_root / "scripts" / "generate_testset.py"),
            "--config",
            str(config_path),
        ]
        if documents is not None:
            if documents <= 0:
                raise ValueError("documents 必须为正整数")
            command.extend(["--documents", str(documents)])
        completed = subprocess.run(
            command,
            cwd=str(self.workspace_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.stdout:
            for line in completed.stdout.strip().splitlines()[-20:]:
                self._append_log(line)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "generate_testset failed").strip()
            raise RuntimeError(detail[-800:])
        config = load_testset_config(config_path)
        return {
            "prepare_only": False,
            "cases": _count_jsonl(config.benchmark_output),
            "cases_path": str(config.benchmark_output),
            "testset_path": str(config.testset_output),
        }

    def _run_mcp_start(self, params: dict[str, Any]) -> dict[str, Any]:
        port = int(params.get("port") or 8000)
        if not 1 <= port <= 65535:
            raise ValueError("端口必须在 1–65535")
        enable_dense = bool(params.get("enable_dense", False))
        with self._lock:
            if self._mcp_process is not None and self._mcp_process.poll() is None:
                raise ValueError("MCP 已在运行")
        index_path = self.workspace_root / "data" / "index" / "lexical.sqlite3"
        if not index_path.is_file():
            raise ValueError("缺少 Lexical 索引，请先构建索引")
        self._set_phase(f"启动 MCP :{port}…", completed=0, total=1)
        command = [
            sys.executable,
            str(self.workspace_root / "scripts" / "mcp_server.py"),
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if enable_dense:
            command.append("--enable-dense")
        process = subprocess.Popen(
            command,
            cwd=str(self.workspace_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._lock:
            self._mcp_process = process
            self._mcp_meta = {
                "transport": "streamable-http",
                "host": "127.0.0.1",
                "port": port,
                "enable_dense": enable_dense,
                "pid": process.pid,
            }
        return dict(self._mcp_meta)

    def _run_mcp_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        self._set_phase("停止 MCP…", completed=0, total=1)
        with self._lock:
            process = self._mcp_process
            self._mcp_process = None
        if process is None or process.poll() is not None:
            return {"stopped": False, "note": "没有由本管理台启动的 MCP 进程"}
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        return {"stopped": True, "pid": process.pid}

    # ------------------------------------------------------------------ helpers

    def _source_paths(self, source: str) -> tuple[Path, Path]:
        mapping = {
            "engine": (
                self.workspace_root / "data" / "chunks" / "engine" / "chunks.jsonl",
                self.workspace_root / "data" / "embeddings" / "engine" / "embeddings.npy",
            ),
            "docs": (
                self.workspace_root / "data" / "chunks" / "docs" / "chunks.jsonl",
                self.workspace_root / "data" / "embeddings" / "docs" / "embeddings.npy",
            ),
            "blueprint": (
                self.workspace_root / "data" / "chunks" / "project" / "blueprints.jsonl",
                self.workspace_root / "data" / "embeddings" / "blueprint" / "embeddings.npy",
            ),
        }
        if source not in mapping:
            raise ValueError("source 必须是 engine、docs 或 blueprint")
        return mapping[source]

    def _artifacts(self) -> dict[str, Any]:
        root = self.workspace_root
        docs_manifest = root / "data" / "raw" / "docs" / "manifest.jsonl"
        # docs crawler may nest by locale; also accept common layout
        if not docs_manifest.is_file():
            nested = list((root / "data" / "raw" / "docs").rglob("manifest.jsonl")) if (root / "data" / "raw" / "docs").is_dir() else []
            if nested:
                docs_manifest = nested[0]
        blueprint_raw = root / "data" / "raw" / "project" / "blueprints.jsonl"
        if not blueprint_raw.is_file():
            cfg = self._read_yaml(root / "config" / "blueprint.yaml")
            candidate = cfg.get("input_path")
            if candidate:
                path = Path(str(candidate))
                if not path.is_absolute():
                    path = root / path
                blueprint_raw = path
        return {
            "engine_inventory": _file_meta(root / "data" / "parsed" / "engine" / "files.jsonl"),
            "engine_documents": _file_meta(root / "data" / "parsed" / "engine" / "documents.jsonl"),
            "engine_chunks": _file_meta(root / "data" / "chunks" / "engine" / "chunks.jsonl"),
            "docs_manifest": _file_meta(docs_manifest),
            "docs_documents": _file_meta(root / "data" / "parsed" / "docs" / "documents.jsonl"),
            "docs_chunks": _file_meta(root / "data" / "chunks" / "docs" / "chunks.jsonl"),
            "blueprint_raw": _file_meta(blueprint_raw),
            "blueprint_documents": _file_meta(root / "data" / "parsed" / "project" / "blueprints.jsonl"),
            "blueprint_chunks": _file_meta(root / "data" / "chunks" / "project" / "blueprints.jsonl"),
            "embeddings_engine": _file_meta(
                root / "data" / "embeddings" / "engine" / "embeddings.npy.manifest.json"
            ),
            "lexical_index": {
                "path": str(root / "data" / "index" / "lexical.sqlite3"),
                "exists": (root / "data" / "index" / "lexical.sqlite3").is_file(),
                "records": 0,
                "updated_at": (
                    datetime.fromtimestamp(
                        (root / "data" / "index" / "lexical.sqlite3").stat().st_mtime, tz=UTC
                    ).isoformat()
                    if (root / "data" / "index" / "lexical.sqlite3").is_file()
                    else None
                ),
            },
            "ragas_cases": _file_meta(root / "data" / "benchmark" / "ragas_cases.jsonl"),
        }

    def _mcp_status(self) -> dict[str, Any]:
        with self._lock:
            process = self._mcp_process
            meta = dict(self._mcp_meta)
        running = process is not None and process.poll() is None
        port = int(meta.get("port") or 8000)
        listening = self._port_open("127.0.0.1", port)
        return {
            **meta,
            "managed": running,
            "listening": listening,
            "status": "running" if running or listening else "stopped",
        }

    def _probe_qdrant(self) -> dict[str, Any]:
        config = self._read_yaml(self.workspace_root / "config" / "qdrant_server.yaml")
        url = str(config.get("url") or "http://127.0.0.1:6333").rstrip("/")
        result: dict[str, Any] = {
            "url": url,
            "collection": config.get("collection"),
            "reachable": False,
            "points": None,
            "status": None,
        }
        try:
            import urllib.request

            with urllib.request.urlopen(f"{url}/collections", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result["reachable"] = True
            collections = payload.get("result", {}).get("collections", [])
            names = {item.get("name") for item in collections}
            target = config.get("collection")
            if target in names:
                with urllib.request.urlopen(f"{url}/collections/{target}", timeout=2) as response:
                    detail = json.loads(response.read().decode("utf-8")).get("result", {})
                result["status"] = detail.get("status")
                result["points"] = (detail.get("points_count") or detail.get("vectors_count"))
        except Exception as error:  # noqa: BLE001
            result["error"] = f"{type(error).__name__}: {error}"
        return result

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.4):
                return True
        except OSError:
            return False

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _step(
        step_id: str,
        name: str,
        description: str,
        *,
        ready: bool,
        artifact: dict[str, Any] | None,
        params: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": step_id,
            "name": name,
            "description": description,
            "ready": ready,
            "artifact": artifact,
            "params": params,
        }

    def _idle_job(self) -> dict[str, Any]:
        return {
            "status": "idle",
            "step": None,
            "phase": None,
            "completed": 0,
            "total": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "summary": None,
            "params": {},
        }

    def _job_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._job)

    def _append_log(self, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            self._log.append(line)
