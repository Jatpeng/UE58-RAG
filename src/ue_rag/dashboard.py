"""Local browser dashboard with project-data synchronization APIs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ue_rag.project_sync import ProjectSyncService
from ue_rag.retrieval import HybridRetriever, LexicalIndex, load_retrieval_config
from ue_rag.visualization import (
    _discover_benchmarks,
    _read_engine_version,
    collect_dashboard_data,
    collect_data_sources,
    render_dashboard,
)
from ue_rag.workflow import DashboardWorkflowService


class DashboardApplication:
    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.sync = ProjectSyncService(self.workspace_root)
        self.search = DashboardSearchService(self.workspace_root)
        self.benchmarks = DashboardBenchmarkService(self.workspace_root, self.search)
        self.workflow = DashboardWorkflowService(self.workspace_root)

    def render_page(self) -> bytes:
        """Render from the newest cached corpus stats and benchmark reports."""

        data = self._dashboard_data()
        data["data_sources"] = collect_data_sources(self.workspace_root, data["corpus"])
        return render_dashboard(data).encode("utf-8")

    def _dashboard_data(self) -> dict[str, Any]:
        output_dir = self.workspace_root / "outputs"
        stats_path = output_dir / "rag_dashboard.stats.json"
        index_path = self.workspace_root / "data/index/lexical.sqlite3"
        benchmark_paths = _discover_benchmarks(self.workspace_root / "data/benchmark")
        dependencies = [path for path in [index_path, *benchmark_paths] if path.is_file()]
        newest_dependency_mtime = max(
            (path.stat().st_mtime for path in dependencies),
            default=0.0,
        )
        cache_is_fresh = (
            stats_path.is_file()
            and stats_path.stat().st_mtime >= newest_dependency_mtime
        )
        if cache_is_fresh:
            try:
                return json.loads(stats_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        legacy_page = output_dir / "rag_dashboard.html"
        legacy_is_fresh = (
            legacy_page.is_file()
            and legacy_page.stat().st_mtime >= newest_dependency_mtime
        )
        if legacy_is_fresh:
            try:
                page = legacy_page.read_text(encoding="utf-8")
                start = page.index("<script>const D=") + len("<script>const D=")
                end = page.index(";const C=", start)
                data = json.loads(page[start:end])
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                stats_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                return data
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        data = collect_dashboard_data(
            index_path=self.workspace_root / "data/index/lexical.sqlite3",
            chunks_dir=self.workspace_root / "data/chunks",
            benchmark_paths=benchmark_paths,
            embeddings_dir=self.workspace_root / "data/embeddings",
            engine_version=_read_engine_version(self.workspace_root / "config/ue58.yaml"),
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return data

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/api/pick-directory":
            return {"path": _pick_directory()}
        if path == "/api/project/preview":
            project_root = str(payload.get("project_root", "")).strip()
            if not project_root:
                raise ValueError("请选择 UE 项目目录")
            return self.sync.preview(project_root)
        if path == "/api/project/apply":
            plan_id = str(payload.get("plan_id", "")).strip()
            if not plan_id:
                raise ValueError("缺少同步计划，请重新扫描")
            return self.sync.apply(plan_id, sync_vectors=bool(payload.get("sync_vectors", True)))
        if path == "/api/search":
            return self.search.query(
                question=str(payload.get("question", "")),
                mode=str(payload.get("mode", "lexical")),
                limit=int(payload.get("limit", 8)),
                source_type=str(payload.get("source_type", "")).strip() or None,
                module=str(payload.get("module", "")).strip() or None,
            )
        if path == "/api/benchmark/run":
            return self.benchmarks.start(str(payload.get("mode", "")).strip())
        if path == "/api/workflow/start":
            params = payload.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params 必须是对象")
            return self.workflow.start(str(payload.get("step", "")).strip(), params)
        if path == "/api/dashboard/refresh":
            self.workflow.invalidate_stats_cache()
            data = self._dashboard_data()
            data["data_sources"] = collect_data_sources(self.workspace_root, data["corpus"])
            return {
                "ok": True,
                "corpus": data["corpus"],
                "embedding": data["embedding"],
                "benchmarks": data["benchmarks"],
                "health": data["health"],
                "data_sources": data["data_sources"],
                "generated_at": data["generated_at"],
            }
        raise FileNotFoundError(path)

    def get(self, path: str) -> dict[str, Any]:
        route = urlparse(path).path
        if route == "/api/benchmark/status":
            return self.benchmarks.status()
        if route == "/api/workflow/status":
            return self.workflow.status()
        if route == "/api/workflow/catalog":
            return self.workflow.catalog()
        if route == "/api/system/status":
            return self.workflow.system_status()
        raise FileNotFoundError(path)

    def close(self) -> None:
        self.search.close()


class DashboardBenchmarkService:
    """Run and expose retrieval benchmarks without requiring a terminal."""

    def __init__(self, workspace_root: str | Path, search: "DashboardSearchService") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.search = search
        self.cases_path = self.workspace_root / "data" / "benchmark" / "ragas_cases.jsonl"
        self.report_dir = self.workspace_root / "data" / "benchmark"
        self._lock = threading.Lock()
        self._job: dict[str, Any] = {
            "status": "idle",
            "mode": None,
            "completed": 0,
            "total": 0,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            job = dict(self._job)
        reports = _collect_benchmark_payloads(self.report_dir)
        return {
            "testset": {
                "path": str(self.cases_path),
                "cases": _count_jsonl(self.cases_path),
                "available": self.cases_path.is_file(),
                "updated_at": _file_timestamp(self.cases_path),
            },
            "reports": reports,
            "job": job,
        }

    def start(self, mode: str) -> dict[str, Any]:
        if mode not in {"lexical", "hybrid"}:
            raise ValueError("评测模式必须是 lexical 或 hybrid")
        total = _count_jsonl(self.cases_path)
        if total == 0:
            raise ValueError("尚未发现可用测试集，请先生成 ragas_cases.jsonl")
        with self._lock:
            if self._job["status"] == "running":
                raise ValueError("已有评测正在运行，请等待完成")
            self._job = {
                "status": "running",
                "mode": mode,
                "completed": 0,
                "total": total,
                "started_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
                "error": None,
            }
        threading.Thread(target=self._run, args=(mode,), daemon=True).start()
        return self.status()

    def _run(self, mode: str) -> None:
        try:
            output_json = self.report_dir / f"{mode}_benchmark_results.json"
            output_markdown = self.report_dir / f"{mode}_benchmark_results.md"

            def progress(completed: int, total: int) -> None:
                with self._lock:
                    self._job["completed"] = completed
                    self._job["total"] = total

            self.search.evaluate_benchmark(
                mode=mode,
                cases_path=self.cases_path,
                output_json=output_json,
                output_markdown=output_markdown,
                progress=progress,
            )
        except Exception as error:  # noqa: BLE001 - report background failures to the page
            with self._lock:
                self._job["status"] = "failed"
                self._job["error"] = f"{type(error).__name__}: {error}"
                self._job["finished_at"] = datetime.now(UTC).isoformat()
            return
        with self._lock:
            self._job["status"] = "succeeded"
            self._job["completed"] = self._job["total"]
            self._job["finished_at"] = datetime.now(UTC).isoformat()


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _file_timestamp(path: Path) -> str | None:
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _collect_benchmark_payloads(directory: Path) -> list[dict[str, Any]]:
    from ue_rag.visualization import _collect_benchmarks

    return _collect_benchmarks(_discover_benchmarks(directory))


class DashboardSearchService:
    """Run read-only retrieval tests for the visual dashboard."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.index_path = self.workspace_root / "data" / "index" / "lexical.sqlite3"
        self.ue_config = self.workspace_root / "config" / "ue58.yaml"
        self.retrieval_config = self.workspace_root / "config" / "retrieval.yaml"
        self.embedding_config = self.workspace_root / "config" / "embedding.yaml"
        self.qdrant_config = self.workspace_root / "config" / "qdrant_server.yaml"
        self._provider: Any | None = None
        self._store: Any | None = None
        self._lock = threading.Lock()
        self._hybrid_lock = threading.Lock()
        self._ready = threading.Event()
        self._preload_error: str | None = None
        self._preload_thread = threading.Thread(target=self._preload, daemon=True)
        self._preload_thread.start()

    def query(
        self,
        *,
        question: str,
        mode: str = "lexical",
        limit: int = 8,
        source_type: str | None = None,
        module: str | None = None,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("请输入要测试的问题")
        if mode not in {"symbol", "lexical", "hybrid"}:
            raise ValueError("检索模式必须是 symbol、lexical 或 hybrid")
        if not 1 <= limit <= 20:
            raise ValueError("返回数量必须在 1 到 20 之间")
        filters = {
            key: value
            for key, value in {"source_type": source_type, "module": module}.items()
            if value
        }
        started = time.perf_counter()
        with self._lock, LexicalIndex(self.index_path) as lexical:
            if mode == "symbol":
                results = lexical.search_symbol(question, limit=limit, filters=filters)
            elif mode == "lexical":
                results = lexical.search(question, limit=limit, filters=filters)
            else:
                self._ready.wait(timeout=60)
                if self._preload_error is not None:
                    raise RuntimeError(f"模型预热失败：{self._preload_error}")
                if self._provider is None or self._store is None:
                    raise RuntimeError("模型预热未完成，请稍后重试")
                retriever = HybridRetriever(
                    self._store,
                    lexical,
                    embedding_provider=self._provider,
                    config=load_retrieval_config(self.ue_config, self.retrieval_config),
                )
                results = retriever.retrieve(question, fusion_top_k=limit, filters=filters)
        duration_ms = round((time.perf_counter() - started) * 1000)
        return {
            "question": question,
            "mode": mode,
            "duration_ms": duration_ms,
            "count": len(results),
            "results": [self._result(position, result) for position, result in enumerate(results, 1)],
        }

    def evaluate_benchmark(
        self,
        *,
        mode: str,
        cases_path: Path,
        output_json: Path,
        output_markdown: Path,
        progress: Any | None = None,
    ) -> None:
        """Evaluate the shared indexes and atomically publish a dashboard report."""

        from ue_rag.eval import BenchmarkEvaluator, load_benchmark_config, load_cases

        cases = load_cases([cases_path])
        config = load_benchmark_config(
            self.workspace_root / "config" / "ue58.yaml",
            self.workspace_root / "config" / "benchmark.yaml",
        ).model_copy(update={"output_json": output_json, "output_markdown": output_markdown})
        completed = 0

        with self._lock, LexicalIndex(self.index_path) as lexical:
            search: Any = lexical.search
            if mode == "hybrid":
                self._ready.wait(timeout=60)
                if self._preload_error is not None:
                    raise RuntimeError(f"模型预热失败：{self._preload_error}")
                if self._provider is None or self._store is None:
                    raise RuntimeError("模型预热未完成，请稍后重试")
                hybrid = HybridRetriever(
                    self._store,
                    lexical,
                    embedding_provider=self._provider,
                    config=load_retrieval_config(self.ue_config, self.retrieval_config),
                )
                search = lambda query, limit=10, filters=None: hybrid.retrieve(
                    query, fusion_top_k=limit, filters=filters
                )

            def tracked_search(query: str, limit: int = 10, filters: Any = None) -> Any:
                nonlocal completed
                results = search(query, limit=limit, filters=filters)
                completed += 1
                if progress is not None:
                    progress(completed, len(cases))
                return results

            evaluator = BenchmarkEvaluator(tracked_search, config)
            evaluator.write(evaluator.evaluate(cases))

    def _preload(self) -> None:
        '''Warm up the embedding model and vector store in the background.'''
        try:
            self._ensure_hybrid()
            # Constructing the provider is lazy; force one inference so the first
            # unseen Hybrid query does not pay the model's cold-start cost.
            _ = self._provider.dimension
            self._provider.embed_query("Unreal Engine RAG warmup")
            self._ready.set()
        except Exception as error:  # noqa: BLE001 - surface to the UI later
            self._preload_error = f"{type(error).__name__}: {error}"
            self._ready.set()

    def _ensure_hybrid(self) -> None:
        with self._hybrid_lock:
            if self._provider is not None and self._store is not None:
                return
            from ue_rag.embedding import QwenEmbeddingProvider, load_embedding_config
            from ue_rag.index import QdrantVectorStore, load_qdrant_config

            self._provider = QwenEmbeddingProvider(
                load_embedding_config(self.ue_config, self.embedding_config)
            )
            self._store = QdrantVectorStore.from_config(load_qdrant_config(self.qdrant_config))

    @staticmethod
    def _result(position: int, result: Any) -> dict[str, Any]:
        metadata = result.metadata or {}
        content = result.content.strip()
        if len(content) > 1_600:
            content = content[:1_597].rstrip() + "..."
        return {
            "position": position,
            "chunk_id": result.chunk_id,
            "symbol": metadata.get("symbol") or metadata.get("class_name") or result.chunk_id,
            "file_path": metadata.get("file_path") or "",
            "source_type": result.source_type.value,
            "module": metadata.get("module") or "",
            "score": float(result.score),
            "dense_rank": result.dense_rank,
            "sparse_rank": result.sparse_rank,
            "fusion_score": result.fusion_score,
            "content": content,
        }

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()
            self._provider = None
        if self._store is not None and getattr(self._store, "client", None) is not None:
            self._store.client.close()
        self._store = None


def _pick_directory() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="选择 Unreal Engine 项目目录", mustexist=True)
        root.destroy()
        return selected
    except Exception as error:
        raise RuntimeError("无法打开系统文件夹选择器，请直接输入项目目录") from error


def _handler(application: DashboardApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "UERAGDashboard/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            if route in {
                "/api/benchmark/status",
                "/api/workflow/status",
                "/api/workflow/catalog",
                "/api/system/status",
            }:
                try:
                    self._json(HTTPStatus.OK, application.get(self.path))
                except (OSError, ValueError, RuntimeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if route not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            page = application.render_page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 64 * 1024:
                    raise ValueError("请求内容过大")
                raw = self.rfile.read(length)
                payload = json.loads(raw or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("请求格式无效")
                result = application.post(self.path, payload)
                self._json(HTTPStatus.OK, result)
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            except (OSError, KeyError, ValueError, RuntimeError, sqlite3.Error) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"同步失败：{type(error).__name__}: {error}"},
                )

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def serve_dashboard(
    workspace_root: str | Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    print("正在加载 RAG 统计与管理服务…", flush=True)
    application = DashboardApplication(workspace_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(application))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"RAG 数据管理台：{url}", flush=True)
    print("关闭此窗口即可停止管理服务。", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        application.close()
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the local RAG data management dashboard.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        print("Error: port must be between 1 and 65535")
        return 2
    try:
        serve_dashboard(args.workspace, port=args.port, open_browser=not args.no_open)
    except (OSError, KeyError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"Error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
