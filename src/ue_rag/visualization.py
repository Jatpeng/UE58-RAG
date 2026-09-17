"""Generate a private, self-contained dashboard for local RAG data."""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import webbrowser
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


LENGTH_BUCKETS = (
    ("< 500", 0, 500),
    ("500–1.5K", 500, 1_500),
    ("1.5–3K", 1_500, 3_000),
    ("3–6K", 3_000, 6_000),
    ("≥ 6K", 6_000, None),
)


def collect_dashboard_data(
    *,
    index_path: str | Path = "data/index/lexical.sqlite3",
    chunks_dir: str | Path = "data/chunks",
    benchmark_paths: list[str | Path] | None = None,
    embeddings_dir: str | Path = "data/embeddings",
    engine_version: str = "5.8",
) -> dict[str, Any]:
    """Collect aggregate metadata without copying source content into the report."""

    index = Path(index_path)
    chunks = Path(chunks_dir)
    embeddings = Path(embeddings_dir)
    corpus = _collect_from_index(index) if index.is_file() else _collect_from_jsonl(chunks)
    reports = _collect_benchmarks(benchmark_paths or [])
    embedding = _collect_embeddings(embeddings)
    workspace_root = index.resolve().parents[2] if len(index.resolve().parents) > 2 else Path.cwd()
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine_version": engine_version,
        "corpus": corpus,
        "embedding": embedding,
        "benchmarks": reports,
        "data_sources": collect_data_sources(workspace_root, corpus),
        "health": {
            "lexical": index.is_file(),
            "vectors": embedding["records"] > 0,
            "benchmark": bool(reports),
        },
    }


def collect_data_sources(workspace_root: str | Path, corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe indexed and locally prepared data sources without reading their content."""

    root = Path(workspace_root).resolve()
    counts = {str(item["name"]): int(item["count"]) for item in corpus.get("sources", [])}
    sources: list[dict[str, Any]] = []
    ue_root = os.environ.get("UE_ROOT", "").strip()
    config_path = root / "config" / "ue58.yaml"
    if not ue_root and config_path.is_file():
        try:
            ue_root = str(((yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("engine") or {}).get("root") or "").strip()
        except (OSError, yaml.YAMLError):
            ue_root = ""
    engine_artifact = root / "data" / "chunks" / "engine" / "chunks.jsonl"
    engine_count = counts.get("engine_source", 0)
    if engine_count or engine_artifact.is_file():
        sources.append(
            {
                "id": "engine-source",
                "type": "engine_source",
                "name": "UE 5.8 引擎 C++ 源码",
                "location": str(Path(ue_root).expanduser()) if ue_root else str(engine_artifact.parent),
                "chunks": engine_count,
                "status": "indexed" if engine_count else "available",
                "detail": "类、方法、字段与 UE 反射宏语义索引",
            }
        )

    docs_artifact = root / "data" / "chunks" / "docs" / "chunks.jsonl"
    docs_count = counts.get("docs", 0) + counts.get("api", 0)
    if docs_count or docs_artifact.is_file():
        sources.append(
            {
                "id": "epic-docs",
                "type": "docs",
                "name": "Epic Games UE 官方文档",
                "location": "dev.epicgames.com / " + str(docs_artifact.parent),
                "chunks": docs_count,
                "status": "indexed" if docs_count else "available",
                "detail": "官方文档与 API 页面",
            }
        )

    blueprint_artifact = root / "data" / "chunks" / "project" / "blueprints.jsonl"
    blueprint_count = counts.get("blueprint", 0)
    if blueprint_count or blueprint_artifact.is_file():
        sources.append(
            {
                "id": "blueprints",
                "type": "blueprint",
                "name": "Blueprint 资产",
                "location": str(blueprint_artifact),
                "chunks": blueprint_count,
                "status": "indexed" if blueprint_count else "available",
                "detail": "资产、图、函数、变量与组件视图",
            }
        )

    state_dir = root / "data" / "parsed" / "project" / "sync_state"
    if state_dir.is_dir():
        for meta_path in sorted(state_dir.glob("*.meta.json")):
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sources.append(
                {
                    "id": f"project-{metadata.get('project_key', meta_path.stem)}",
                    "type": "project_source",
                    "name": str(metadata.get("project_name") or "UE 项目"),
                    "location": str(metadata.get("project_root") or "未知位置"),
                    "chunks": int(metadata.get("chunks", 0)),
                    "status": "indexed",
                    "detail": f"{int(metadata.get('files', 0)):,} 个文件 · 最近同步 {metadata.get('synced_at', '未知')}",
                }
            )

    represented = {item["type"] for item in sources}
    labels = {"lyra": "Lyra 示例项目", "project_source": "项目源码"}
    for source_type, count in sorted(counts.items()):
        if count and source_type not in represented:
            sources.append(
                {
                    "id": f"source-{source_type}",
                    "type": source_type,
                    "name": labels.get(source_type, source_type),
                    "location": "SQLite 检索索引",
                    "chunks": count,
                    "status": "indexed",
                    "detail": "已进入当前 RAG 语料库",
                }
            )
    return sources


def _collect_from_index(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(str(path))
    try:
        total, documents, characters, symbols = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT document_id),
                   COALESCE(SUM(LENGTH(content)), 0),
                   COALESCE(SUM(CASE WHEN symbol IS NOT NULL AND symbol <> '' THEN 1 ELSE 0 END), 0)
            FROM chunks
            """
        ).fetchone()
        sources = _grouped_counts(connection, "source_type", 20)
        modules = _grouped_counts(connection, "module", 12)
        plugins = _grouped_counts(connection, "plugin", 10)
        symbol_types = _grouped_counts(connection, "symbol_type", 10)
        rows = connection.execute(
            """
            SELECT CASE
                WHEN LENGTH(content) < 500 THEN '< 500'
                WHEN LENGTH(content) < 1500 THEN '500–1.5K'
                WHEN LENGTH(content) < 3000 THEN '1.5–3K'
                WHEN LENGTH(content) < 6000 THEN '3–6K'
                ELSE '≥ 6K' END AS bucket,
                COUNT(*)
            FROM chunks GROUP BY bucket
            """
        ).fetchall()
        length_counts = dict(rows)
    finally:
        connection.close()
    return {
        "origin": "SQLite 索引",
        "chunks": total,
        "documents": documents,
        "characters": characters,
        "symbols": symbols,
        "index_bytes": path.stat().st_size,
        "sources": sources,
        "modules": modules,
        "plugins": plugins,
        "symbol_types": symbol_types,
        "lengths": [{"label": label, "count": length_counts.get(label, 0)} for label, _, _ in LENGTH_BUCKETS],
    }


def _grouped_counts(connection: sqlite3.Connection, field: str, limit: int) -> list[dict[str, Any]]:
    allowed = {"source_type", "module", "plugin", "symbol_type"}
    if field not in allowed:
        raise ValueError(f"unsupported aggregate field: {field}")
    rows = connection.execute(
        f"""SELECT {field}, COUNT(*) AS amount FROM chunks
             WHERE {field} IS NOT NULL AND {field} <> ''
             GROUP BY {field} ORDER BY amount DESC, {field} ASC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [{"name": name, "count": count} for name, count in rows]


def _collect_from_jsonl(directory: Path) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    modules: Counter[str] = Counter()
    plugins: Counter[str] = Counter()
    symbol_types: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    document_ids: set[str] = set()
    total = characters = symbols = 0
    for path in sorted(directory.rglob("*.jsonl")) if directory.is_dir() else []:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
                total += 1
                document_id = record.get("document_id")
                if document_id:
                    document_ids.add(str(document_id))
                content_length = len(record.get("content") or "")
                characters += content_length
                lengths[_length_bucket(content_length)] += 1
                if record.get("symbol"):
                    symbols += 1
                for key, counter in (
                    ("source_type", sources),
                    ("module", modules),
                    ("plugin", plugins),
                    ("symbol_type", symbol_types),
                ):
                    value = record.get(key)
                    if value:
                        counter[str(value)] += 1
    return {
        "origin": "JSONL 语料",
        "chunks": total,
        "documents": len(document_ids),
        "characters": characters,
        "symbols": symbols,
        "index_bytes": 0,
        "sources": _counter_rows(sources, 20),
        "modules": _counter_rows(modules, 12),
        "plugins": _counter_rows(plugins, 10),
        "symbol_types": _counter_rows(symbol_types, 10),
        "lengths": [{"label": label, "count": lengths[label]} for label, _, _ in LENGTH_BUCKETS],
    }


def _length_bucket(length: int) -> str:
    for label, minimum, maximum in LENGTH_BUCKETS:
        if length >= minimum and (maximum is None or length < maximum):
            return label
    raise AssertionError("unreachable length bucket")


def _counter_rows(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def _collect_benchmarks(paths: list[str | Path]) -> list[dict[str, Any]]:
    reports_by_name: dict[str, dict[str, Any]] = {}
    for value in sorted(paths, key=lambda item: Path(item).stat().st_mtime if Path(item).is_file() else 0):
        path = Path(value)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            overall = payload["overall"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid benchmark report: {path}") from error
        label = path.stem.replace("_benchmark_results", "").replace("benchmark_results", "lexical")
        name = label.replace("_", " ").strip().title()
        reports_by_name[name] = {
                "name": name,
                "cases": int(payload.get("cases", 0)),
                "passed": bool(payload.get("baseline_passed", False)),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
                "source_file": str(path.resolve()),
                "overall": {key: float(value) for key, value in overall.items()},
                "categories": [
                    {"name": name, **{key: float(value) for key, value in scores.items()}}
                    for name, scores in payload.get("by_category", {}).items()
                ],
            }
    return list(reports_by_name.values())


def _collect_embeddings(directory: Path) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.manifest.json")) if directory.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        manifests.append(
            {
                "name": path.parent.name,
                "records": int(payload.get("records", 0)),
                "dimension": int(payload.get("dimension", 0)),
                "model": str(payload.get("model", "unknown")),
            }
        )
    return {
        "records": sum(item["records"] for item in manifests),
        "dimensions": sorted({item["dimension"] for item in manifests if item["dimension"]}),
        "models": sorted({item["model"] for item in manifests}),
        "manifests": manifests,
    }


def render_dashboard(data: dict[str, Any]) -> str:
    """Render dashboard data as a self-contained HTML document."""

    safe_json = (
        json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    title = html.escape(f"UE {data['engine_version']} RAG 数据观测台")
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", safe_json)


def write_dashboard(data: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically write the generated dashboard and return its resolved path."""

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    stats_output = output.with_suffix(".stats.json")
    temporary_stats = stats_output.with_suffix(stats_output.suffix + ".tmp")
    try:
        temporary.write_text(render_dashboard(data), encoding="utf-8")
        temporary_stats.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        temporary_stats.replace(stats_output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_stats.unlink(missing_ok=True)
        raise
    return output


def _discover_benchmarks(directory: Path) -> list[Path]:
    return sorted(directory.glob("*benchmark_results.json")) if directory.is_dir() else []


def _read_engine_version(path: Path) -> str:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return str(payload["engine"]["version"])
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local RAG data dashboard.")
    parser.add_argument("--index", type=Path, default=Path("data/index/lexical.sqlite3"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--benchmark", type=Path, action="append", help="Benchmark JSON; repeat to compare.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/rag_dashboard.html"))
    parser.add_argument("--open", action="store_true", dest="open_browser", help="Open the report in a browser.")
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmarks = args.benchmark or _discover_benchmarks(args.benchmark_dir)
    try:
        data = collect_dashboard_data(
            index_path=args.index,
            chunks_dir=args.chunks_dir,
            benchmark_paths=benchmarks,
            embeddings_dir=args.embeddings_dir,
            engine_version=_read_engine_version(args.ue_config),
        )
        output = write_dashboard(data, args.output)
    except (OSError, sqlite3.Error, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Dashboard: {output}")
    print(f"Chunks: {data['corpus']['chunks']:,}")
    print(f"Benchmarks: {len(data['benchmarks'])}")
    if args.open_browser:
        webbrowser.open(output.as_uri())
    return 0


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0f172a;--bg-deep:#090f1d;--panel:#151f33;--panel-strong:#1b263b;--surface:#10192a;--line:#334155;--line-soft:#273449;--text:#f8fafc;--muted:#94a3b8;--accent:#22c55e;--accent-soft:#163927;--blue:#38bdf8;--blue-soft:#102f45;--orange:#f59e0b;--red:#ef4444;--red-soft:#401c24;--radius:14px;--shadow:0 18px 50px rgba(2,6,23,.26)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;min-width:320px;min-height:100dvh;background:radial-gradient(circle at 90% -10%,rgba(56,189,248,.12),transparent 32%),linear-gradient(180deg,var(--bg-deep),var(--bg));color:var(--text);font:400 16px/1.55 "IBM Plex Sans","Segoe UI","Microsoft YaHei",sans-serif}button,input,select,textarea{font:inherit}.skip-link{position:fixed;left:16px;top:-64px;z-index:1000;padding:10px 14px;border-radius:8px;background:var(--accent);color:#052e16;font-weight:800;text-decoration:none}.skip-link:focus{top:12px}.shell{width:min(100%,1480px);margin:auto;padding:24px}.topline{height:2px;background:linear-gradient(90deg,var(--accent),var(--blue),transparent 72%);position:fixed;inset:0 0 auto;z-index:100}
header{display:flex;justify-content:space-between;gap:32px;align-items:flex-end;padding:34px 0 24px}.brand-lockup{display:flex;align-items:center;gap:16px;min-width:0}.brand-mark{display:grid;place-items:center;width:48px;height:48px;flex:0 0 auto;border:1px solid #416076;border-radius:12px;background:linear-gradient(145deg,#1d3046,#101b2e);color:var(--blue);font:800 15px/1 "JetBrains Mono",ui-monospace,monospace;box-shadow:inset 0 1px rgba(255,255,255,.05)}.eyebrow{color:var(--blue);font:700 11px/1.4 "JetBrains Mono",ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase}.brand h1{font-size:clamp(28px,3.6vw,48px);line-height:1.08;margin:6px 0;letter-spacing:-.045em;text-wrap:balance}.brand p,.timestamp{color:var(--muted);margin:0;font-size:13px}.stamp{text-align:right}.status{display:inline-flex;align-items:center;gap:9px;padding:8px 12px;border:1px solid #28513b;border-radius:999px;background:rgba(22,57,39,.72);margin-bottom:8px;color:#bbf7d0;font:700 11px "JetBrains Mono",ui-monospace,monospace}.dot{width:7px;height:7px;background:var(--accent);box-shadow:0 0 12px rgba(34,197,94,.8);border-radius:50%}
.section-nav{position:sticky;top:10px;z-index:40;display:flex;align-items:center;gap:4px;margin:0 0 12px;padding:6px;border:1px solid rgba(51,65,85,.88);border-radius:12px;background:rgba(9,15,29,.88);box-shadow:0 8px 30px rgba(2,6,23,.24);backdrop-filter:blur(14px);overflow:auto}.section-nav a{min-height:36px;display:inline-flex;align-items:center;padding:7px 12px;border-radius:8px;color:var(--muted);font-size:12px;font-weight:700;text-decoration:none;white-space:nowrap;transition:background .18s,color .18s}.section-nav a:hover{background:var(--panel-strong);color:var(--text)}
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:10px;scroll-margin-top:72px}.card,.panel{background:linear-gradient(145deg,rgba(27,38,59,.96),rgba(16,25,42,.98));border:1px solid var(--line-soft);border-radius:var(--radius);box-shadow:var(--shadow)}.card{padding:16px;min-height:116px;position:relative;overflow:hidden}.card:before{content:"";position:absolute;inset:0 auto auto 0;width:100%;height:2px;background:linear-gradient(90deg,var(--blue),transparent 70%);opacity:.7}.label{color:var(--muted);font-size:12px;font-weight:600}.value{font:700 clamp(22px,2.4vw,32px)/1.05 "JetBrains Mono",ui-monospace,monospace;margin:16px 0 7px;white-space:nowrap;font-variant-numeric:tabular-nums}.sub{font-size:12px;color:var(--muted);overflow-wrap:anywhere}
.flow{display:grid;grid-template-columns:repeat(7,auto);align-items:center;justify-content:center;gap:10px;padding:12px;margin-bottom:10px;border:1px solid var(--line-soft);border-radius:var(--radius);background:rgba(9,15,29,.48)}.node{padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--surface);font:600 12px "JetBrains Mono",ui-monospace,monospace}.arrow{color:var(--blue)}.grid{display:grid;grid-template-columns:1.05fr 1.4fr 1fr;gap:10px;margin-bottom:10px;scroll-margin-top:72px}.panel{padding:18px;min-width:0;scroll-margin-top:72px}.panel.wide{grid-column:span 2}.panel.full{grid-column:1/-1}.panel h2{font-size:15px;line-height:1.3;margin:0}.panel-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:16px}.tag{flex:0 0 auto;font:700 9px "JetBrains Mono",ui-monospace,monospace;letter-spacing:.06em;color:#bae6fd;border:1px solid #28506b;border-radius:999px;padding:5px 8px;background:var(--blue-soft)}.empty{color:var(--muted);display:grid;place-items:center;min-height:140px;text-align:center}
.source-layout{display:grid;grid-template-columns:150px 1fr;gap:24px;align-items:center}.donut{width:144px;height:144px;border-radius:50%;display:grid;place-items:center;position:relative}.donut:after{content:"";position:absolute;inset:25px;border-radius:50%;background:var(--panel)}.donut-center{z-index:1;text-align:center;font:700 20px "JetBrains Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}.donut-center small{display:block;font:9px "JetBrains Mono",ui-monospace,monospace;color:var(--muted);margin-top:4px}.rows{display:grid;gap:9px}.bar-row{display:grid;grid-template-columns:minmax(82px,1fr) minmax(90px,2fr) auto;gap:10px;align-items:center;font-size:12px}.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#cbd5e1}.track{height:6px;background:#243147;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),#60a5fa)}.bar-value{font:11px "JetBrains Mono",ui-monospace,monospace;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.histogram{height:180px;display:flex;gap:10px;align-items:flex-end;padding-top:10px}.column{height:100%;flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:7px}.column-value{font:9px "JetBrains Mono",ui-monospace,monospace;color:var(--muted);font-variant-numeric:tabular-nums}.column-bar{width:min(42px,78%);min-height:3px;border-radius:6px 6px 2px 2px;background:linear-gradient(180deg,#fbbf24,#d97706)}.column-label{font-size:9px;color:var(--muted);white-space:nowrap}
.metrics{display:grid;gap:10px}.metric-group{border:1px solid var(--line-soft);border-radius:10px;padding:12px;background:rgba(9,15,29,.34)}.metric-title{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px;font-size:12px}.pass{color:#86efac}.fail{color:#fca5a5}.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.metric{background:var(--surface);border:1px solid rgba(51,65,85,.5);border-radius:7px;padding:8px}.metric b{display:block;font:700 15px "JetBrains Mono",ui-monospace,monospace;margin-top:5px;font-variant-numeric:tabular-nums}.metric span{font-size:9px;color:var(--muted)}
.health{display:grid;gap:11px}.health-row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line-soft);padding-bottom:10px;font-size:12px}.pill{font:800 9px "JetBrains Mono",ui-monospace,monospace;padding:4px 7px;border-radius:6px;background:var(--accent-soft);color:#86efac}.pill.off{background:var(--red-soft);color:#fca5a5}.model{margin-top:12px;padding:11px;background:var(--surface);border:1px solid rgba(51,65,85,.5);border-radius:8px;color:#b6c4d6;font:10px/1.65 "JetBrains Mono",ui-monospace,monospace;overflow-wrap:anywhere}
.operations{display:grid;grid-template-columns:1.35fr 1fr;gap:10px;margin-bottom:10px}.manager,.benchmark-console,.source-panel,.rag-test{margin-bottom:10px}.manager{padding:18px}.manager-grid{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:end}.field label{display:block;color:#cbd5e1;font-size:11px;font-weight:600;margin:0 0 7px}.input-wrap{display:flex;gap:8px}.input-wrap input,.question-box,.query-controls select,.query-controls input{width:100%;min-width:0;background:#0b1424;border:1px solid var(--line);border-radius:9px;color:var(--text);outline:none}.input-wrap input{padding:11px 12px}.input-wrap input:focus,.question-box:focus,.query-controls select:focus,.query-controls input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(56,189,248,.14)}button{min-height:40px;border:0;border-radius:9px;padding:9px 14px;font-weight:750;cursor:pointer;white-space:nowrap;touch-action:manipulation;transition:background .18s,border-color .18s,color .18s,opacity .18s}.secondary{background:#243147;color:#dbeafe;border:1px solid #3b4b64}.secondary:hover{background:#2d3b53}.primary{background:var(--accent);color:#052e16}.primary:hover{background:#4ade80}.primary:disabled,.secondary:disabled{opacity:.45;cursor:not-allowed}.manager-actions{display:flex;gap:8px}.sync-options{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;margin-top:11px}.sync-options input{accent-color:var(--accent)}.sync-status{margin-top:14px;border:1px solid var(--line-soft);border-radius:10px;background:var(--surface);padding:12px;display:none}.sync-status.show{display:block}.sync-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.sync-stat{padding:9px;border-radius:7px;background:var(--panel-strong)}.sync-stat b{font:700 18px "JetBrains Mono",ui-monospace,monospace;display:block;margin-top:3px}.sync-stat span{font-size:9px;color:var(--muted)}.change-preview{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:10px}.change-col{min-width:0}.change-col h3{font-size:11px;margin:0 0 6px}.change-col ul{padding:0;margin:0;list-style:none;max-height:110px;overflow:auto}.change-col li{font:10px/1.6 "JetBrains Mono",ui-monospace,monospace;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sync-message{font-size:12px;color:#cbd5e1;margin-top:10px}.sync-message.error{color:#fca5a5}
.source-catalog{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:9px}.source-card{background:var(--surface);border:1px solid var(--line-soft);border-radius:10px;padding:14px;min-width:0}.source-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}.source-card h3{font-size:13px;margin:0}.source-state{font:800 9px "JetBrains Mono",ui-monospace,monospace;border-radius:6px;padding:4px 6px;color:#86efac;background:var(--accent-soft)}.source-state.available{color:#fcd34d;background:#3d2c12}.source-count{font:700 22px "JetBrains Mono",ui-monospace,monospace;margin:12px 0 4px;font-variant-numeric:tabular-nums}.source-count small{font:9px "JetBrains Mono",ui-monospace,monospace;color:var(--muted);margin-left:5px}.source-location{font:10px/1.5 "JetBrains Mono",ui-monospace,monospace;color:#93c5fd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:9px}.source-detail{font-size:10px;color:var(--muted);margin-top:5px}
.rag-test>label.field{display:block;color:#cbd5e1;font-size:11px;font-weight:600;margin:0 0 7px}.question-box{min-height:88px;resize:vertical;padding:12px;font-size:13px;line-height:1.6}.query-controls{display:grid;grid-template-columns:150px 170px minmax(140px,1fr) auto;gap:8px;margin-top:8px}.query-controls select,.query-controls input{padding:10px}.examples{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.example-chip,.history-item{min-height:34px;background:#17253a;color:#bcd4e8;border:1px solid #30445f;padding:6px 10px;border-radius:999px;font-size:10px;font-weight:600}.example-chip:hover,.history-item:hover{border-color:var(--blue);color:var(--text)}.query-status{font-size:11px;color:var(--muted);margin:12px 0 0}.search-results{display:grid;gap:8px;margin-top:10px}.result-card{background:var(--surface);border:1px solid var(--line-soft);border-radius:10px;padding:13px}.result-head{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center}.result-rank{display:grid;place-items:center;width:28px;height:28px;border-radius:7px;background:var(--blue-soft);color:#7dd3fc;font:700 12px "JetBrains Mono",ui-monospace,monospace}.result-title{font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.result-score{font:11px "JetBrains Mono",ui-monospace,monospace;color:#fbbf24}.result-meta{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0;color:var(--muted);font:9px "JetBrains Mono",ui-monospace,monospace}.result-meta span{background:#1a2638;border-radius:5px;padding:3px 6px}.result-content{white-space:pre-wrap;max-height:220px;overflow:auto;margin:0;padding:10px;background:#080f1c;border-radius:8px;color:#c1cede;font:10px/1.6 "JetBrains Mono",ui-monospace,monospace;overflow-wrap:anywhere}
.benchmark-controls{display:grid;grid-template-columns:minmax(240px,1fr) auto;gap:14px;align-items:center}.benchmark-state{display:grid;gap:7px}.benchmark-state-line{display:flex;justify-content:space-between;gap:12px;font-size:12px}.benchmark-progress{height:7px;background:#243147;border-radius:99px;overflow:hidden}.benchmark-progress div{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--blue));transition:width .25s}.benchmark-actions{display:flex;gap:7px;flex-wrap:wrap}.benchmark-message{font-size:11px;color:var(--muted)}.benchmark-message.error{color:#fca5a5}
.workflow-shell{display:grid;gap:10px;margin-bottom:10px;scroll-margin-top:72px}.system-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.system-card{background:var(--surface);border:1px solid var(--line-soft);border-radius:10px;padding:12px}.system-card .label{margin-bottom:8px}.system-card b{display:block;font:700 15px/1.35 "JetBrains Mono",ui-monospace,monospace;margin-bottom:4px;overflow-wrap:anywhere}.system-card span{font-size:11px;color:var(--muted)}.workflow-viewport{border:1px solid var(--line-soft);border-radius:12px;background:rgba(9,15,29,.55);padding:14px}.workflow-viewport-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px}.workflow-active{display:grid;grid-template-columns:auto minmax(0,1fr) minmax(180px,.7fr);gap:14px;align-items:center;margin-bottom:12px;padding:12px;border:1px solid #2b5b76;border-radius:12px;background:linear-gradient(110deg,rgba(16,47,69,.72),rgba(22,57,39,.38));min-height:88px}.workflow-active-mark{display:grid;place-items:center;width:48px;height:48px;border-radius:50%;border:2px solid var(--blue);color:var(--blue);font:800 12px "JetBrains Mono",ui-monospace,monospace;box-shadow:0 0 0 5px rgba(56,189,248,.08)}.workflow-active-mark.running{animation:pulse 1.8s ease-in-out infinite}.workflow-active h3{margin:3px 0;font-size:16px}.workflow-active p{margin:0;color:#cbd5e1;font-size:12px}.workflow-active-meta{text-align:right;color:var(--muted);font:11px/1.6 "JetBrains Mono",ui-monospace,monospace}.workflow-active-meta b{display:block;color:var(--text);font-size:18px}.workflow-progress{height:8px;background:#243147;border-radius:99px;overflow:hidden;margin:8px 0}.workflow-progress div{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--blue));transition:width .25s}.workflow-log{margin:0;max-height:160px;overflow:auto;padding:10px;border-radius:8px;background:#080f1c;color:#9fb2c9;font:10px/1.65 "JetBrains Mono",ui-monospace,monospace;white-space:pre-wrap}.pipeline-grid{display:grid;gap:10px}.pipeline-card{border:1px solid var(--line-soft);border-radius:12px;background:rgba(16,25,42,.88);padding:14px}.pipeline-card h3{margin:0 0 4px;font-size:14px}.pipeline-desc{color:var(--muted);font-size:12px;margin-bottom:12px}.step-list{display:grid;gap:8px}.step-row{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(180px,1fr) auto;gap:10px;align-items:start;padding:10px;border:1px solid rgba(51,65,85,.55);border-radius:10px;background:var(--surface);transition:border-color .2s,background .2s,transform .2s}.step-row.current{border-color:var(--blue);background:linear-gradient(90deg,rgba(16,47,69,.8),var(--surface));transform:translateX(2px)}.step-row.done{border-color:#28513b}.step-row.current .step-main h4{color:#bae6fd}.step-row.done .step-main h4{color:#86efac}.step-main h4{margin:0 0 4px;font-size:13px}.step-main p{margin:0;color:var(--muted);font-size:11px}.step-artifact{margin-top:6px;font:10px/1.5 "JetBrains Mono",ui-monospace,monospace;color:#93c5fd;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.step-params{display:grid;gap:6px}.step-params label{display:grid;gap:4px;font-size:10px;color:#cbd5e1}.step-params input,.step-params select{width:100%;min-height:34px;padding:7px 9px;border-radius:8px;border:1px solid var(--line);background:#0b1424;color:var(--text)}.step-params .check{display:flex;align-items:center;gap:8px;min-height:34px}.step-params .check input{width:auto;min-height:0;accent-color:var(--accent)}.step-actions{display:flex;flex-direction:column;gap:6px;align-items:stretch}.step-ready{font:800 9px "JetBrains Mono",ui-monospace,monospace;padding:4px 6px;border-radius:6px;text-align:center}.step-ready.ok{background:var(--accent-soft);color:#86efac}.step-ready.wait{background:#3d2c12;color:#fcd34d}@keyframes pulse{50%{box-shadow:0 0 0 10px rgba(56,189,248,0)}}
.history-bar{display:flex;align-items:center;gap:8px;margin-top:10px;flex-wrap:wrap}.history-label{font:700 9px "JetBrains Mono",ui-monospace,monospace;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}.history-list{display:flex;gap:6px;flex-wrap:wrap;flex:1;min-width:0}.history-item{max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.history-empty{font-size:11px;color:var(--muted)}.history-clear{min-height:34px;background:transparent;border:1px solid var(--line);color:var(--muted);padding:6px 10px;font-size:10px}.history-clear:hover{color:#fca5a5;border-color:var(--red)}
footer{color:#64748b;font-size:11px;display:flex;justify-content:space-between;gap:16px;border-top:1px solid var(--line-soft);padding:20px 2px;margin-top:20px}a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #7dd3fc;outline-offset:2px}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*:before,*:after{transition-duration:.01ms!important;animation-duration:.01ms!important;animation-iteration-count:1!important}}
@media(max-width:1050px){.kpis{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr 1fr}.panel.wide{grid-column:span 2}.operations{grid-template-columns:1fr}.flow{overflow:auto;justify-content:start}.system-grid{grid-template-columns:1fr 1fr}.step-row{grid-template-columns:1fr}}
@media(max-width:720px){body{font-size:16px}.shell{padding:16px}.kpis,.grid{grid-template-columns:1fr 1fr}.panel.wide,.panel.full{grid-column:1/-1}header{align-items:flex-start;flex-direction:column}.stamp{text-align:left}.section-nav{top:6px}.source-layout{grid-template-columns:1fr}.donut{margin:auto}.metric-grid{grid-template-columns:1fr 1fr}.flow{display:none}.manager-grid,.benchmark-controls{grid-template-columns:1fr}.manager-actions{display:grid;grid-template-columns:1fr 1fr}.input-wrap{display:grid;grid-template-columns:1fr auto}.sync-summary{grid-template-columns:1fr 1fr}.change-preview{grid-template-columns:1fr}.query-controls{grid-template-columns:1fr 1fr}.query-controls input,.query-controls button{grid-column:1/-1}.system-grid{grid-template-columns:1fr}.workflow-active{grid-template-columns:auto 1fr}.workflow-active-meta{text-align:left;grid-column:2}.workflow-active-meta b{display:inline;margin-right:8px}footer{display:block;line-height:1.8}}
@media(max-width:440px){.shell{padding:12px}.brand-lockup{align-items:flex-start}.brand-mark{width:42px;height:42px}.kpis,.grid{grid-template-columns:1fr}.panel.wide,.panel.full{grid-column:auto}.panel{padding:15px}.input-wrap,.query-controls,.manager-actions{grid-template-columns:1fr}.input-wrap button,.query-controls>*{grid-column:auto}.bar-row{grid-template-columns:minmax(76px,1fr) minmax(70px,1.4fr) auto}}
</style>
</head><body><a class="skip-link" href="#main-content">跳到主要内容</a><div class="topline"></div><main class="shell" id="main-content"><header><div class="brand-lockup"><div class="brand-mark" aria-hidden="true">UE</div><div class="brand"><div class="eyebrow">Retrieval observability console</div><h1>UE <span id="version"></span> RAG 数据观测台</h1><p>语料观测、全流程流水线、检索验证与本地运维</p></div></div><div class="stamp"><div class="status"><span class="dot" aria-hidden="true"></span>本地服务在线</div><div class="timestamp" id="timestamp"></div></div></header>
<nav class="section-nav" aria-label="观测台区域"><a href="#kpis">总览</a><a href="#pipeline-console">全流程</a><a href="#observability">健康与分布</a><a href="#sources-panel">数据来源</a><a href="#query-lab">检索验证</a><a href="#operations">运维操作</a></nav>
<section class="kpis" id="kpis" aria-label="核心指标"></section><section class="flow" aria-label="RAG 数据处理链路"><div class="node">源数据</div><div class="arrow" aria-hidden="true">→</div><div class="node">语义切块</div><div class="arrow" aria-hidden="true">→</div><div class="node">Lexical + Vector</div><div class="arrow" aria-hidden="true">→</div><div class="node">Benchmark</div></section>
<section class="panel workflow-shell" id="pipeline-console"><div class="panel-head"><div><h2>RAG 全流程控制台</h2><div class="sub" style="margin-top:6px">启动采集、解析、切块、Embedding、索引与 MCP；进度与日志在下方视口实时展示</div></div><span class="tag">PIPELINE</span></div><div class="system-grid" id="system-status"></div><div class="workflow-viewport" style="margin-top:10px"><div class="workflow-viewport-head"><div><div class="label">任务视口</div><div class="benchmark-state-line" style="margin-top:6px"><span id="workflow-phase">待机 · 选择下方任一步骤开始</span><b id="workflow-job">IDLE</b></div></div><button class="secondary" id="refresh-dashboard" type="button">刷新统计</button></div><div class="workflow-active" id="workflow-active" aria-live="polite"><div class="workflow-active-mark" id="workflow-active-mark">IDLE</div><div><div class="label" id="workflow-active-label">当前没有运行中的功能</div><h3 id="workflow-active-title">等待启动流水线</h3><p id="workflow-active-detail">从下方任意步骤启动后，这里会显示正在处理的功能和实时进度。</p></div><div class="workflow-active-meta"><span>完成度</span><b id="workflow-active-percent">0%</b><span id="workflow-active-count">尚未开始</span></div></div><div class="workflow-progress" role="progressbar" aria-label="流水线进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="workflow-progress-bar"></div></div><div class="benchmark-message" id="workflow-message" role="status" aria-live="polite">同一时间只运行一个后台任务。</div><pre class="workflow-log" id="workflow-log">尚无日志</pre></div><div class="pipeline-grid" id="pipeline-list" style="margin-top:10px"></div></section>
<section class="grid" id="observability"><article class="panel wide"><div class="panel-head"><h2>语料构成</h2><span class="tag" id="origin"></span></div><div id="sources"></div></article><article class="panel"><div class="panel-head"><h2>系统就绪度</h2><span class="tag">STATUS</span></div><div id="health"></div></article><article class="panel"><div class="panel-head"><h2>切块字符长度</h2><span class="tag">DISTRIBUTION</span></div><div id="lengths"></div></article><article class="panel"><div class="panel-head"><h2>Top 模块</h2><span class="tag">MODULES</span></div><div id="modules"></div></article><article class="panel"><div class="panel-head"><h2>检索质量</h2><span class="tag">BENCHMARK</span></div><div id="benchmarks"></div></article><article class="panel full"><div class="panel-head"><h2>符号类型</h2><span class="tag">CODE INTELLIGENCE</span></div><div id="symbols"></div></article></section>
<section class="panel source-panel" id="sources-panel"><div class="panel-head"><div><h2>当前数据来源</h2><div class="sub" style="margin-top:6px">追踪已进入索引或已准备待索引的本地语料</div></div><span class="tag">PROVENANCE</span></div><div class="source-catalog" id="data-source-list"></div></section>
<section class="panel rag-test" id="query-lab"><div class="panel-head"><div><h2>RAG 检索测试</h2><div class="sub" style="margin-top:6px">输入问题，验证实际召回的符号、文件和上下文</div></div><span class="tag">QUERY LAB</span></div><label class="field" for="test-question">测试问题</label><textarea class="question-box" id="test-question" placeholder="例如：UCharacterMovementComponent 如何限制角色最大行走速度？"></textarea><div class="query-controls"><select id="test-mode" aria-label="检索模式"><option value="lexical">关键词 + 符号</option><option value="symbol">精确符号</option><option value="hybrid">Hybrid 混合检索</option></select><select id="test-source" aria-label="数据来源"><option value="">全部来源</option><option value="engine_source">引擎源码</option><option value="docs">官方文档</option><option value="project_source">项目源码</option><option value="blueprint">Blueprint</option></select><input id="test-module" aria-label="模块筛选" placeholder="模块筛选（可选）"><button class="primary" id="run-query">开始测试</button></div><div class="examples"><button class="example-chip">UCharacterMovementComponent::MaxWalkSpeed</button><button class="example-chip">角色移动网络预测如何工作？</button><button class="example-chip">Enhanced Input 如何绑定输入动作？</button><button class="example-chip">UPROPERTY EditAnywhere 和 BlueprintReadWrite 有什么区别？</button></div><div class="history-bar" id="history-bar"><span class="history-label">检索历史</span><div class="history-list" id="history-list"></div><button class="history-clear" id="history-clear" type="button">清空</button></div><div class="query-status" id="query-status" role="status" aria-live="polite">选择示例问题或输入自己的问题。</div><div class="search-results" id="search-results"></div></section>
<section class="operations" id="operations"><section class="panel manager"><div class="panel-head"><div><h2>增量添加项目数据</h2><div class="sub" style="margin-top:6px">扫描 UE 项目，预览变更后再写入检索索引</div></div><span class="tag">DATA SYNC</span></div><div class="manager-grid"><div class="field"><label for="project-root">UE 项目目录</label><div class="input-wrap"><input id="project-root" placeholder="例如 D:\Work\MyGame" autocomplete="off"><button class="secondary" id="pick-root">选择文件夹</button></div></div><div class="manager-actions"><button class="secondary" id="scan-project">扫描变更</button><button class="primary" id="apply-sync" disabled>确认同步</button></div></div><label class="sync-options"><input type="checkbox" id="sync-vectors" aria-label="同步向量索引" checked> 同步向量索引（需要本地 Embedding 模型和 Qdrant）</label><div class="sync-status" id="sync-status"><div class="sync-summary" id="sync-summary"></div><div class="change-preview" id="change-preview"></div><div class="sync-message" id="sync-message" role="status" aria-live="polite"></div></div></section>
<section class="panel benchmark-console"><div class="panel-head"><div><h2>检索评测中心</h2><div class="sub" style="margin-top:6px">同步报告或运行当前测试集</div></div><span class="tag">EVALUATION</span></div><div class="benchmark-controls"><div class="benchmark-state"><div class="benchmark-state-line"><span id="benchmark-testset">正在读取测试集…</span><b id="benchmark-job">待机</b></div><div class="benchmark-progress" role="progressbar" aria-label="评测进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="benchmark-progress-bar"></div></div><div class="benchmark-message" id="benchmark-message" role="status" aria-live="polite">评测在后台运行，完成后指标会自动同步。</div></div><div class="benchmark-actions"><button class="secondary" id="sync-benchmark">同步结果</button><button class="secondary" id="run-lexical-benchmark">测试 Lexical</button><button class="primary" id="run-hybrid-benchmark">测试 Hybrid</button></div></div></section></section>
<footer><span>仅包含聚合统计，不包含 UE 源码或文档正文。</span><span>UE RAG Observatory · Local only</span></footer></main>
<script>const D=__DATA__;const C=['#38bdf8','#22c55e','#f59e0b','#a78bfa','#fb7185','#60a5fa','#34d399'];const fmt=n=>Intl.NumberFormat('zh-CN',{notation:n>=1e6?'compact':'standard',maximumFractionDigits:1}).format(n||0);const pct=n=>(100*(n||0)).toFixed(1)+'%';const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));document.querySelector('#version').textContent=D.engine_version;document.querySelector('#timestamp').textContent='快照更新于 '+new Date(D.generated_at).toLocaleString('zh-CN');document.querySelector('#origin').textContent=D.corpus.origin;
const embCoverage=D.corpus.chunks?D.embedding.records/D.corpus.chunks:0;const cards=[['语义切块',fmt(D.corpus.chunks),'条可检索上下文'],['源文档',fmt(D.corpus.documents),'个唯一文档'],['代码符号',fmt(D.corpus.symbols),pct(D.corpus.symbols/Math.max(D.corpus.chunks,1))+' 符号覆盖'],['向量记录',fmt(D.embedding.records),pct(embCoverage)+' 向量覆盖'],['语料字符',fmt(D.corpus.characters),'聚合统计，不含正文']];document.querySelector('#kpis').innerHTML=cards.map(x=>`<article class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div><div class="sub">${x[2]}</div></article>`).join('');
function rows(items){if(!items.length)return '<div class="empty">暂无数据</div>';const max=Math.max(...items.map(x=>x.count),1);return `<div class="rows">${items.map(x=>`<div class="bar-row" title="${esc(x.name)}: ${x.count.toLocaleString()}"><div class="bar-label">${esc(x.name)}</div><div class="track"><div class="fill" style="width:${100*x.count/max}%"></div></div><div class="bar-value">${fmt(x.count)}</div></div>`).join('')}</div>`}
const src=D.corpus.sources,total=Math.max(D.corpus.chunks,1);let angle=0;const stops=src.map((x,i)=>{const start=angle;angle+=x.count/total*360;return `${C[i%C.length]} ${start}deg ${angle}deg`});const sourceSummary=src.map(x=>`${x.name} ${x.count.toLocaleString()} 条`).join('，');document.querySelector('#sources').innerHTML=src.length?`<div class="source-layout"><div class="donut" role="img" aria-label="语料构成：${esc(sourceSummary)}" style="background:conic-gradient(${stops.join(',')})"><div class="donut-center">${fmt(D.corpus.chunks)}<small>CHUNKS</small></div></div>${rows(src)}</div>`:'<div class="empty">未发现语料</div>';document.querySelector('#modules').innerHTML=rows(D.corpus.modules);document.querySelector('#symbols').innerHTML=rows(D.corpus.symbol_types);
const lens=D.corpus.lengths,maxLen=Math.max(...lens.map(x=>x.count),1);document.querySelector('#lengths').innerHTML=`<div class="histogram">${lens.map(x=>`<div class="column"><div class="column-value">${fmt(x.count)}</div><div class="column-bar" style="height:${Math.max(2,145*x.count/maxLen)}px"></div><div class="column-label">${x.label}</div></div>`).join('')}</div>`;
const healthNames={lexical:'关键词索引',vectors:'向量索引',benchmark:'评测报告'};document.querySelector('#health').innerHTML=`<div class="health">${Object.entries(D.health).map(([k,v])=>`<div class="health-row"><span>${healthNames[k]}</span><span class="pill ${v?'':'off'}">${v?'就绪':'缺失'}</span></div>`).join('')}</div><div class="model">Embedding<br>${D.embedding.models.map(esc).join('<br>')||'尚未发现模型清单'}<br>${D.embedding.dimensions.length?'DIM '+D.embedding.dimensions.join(', '):''}</div>`;
const metricNames={hit_at_1:'Hit@1',hit_at_5:'Hit@5',hit_at_10:'Hit@10',recall_at_10:'Recall@10',mrr:'MRR',ndcg:'nDCG'};function renderBenchmarks(){document.querySelector('#benchmarks').innerHTML=D.benchmarks.length?`<div class="metrics">${D.benchmarks.map(b=>`<div class="metric-group"><div class="metric-title"><b>${esc(b.name)}</b><span class="${b.passed?'pass':'fail'}">${b.cases} CASES · ${b.passed?'PASS':'CHECK'}</span></div><div class="sub" style="margin:5px 0 10px">${b.updated_at?'更新于 '+new Date(b.updated_at).toLocaleString('zh-CN'):'尚无更新时间'}</div><div class="metric-grid">${Object.entries(metricNames).map(([k,n])=>`<div class="metric"><span>${n}</span><b>${pct(b.overall[k])}</b></div>`).join('')}</div></div>`).join('')}</div>`:'<div class="empty">运行 Benchmark 后将在这里展示质量指标</div>'}renderBenchmarks();
function renderDataSources(){const items=D.data_sources||[];document.querySelector('#data-source-list').innerHTML=items.length?items.map(item=>`<article class="source-card"><div class="source-card-head"><h3>${esc(item.name)}</h3><span class="source-state ${item.status==='indexed'?'':'available'}">${item.status==='indexed'?'已索引':'待索引'}</span></div><div class="source-count">${fmt(item.chunks)}<small>CHUNKS</small></div><div class="source-location" title="${esc(item.location)}">${esc(item.location)}</div><div class="source-detail">${esc(item.detail)}</div></article>`).join(''):'<div class="empty">尚未发现数据来源</div>'}
renderDataSources();
const benchmarkTestset=document.querySelector('#benchmark-testset'),benchmarkJob=document.querySelector('#benchmark-job'),benchmarkBar=document.querySelector('#benchmark-progress-bar'),benchmarkMessage=document.querySelector('#benchmark-message'),syncBenchmarkButton=document.querySelector('#sync-benchmark'),runLexicalButton=document.querySelector('#run-lexical-benchmark'),runHybridButton=document.querySelector('#run-hybrid-benchmark');let benchmarkPoll=null;
async function getApi(path){const response=await fetch(path,{cache:'no-store'});const body=await response.json().catch(()=>({error:'服务返回了无效响应'}));if(!response.ok)throw new Error(body.error||'操作失败');return body}
function showBenchmarkStatus(state){const set=state.testset,job=state.job;benchmarkTestset.textContent=set.available?`测试集 ${set.cases} 题 · ${set.updated_at?new Date(set.updated_at).toLocaleString('zh-CN'):'时间未知'}`:'测试集尚未生成';const running=job.status==='running',percent=job.total?Math.round(job.completed/job.total*100):0,displayPercent=running?percent:(job.status==='succeeded'?100:0);benchmarkBar.style.width=displayPercent+'%';benchmarkBar.parentElement.setAttribute('aria-valuenow',String(displayPercent));benchmarkJob.textContent=running?`${job.mode.toUpperCase()} ${job.completed}/${job.total}`:job.status==='succeeded'?`${job.mode.toUpperCase()} 已完成`:job.status==='failed'?'测试失败':'待机';benchmarkMessage.textContent=job.error||((running?'正在逐题检索，页面可以保持打开…':job.status==='succeeded'?'测试完成，最新结果已同步。':'选择一种检索模式开始测试。'));benchmarkMessage.className='benchmark-message'+(job.error?' error':'');syncBenchmarkButton.disabled=running;runLexicalButton.disabled=running||!set.available;runHybridButton.disabled=running||!set.available;D.benchmarks=state.reports||[];renderBenchmarks();if(running){clearTimeout(benchmarkPoll);benchmarkPoll=setTimeout(syncBenchmarkStatus,1500)}}
async function syncBenchmarkStatus(){try{showBenchmarkStatus(await getApi('/api/benchmark/status'))}catch(error){benchmarkMessage.textContent=error.message;benchmarkMessage.className='benchmark-message error'}}
async function runBenchmark(mode){benchmarkMessage.textContent='正在启动评测…';try{showBenchmarkStatus(await api('/api/benchmark/run',{mode}))}catch(error){benchmarkMessage.textContent=error.message;benchmarkMessage.className='benchmark-message error'}}
syncBenchmarkButton.addEventListener('click',syncBenchmarkStatus);runLexicalButton.addEventListener('click',()=>runBenchmark('lexical'));runHybridButton.addEventListener('click',()=>runBenchmark('hybrid'));syncBenchmarkStatus();
function renderSystemStatus(system){const gpu=system.gpu||{},qdrant=system.qdrant||{},emb=system.embedding||{},paths=system.paths||{};const device=(gpu.devices&&gpu.devices[0])?`${gpu.devices[0].name} · ${gpu.devices[0].total_memory_gb} GB`:(gpu.cuda_available?'CUDA 可用':'CUDA 不可用');const cards=[['GPU / CUDA',device,gpu.cuda_available?`PyTorch CUDA ${gpu.cuda_version||''}`:(gpu.error||'检查 PyTorch 安装')],['Embedding',esc(emb.model||'未配置'),`device=${esc(emb.device||'?')} · batch=${esc(emb.batch_size||'?')}`],['Qdrant',qdrant.reachable?`${qdrant.status||'online'} · ${fmt(qdrant.points||0)} pts`:'不可达',esc(qdrant.url||'')],['UE_ROOT',esc(paths.ue_root||'未设置'),'引擎全量扫描需要此路径']];document.querySelector('#system-status').innerHTML=cards.map(x=>`<article class="system-card"><div class="label">${x[0]}</div><b>${x[1]}</b><span>${x[2]}</span></article>`).join('')}
function renderParam(param,stepId){const id=`param-${stepId}-${param.key}`;if(param.type==='checkbox'){const checked=param.default===true?'checked':'';return `<label class="check"><input type="checkbox" id="${id}" data-key="${esc(param.key)}" ${checked}> ${esc(param.label)}</label>`}if(param.type==='select'){const options=(param.options||[]).map(o=>`<option value="${esc(o.value)}" ${o.value===param.default?'selected':''}>${esc(o.label)}</option>`).join('');return `<label for="${id}">${esc(param.label)}<select id="${id}" data-key="${esc(param.key)}">${options}</select></label>`}return `<label for="${id}">${esc(param.label)}<input id="${id}" data-key="${esc(param.key)}" type="${param.type==='number'?'number':'text'}" placeholder="${esc(param.placeholder||'')}" value="${param.default!=null?esc(param.default):''}"></label>`}
function renderPipelines(catalog){const job=catalog.job||{},running=job.status==='running',activeId=job.step;document.querySelector('#pipeline-list').innerHTML=(catalog.pipelines||[]).map(function(pipeline){const steps=(pipeline.steps||[]).map(function(step){const art=step.artifact;let artMeta='';if(art){if(art.exists){artMeta=(art.records?fmt(art.records)+' 条 · ':'')+(art.updated_at?new Date(art.updated_at).toLocaleString('zh-CN'):'已就绪')}else{artMeta='产物尚未生成'}}const artHtml=art?('<div class="step-artifact" title="'+esc(art.path||'')+'">'+esc(art.path||'')+(artMeta?' · '+artMeta:'')+'</div>'):'';const paramsHtml=(step.params||[]).map(function(p){return renderParam(p,step.id)}).join('')||'<span class="sub">无额外参数</span>';const isCurrent=step.id===activeId&&running;const isDone=step.id===activeId&&job.status==='succeeded';const stateLabel=isCurrent?'进行中':isDone?'已完成':(step.ready?'可启动':'缺前置');return '<div class="step-row '+(isCurrent?'current ':'')+(isDone?'done':'')+'"><div class="step-main"><h4>'+esc(step.name)+'</h4><p>'+esc(step.description)+'</p>'+artHtml+'</div><div class="step-params">'+paramsHtml+'</div><div class="step-actions"><span class="step-ready '+(isCurrent?'ok':isDone?'ok':step.ready?'ok':'wait')+'">'+stateLabel+'</span><button class="primary workflow-run" data-step="'+esc(step.id)+'" '+(running?'disabled':'')+'>'+ (isCurrent?'运行中':'启动') +'</button></div></div>'}).join('');return '<article class="pipeline-card"><h3>'+esc(pipeline.name)+'</h3><div class="pipeline-desc">'+esc(pipeline.description)+'</div><div class="step-list">'+steps+'</div></article>'}).join('')}
let workflowPoll=null,workflowBusy=false;
function showWorkflowStatus(state){const job=state.job||{};const running=job.status==='running';const percent=job.total?Math.round(job.completed/job.total*100):0;const display=running?percent:(job.status==='succeeded'?100:0);document.querySelector('#workflow-progress-bar').style.width=display+'%';document.querySelector('#workflow-progress-bar').parentElement.setAttribute('aria-valuenow',String(display));document.querySelector('#workflow-job').textContent=running?((job.step||'').toUpperCase()+' '+job.completed+'/'+(job.total||'?')):(job.status==='succeeded'?'SUCCEEDED':job.status==='failed'?'FAILED':'IDLE');document.querySelector('#workflow-phase').textContent=job.phase||(job.status==='succeeded'?'任务完成':job.status==='failed'?'任务失败':'待机 · 选择下方任一步骤开始');const active=document.querySelector('#workflow-active'),mark=document.querySelector('#workflow-active-mark'),label=document.querySelector('#workflow-active-label'),title=document.querySelector('#workflow-active-title'),detail=document.querySelector('#workflow-active-detail'),activePercent=document.querySelector('#workflow-active-percent'),activeCount=document.querySelector('#workflow-active-count');const activeName=job.step?String(job.step).replaceAll('_',' '):'';active.classList.toggle('running',running);mark.classList.toggle('running',running);mark.textContent=running?'LIVE':job.status==='succeeded'?'DONE':'IDLE';label.textContent=running?'正在进行':'最近状态';title.textContent=running?(job.phase||activeName||'后台任务运行中'):job.status==='succeeded'?'刚刚完成：'+activeName:job.status==='failed'?'任务失败：'+activeName:'当前没有运行中的功能';detail.textContent=running?'管理台正在处理 '+activeName+'，下方对应步骤已高亮。':job.status==='succeeded'?'该功能已完成，刷新统计即可查看最新结果。':'从下方任意步骤启动后，这里会显示正在处理的功能和实时进度。';activePercent.textContent=display+'%';activeCount.textContent=job.total?(job.completed+' / '+job.total+' 项'):'尚未开始';const msg=document.querySelector('#workflow-message');if(job.error){msg.textContent=job.error;msg.className='benchmark-message error'}else if(job.summary){msg.textContent=typeof job.summary==='string'?job.summary:JSON.stringify(job.summary);msg.className='benchmark-message'}else{msg.textContent=running?'任务运行中，可保持页面打开…':'同一时间只运行一个后台任务。';msg.className='benchmark-message'}document.querySelector('#workflow-log').textContent=(state.log&&state.log.length)?state.log.join(String.fromCharCode(10)):'尚无日志';document.querySelectorAll('.workflow-run').forEach(function(btn){btn.disabled=running||workflowBusy});if(running){clearTimeout(workflowPoll);workflowPoll=setTimeout(pollWorkflowStatus,1500)}else if(job.status==='succeeded'){refreshCatalog()}}
function syncActiveStep(job){const running=job&&job.status==='running';document.querySelectorAll('.workflow-run').forEach(function(btn){const row=btn.closest('.step-row'),current=running&&btn.dataset.step===job.step,done=!running&&job&&job.status==='succeeded'&&btn.dataset.step===job.step;row.classList.toggle('current',current);row.classList.toggle('done',done);if(current){btn.textContent='运行中';const badge=row.querySelector('.step-ready');if(badge){badge.textContent='进行中';badge.className='step-ready ok'}}})}
async function pollWorkflowStatus(){try{const state=await getApi('/api/workflow/status');showWorkflowStatus(state);syncActiveStep(state.job)}catch(error){document.querySelector('#workflow-message').textContent=error.message;document.querySelector('#workflow-message').className='benchmark-message error'}}
async function refreshCatalog(){try{const catalog=await getApi('/api/workflow/catalog');renderSystemStatus(catalog.system||{});renderPipelines(catalog);showWorkflowStatus(catalog)}catch(error){document.querySelector('#workflow-message').textContent=error.message;document.querySelector('#workflow-message').className='benchmark-message error'}}
document.querySelector('#pipeline-list').addEventListener('click',async function(event){const button=event.target.closest('.workflow-run');if(!button||button.disabled)return;const step=button.dataset.step;const card=button.closest('.step-row');const collected={};card.querySelectorAll('.step-params [data-key]').forEach(function(el){const key=el.dataset.key;if(el.type==='checkbox')collected[key]=el.checked;else if(el.type==='number'){if(el.value.trim()!=='')collected[key]=Number(el.value)}else if(el.value.trim()!=='')collected[key]=el.value.trim()});workflowBusy=true;button.disabled=true;document.querySelector('#workflow-message').textContent='正在启动 '+step+'…';try{const state=await api('/api/workflow/start',{step:step,params:collected});showWorkflowStatus(state);syncActiveStep(state.job)}catch(error){document.querySelector('#workflow-message').textContent=error.message;document.querySelector('#workflow-message').className='benchmark-message error'}finally{workflowBusy=false}});
document.querySelector('#refresh-dashboard').addEventListener('click',async function(){document.querySelector('#workflow-message').textContent='正在刷新语料统计…';try{await api('/api/dashboard/refresh',{});location.reload()}catch(error){document.querySelector('#workflow-message').textContent=error.message;document.querySelector('#workflow-message').className='benchmark-message error'}});
refreshCatalog();
let activePlan=null;const rootInput=document.querySelector('#project-root'),scanButton=document.querySelector('#scan-project'),applyButton=document.querySelector('#apply-sync'),pickButton=document.querySelector('#pick-root'),statusBox=document.querySelector('#sync-status'),summaryBox=document.querySelector('#sync-summary'),previewBox=document.querySelector('#change-preview'),messageBox=document.querySelector('#sync-message');
function syncMessage(value,error=false){statusBox.classList.add('show');messageBox.textContent=value;messageBox.className='sync-message'+(error?' error':'')}
function busy(value,label){scanButton.disabled=value;pickButton.disabled=value;applyButton.disabled=value||!activePlan;if(label)syncMessage(label)}
async function api(path,payload={}){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const body=await response.json().catch(()=>({error:'服务返回了无效响应'}));if(!response.ok)throw new Error(body.error||'操作失败');return body}
const questionInput=document.querySelector('#test-question'),queryButton=document.querySelector('#run-query'),queryStatus=document.querySelector('#query-status'),searchResults=document.querySelector('#search-results');
document.querySelectorAll('.example-chip').forEach(button=>button.addEventListener('click',()=>{questionInput.value=button.textContent;questionInput.focus()}));
function renderSearchResults(response){queryStatus.textContent=`${response.mode.toUpperCase()} · ${response.count} 条结果 · ${response.duration_ms} ms`;searchResults.innerHTML=response.results.length?response.results.map(item=>{const ranks=[item.dense_rank?`Dense #${item.dense_rank}`:'',item.sparse_rank?`Sparse #${item.sparse_rank}`:''].filter(Boolean);return `<article class="result-card"><div class="result-head"><span class="result-rank">${item.position}</span><div class="result-title" title="${esc(item.symbol||'未命名切块')}">${esc(item.symbol||'未命名切块')}</div><span class="result-score">${Number(item.score).toFixed(6)}</span></div><div class="result-meta"><span>${esc(item.source_type)}</span>${item.module?`<span>${esc(item.module)}</span>`:''}${item.file_path?`<span title="${esc(item.file_path)}">${esc(item.file_path)}</span>`:''}${ranks.map(rank=>`<span>${rank}</span>`).join('')}</div><pre class="result-content">${esc(item.content)}</pre></article>`}).join(''):'<div class="empty">没有召回结果，请调整问题、来源或检索模式。</div>'}
async function runRagTest(){const question=questionInput.value.trim();if(!question){queryStatus.textContent='请输入要测试的问题。';questionInput.focus();return}queryButton.disabled=true;searchResults.innerHTML='';queryStatus.textContent=document.querySelector('#test-mode').value==='hybrid'?'正在加载模型并执行混合检索，首次运行可能稍慢…':'正在检索当前索引…';try{const response=await api('/api/search',{question,mode:document.querySelector('#test-mode').value,source_type:document.querySelector('#test-source').value,module:document.querySelector('#test-module').value,limit:8});renderSearchResults(response);pushHistory({question,mode:document.querySelector('#test-mode').value,source_type:document.querySelector('#test-source').value,module:document.querySelector('#test-module').value,count:response.count,duration_ms:response.duration_ms})}catch(error){queryStatus.textContent=error.message;searchResults.innerHTML=''}finally{queryButton.disabled=false}}
queryButton.addEventListener('click',runRagTest);questionInput.addEventListener('keydown',event=>{if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();runRagTest()}});const HISTORY_KEY='ue_rag_query_history';function loadHistory(){try{return JSON.parse(localStorage.getItem(HISTORY_KEY)||'[]')}catch(e){return[]}}function saveHistory(list){try{localStorage.setItem(HISTORY_KEY,JSON.stringify(list.slice(0,20)))}catch(e){}}function renderHistory(){const list=loadHistory();const box=document.querySelector('#history-list');if(!list.length){box.innerHTML='<span class="history-empty">尚无历史记录</span>';return}box.innerHTML=list.map(function(h){return '<button class="history-item" title="'+esc(h.q)+'" data-q="'+esc(h.q)+'" data-mode="'+esc(h.mode||'lexical')+'" data-source="'+esc(h.source_type||'')+'" data-module="'+esc(h.module||'')+'">'+esc(h.q)+'</button>'}).join('')}function pushHistory(ctx){const list=loadHistory();const key=ctx.question+'|'+ctx.mode+'|'+(ctx.source_type||'')+'|'+(ctx.module||'');const filtered=list.filter(function(h){return (h.q+'|'+h.mode+'|'+(h.source_type||'')+'|'+(h.module||''))!==key});filtered.unshift({q:ctx.question,mode:ctx.mode,source_type:ctx.source_type||'',module:ctx.module||'',count:ctx.count,duration_ms:ctx.duration_ms,at:Date.now()});saveHistory(filtered);renderHistory()}document.querySelector('#history-list').addEventListener('click',function(event){const btn=event.target.closest('.history-item');if(!btn)return;questionInput.value=btn.dataset.q;document.querySelector('#test-mode').value=btn.dataset.mode;document.querySelector('#test-source').value=btn.dataset.source;document.querySelector('#test-module').value=btn.dataset.module;runRagTest()});document.querySelector('#history-clear').addEventListener('click',function(){saveHistory([]);renderHistory()});renderHistory();
function renderPlan(plan){const labels={added:'新增',modified:'修改',deleted:'删除',unchanged:'未变化'};summaryBox.innerHTML=Object.entries(labels).map(([key,label])=>`<div class="sync-stat"><span>${label}</span><b>${plan.counts[key]}</b></div>`).join('');previewBox.innerHTML=['added','modified','deleted'].map(key=>`<div class="change-col"><h3>${labels[key]}</h3><ul>${(plan.changes[key]||[]).map(path=>`<li title="${esc(path)}">${esc(path)}</li>`).join('')||'<li>—</li>'}</ul></div>`).join('');syncMessage(plan.total_changes?`已发现 ${plan.total_changes} 个变化，请确认后同步。`:'数据已是最新状态。');applyButton.disabled=!plan.total_changes}
pickButton.addEventListener('click',async()=>{busy(true,'正在打开文件夹选择器…');try{const result=await api('/api/pick-directory');if(result.path)rootInput.value=result.path;syncMessage(result.path?'已选择项目目录。':'未选择目录。')}catch(error){syncMessage(error.message+'；也可以直接输入目录。',true)}finally{busy(false)}});
scanButton.addEventListener('click',async()=>{const project_root=rootInput.value.trim();if(!project_root){syncMessage('请先选择或输入 UE 项目目录。',true);return}busy(true,'正在计算文件哈希并对比上次同步状态…');try{const plan=await api('/api/project/preview',{project_root});activePlan=plan.plan_id;renderPlan(plan)}catch(error){activePlan=null;summaryBox.innerHTML='';previewBox.innerHTML='';syncMessage(error.message,true)}finally{busy(false)}});
applyButton.addEventListener('click',async()=>{if(!activePlan)return;busy(true,'正在解析、切块并更新索引，请保持页面打开…');try{const result=await api('/api/project/apply',{plan_id:activePlan,sync_vectors:document.querySelector('#sync-vectors').checked});activePlan=null;summaryBox.innerHTML=[['变更文件',result.changed_files],['生成文档',result.documents],['新切块',result.chunks],['清理旧块',result.removed_chunks]].map(x=>`<div class="sync-stat"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('');previewBox.innerHTML='';document.querySelector('#kpis .card .value').textContent=fmt(result.total_chunks);const sourceIndex=(D.data_sources||[]).findIndex(item=>item.id===result.data_source.id);if(sourceIndex>=0)D.data_sources[sourceIndex]=result.data_source;else(D.data_sources||(D.data_sources=[])).push(result.data_source);renderDataSources();syncMessage(`同步完成：关键词索引新增 ${result.lexical.added} 条${result.vectors.enabled?'，向量索引新增 '+result.vectors.added+' 条':'；本次未同步向量索引'}。`)}catch(error){syncMessage(error.message,true)}finally{busy(false)}});
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(cli())
