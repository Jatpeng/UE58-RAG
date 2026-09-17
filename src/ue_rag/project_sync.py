"""Visual-first incremental synchronization for local Unreal projects."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ue_rag.chunker.cpp import CPPChunkerConfig, CPPSemanticChunker
from ue_rag.chunker.docs import HeuristicTokenCounter
from ue_rag.crawler.engine import EngineFileRecord, EngineFileType
from ue_rag.crawler.incremental import ProjectDiff, diff_inventories, load_inventory
from ue_rag.crawler.project import (
    ProjectFileRecord,
    ProjectFileType,
    ProjectSourceScanner,
    load_project_scanner_config,
)
from ue_rag.jsonl import save_jsonl
from ue_rag.parser.cpp import CPPParserConfig, UnrealCPPParser
from ue_rag.retrieval import LexicalIndex, load_lexical_config
from ue_rag.schema import SourceScope, SourceType, UEChunk, UEDocument


@dataclass(frozen=True)
class ProjectSyncPlan:
    id: str
    project_root: Path
    project_key: str
    inventory_path: Path
    records: tuple[ProjectFileRecord, ...]
    diff: ProjectDiff

    def public(self) -> dict[str, Any]:
        return {
            "plan_id": self.id,
            "project_root": str(self.project_root),
            "counts": {
                "added": len(self.diff.added),
                "modified": len(self.diff.modified),
                "deleted": len(self.diff.deleted),
                "unchanged": self.diff.unchanged,
            },
            "changes": {
                "added": [item.relative_path for item in self.diff.added[:30]],
                "modified": [item.relative_path for item in self.diff.modified[:30]],
                "deleted": [item.relative_path for item in self.diff.deleted[:30]],
            },
            "total_changes": len(self.diff.added) + len(self.diff.modified) + len(self.diff.deleted),
        }


class ProjectSyncService:
    """Preview and apply project changes while keeping old chunks out of both indexes."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        state_dir: str | Path | None = None,
        lexical_index_path: str | Path | None = None,
        qdrant_config_path: str | Path = "config/qdrant_server.yaml",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.ue_config_path = self.workspace_root / "config" / "ue58.yaml"
        self.scanner_config_path = self.workspace_root / "config" / "project_scanner.yaml"
        self.cpp_parser_config_path = self.workspace_root / "config" / "cpp_parser.yaml"
        self.cpp_chunker_config_path = self.workspace_root / "config" / "cpp_chunker.yaml"
        self.embedding_config_path = self.workspace_root / "config" / "embedding.yaml"
        self.qdrant_config_path = self._resolve(qdrant_config_path)
        self.state_dir = self._resolve(state_dir or "data/parsed/project/sync_state")
        self.plan_dir = self._resolve("work/dashboard/plans")
        lexical_config = load_lexical_config(self.ue_config_path, self.workspace_root / "config" / "lexical.yaml")
        self.lexical_index_path = self._resolve(lexical_index_path or lexical_config.index_path)
        with self.ue_config_path.open(encoding="utf-8") as stream:
            self.engine_version = str((yaml.safe_load(stream) or {})["engine"]["version"])
        self._plans: dict[str, ProjectSyncPlan] = {}
        self._sync_lock = threading.Lock()

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.workspace_root / path).resolve()

    def preview(self, project_root: str | Path) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        key = _project_key(root)
        plan_id = uuid.uuid4().hex
        inventory_path = self.plan_dir / f"{plan_id}.jsonl"
        issues_path = self.plan_dir / f"{plan_id}.issues.jsonl"
        config = load_project_scanner_config(self.ue_config_path, self.scanner_config_path)
        config = config.model_copy(
            update={
                "project_root": root,
                "output_path": inventory_path,
                "issues_path": issues_path,
            }
        )
        summary = ProjectSourceScanner(config).scan()
        current = tuple(load_inventory(inventory_path))
        state_path = self._state_path(key)
        previous = load_inventory(state_path) if state_path.is_file() else []
        diff = diff_inventories(previous, current)
        plan = ProjectSyncPlan(plan_id, root, key, inventory_path, current, diff)
        self._plans[plan_id] = plan
        while len(self._plans) > 20:
            self._plans.pop(next(iter(self._plans)))
        result = plan.public()
        result["scan"] = {"files": summary.files, "issues": summary.issues}
        return result

    def apply(self, plan_id: str, *, sync_vectors: bool = True) -> dict[str, Any]:
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError("同步计划已失效，请重新扫描")
        if not self._sync_lock.acquire(blocking=False):
            raise RuntimeError("已有同步任务正在运行")
        try:
            return self._apply_locked(plan, sync_vectors=sync_vectors)
        finally:
            self._sync_lock.release()

    def _apply_locked(self, plan: ProjectSyncPlan, *, sync_vectors: bool) -> dict[str, Any]:
        changed = list(plan.diff.changed)
        for change in changed:
            if change.current is None:
                continue
            actual = _sha256(plan.project_root / change.relative_path)
            if actual != change.current.sha256:
                raise ValueError(f"扫描后文件又发生变化，请重新扫描：{change.relative_path}")

        parser = self._cpp_parser(plan.project_root)
        chunker = self._cpp_chunker()
        chunks: list[UEChunk] = []
        chunks_by_path: dict[str, list[UEChunk]] = defaultdict(list)
        parsed_documents = 0
        for change in changed:
            record = change.current
            if record is None:
                continue
            documents = self._parse_record(plan, record, parser)
            parsed_documents += len(documents)
            file_chunks = self._chunk_documents(documents, chunker)
            chunks.extend(file_chunks)
            chunks_by_path[record.relative_path].extend(file_chunks)

        replaced_paths = [
            self._indexed_path(plan, item.relative_path)
            for item in (*plan.diff.modified, *plan.diff.deleted)
        ]

        with LexicalIndex(self.lexical_index_path) as lexical:
            old_ids = lexical.chunk_ids_for_file_paths(replaced_paths)

        vectors = None
        provider = None
        if sync_vectors and chunks:
            from ue_rag.embedding import QwenEmbeddingProvider, load_embedding_config

            provider = QwenEmbeddingProvider(
                load_embedding_config(self.ue_config_path, self.embedding_config_path)
            )
            vectors = provider.embed_documents([chunk.content for chunk in chunks])

        vector_summary: dict[str, Any] = {"enabled": sync_vectors, "deleted": 0, "added": 0, "updated": 0}
        store = None
        try:
            if sync_vectors:
                from ue_rag.index import QdrantVectorStore, load_qdrant_config

                store = QdrantVectorStore.from_config(load_qdrant_config(self.qdrant_config_path))
                if old_ids:
                    store.delete(chunk_ids=old_ids)
                    vector_summary["deleted"] = len(old_ids)
                if chunks:
                    result = store.upsert(chunks, vectors)
                    if result.failed:
                        raise RuntimeError(f"向量索引写入失败：{result.failed} 条")
                    vector_summary.update({"added": result.added, "updated": result.updated})
        finally:
            if provider is not None:
                provider.close()
            if store is not None and getattr(store, "client", None) is not None:
                store.client.close()

        with LexicalIndex(self.lexical_index_path) as lexical:
            _, lexical_summary = lexical.replace_file_chunks(replaced_paths, chunks)
            total_chunks = int(lexical.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

        for change in plan.diff.deleted:
            self._shard_path(plan.project_key, change.relative_path).unlink(missing_ok=True)
        for relative_path, file_chunks in chunks_by_path.items():
            save_jsonl(self._shard_path(plan.project_key, relative_path), file_chunks)
        save_jsonl(self._state_path(plan.project_key), plan.records)
        project_chunk_count = _count_jsonl_records(
            self.workspace_root / "data" / "chunks" / "project" / plan.project_key
        )
        source_metadata = {
            "project_key": plan.project_key,
            "project_name": plan.project_root.name,
            "project_root": str(plan.project_root),
            "files": len(plan.records),
            "chunks": project_chunk_count,
            "synced_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "vectors_enabled": sync_vectors,
        }
        _write_json(self._meta_path(plan.project_key), source_metadata)
        self._plans.pop(plan.id, None)
        plan.inventory_path.unlink(missing_ok=True)
        plan.inventory_path.with_name(f"{plan.inventory_path.stem}.issues.jsonl").unlink(missing_ok=True)
        return {
            "project_root": str(plan.project_root),
            "changed_files": len(changed),
            "deleted_files": len(plan.diff.deleted),
            "documents": parsed_documents,
            "chunks": len(chunks),
            "total_chunks": total_chunks,
            "removed_chunks": len(old_ids),
            "lexical": {
                "added": lexical_summary.added,
                "updated": lexical_summary.updated,
                "skipped": lexical_summary.skipped,
            },
            "vectors": vector_summary,
            "data_source": {
                "id": f"project-{plan.project_key}",
                "type": "project_source",
                "name": plan.project_root.name,
                "location": str(plan.project_root),
                "chunks": project_chunk_count,
                "status": "indexed",
                "detail": f"{len(plan.records):,} 个文件 · 刚刚同步",
            },
        }

    def _parse_record(
        self,
        plan: ProjectSyncPlan,
        record: ProjectFileRecord,
        parser: UnrealCPPParser,
    ) -> list[UEDocument]:
        source_path = plan.project_root / record.relative_path
        indexed_path = self._indexed_path(plan, record.relative_path)
        if record.file_type in {ProjectFileType.HEADER, ProjectFileType.CPP, ProjectFileType.INL}:
            engine_record = EngineFileRecord(
                relative_path=indexed_path,
                module=record.module,
                plugin=record.plugin,
                file_type=EngineFileType(record.file_type.value),
                engine_version=record.engine_version,
                size=record.size,
                sha256=record.sha256,
            )
            documents, _ = parser.parse_source(
                source_path.read_bytes(),
                engine_record,
                source_scope=SourceScope.PROJECT,
                source_type=SourceType.PROJECT_SOURCE,
            )
            return [
                document.model_copy(
                    update={
                        "metadata": {
                            **document.metadata,
                            "project_relative_path": record.relative_path,
                            "project_key": plan.project_key,
                        }
                    }
                )
                for document in documents
            ]

        content = source_path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return []
        identifier = "project-doc-" + hashlib.sha256(
            f"{plan.project_key}:{record.relative_path}:{record.sha256}".encode("utf-8")
        ).hexdigest()
        return [
            UEDocument(
                id=identifier,
                engine_version=record.engine_version,
                source_scope=SourceScope.PROJECT,
                source_type=record.source_type,
                content=content,
                title=Path(record.relative_path).name,
                module=record.module,
                plugin=record.plugin,
                file_path=indexed_path,
                metadata={
                    "project_relative_path": record.relative_path,
                    "project_key": plan.project_key,
                    "file_sha256": record.sha256,
                    "file_type": record.file_type.value,
                    "parser_version": "project-text-1",
                },
            )
        ]

    def _chunk_documents(
        self, documents: list[UEDocument], chunker: CPPSemanticChunker
    ) -> list[UEChunk]:
        cpp_documents = [item for item in documents if item.source_type is SourceType.PROJECT_SOURCE and item.symbol_type]
        text_documents = [item for item in documents if item not in cpp_documents]
        chunks: list[UEChunk] = []
        fields: dict[tuple[str | None, str | None], list[UEDocument]] = defaultdict(list)
        for document in cpp_documents:
            if document.symbol_type == "field":
                fields[(document.class_name, document.file_path)].append(document)
            else:
                chunks.extend(chunker.chunk_document(document))
        for grouped in fields.values():
            chunks.append(chunker.chunk_property_group(grouped))
        for document in text_documents:
            chunks.extend(_chunk_text_document(document))
        return chunks

    def _cpp_parser(self, project_root: Path) -> UnrealCPPParser:
        values = yaml.safe_load(self.cpp_parser_config_path.read_text(encoding="utf-8")) or {}
        scratch = self.plan_dir / "unused.jsonl"
        config = CPPParserConfig(
            engine_version=self.engine_version,
            engine_root=project_root,
            inventory_path=scratch,
            output_path=scratch,
            issues_path=scratch,
            **values,
        )
        return UnrealCPPParser(config)

    def _cpp_chunker(self) -> CPPSemanticChunker:
        values = yaml.safe_load(self.cpp_chunker_config_path.read_text(encoding="utf-8")) or {}
        scratch = self.plan_dir / "unused.jsonl"
        return CPPSemanticChunker(
            CPPChunkerConfig(
                engine_version=self.engine_version,
                input_path=scratch,
                output_path=scratch,
                **values,
            )
        )

    def _indexed_path(self, plan: ProjectSyncPlan, relative_path: str) -> str:
        return f"Project/{plan.project_root.name}-{plan.project_key[:8]}/{relative_path}"

    def _state_path(self, project_key: str) -> Path:
        return self.state_dir / f"{project_key}.jsonl"

    def _meta_path(self, project_key: str) -> Path:
        return self.state_dir / f"{project_key}.meta.json"

    def _shard_path(self, project_key: str, relative_path: str) -> Path:
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
        return self.workspace_root / "data" / "chunks" / "project" / project_key / f"{digest}.jsonl"


def _project_key(root: Path) -> str:
    return hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:20]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _count_jsonl_records(directory: Path) -> int:
    total = 0
    for path in directory.glob("*.jsonl") if directory.is_dir() else []:
        with path.open(encoding="utf-8") as stream:
            total += sum(1 for line in stream if line.strip())
    return total


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _chunk_text_document(document: UEDocument) -> list[UEChunk]:
    counter = HeuristicTokenCounter()
    remaining = document.content.strip()
    parts: list[str] = []
    while counter.count(remaining) > 1_200:
        head = counter.head(remaining, 1_000)
        if not head:
            break
        parts.append(head)
        remaining = remaining[len(head) :].strip()
    if remaining:
        parts.append(remaining)
    chunks: list[UEChunk] = []
    for index, body in enumerate(parts):
        content = f"File: {document.file_path}\n\n{body}".strip()
        chunk_id = "project-chunk-" + hashlib.sha256(
            f"{document.id}:{index}:{content}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            UEChunk(
                id=chunk_id,
                document_id=document.id,
                chunk_index=index,
                engine_version=document.engine_version,
                source_scope=document.source_scope,
                source_type=document.source_type,
                content=content,
                title=document.title,
                module=document.module,
                plugin=document.plugin,
                file_path=document.file_path,
                token_count=counter.count(content),
                metadata={
                    **document.metadata,
                    "chunker_version": "project-text-1",
                    "part_index": index,
                    "part_count": len(parts),
                },
            )
        )
    return chunks
