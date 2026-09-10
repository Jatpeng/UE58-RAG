"""Persistent lexical and exact-symbol retrieval for UE chunks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ue_rag.schema import RetrievalResult, SourceType, UEChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[\u3400-\u4dbf\u4e00-\u9fff]+|\d+")


class LexicalConfig(BaseModel):
    """Paths and retrieval limits for the lexical index."""

    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    input_path: Path
    index_path: Path
    batch_size: int = Field(gt=0)
    symbol_top_k: int = Field(gt=0)
    lexical_top_k: int = Field(gt=0)


def load_lexical_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    lexical_config_path: str | Path = "config/lexical.yaml",
) -> LexicalConfig:
    """Load UE version and project-relative chunk/index paths."""

    ue_path = Path(ue_config_path).resolve()
    lexical_path = Path(lexical_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with lexical_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    project_root = ue_path.parent.parent
    chunks_dir = Path(ue_config["data"]["chunks"])
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir
    index_path = Path(values["index_path"])
    if not index_path.is_absolute():
        index_path = project_root / index_path
    values.update(
        {
            "engine_version": str(ue_config["engine"]["version"]),
            "input_path": chunks_dir / "engine" / "chunks.jsonl",
            "index_path": index_path,
        }
    )
    return LexicalConfig(**values)


@dataclass(frozen=True)
class LexicalIngestSummary:
    total: int
    added: int
    updated: int
    skipped: int
    failed: int

    def __add__(self, other: "LexicalIngestSummary") -> "LexicalIngestSummary":
        return LexicalIngestSummary(
            self.total + other.total,
            self.added + other.added,
            self.updated + other.updated,
            self.skipped + other.skipped,
            self.failed + other.failed,
        )


class LexicalIndex:
    """SQLite FTS5 index with an exact-symbol side table."""

    FILTER_FIELDS = ("engine_version", "source_type", "module", "plugin", "class", "class_name", "symbol")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                source_type TEXT NOT NULL,
                module TEXT,
                plugin TEXT,
                class_name TEXT,
                function_name TEXT,
                symbol TEXT,
                symbol_type TEXT,
                file_path TEXT,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                content_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exact_symbols (
                symbol_folded TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                PRIMARY KEY(symbol_folded, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS exact_symbols_lookup ON exact_symbols(symbol_folded);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                symbol,
                class_name,
                function_name,
                ue_macros,
                file_path,
                content,
                tokenize='unicode61'
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LexicalIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert(self, chunks: Sequence[UEChunk]) -> LexicalIngestSummary:
        """Insert/update chunks, skipping rows whose content is unchanged."""

        summary = LexicalIngestSummary(0, 0, 0, 0, 0)
        self.connection.execute("BEGIN")
        try:
            for chunk in chunks:
                summary = self._upsert_one(summary, chunk)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return summary

    def index_jsonl(self, input_path: str | Path, *, batch_size: int = 512) -> LexicalIngestSummary:
        """Stream and validate UEChunk JSONL in bounded batches."""

        summary = LexicalIngestSummary(0, 0, 0, 0, 0)
        batch: list[UEChunk] = []
        with Path(input_path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = UEChunk.model_validate_json(line)
                except Exception as error:
                    raise ValueError(f"Invalid UEChunk at line {line_number}: {input_path}") from error
                batch.append(chunk)
                if len(batch) >= batch_size:
                    summary += self.upsert(batch)
                    batch = []
            if batch:
                summary += self.upsert(batch)
        return summary

    def search_symbol(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Return exact symbol/class/function matches with deterministic priority."""

        self._validate_query(query, limit, filters)
        folded = query.strip().casefold()
        where, parameters = _where_clause(filters, alias="c")
        rows = self.connection.execute(
            "SELECT c.chunk_id, c.symbol, c.class_name, c.function_name, "
            "c.document_id, c.engine_version, c.source_type, c.content, c.module, c.plugin, c.file_path, "
            "c.symbol_type, c.metadata_json FROM chunks c "
            "JOIN exact_symbols e ON e.chunk_id = c.chunk_id "
            f"WHERE e.symbol_folded = ? AND {where} ORDER BY CASE WHEN lower(c.symbol) = ? THEN 0 "
            "WHEN lower(c.class_name) = ? THEN 1 ELSE 2 END, c.chunk_id LIMIT ?",
            [folded, *parameters, folded, folded, limit],
        ).fetchall()
        return [_row_to_result(row, score=100.0 - index) for index, row in enumerate(rows)]

    def search_lexical(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search symbols, paths, macros, and source content with SQLite FTS5."""

        self._validate_query(query, limit, filters)
        tokens = TOKEN_PATTERN.findall(query)
        if not tokens:
            return []
        match_query = " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        where, parameters = _where_clause(filters, alias="c")
        rows = self.connection.execute(
            "SELECT c.chunk_id, c.symbol, c.class_name, c.function_name, c.document_id, "
            "c.engine_version, c.source_type, c.content, c.module, c.plugin, c.file_path, "
            "c.symbol_type, c.metadata_json, bm25(chunks_fts) AS rank_score "
            "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
            "WHERE chunks_fts MATCH ? AND " + where + " ORDER BY rank_score LIMIT ?",
            [match_query, *parameters, limit],
        ).fetchall()
        return [_row_to_result(row[:13], score=1.0 / (index + 1), lexical_rank=index + 1) for index, row in enumerate(rows)]

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Exact symbol search first, then lexical matches without duplicates."""

        exact = self.search_symbol(query, limit=limit, filters=filters)
        if len(exact) >= limit:
            return exact[:limit]
        remaining = limit - len(exact)
        lexical = self.search_lexical(query, limit=limit * 2, filters=filters)
        seen = {result.chunk_id for result in exact}
        for result in lexical:
            if result.chunk_id not in seen:
                exact.append(result)
                seen.add(result.chunk_id)
            if len(exact) == limit:
                break
        return exact

    def _upsert_one(self, summary: LexicalIngestSummary, chunk: UEChunk) -> LexicalIngestSummary:
        digest = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        previous = self.connection.execute(
            "SELECT content_sha256 FROM chunks WHERE chunk_id = ?", (chunk.id,)
        ).fetchone()
        if previous and previous[0] == digest:
            return summary + LexicalIngestSummary(1, 0, 0, 1, 0)
        payload = _chunk_fields(chunk, digest)
        self.connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk.id,))
        self.connection.execute("DELETE FROM exact_symbols WHERE chunk_id = ?", (chunk.id,))
        self.connection.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", payload
        )
        self.connection.execute(
            "INSERT INTO chunks_fts(chunk_id, symbol, class_name, function_name, ue_macros, file_path, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk.id, chunk.symbol or "", chunk.class_name or "", chunk.function_name or "", _macro_text(chunk), chunk.file_path or "", chunk.content),
        )
        aliases = [chunk.symbol, chunk.class_name, chunk.function_name]
        field_symbols = chunk.metadata.get("field_symbols", [])
        if isinstance(field_symbols, list):
            aliases.extend(value for value in field_symbols if isinstance(value, str))
        for value in aliases:
            if value:
                self.connection.execute(
                    "INSERT OR IGNORE INTO exact_symbols(symbol_folded, chunk_id) VALUES (?, ?)",
                    (value.casefold(), chunk.id),
                )
        return summary + LexicalIngestSummary(1, int(previous is None), int(previous is not None), 0, 0)

    def _validate_query(self, query: str, limit: int, filters: dict[str, str] | None) -> None:
        if not query.strip():
            raise ValueError("query must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if filters:
            unknown = set(filters) - set(self.FILTER_FIELDS)
            if unknown:
                raise ValueError(f"unsupported lexical filter fields: {sorted(unknown)}")


def _chunk_fields(chunk: UEChunk, digest: str) -> tuple[Any, ...]:
    return (
        chunk.id,
        chunk.document_id,
        chunk.engine_version,
        chunk.source_type.value,
        chunk.module,
        chunk.plugin,
        chunk.class_name,
        chunk.function_name,
        chunk.symbol,
        chunk.symbol_type,
        chunk.file_path,
        chunk.content,
        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
        digest,
    )


def _macro_text(chunk: UEChunk) -> str:
    macros = chunk.metadata.get("ue_macros", [])
    return " ".join(str(item.get("name", "")) for item in macros if isinstance(item, dict))


def _where_clause(filters: dict[str, str] | None, *, alias: str = "") -> tuple[str, list[str]]:
    if not filters:
        return "1=1", []
    clauses: list[str] = []
    values: list[str] = []
    for key, value in filters.items():
        column = "class_name" if key == "class" else key
        qualified = f"{alias}.{column}" if alias else column
        clauses.append(f"{qualified} = ?")
        values.append(value)
    return " AND ".join(clauses), values


def _row_to_result(row: tuple[Any, ...], *, score: float, lexical_rank: int | None = None) -> RetrievalResult:
    (
        chunk_id,
        symbol,
        class_name,
        function_name,
        document_id,
        engine_version,
        source_type,
        content,
        module,
        plugin,
        file_path,
        symbol_type,
        metadata_json,
    ) = row
    metadata = json.loads(metadata_json)
    metadata.update(
        {
            "module": module,
            "plugin": plugin,
            "class": class_name,
            "symbol": symbol,
            "symbol_type": symbol_type,
            "file_path": file_path,
            "retrieval": "lexical" if lexical_rank else "symbol",
        }
    )
    if lexical_rank is not None:
        metadata["lexical_rank"] = lexical_rank
    return RetrievalResult(
        document_id=document_id,
        chunk_id=chunk_id,
        engine_version=engine_version,
        source_type=SourceType(source_type),
        content=content,
        score=score,
        metadata=metadata,
    )
