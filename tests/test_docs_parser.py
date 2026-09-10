"""Snapshot and pipeline tests for Epic documentation parsing."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ue_rag.crawler.docs import CrawlRecord, CrawlStatus
from ue_rag.jsonl import load_jsonl
from ue_rag.parser.docs import (
    DocumentationParser,
    DocumentationParserConfig,
    load_docs_parser_config,
)
from ue_rag.schema import SourceScope, SourceType, UEDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_HTML = PROJECT_ROOT / "tests" / "fixtures" / "docs" / "epic_page.html"
SNAPSHOT_MARKDOWN = (
    PROJECT_ROOT / "tests" / "snapshots" / "docs_parser_expected.md"
)
PAGE_URL = (
    "https://dev.epicgames.com/documentation/en-us/unreal-engine/example"
    "?application_version=5.8"
)


def make_record(
    *, engine_version: str = "5.8", file_path: str = "pages/example.html"
) -> CrawlRecord:
    html = FIXTURE_HTML.read_bytes()
    timestamp = datetime(2026, 9, 10, 3, 0, tzinfo=timezone.utc)
    return CrawlRecord(
        url=PAGE_URL,
        final_url=PAGE_URL,
        topic="gameplay",
        engine_version=engine_version,
        status=CrawlStatus.DOWNLOADED,
        recorded_at=timestamp,
        fetched_at=timestamp,
        file_path=file_path,
        content_sha256=hashlib.sha256(html).hexdigest(),
        size_bytes=len(html),
        content_type="text/html; charset=utf-8",
        http_status=200,
        attempts=1,
    )


def test_parser_markdown_matches_snapshot() -> None:
    document = DocumentationParser().parse_html(FIXTURE_HTML.read_bytes(), make_record())
    expected = SNAPSHOT_MARKDOWN.read_text(encoding="utf-8").rstrip("\n")

    assert document.content == expected


def test_parser_preserves_provenance_headings_and_code_metadata() -> None:
    document = DocumentationParser().parse_html(FIXTURE_HTML.read_bytes(), make_record())

    assert document.engine_version == "5.8"
    assert document.source_scope is SourceScope.GLOBAL
    assert document.source_type is SourceType.DOCS
    assert document.title == "Gameplay & C++"
    assert document.metadata["url"] == PAGE_URL
    assert document.metadata["fetched_at"] == "2026-09-10T03:00:00+00:00"
    assert document.metadata["code_block_count"] == 1
    assert document.metadata["code_languages"] == ["cpp"]
    assert document.metadata["heading_paths"] == [
        {"level": 1, "title": "Gameplay & C++", "path": ["Gameplay & C++"]},
        {
            "level": 2,
            "title": "Overview",
            "path": ["Gameplay & C++", "Overview"],
        },
        {
            "level": 3,
            "title": "Example",
            "path": ["Gameplay & C++", "Overview", "Example"],
        },
    ]


def test_parser_removes_navigation_footer_cookie_and_copy_ui() -> None:
    content = DocumentationParser().parse_html(
        FIXTURE_HTML.read_bytes(), make_record()
    ).content

    for noise in (
        "Duplicate global navigation",
        "Duplicate table of contents",
        "Duplicate sidebar menu",
        "Duplicate footer",
        "Accept cookies",
        "Copy full snippet",
    ):
        assert noise not in content


def test_manifest_pipeline_writes_idempotent_uedocument_jsonl(tmp_path: Path) -> None:
    raw_docs = tmp_path / "raw" / "docs"
    raw_page = raw_docs / "pages" / "example.html"
    raw_page.parent.mkdir(parents=True)
    raw_page.write_bytes(FIXTURE_HTML.read_bytes())
    record = make_record()
    manifest = raw_docs / "manifest.jsonl"
    manifest.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "parsed" / "docs" / "documents.jsonl"
    config = DocumentationParserConfig(
        engine_version="5.8",
        raw_docs_dir=raw_docs,
        output_path=output,
    )
    parser = DocumentationParser()

    first = parser.parse_manifest(config)
    first_bytes = output.read_bytes()
    second = parser.parse_manifest(config)
    documents = load_jsonl(output, UEDocument)

    assert first.documents == second.documents == 1
    assert first.headings == 3
    assert first.code_blocks == 1
    assert first.skipped == 0
    assert output.read_bytes() == first_bytes
    assert len(documents) == 1
    assert documents[0].id.startswith("docs-")
    assert documents[0].content.startswith("# Gameplay & C++")


def test_manifest_records_unusable_dynamic_page_as_issue(tmp_path: Path) -> None:
    raw_docs = tmp_path / "raw" / "docs"
    raw_page = raw_docs / "pages" / "empty.html"
    raw_page.parent.mkdir(parents=True)
    raw_page.write_text(
        "<main><header class='section-page-header'><h1></h1></header>"
        "<spinner>Loading</spinner></main>",
        encoding="utf-8",
    )
    record = make_record(file_path="pages/empty.html")
    manifest = raw_docs / "manifest.jsonl"
    manifest.write_text(record.model_dump_json(), encoding="utf-8")
    output = tmp_path / "parsed" / "docs" / "documents.jsonl"
    config = DocumentationParserConfig(
        engine_version="5.8",
        raw_docs_dir=raw_docs,
        output_path=output,
    )

    summary = DocumentationParser().parse_manifest(config)

    assert summary.documents == 0
    assert summary.skipped == 1
    assert output.read_text(encoding="utf-8") == ""
    assert PAGE_URL in summary.issues_path.read_text(encoding="utf-8")
    assert "no usable title" in summary.issues_path.read_text(encoding="utf-8")


def test_manifest_version_must_match_config(tmp_path: Path) -> None:
    raw_docs = tmp_path / "raw" / "docs"
    raw_page = raw_docs / "pages" / "example.html"
    raw_page.parent.mkdir(parents=True)
    raw_page.write_bytes(FIXTURE_HTML.read_bytes())
    manifest = raw_docs / "manifest.jsonl"
    manifest.write_text(make_record(engine_version="5.7").model_dump_json(), encoding="utf-8")
    config = DocumentationParserConfig(
        engine_version="5.8",
        raw_docs_dir=raw_docs,
        output_path=tmp_path / "documents.jsonl",
    )

    with pytest.raises(ValueError, match="does not match configured version"):
        DocumentationParser().parse_manifest(config)


def test_parser_config_uses_ue_paths_and_version() -> None:
    config = load_docs_parser_config()

    assert config.engine_version == "5.8"
    assert config.raw_docs_dir == PROJECT_ROOT / "data" / "raw" / "docs"
    assert config.output_path == (
        PROJECT_ROOT / "data" / "parsed" / "docs" / "documents.jsonl"
    )
