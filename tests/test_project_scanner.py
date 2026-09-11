"""Tests for the Project RAG source scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from ue_rag.crawler.project import (
    ProjectFileType,
    ProjectFileRecord,
    ProjectSourceScanner,
    ProjectScannerConfig,
    load_project_scanner_config,
)
from ue_rag.jsonl import load_jsonl


def make_config(tmp_path: Path, project_root: Path | None = None) -> ProjectScannerConfig:
    return ProjectScannerConfig(
        engine_version="5.8",
        project_root=project_root or tmp_path / "Project",
        output_path=tmp_path / "parsed" / "project" / "files.jsonl",
        issues_path=tmp_path / "parsed" / "project" / "issues.jsonl",
        source_directories=["Source", "Plugins", "Config", "Docs"],
        include_suffixes={".h", ".hpp", ".cpp", ".inl", ".cs", ".ini", ".json", ".md", ".txt"},
        ignored_directories={".git", "Binaries", "DerivedDataCache", "Intermediate", "Saved", "Build"},
        hash_chunk_size=32,
        hash_workers=2,
    )


def write(root: Path, relative: str, content: str = "content") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scan_covers_source_plugins_config_and_docs(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    write(root, "Source/Hero/Hero.Build.cs", "class Hero : ModuleRules {}")
    write(root, "Source/Hero/Public/Hero.h", "class AHero {};")
    write(root, "Source/Hero/Private/Hero.cpp", "void Tick() {}")
    write(root, "Plugins/Combat/Source/Combat/Combat.Build.cs", "class Combat : ModuleRules {}")
    write(root, "Plugins/Combat/Source/Combat/Combat.h", "struct FCombat {};")
    write(root, "Config/DefaultGame.ini", "[/Script/EngineSettings.GeneralProjectSettings]")
    write(root, "Docs/README.md", "# Project notes")
    scanner = ProjectSourceScanner(make_config(tmp_path, root))

    summary = scanner.scan()
    records = load_jsonl(summary.output_path, ProjectFileRecord)

    assert summary.files == 7
    assert summary.issues == 0
    assert summary.modules >= 2
    assert summary.plugins == 1
    assert {record.file_type for record in records} == {
        ProjectFileType.BUILD_CS,
        ProjectFileType.HEADER,
        ProjectFileType.CPP,
        ProjectFileType.CONFIG,
        ProjectFileType.DOCUMENT,
    }
    assert records[0].engine_version == "5.8"
    assert all(len(record.sha256) == 64 for record in records)


def test_scan_ignores_generated_and_binary_directories(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    write(root, "Source/Hero/Hero.h")
    write(root, "Binaries/Hero/generated.cpp")
    write(root, "Intermediate/Hero/generated.cpp")
    write(root, "Saved/trace.txt")
    write(root, "Source/Hero/Content.uasset")

    summary = ProjectSourceScanner(make_config(tmp_path, root)).scan()
    records = load_jsonl(summary.output_path, ProjectFileRecord)

    assert [record.relative_path for record in records] == ["Source/Hero/Hero.h"]


def test_missing_optional_docs_and_config_directories_are_allowed(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    write(root, "Source/Hero/Hero.h")

    summary = ProjectSourceScanner(make_config(tmp_path, root)).scan()

    assert summary.files == 1


def test_project_root_is_required_and_must_exist(tmp_path: Path) -> None:
    config = make_config(tmp_path, None)
    config.project_root = None
    with pytest.raises(ValueError, match="project root is not configured"):
        ProjectSourceScanner(config).validate()
    config.project_root = tmp_path / "Missing"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ProjectSourceScanner(config).validate()


def test_source_directory_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "Project"
    root.mkdir()
    config = make_config(tmp_path, root)
    config.source_directories = ["../outside"]
    with pytest.raises(ValueError, match="escapes"):
        ProjectSourceScanner(config).validate()


def test_load_config_reads_ue_version_and_output_paths() -> None:
    config = load_project_scanner_config()

    assert config.engine_version == "5.8"
    assert config.output_path == Path.cwd() / "data/parsed/project/files.jsonl"
