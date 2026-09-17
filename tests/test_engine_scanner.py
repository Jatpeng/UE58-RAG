"""Tests for Unreal Engine source inventory scanning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ue_rag.crawler.engine import (
    EngineFileRecord,
    EngineFileType,
    EngineScannerConfig,
    EngineSourceScanner,
    load_engine_scanner_config,
)
from ue_rag.jsonl import load_jsonl


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_engine_fixture(tmp_path: Path, version: str = "5.8") -> Path:
    root = tmp_path / "UE_5.8"
    major, minor = version.split(".")
    write_file(
        root / "Engine" / "Build" / "Build.version",
        json.dumps({"MajorVersion": int(major), "MinorVersion": int(minor)}),
    )

    write_file(root / "Engine" / "Source" / "Runtime" / "Core" / "Core.Build.cs", "Core build")
    write_file(root / "Engine" / "Source" / "Runtime" / "Core" / "Public" / "CoreThing.h", "header")
    write_file(root / "Engine" / "Source" / "Runtime" / "Core" / "Private" / "CoreThing.cpp", "cpp")
    write_file(root / "Engine" / "Source" / "Runtime" / "Core" / "Private" / "CoreThing.inl", "inline")
    write_file(root / "Engine" / "Source" / "Editor" / "Loose" / "Loose.cpp", "loose")
    write_file(root / "Engine" / "Source" / "Developer" / "Tool" / "Tool.Build.cs", "tool build")
    write_file(root / "Engine" / "Source" / "Developer" / "Tool" / "Tool.h", "tool header")
    write_file(root / "Engine" / "Source" / "Programs" / "HeaderTool" / "HeaderTool.Build.cs", "program build")
    write_file(root / "Engine" / "Source" / "Programs" / "HeaderTool" / "HeaderTool.cpp", "program cpp")
    write_file(root / "Engine" / "Source" / "ThirdParty" / "Library" / "ThirdParty.h", "third party")
    write_file(root / "Engine" / "Source" / "ThirdParty" / "Library" / "Library.h", "third party")

    plugin = root / "Engine" / "Plugins" / "Runtime" / "TestPlugin"
    write_file(plugin / "TestPlugin.uplugin", "{}")
    write_file(plugin / "Source" / "TestModule" / "TestModule.Build.cs", "module build")
    write_file(plugin / "Source" / "TestModule" / "Public" / "TestModule.h", "plugin header")
    write_file(plugin / "Source" / "TestModule" / "Private" / "TestModule.cpp", "plugin cpp")
    write_file(plugin / "Shaders" / "ShaderHelper.h", "not plugin source")

    for ignored in ("Intermediate", "Binaries", "DerivedDataCache", "Saved", "Content"):
        write_file(plugin / ignored / "ShouldNotAppear.cpp", "ignored")
    write_file(plugin / "interMEDIATE" / "AlsoIgnored.h", "ignored case")
    write_file(plugin / "Source" / "TestModule" / "Readme.txt", "not source")
    return root


def make_config(tmp_path: Path, root: Path) -> EngineScannerConfig:
    output_dir = tmp_path / "parsed" / "engine"
    return EngineScannerConfig(
        engine_version="5.8",
        engine_root=root,
        source_root=root / "Engine" / "Source",
        plugins_root=root / "Engine" / "Plugins",
        output_path=output_dir / "files.jsonl",
        issues_path=output_dir / "issues.jsonl",
        include_suffixes={".h", ".cpp", ".inl", ".build.cs"},
        source_categories=["Runtime", "Editor", "Developer", "Programs"],
        ignored_directories={
            "Intermediate",
            "Binaries",
            "DerivedDataCache",
            "Saved",
            "Content",
        },
        hash_chunk_size=4,
        hash_workers=2,
    )


def test_scanner_inventories_source_modules_plugins_and_hashes(tmp_path: Path) -> None:
    root = make_engine_fixture(tmp_path)
    config = make_config(tmp_path, root)

    summary = EngineSourceScanner(config).scan()
    records = load_jsonl(config.output_path, EngineFileRecord)

    assert summary.files == len(records) == 12
    assert summary.modules == 5
    assert summary.plugins == 1
    assert summary.issues == 0
    assert summary.by_type == {
        EngineFileType.HEADER: 3,
        EngineFileType.CPP: 4,
        EngineFileType.INL: 1,
        EngineFileType.BUILD_CS: 4,
    }
    assert config.issues_path.read_text(encoding="utf-8") == ""
    assert [record.relative_path for record in records] == sorted(
        (record.relative_path for record in records), key=str.casefold
    )
    assert all(record.engine_version == "5.8" for record in records)
    assert all(len(record.sha256) == 64 for record in records)

    core_header = next(
        record for record in records if record.relative_path.endswith("CoreThing.h")
    )
    assert core_header.module == "Core"
    assert core_header.plugin is None
    assert core_header.file_type is EngineFileType.HEADER
    assert core_header.sha256 == hashlib.sha256(b"header").hexdigest()

    plugin_cpp = next(
        record for record in records if record.relative_path.endswith("TestModule.cpp")
    )
    assert plugin_cpp.module == "TestModule"
    assert plugin_cpp.plugin == "TestPlugin"
    assert plugin_cpp.relative_path.startswith("Engine/Plugins/")


def test_scanner_ignores_generated_cache_and_non_source_files(tmp_path: Path) -> None:
    root = make_engine_fixture(tmp_path)
    config = make_config(tmp_path, root)

    EngineSourceScanner(config).scan()
    records = load_jsonl(config.output_path, EngineFileRecord)
    paths = [record.relative_path.casefold() for record in records]

    for ignored in (
        "intermediate",
        "binaries",
        "deriveddatacache",
        "saved",
        "content",
        "readme.txt",
        "thirdparty",
        "shaderhelper.h",
    ):
        assert all(ignored not in path for path in paths)


def test_module_falls_back_to_source_layout_without_build_file(tmp_path: Path) -> None:
    root = make_engine_fixture(tmp_path)
    config = make_config(tmp_path, root)

    EngineSourceScanner(config).scan()
    records = load_jsonl(config.output_path, EngineFileRecord)
    loose = next(record for record in records if record.relative_path.endswith("Loose.cpp"))

    assert loose.module == "Loose"
    assert loose.plugin is None


def test_repeated_scan_is_deterministic_and_hash_tracks_changes(tmp_path: Path) -> None:
    root = make_engine_fixture(tmp_path)
    config = make_config(tmp_path, root)
    scanner = EngineSourceScanner(config)

    scanner.scan()
    first_bytes = config.output_path.read_bytes()
    scanner.scan()
    assert config.output_path.read_bytes() == first_bytes

    changed_file = root / "Engine" / "Source" / "Runtime" / "Core" / "Public" / "CoreThing.h"
    changed_file.write_text("changed header", encoding="utf-8")
    scanner.scan()
    assert config.output_path.read_bytes() != first_bytes


def test_installed_version_must_match_config(tmp_path: Path) -> None:
    root = make_engine_fixture(tmp_path, version="5.7")
    config = make_config(tmp_path, root)

    with pytest.raises(ValueError, match="does not match installed engine version"):
        EngineSourceScanner(config).scan()

    assert not config.output_path.exists()


def test_scan_roots_must_exist_under_engine_root(tmp_path: Path) -> None:
    root = make_engine_fixture(tmp_path)
    config = make_config(tmp_path, root)
    config.source_root = tmp_path / "outside"

    with pytest.raises(ValueError, match="escapes configured engine root"):
        EngineSourceScanner(config).validate()


def test_workspace_config_accepts_ue_root_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_engine_fixture(tmp_path)
    monkeypatch.setenv("UE_ROOT", str(root))

    config = load_engine_scanner_config()

    assert config.engine_version == "5.8"
    assert config.engine_root == root
    assert config.source_root == config.engine_root / "Engine" / "Source"
    assert config.plugins_root == config.engine_root / "Engine" / "Plugins"
