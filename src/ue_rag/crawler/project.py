"""Scan a user's Unreal project for source, config, and documentation files."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ue_rag.jsonl import save_jsonl
from ue_rag.schema import SourceType


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ProjectFileType(str, Enum):
    HEADER = "header"
    CPP = "cpp"
    INL = "inl"
    BUILD_CS = "build_cs"
    CONFIG = "config"
    DOCUMENT = "document"


class ProjectFileRecord(BaseModel):
    """One deterministic project file inventory record."""

    model_config = ConfigDict(extra="forbid")

    relative_path: NonEmptyString
    root_kind: NonEmptyString
    module: str | None
    plugin: str | None
    file_type: ProjectFileType
    source_type: SourceType
    engine_version: NonEmptyString
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProjectScanIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    error: str


class ProjectScannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: NonEmptyString
    project_root: Path | None = None
    output_path: Path
    issues_path: Path
    source_directories: list[str] = Field(min_length=1)
    include_suffixes: set[str] = Field(min_length=1)
    ignored_directories: set[str]
    hash_chunk_size: int = Field(gt=0)
    hash_workers: int = Field(gt=0)


def load_project_scanner_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    scanner_config_path: str | Path = "config/project_scanner.yaml",
) -> ProjectScannerConfig:
    """Load project root/rules and UE version/data paths from configuration."""

    ue_path = Path(ue_config_path).resolve()
    scanner_path = Path(scanner_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with scanner_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    project_root = values.get("root")
    values.pop("root", None)
    if project_root:
        root = Path(project_root)
        values["project_root"] = root if root.is_absolute() else ue_path.parent.parent / root
    else:
        values["project_root"] = None
    project_root_dir = ue_path.parent.parent
    parsed_dir = Path(ue_config["data"]["parsed"])
    if not parsed_dir.is_absolute():
        parsed_dir = project_root_dir / parsed_dir
    values["output_path"] = parsed_dir / "project" / "files.jsonl"
    values["issues_path"] = parsed_dir / "project" / "issues.jsonl"
    values["engine_version"] = str(ue_config["engine"]["version"])
    return ProjectScannerConfig(**values)


@dataclass(frozen=True)
class ProjectScanSummary:
    files: int
    modules: int
    plugins: int
    total_bytes: int
    issues: int
    by_type: dict[ProjectFileType, int]
    output_path: Path
    issues_path: Path


class ProjectSourceScanner:
    """Scan project roots while preserving provenance and content hashes."""

    def __init__(self, config: ProjectScannerConfig) -> None:
        self.config = config
        self._suffixes = {suffix.casefold() for suffix in config.include_suffixes}
        self._ignored = {name.casefold() for name in config.ignored_directories}

    def validate(self) -> Path:
        if self.config.project_root is None:
            raise ValueError("project root is not configured; pass --project-root")
        root = self.config.project_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Project root does not exist: {root}")
        for directory in self.config.source_directories:
            child = (root / directory).resolve()
            if not child.is_relative_to(root):
                raise ValueError(f"Project directory escapes configured root: {directory}")
            if not child.is_dir():
                if directory.casefold() == "source":
                    raise FileNotFoundError(f"Project source directory does not exist: {child}")
        return root

    def scan(self) -> ProjectScanSummary:
        root = self.validate()
        candidates: list[tuple[Path, str]] = []
        issues: list[ProjectScanIssue] = []
        for directory in self.config.source_directories:
            scan_root = root / directory
            for current, directory_names, file_names in os.walk(scan_root, topdown=True, followlinks=False):
                current_path = Path(current)
                directory_names[:] = sorted(
                    (name for name in directory_names if name.casefold() not in self._ignored and not (current_path / name).is_symlink()),
                    key=str.casefold,
                )
                for file_name in sorted(file_names, key=str.casefold):
                    path = current_path / file_name
                    if path.suffix.casefold() in self._suffixes:
                        candidates.append((path, directory.casefold()))
        ordered = sorted(set(candidates), key=lambda item: item[0].relative_to(root).as_posix().casefold())
        records: list[ProjectFileRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.hash_workers) as executor:
            for record, issue in executor.map(lambda item: self._record(item, root), ordered):
                if record:
                    records.append(record)
                if issue:
                    issues.append(issue)
        records.sort(key=lambda item: item.relative_path.casefold())
        issues.sort(key=lambda item: item.path.casefold())
        save_jsonl(self.config.output_path, records)
        save_jsonl(self.config.issues_path, issues)
        by_type = {file_type: 0 for file_type in ProjectFileType}
        for record in records:
            by_type[record.file_type] += 1
        return ProjectScanSummary(
            files=len(records),
            modules=len({record.module for record in records if record.module}),
            plugins=len({record.plugin for record in records if record.plugin}),
            total_bytes=sum(record.size for record in records),
            issues=len(issues),
            by_type=by_type,
            output_path=self.config.output_path,
            issues_path=self.config.issues_path,
        )

    def _record(self, item: tuple[Path, str], root: Path) -> tuple[ProjectFileRecord | None, ProjectScanIssue | None]:
        path, root_kind = item
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            return ProjectFileRecord(
                relative_path=relative,
                root_kind=root_kind,
                module=_module_for(path, root),
                plugin=_plugin_for(path, root),
                file_type=_file_type(path),
                source_type=SourceType.DOCS if root_kind.casefold() == "docs" else SourceType.PROJECT_SOURCE,
                engine_version=self.config.engine_version,
                size=stat.st_size,
                sha256=self._sha256(path),
            ), None
        except OSError as error:
            return None, ProjectScanIssue(path=str(path), error=str(error))

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(self.config.hash_chunk_size):
                digest.update(block)
        return digest.hexdigest()


def _file_type(path: Path) -> ProjectFileType:
    suffix = path.suffix.casefold()
    if path.name.casefold().endswith(".build.cs"):
        return ProjectFileType.BUILD_CS
    if suffix in {".h", ".hpp"}:
        return ProjectFileType.HEADER
    if suffix == ".cpp":
        return ProjectFileType.CPP
    if suffix == ".inl":
        return ProjectFileType.INL
    if suffix in {".ini", ".json", ".yaml", ".yml"}:
        return ProjectFileType.CONFIG
    return ProjectFileType.DOCUMENT


def _module_for(path: Path, root: Path) -> str | None:
    for parent in (path.parent, *path.parents):
        if parent == root:
            break
        for candidate in parent.glob("*.Build.cs"):
            return candidate.name[: -len(".Build.cs")]
    parts = path.relative_to(root).parts
    if len(parts) >= 2 and parts[0].casefold() == "source":
        return parts[1]
    if "source" in {part.casefold() for part in parts}:
        index = next(index for index, part in enumerate(parts) if part.casefold() == "source")
        return parts[index + 1] if index + 1 < len(parts) else None
    return None


def _plugin_for(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts):
        if part.casefold() == "plugins" and index + 1 < len(parts):
            return parts[index + 1]
    return None
