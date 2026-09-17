"""Inventory Unreal Engine source files with module/plugin provenance and hashes."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ue_rag.jsonl import save_jsonl


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EngineFileType(str, Enum):
    """Source file types included in the engine inventory."""

    HEADER = "header"
    CPP = "cpp"
    INL = "inl"
    BUILD_CS = "build_cs"


class EngineFileRecord(BaseModel):
    """One deterministic source inventory record."""

    model_config = ConfigDict(extra="forbid")

    relative_path: NonEmptyString
    module: str | None
    plugin: str | None
    file_type: EngineFileType
    engine_version: NonEmptyString
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EngineScanIssue(BaseModel):
    """Auditable file-system error encountered during scanning or hashing."""

    model_config = ConfigDict(extra="forbid")

    path: str
    error: str


class EngineScannerConfig(BaseModel):
    """Resolved paths and scanning rules."""

    model_config = ConfigDict(extra="forbid")

    engine_version: NonEmptyString
    engine_root: Path
    source_root: Path
    plugins_root: Path
    output_path: Path
    issues_path: Path
    include_suffixes: set[str] = Field(min_length=1)
    source_categories: list[str] = Field(min_length=1)
    ignored_directories: set[str]
    hash_chunk_size: int = Field(gt=0)
    hash_workers: int = Field(gt=0)


def load_engine_scanner_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    scanner_config_path: str | Path = "config/engine_scanner.yaml",
) -> EngineScannerConfig:
    """Load engine paths/version from UE config and scanner rules separately."""

    ue_path = Path(ue_config_path).resolve()
    scanner_path = Path(scanner_config_path).resolve()
    with ue_path.open(encoding="utf-8") as config_file:
        ue_config = yaml.safe_load(config_file)
    with scanner_path.open(encoding="utf-8") as config_file:
        scanner_config = yaml.safe_load(config_file)

    project_root = ue_path.parent.parent
    # Environment configuration makes a cloned repository portable while the
    # YAML value remains a convenient fallback for a private/local checkout.
    configured_root = os.environ.get("UE_ROOT", "").strip()
    if not configured_root:
        configured_root = str(ue_config["engine"]["root"]).strip()
    if not configured_root:
        raise ValueError(
            f"Unreal Engine root is not configured; set UE_ROOT or engine.root in {ue_path}"
        )
    engine_root = Path(configured_root).expanduser()
    if not engine_root.is_absolute():
        engine_root = project_root / engine_root
    engine_root = engine_root.resolve()

    source_root = (engine_root / ue_config["paths"]["engine_source"]).resolve()
    plugins_root = (engine_root / ue_config["paths"]["engine_plugins"]).resolve()
    parsed_dir = Path(ue_config["data"]["parsed"])
    if not parsed_dir.is_absolute():
        parsed_dir = project_root / parsed_dir
    output_dir = parsed_dir / "engine"

    return EngineScannerConfig(
        engine_version=str(ue_config["engine"]["version"]),
        engine_root=engine_root,
        source_root=source_root,
        plugins_root=plugins_root,
        output_path=output_dir / "files.jsonl",
        issues_path=output_dir / "issues.jsonl",
        **scanner_config,
    )


@dataclass(frozen=True)
class EngineScanSummary:
    """Aggregate inventory counts returned to the CLI."""

    files: int
    modules: int
    plugins: int
    total_bytes: int
    issues: int
    by_type: dict[EngineFileType, int]
    output_path: Path
    issues_path: Path


class EngineSourceScanner:
    """Scan Engine/Source and Engine/Plugins without parsing file contents."""

    def __init__(self, config: EngineScannerConfig) -> None:
        self.config = config
        self._suffixes = {suffix.casefold() for suffix in config.include_suffixes}
        self._ignored = {name.casefold() for name in config.ignored_directories}

    def validate(self) -> None:
        """Verify configured roots and installed engine version before scanning."""

        root = self.config.engine_root
        if not root.is_dir():
            raise FileNotFoundError(f"Engine root does not exist: {root}")
        for scan_root in (self.config.source_root, self.config.plugins_root):
            if not scan_root.is_relative_to(root):
                raise ValueError(f"Scan root escapes configured engine root: {scan_root}")
            if not scan_root.is_dir():
                raise FileNotFoundError(f"Engine scan root does not exist: {scan_root}")
        for category in self.config.source_categories:
            category_root = (self.config.source_root / category).resolve()
            if not category_root.is_relative_to(self.config.source_root):
                raise ValueError(f"Source category escapes Engine/Source: {category}")
            if not category_root.is_dir():
                raise FileNotFoundError(f"Engine source category does not exist: {category_root}")

        build_version_path = root / "Engine" / "Build" / "Build.version"
        if not build_version_path.is_file():
            raise FileNotFoundError(f"Engine Build.version is missing: {build_version_path}")
        try:
            build_version = json.loads(build_version_path.read_text(encoding="utf-8-sig"))
            installed_version = (
                f"{int(build_version['MajorVersion'])}.{int(build_version['MinorVersion'])}"
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Engine Build.version: {build_version_path}") from error
        if installed_version != self.config.engine_version:
            raise ValueError(
                f"Configured UE version {self.config.engine_version!r} does not match "
                f"installed engine version {installed_version!r}: {root}"
            )

    def scan(self) -> EngineScanSummary:
        """Create and persist a deterministic inventory for both configured roots."""

        self.validate()
        candidate_files: list[Path] = []
        descriptor_paths: list[Path] = []
        discovery_issues: list[EngineScanIssue] = []
        scan_targets = [
            (self.config.source_root / category, False)
            for category in self.config.source_categories
        ]
        scan_targets.append((self.config.plugins_root, True))
        for scan_root, plugins_source_only in scan_targets:
            files, descriptors, issues = self._discover(
                scan_root, plugins_source_only=plugins_source_only
            )
            candidate_files.extend(files)
            descriptor_paths.extend(descriptors)
            discovery_issues.extend(issues)

        module_roots = self._module_roots(candidate_files)
        plugin_roots = {
            path.parent.resolve(): path.stem for path in descriptor_paths
        }
        records: list[EngineFileRecord] = []
        issues = list(discovery_issues)
        ordered_paths = sorted(
            set(candidate_files), key=lambda item: item.as_posix().casefold()
        )
        with ThreadPoolExecutor(max_workers=self.config.hash_workers) as executor:
            for record, issue in executor.map(
                lambda path: self._record_for_path(path, module_roots, plugin_roots),
                ordered_paths,
            ):
                if record:
                    records.append(record)
                if issue:
                    issues.append(issue)

        records.sort(key=lambda record: record.relative_path.casefold())
        issues.sort(key=lambda issue: issue.path.casefold())
        save_jsonl(self.config.output_path, records)
        save_jsonl(self.config.issues_path, issues)

        by_type = {file_type: 0 for file_type in EngineFileType}
        for record in records:
            by_type[record.file_type] += 1
        return EngineScanSummary(
            files=len(records),
            modules=len({record.module for record in records if record.module}),
            plugins=len({record.plugin for record in records if record.plugin}),
            total_bytes=sum(record.size for record in records),
            issues=len(issues),
            by_type=by_type,
            output_path=self.config.output_path,
            issues_path=self.config.issues_path,
        )

    def _discover(
        self, scan_root: Path, *, plugins_source_only: bool
    ) -> tuple[list[Path], list[Path], list[EngineScanIssue]]:
        candidates: list[Path] = []
        descriptors: list[Path] = []
        issues: list[EngineScanIssue] = []

        def on_error(error: OSError) -> None:
            issues.append(
                EngineScanIssue(path=error.filename or str(scan_root), error=str(error))
            )

        for current, directory_names, file_names in os.walk(
            scan_root, topdown=True, onerror=on_error, followlinks=False
        ):
            current_path = Path(current)
            directory_names[:] = sorted(
                (
                    name
                    for name in directory_names
                    if name.casefold() not in self._ignored
                    and not (current_path / name).is_symlink()
                ),
                key=str.casefold,
            )
            for file_name in sorted(file_names, key=str.casefold):
                path = current_path / file_name
                lower_name = file_name.casefold()
                if lower_name.endswith(".uplugin"):
                    descriptors.append(path)
                relative_parts = path.relative_to(scan_root).parts
                is_plugin_source = any(
                    part.casefold() == "source" for part in relative_parts[:-1]
                )
                if (
                    any(lower_name.endswith(suffix) for suffix in self._suffixes)
                    and (not plugins_source_only or is_plugin_source)
                ):
                    candidates.append(path)
        return candidates, descriptors, issues

    def _record_for_path(
        self,
        path: Path,
        module_roots: dict[Path, str],
        plugin_roots: dict[Path, str],
    ) -> tuple[EngineFileRecord | None, EngineScanIssue | None]:
        try:
            stat = path.stat()
            return (
                EngineFileRecord(
                    relative_path=path.relative_to(self.config.engine_root).as_posix(),
                    module=self._find_module(path, module_roots),
                    plugin=self._find_plugin(path, plugin_roots),
                    file_type=_file_type(path),
                    engine_version=self.config.engine_version,
                    size=stat.st_size,
                    sha256=self._sha256(path),
                ),
                None,
            )
        except OSError as error:
            return None, EngineScanIssue(path=str(path), error=str(error))

    @staticmethod
    def _module_roots(candidate_files: list[Path]) -> dict[Path, str]:
        roots: dict[Path, str] = {}
        for path in candidate_files:
            lower_name = path.name.casefold()
            if lower_name.endswith(".build.cs"):
                roots[path.parent.resolve()] = path.name[: -len(".Build.cs")]
        return roots

    def _find_module(self, path: Path, module_roots: dict[Path, str]) -> str | None:
        discovered = self._nearest_named_root(path, module_roots)
        if discovered:
            return discovered

        if path.is_relative_to(self.config.source_root):
            relative_parts = path.relative_to(self.config.source_root).parts
            if len(relative_parts) >= 2:
                return relative_parts[1]
        if path.is_relative_to(self.config.plugins_root):
            relative_parts = path.relative_to(self.config.plugins_root).parts
            source_indices = [
                index
                for index, part in enumerate(relative_parts)
                if part.casefold() == "source"
            ]
            if source_indices and source_indices[-1] + 1 < len(relative_parts):
                return relative_parts[source_indices[-1] + 1]
        return None

    def _find_plugin(self, path: Path, plugin_roots: dict[Path, str]) -> str | None:
        discovered = self._nearest_named_root(path, plugin_roots)
        if discovered:
            return discovered
        if not path.is_relative_to(self.config.plugins_root):
            return None
        relative_parts = path.relative_to(self.config.plugins_root).parts
        source_indices = [
            index for index, part in enumerate(relative_parts) if part.casefold() == "source"
        ]
        if source_indices and source_indices[-1] > 0:
            return relative_parts[source_indices[-1] - 1]
        return None

    def _nearest_named_root(
        self, path: Path, roots: dict[Path, str]
    ) -> str | None:
        current = path.parent.resolve()
        engine_root = self.config.engine_root
        while current.is_relative_to(engine_root):
            if current in roots:
                return roots[current]
            if current == engine_root:
                break
            current = current.parent
        return None

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source_file:
            while block := source_file.read(self.config.hash_chunk_size):
                digest.update(block)
        return digest.hexdigest()


def _file_type(path: Path) -> EngineFileType:
    name = path.name.casefold()
    if name.endswith(".build.cs"):
        return EngineFileType.BUILD_CS
    if name.endswith(".cpp"):
        return EngineFileType.CPP
    if name.endswith(".inl"):
        return EngineFileType.INL
    if name.endswith(".h"):
        return EngineFileType.HEADER
    raise ValueError(f"Unsupported engine source file type: {path}")
