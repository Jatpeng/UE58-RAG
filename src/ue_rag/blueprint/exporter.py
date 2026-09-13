"""Normalize Blueprint summaries/graphs into the shared UEDocument schema."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ue_rag.jsonl import save_jsonl
from ue_rag.schema import SourceScope, SourceType, UEDocument


class BlueprintNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    node_type: str = Field(default="", alias="type")
    pins: list[dict[str, Any]] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class BlueprintGraph(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    nodes: list[BlueprintNode] = Field(default_factory=list)
    links: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_edges_alias(cls, value: Any) -> Any:
        if isinstance(value, dict) and "links" not in value and "edges" in value:
            value = {**value, "links": value["edges"]}
        return value


class BlueprintFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    graph_name: str | None = None
    graph: BlueprintGraph | None = None
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    description: str | None = None


class BlueprintVariable(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    type: str = ""
    default_value: Any = None
    category: str | None = None


class BlueprintComponent(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    class_name: str = Field(default="", alias="class")
    parent: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class BlueprintAsset(BaseModel):
    """Connector-neutral Blueprint export payload."""

    model_config = ConfigDict(extra="allow")

    asset_name: str = Field(min_length=1)
    package_path: str = ""
    parent_class: str | None = None
    graphs: list[BlueprintGraph] = Field(default_factory=list)
    functions: list[BlueprintFunction] = Field(default_factory=list)
    variables: list[BlueprintVariable] = Field(default_factory=list)
    components: list[BlueprintComponent] = Field(default_factory=list)


class BlueprintExporterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_version: str = Field(min_length=1)
    input_path: Path
    output_path: Path


def load_blueprint_config(
    ue_config_path: str | Path = "config/ue58.yaml",
    blueprint_config_path: str | Path = "config/blueprint.yaml",
) -> BlueprintExporterConfig:
    ue_path = Path(ue_config_path).resolve()
    blueprint_path = Path(blueprint_config_path).resolve()
    with ue_path.open(encoding="utf-8") as stream:
        ue_config = yaml.safe_load(stream)
    with blueprint_path.open(encoding="utf-8") as stream:
        values = dict(yaml.safe_load(stream) or {})
    project_root = ue_path.parent.parent
    for key in ("input_path", "output_path"):
        path = Path(values[key])
        values[key] = path if path.is_absolute() else project_root / path
    values["engine_version"] = str(ue_config["engine"]["version"])
    return BlueprintExporterConfig(**values)


class BlueprintExportSummary(BaseModel):
    assets: int
    documents: int
    output_path: Path


class BlueprintExporter(ABC):
    """Export five normalized Blueprint views for MCP or file connectors."""

    def __init__(self, engine_version: str) -> None:
        self.engine_version = engine_version

    @abstractmethod
    def export_blueprint_summary(self, asset: BlueprintAsset) -> UEDocument:
        """Export one high-level asset summary."""

    @abstractmethod
    def export_blueprint_graph(self, asset: BlueprintAsset, graph_name: str | None = None) -> list[UEDocument]:
        """Export one document per Blueprint graph."""

    @abstractmethod
    def export_blueprint_function(self, asset: BlueprintAsset, function_name: str | None = None) -> list[UEDocument]:
        """Export one document per Blueprint function."""

    @abstractmethod
    def export_blueprint_variables(self, asset: BlueprintAsset) -> UEDocument:
        """Export all Blueprint variables as one document."""

    @abstractmethod
    def export_blueprint_components(self, asset: BlueprintAsset) -> UEDocument:
        """Export all Blueprint components as one document."""

    def export_asset(self, asset: BlueprintAsset) -> list[UEDocument]:
        documents = [self.export_blueprint_summary(asset)]
        documents.extend(self.export_blueprint_graph(asset))
        documents.extend(self.export_blueprint_function(asset))
        documents.append(self.export_blueprint_variables(asset))
        documents.append(self.export_blueprint_components(asset))
        return documents


class JsonBlueprintExporter(BlueprintExporter):
    """Exporter for connector-neutral JSON payloads produced by Unreal tooling."""

    def export_blueprint_summary(self, asset: BlueprintAsset) -> UEDocument:
        content = "\n".join(
            [
                f"Blueprint: {asset.asset_name}",
                f"Package: {asset.package_path or '<unknown>'}",
                f"Parent Class: {asset.parent_class or '<unknown>'}",
                f"Graphs: {len(asset.graphs)}",
                f"Functions: {len(asset.functions)}",
                f"Variables: {len(asset.variables)}",
                f"Components: {len(asset.components)}",
            ]
        )
        return self._document(asset, "summary", asset.asset_name, content, metadata={"graph_count": len(asset.graphs)})

    def export_blueprint_graph(self, asset: BlueprintAsset, graph_name: str | None = None) -> list[UEDocument]:
        graphs = [graph for graph in asset.graphs if graph_name is None or graph.name == graph_name]
        return [
            self._document(
                asset,
                "graph",
                graph.name,
                _graph_content(graph),
                graph_name=graph.name,
                metadata={"node_count": len(graph.nodes), "edge_count": len(graph.links)},
            )
            for graph in graphs
        ]

    def export_blueprint_function(self, asset: BlueprintAsset, function_name: str | None = None) -> list[UEDocument]:
        functions = [function for function in asset.functions if function_name is None or function.name == function_name]
        return [
            self._document(
                asset,
                "function",
                function.name,
                _function_content(function),
                graph_name=function.graph_name or (function.graph.name if function.graph else None),
                metadata={"input_count": len(function.inputs), "output_count": len(function.outputs)},
            )
            for function in functions
        ]

    def export_blueprint_variables(self, asset: BlueprintAsset) -> UEDocument:
        content = _json_lines("Variables", [variable.model_dump(by_alias=True, mode="json") for variable in asset.variables])
        return self._document(asset, "variables", "Variables", content, metadata={"variable_count": len(asset.variables)})

    def export_blueprint_components(self, asset: BlueprintAsset) -> UEDocument:
        content = _json_lines("Components", [component.model_dump(by_alias=True, mode="json") for component in asset.components])
        return self._document(asset, "components", "Components", content, metadata={"component_count": len(asset.components)})

    def _document(
        self,
        asset: BlueprintAsset,
        kind: str,
        name: str,
        content: str,
        *,
        graph_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UEDocument:
        symbol = f"{asset.asset_name}::{name}" if kind != "summary" else asset.asset_name
        return UEDocument(
            id=_document_id(self.engine_version, asset.package_path, kind, name, content),
            engine_version=self.engine_version,
            source_scope=SourceScope.PROJECT,
            source_type=SourceType.BLUEPRINT,
            content=content,
            title=symbol,
            file_path=asset.package_path or None,
            symbol=symbol,
            symbol_type=f"blueprint_{kind}",
            asset_name=asset.asset_name,
            graph_name=graph_name,
            metadata={
                "export_kind": kind,
                "parent_class": asset.parent_class,
                "package_path": asset.package_path,
                **(metadata or {}),
            },
        )

    def export_jsonl(self, input_path: str | Path, output_path: str | Path) -> BlueprintExportSummary:
        assets = list(_load_assets(Path(input_path)))
        documents = [document for asset in assets for document in self.export_asset(asset)]
        save_jsonl(output_path, documents)
        return BlueprintExportSummary(assets=len(assets), documents=len(documents), output_path=Path(output_path))


def _load_assets(path: Path) -> Iterable[BlueprintAsset]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.casefold() == ".jsonl":
        return (BlueprintAsset.model_validate_json(line) for line in text.splitlines() if line.strip())
    payload = json.loads(text)
    values = payload if isinstance(payload, list) else [payload]
    return (BlueprintAsset.model_validate(value) for value in values)


def _graph_content(graph: BlueprintGraph) -> str:
    return _json_lines(
        f"Graph: {graph.name}",
        [{"node": node.model_dump(by_alias=True, mode="json")} for node in graph.nodes]
        + [{"link": link} for link in graph.links],
    )


def _function_content(function: BlueprintFunction) -> str:
    payload = function.model_dump(mode="json", exclude_none=True)
    return _json_lines(f"Function: {function.name}", [payload])


def _json_lines(title: str, values: list[dict[str, Any]]) -> str:
    return title + "\n" + "\n".join(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values)


def _document_id(engine_version: str, package: str, kind: str, name: str, content: str) -> str:
    raw = f"{engine_version}:{package}:{kind}:{name}:{content}".encode("utf-8")
    return "bp-" + hashlib.sha256(raw).hexdigest()
