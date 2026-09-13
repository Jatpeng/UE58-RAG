"""Tests for connector-neutral Blueprint export documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ue_rag.blueprint import BlueprintAsset, JsonBlueprintExporter, load_blueprint_config
from ue_rag.jsonl import load_jsonl
from ue_rag.schema import SourceScope, SourceType, UEDocument


def asset() -> BlueprintAsset:
    return BlueprintAsset.model_validate(
        {
            "asset_name": "BP_PlayerCharacter",
            "package_path": "/Game/Characters/BP_PlayerCharacter",
            "parent_class": "Character",
            "graphs": [
                {
                    "name": "EventGraph",
                    "nodes": [
                        {"id": "1", "name": "BeginPlay", "type": "Event", "pins": [{"name": "Exec"}]},
                        {"id": "2", "name": "PrintString", "type": "CallFunction"},
                    ],
                    "edges": [{"from": "1", "to": "2"}],
                }
            ],
            "functions": [
                {"name": "GetHealth", "graph_name": "EventGraph", "outputs": [{"name": "Health", "type": "float"}]}
            ],
            "variables": [{"name": "Health", "type": "float", "default_value": 100.0}],
            "components": [{"name": "Camera", "class": "CameraComponent", "parent": "RootComponent"}],
        }
    )


def test_exporter_emits_five_blueprint_views() -> None:
    documents = JsonBlueprintExporter("5.8").export_asset(asset())

    assert len(documents) == 5
    assert {document.metadata["export_kind"] for document in documents} == {
        "summary",
        "graph",
        "function",
        "variables",
        "components",
    }
    assert all(document.source_scope is SourceScope.PROJECT for document in documents)
    assert all(document.source_type is SourceType.BLUEPRINT for document in documents)


def test_summary_preserves_counts_and_parent_class() -> None:
    document = JsonBlueprintExporter("5.8").export_blueprint_summary(asset())

    assert "Blueprint: BP_PlayerCharacter" in document.content
    assert "Graphs: 1" in document.content
    assert document.metadata["parent_class"] == "Character"
    assert document.symbol == "BP_PlayerCharacter"


def test_graph_accepts_edges_alias_and_records_nodes() -> None:
    document = JsonBlueprintExporter("5.8").export_blueprint_graph(asset())[0]

    assert document.graph_name == "EventGraph"
    assert document.metadata["node_count"] == 2
    assert document.metadata["edge_count"] == 1
    assert '"from": "1"' in document.content


def test_function_can_be_filtered_by_name() -> None:
    exporter = JsonBlueprintExporter("5.8")

    documents = exporter.export_blueprint_function(asset(), "GetHealth")

    assert len(documents) == 1
    assert documents[0].symbol == "BP_PlayerCharacter::GetHealth"
    assert documents[0].graph_name == "EventGraph"


def test_variables_and_components_are_structured_documents() -> None:
    exporter = JsonBlueprintExporter("5.8")
    variables = exporter.export_blueprint_variables(asset())
    components = exporter.export_blueprint_components(asset())

    assert variables.metadata["variable_count"] == 1
    assert '"name": "Health"' in variables.content
    assert components.metadata["component_count"] == 1
    assert '"class": "CameraComponent"' in components.content


def test_ids_are_deterministic_and_change_with_content() -> None:
    exporter = JsonBlueprintExporter("5.8")
    first = exporter.export_asset(asset())
    second = exporter.export_asset(asset())
    changed = asset().model_copy(update={"parent_class": "Pawn"})

    assert [document.id for document in first] == [document.id for document in second]
    assert first[0].id != exporter.export_blueprint_summary(changed).id


def test_json_export_supports_jsonl_and_single_json(tmp_path: Path) -> None:
    exporter = JsonBlueprintExporter("5.8")
    jsonl_path = tmp_path / "blueprints.jsonl"
    jsonl_path.write_text(asset().model_dump_json() + "\n", encoding="utf-8")
    json_path = tmp_path / "blueprints.json"
    json_path.write_text(json.dumps(asset().model_dump(mode="json")), encoding="utf-8")

    jsonl_summary = exporter.export_jsonl(jsonl_path, tmp_path / "jsonl-output.jsonl")
    json_summary = exporter.export_jsonl(json_path, tmp_path / "json-output.jsonl")

    assert jsonl_summary.assets == json_summary.assets == 1
    assert jsonl_summary.documents == json_summary.documents == 5
    assert len(load_jsonl(jsonl_summary.output_path, UEDocument)) == 5


def test_config_uses_project_paths() -> None:
    config = load_blueprint_config()

    assert config.engine_version == "5.8"
    assert config.input_path == Path.cwd() / "data/raw/project/blueprints.jsonl"
    assert config.output_path == Path.cwd() / "data/parsed/project/blueprints.jsonl"


def test_empty_json_input_exports_zero_documents(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")

    summary = JsonBlueprintExporter("5.8").export_jsonl(input_path, tmp_path / "output.jsonl")

    assert summary.assets == summary.documents == 0
