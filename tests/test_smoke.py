"""Smoke tests for the T01 project skeleton."""

from pathlib import Path

import pytest
import yaml

import ue_rag


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports() -> None:
    """The src-layout package is importable."""
    assert ue_rag.__name__ == "ue_rag"


@pytest.mark.parametrize(
    "config_name",
    ["ue58.yaml", "embedding.yaml", "retrieval.yaml"],
)
def test_configuration_can_be_read(config_name: str) -> None:
    """Every T01 YAML configuration file is valid and non-empty."""
    config_path = PROJECT_ROOT / "config" / config_name
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert isinstance(config, dict)
    assert config


def test_ue_version_comes_from_configuration() -> None:
    """The configured engine version is Unreal Engine 5.8."""
    config_path = PROJECT_ROOT / "config" / "ue58.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["engine"]["version"] == "5.8"


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/ue_rag/crawler",
        "src/ue_rag/parser",
        "src/ue_rag/chunker",
        "src/ue_rag/embedding",
        "src/ue_rag/index",
        "src/ue_rag/retrieval",
        "src/ue_rag/reranker",
        "src/ue_rag/eval",
        "src/ue_rag/mcp",
        "data/raw",
        "data/parsed",
        "data/chunks",
        "data/benchmark",
    ],
)
def test_core_directory_exists(relative_path: str) -> None:
    """The planned package and data directories exist."""
    assert (PROJECT_ROOT / relative_path).is_dir()
