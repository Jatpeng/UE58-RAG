"""Tests for MCP tool adapters without requiring a live MCP transport."""

from __future__ import annotations

import pytest

from ue_rag.mcp import MCPTools, create_mcp_server
from ue_rag.schema import RetrievalResult, SourceType


def result() -> RetrievalResult:
    return RetrievalResult(
        document_id="doc-1",
        chunk_id="chunk-1",
        engine_version="5.8",
        source_type=SourceType.ENGINE_SOURCE,
        content="source",
        score=1.0,
        metadata={"symbol": "AHero::Tick"},
    )


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, dict | None]] = []

    def search(self, query: str, *, mode: str, limit: int, query_vector=None, filters=None):
        self.calls.append((query, mode, limit, filters))
        return [result()]


def test_ue_search_returns_json_safe_payload() -> None:
    service = FakeService()
    tools = MCPTools(service)

    payload = tools.ue_search("movement", mode="hybrid", limit=3, query_vector=[1, 0])

    assert payload[0]["chunk_id"] == "chunk-1"
    assert service.calls == [("movement", "hybrid", 3, None)]


def test_find_symbol_and_scoped_search_set_filters() -> None:
    service = FakeService()
    tools = MCPTools(service)

    tools.ue_find_symbol("AHero::Tick", module="Engine")
    tools.ue_search_docs("networking")
    tools.ue_search_source("movement", plugin="Gameplay")

    assert service.calls[0][3] == {"module": "Engine"}
    assert service.calls[1][3] == {"source_type": "docs"}
    assert service.calls[2][3] == {"source_type": "engine_source", "plugin": "Gameplay"}


def test_mcp_tools_validate_limits_and_queries() -> None:
    tools = MCPTools(FakeService(), max_limit=5)

    with pytest.raises(ValueError, match="non-empty"):
        tools.ue_search(" ")
    with pytest.raises(ValueError, match="between 1 and 5"):
        tools.ue_search("query", limit=6)
    with pytest.raises(ValueError, match="between 1 and 5"):
        tools.ue_search("query", limit=0)


def test_mcp_server_registers_when_sdk_is_installed() -> None:
    server = create_mcp_server(MCPTools(FakeService()))

    assert server is not None
