"""Model Context Protocol tools over the unified UE query service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ue_rag.retrieval.query import UnifiedQueryService
from ue_rag.schema import RetrievalResult


class MCPTools:
    """Validated MCP-facing operations, independent of the MCP SDK runtime."""

    def __init__(self, query_service: UnifiedQueryService, *, max_limit: int = 50) -> None:
        if max_limit <= 0:
            raise ValueError("max_limit must be greater than zero")
        self.query_service = query_service
        self.max_limit = max_limit

    def ue_search(
        self,
        query: str,
        *,
        mode: str = "lexical",
        limit: int = 10,
        query_vector: Sequence[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self._run(query, mode=mode, limit=limit, query_vector=query_vector, filters=filters)

    def ue_find_symbol(
        self,
        symbol: str,
        *,
        limit: int = 10,
        module: str | None = None,
        plugin: str | None = None,
        class_name: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = {
            key: value
            for key, value in {"module": module, "plugin": plugin, "class": class_name}.items()
            if value is not None
        }
        return self._run(symbol, mode="symbol", limit=limit, filters=filters)

    def ue_search_docs(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        return self._run(query, mode="lexical", limit=limit, filters={"source_type": "docs"})

    def ue_search_source(
        self,
        query: str,
        *,
        limit: int = 10,
        module: str | None = None,
        plugin: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = {
            key: value
            for key, value in {
                "source_type": "engine_source",
                "module": module,
                "plugin": plugin,
            }.items()
            if value is not None
        }
        return self._run(query, mode="lexical", limit=limit, filters=filters)

    def _run(
        self,
        query: str,
        *,
        mode: str,
        limit: int,
        query_vector: Sequence[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if limit <= 0 or limit > self.max_limit:
            raise ValueError(f"limit must be between 1 and {self.max_limit}")
        results = self.query_service.search(
            query,
            mode=mode,
            limit=limit,
            query_vector=query_vector,
            filters=filters,
        )
        return [result_payload(result) for result in results]


def result_payload(result: RetrievalResult) -> dict[str, Any]:
    """Convert a retrieval result into an MCP-safe JSON object."""

    return result.model_dump(mode="json")


def create_mcp_server(
    tools: MCPTools | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:
    """Register the four UE tools on an official FastMCP server instance."""

    if tools is None:
        raise ValueError("MCPTools instance is required")
    if not host.strip():
        raise ValueError("host must be non-empty")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError("mcp is required to run the MCP server; install mcp[cli]") from error
    server = FastMCP("ue58-rag", host=host, port=port)

    @server.tool()
    def ue_search(
        query: str,
        mode: str = "lexical",
        limit: int = 10,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Search UE knowledge with lexical, symbol, hybrid, or rerank mode."""

        return tools.ue_search(query, mode=mode, limit=limit, query_vector=query_vector)

    @server.tool()
    def ue_find_symbol(
        symbol: str,
        limit: int = 10,
        module: str | None = None,
        plugin: str | None = None,
        class_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find an exact C++ symbol or merged-property alias."""

        return tools.ue_find_symbol(symbol, limit=limit, module=module, plugin=plugin, class_name=class_name)

    @server.tool()
    def ue_search_docs(query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search UE documentation chunks."""

        return tools.ue_search_docs(query, limit=limit)

    @server.tool()
    def ue_search_source(
        query: str,
        limit: int = 10,
        module: str | None = None,
        plugin: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search UE Engine source chunks."""

        return tools.ue_search_source(query, limit=limit, module=module, plugin=plugin)

    return server
