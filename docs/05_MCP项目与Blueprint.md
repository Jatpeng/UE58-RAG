# 05 MCP、项目和 Blueprint

本篇说明如何把 RAG 接入 MCP 客户端，并加入真实项目和 Blueprint 数据。

## 1. MCP stdio 模式

Cursor 或 Claude 可以启动：

```powershell
python scripts/mcp_server.py
```

stdio 模式等待 JSON-RPC 消息。不要在终端中手动敲空行，否则会出现 `Invalid JSON: EOF`。

客户端配置示例：

```json
{
  "mcpServers": {
    "ue-rag": {
      "command": "python",
      "args": ["E:/UE5.8 RAG/scripts/mcp_server.py"]
    }
  }
}
```

## 2. MCP HTTP 模式

```powershell
python scripts/mcp_server.py --transport streamable-http
```

默认端点：

```text
http://127.0.0.1:8000/mcp
```

这是 MCP 协议端点，不是 HTML 搜索页面。浏览器直接 GET 可能返回 406，这是正常的协议行为。

注册工具：

- `ue_search`：统一搜索。
- `ue_find_symbol`：精确符号查询。
- `ue_search_docs`：仅搜索文档。
- `ue_search_source`：仅搜索引擎源码。

## 3. 扫描真实项目

```powershell
python scripts/ingest_project.py `
  --project-root "D:\Work\MyGame" `
  --dry-run
python scripts/ingest_project.py --project-root "D:\Work\MyGame"
```

默认扫描 `Source/`、`Plugins/`、`Config/` 和 `Docs/`，跳过 `Binaries/`、`Intermediate/`、`Saved/` 和 `DerivedDataCache/`。

输出：

```text
data/parsed/project/files.jsonl
data/parsed/project/issues.jsonl
```

## 4. 增量更新

```powershell
python scripts/incremental_project.py `
  --previous data/parsed/project/files.previous.jsonl `
  --current data/parsed/project/files.jsonl `
  --output data/parsed/project/changes.jsonl
```

系统会区分新增、修改、删除和未变化文件，只把变化部分送入后续解析和索引流程。

## 5. Blueprint 数据

当前 Blueprint 导出器接收连接器无关的 JSON，不直接解析二进制 `.uasset`。

```powershell
python scripts/export_blueprint.py `
  --input data/raw/project/blueprints.jsonl `
  --output data/parsed/project/blueprints.jsonl

python scripts/chunk_blueprint.py `
  --input data/parsed/project/blueprints.jsonl `
  --output data/chunks/project/blueprints.jsonl
```

每个 Blueprint 可生成 summary、graph、function、variables 和 components 视图，然后复用既有的 Embedding、Lexical、Qdrant 和 MCP 链路。
