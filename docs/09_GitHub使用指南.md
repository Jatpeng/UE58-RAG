# GitHub 拉取与使用

这个仓库发布程序、配置模板和测试，不发布生成后的 Unreal Engine
源码语料、SQLite 索引、Embedding 或 Qdrant 数据。这样可以避开 GitHub
的大文件限制，也避免重新分发 Epic 或项目私有源码。

## 使用条件

- Windows 10/11
- Python 3.11 或更高版本
- 本机已经安装与仓库配置匹配的 Unreal Engine 5.8 源码
- 足够的磁盘空间；完整关键词索引本身可能达到数 GB

## 拉取后初始化

在 PowerShell 中运行：

```powershell
git clone <仓库地址>
cd ue58-rag
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 `
  -UnrealRoot "C:\Program Files\Epic Games\UE_5.8"
```

脚本会创建 `.venv`、安装依赖、验证 UE 版本，并依次完成源码扫描、
解析、切块和关键词索引构建。完整构建会持续较长时间，具体取决于
CPU、磁盘和 UE 安装内容。

只检查环境、不开始建库：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 `
  -UnrealRoot "C:\Program Files\Epic Games\UE_5.8" `
  -ValidateOnly
```

再次建库时可以跳过依赖安装：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 `
  -UnrealRoot "C:\Program Files\Epic Games\UE_5.8" `
  -SkipInstall
```

## 连接 AI 客户端

本机客户端推荐使用 stdio：

```json
{
  "mcpServers": {
    "ue58-rag": {
      "command": "C:/path/to/ue58-rag/.venv/Scripts/python.exe",
      "args": ["C:/path/to/ue58-rag/scripts/mcp_server.py"],
      "cwd": "C:/path/to/ue58-rag"
    }
  }
}
```

也可以先启动 HTTP MCP 服务：

```powershell
.\scripts\start_mcp.ps1 -Transport streamable-http
```

客户端连接 `http://127.0.0.1:8000/mcp`。

如需供可信局域网中的其他机器连接：

```powershell
.\scripts\start_mcp.ps1 `
  -Transport streamable-http `
  -HostAddress 0.0.0.0 `
  -Port 8000
```

然后放行对应的 Windows 防火墙端口。当前 MCP 服务本身没有身份验证，
不要直接暴露到公网；公网使用应放在 VPN 或带 HTTPS 和身份认证的网关后。

## 可选的语义检索

默认安装只包含关键词/符号检索所需依赖。需要 Embedding 和 Reranker 时：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[semantic]"
```

之后仍需生成 Embedding、建立 Qdrant 集合，再用 `-EnableDense` 或
`-EnableRerank` 启动服务。详细流程参见其他教学文档。

## 发布仓库前检查

1. 保持 `data/` 下生成文件、`.env`、`.venv` 和 `work/` 不进入 Git。
2. 不要提交 Unreal Engine 源码、项目私有源码、模型缓存或 API Key。
3. 公开仓库前选择合适的代码许可证，并确认 UE 相关内容符合 Epic 的许可条款。
4. 运行 `pytest`，确认测试通过。
