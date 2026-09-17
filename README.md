# Unreal Engine 5.8 Developer RAG

**简体中文** | [English](README_EN.md)

面向 Unreal Engine 5.8 开发者的本地 RAG 检索系统。项目将 UE 官方文档、
引擎 C++ 源码、项目源码和 Blueprint 转换为结构化语义块，通过精确符号检索、
全文检索、向量检索、混合召回和重排，为 AI 编程助手提供可追溯的技术上下文。

> 仓库只发布程序、配置模板、测试和文档，不发布 Unreal Engine 源码、生成后的
> 数据集、索引、Embedding、模型缓存或项目私有代码。

## 预生成向量数据

如需跳过本地 Embedding 生成，可以从 Hugging Face 下载预生成的向量和 chunks：

[下载 UE5.8 RAG Embeddings Dataset](https://huggingface.co/datasets/Jatpeng/ue58-rag-embeddings)

下载后可双击 `从HuggingFace下载并重建索引.bat`，自动恢复 engine 向量并建立
Qdrant 索引。使用前请确认数据授权和 Hugging Face 仓库访问权限。

## 主要能力

- 使用 Tree-sitter 解析 UE C++ 类、结构体、枚举、函数、方法和字段。
- 保留 `UCLASS`、`USTRUCT`、`UFUNCTION`、`UPROPERTY` 等 UE 反射宏。
- 按类、函数、属性组和文档章节进行语义切块。
- 使用 SQLite FTS5 完成关键词和精确 C++ 符号检索。
- 使用 Qwen Embedding、Qdrant、RRF 和 Qwen Reranker 完成混合检索。
- 支持模块、插件、类、符号、来源类型和 UE 版本过滤。
- 通过 MCP 提供 `ue_search`、`ue_find_symbol`、`ue_search_docs` 和
  `ue_search_source` 四个工具。
- 包含检索 Benchmark、RAGAS 测试集生成和增量项目扫描能力。
- 提供本地数据观测台，可视化语料构成、切块分布、索引状态和 Benchmark 指标。

## 检索流程

```text
UE 文档 / 引擎源码 / 项目源码 / Blueprint
                    ↓
              解析与语义切块
                    ↓
       ┌────────────┴────────────┐
       ↓                         ↓
关键词与精确符号检索       Embedding + Qdrant
       └────────────┬────────────┘
                    ↓
                  RRF
                    ↓
                 Reranker
                    ↓
              MCP / AI 助手
```

## 环境要求

- Windows 10/11
- Python 3.11 或更高版本
- 本机已安装 Unreal Engine 5.8 源码
- 足够的磁盘空间；完整索引可能达到数 GB
- 语义检索建议使用支持 CUDA 的 NVIDIA GPU

### 本地部署硬件建议

这套 RAG 可以按需要部署为仅关键词检索、Hybrid 检索或完整重排版本。比较均衡的
完整本地配置是：**8 核 CPU、32 GB 内存、12～16 GB NVIDIA 显存，以及至少
100 GB 可用 NVMe 空间**。

| 部署模式 | CPU | 内存 | GPU | 建议可用磁盘 |
|---|---:|---:|---:|---:|
| 仅 Lexical / 符号检索 | 4 核以上 | 8 GB 最低，16 GB 推荐 | 不需要 | 30～50 GB |
| Hybrid，不启用重排 | 8 核以上 | 32 GB 推荐 | 8 GB 最低，12 GB 推荐 | 80～100 GB |
| Hybrid + Reranker | 8～12 核 | 32～64 GB | 12 GB 最低，16 GB 推荐 | 100 GB 以上 |
| RAG + 本地生成式 LLM | 12～16 核 | 64 GB | 24 GB 以上更合适 | 150 GB 以上 |

以上磁盘空间不包含 Unreal Engine 5.8、引擎源码和游戏项目本身。以当前约
1,693,105 个 C++ chunks 的完整语料为例，生成数据的实际占用约为：

| 数据 | 参考占用 |
|---|---:|
| SQLite 关键词索引 | 6.84 GiB |
| Qdrant 数据 | 18.58 GiB |
| Embedding 与缓存 | 14.00 GiB |
| 语义切块 | 3.07 GiB |
| 解析结果 | 2.38 GiB |
| 合计 | 约 44.9 GiB |

完整重建会先生成临时索引，再原子替换旧索引；重建期间可能同时存在新旧数据，
因此完整 Hybrid 部署建议至少预留 80～100 GB，而不是只按最终文件大小准备空间。

默认语义配置使用 `Qwen3-Embedding-0.6B`，批量为 16、最大输入长度为 2048；
Reranker 使用 `Qwen3-Reranker-0.6B`，批量为 8、最大输入长度为 4096。启用重排的
MCP 服务会同时加载 Embedding 和 Reranker：8 GB 显存可以通过降低 batch 和
token 上限运行，12 GB 适合单人 Hybrid 检索，16 GB 更适合完整重排。如果还要在
同一张显卡上运行本地生成式 LLM，建议使用 24 GB 或更多显存。

没有 NVIDIA GPU 时仍可使用 Lexical、符号搜索、源码/Blueprint 解析和切块。
Embedding 与 Reranker 也可以切换到 CPU，但全量生成向量和查询重排会明显变慢；
这种机器建议先部署 Lexical 版本，或改用云端 Embedding。

本项目的 RAG 服务负责返回检索上下文，不包含本地生成式 LLM。通过 MCP 将结果
交给 Codex、Claude 等客户端时，不需要额外为生成式模型预留本机显存。

## 拉取后快速开始

```powershell
git clone <仓库地址>
cd ue58-rag

powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 `
  -UnrealRoot "C:\Program Files\Epic Games\UE_5.8"
```

初始化脚本会：

1. 创建 `.venv` Python 虚拟环境。
2. 安装关键词检索所需依赖。
3. 验证本机 UE 版本和源码路径。
4. 扫描并解析 UE C++ 源码。
5. 生成语义块和本地 SQLite 检索索引。

完整建库可能持续较长时间。只检查环境而不建库：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1 `
  -UnrealRoot "C:\Program Files\Epic Games\UE_5.8" `
  -ValidateOnly
```

## 启动 MCP

本机 AI 客户端推荐使用 stdio：

```powershell
.\scripts\start_mcp.ps1
```

也可以启动 Streamable HTTP：

```powershell
.\scripts\start_mcp.ps1 -Transport streamable-http
```

默认 MCP 地址：

```text
http://127.0.0.1:8000/mcp
```

局域网共享：

```powershell
.\scripts\start_mcp.ps1 `
  -Transport streamable-http `
  -HostAddress 0.0.0.0 `
  -Port 8000
```

当前 MCP HTTP 服务自身不提供身份验证，请勿直接暴露到公网。公网部署应使用
VPN，或在服务前增加带 HTTPS、身份认证和限流的网关。

## 数据可视化

双击仓库根目录的 `启动RAG数据管理台.bat`，系统会自动打开本地管理页面。页面支持：

- 选择 UE 项目目录。
- 查看当前 UE 源码、官方文档、Blueprint 和项目语料的来源位置与索引状态。
- 使用示例问题或自定义问题测试 Symbol、Lexical 与 Hybrid 检索结果。
- 扫描并预览新增、修改、删除和未变化文件。
- 确认后增量解析 C++、配置与项目文档。
- 原子替换关键词索引中的旧切块。
- 可选同步 Embedding 和 Qdrant 向量索引。
- 保存同步状态，下一次只处理变化文件。

只生成无需服务的静态统计报告：

```powershell
python scripts/visualize.py --open
```

默认输出到 `outputs/rag_dashboard.html`。仪表盘从 SQLite 索引、Embedding 清单和
Benchmark JSON 自动汇总语料规模、来源与模块分布、切块长度、向量覆盖率及检索
质量。页面只写入聚合统计，不包含 UE 源码、文档正文或项目私有内容。

比较指定的多份评测报告：

```powershell
python scripts/visualize.py `
  --benchmark data/benchmark/benchmark_results.json `
  --benchmark data/benchmark/hybrid_benchmark_results.json `
  --output outputs/rag_dashboard.html
```

## 手动安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

运行测试：

```powershell
python -m pytest
```

Embedding 和 Reranker 是可选的大模型依赖：

```powershell
pip install -e ".[semantic]"
```

安装语义模型依赖后，仍需生成 Embedding 并建立 Qdrant 集合。完整过程参见
[索引与检索](docs/03_索引与检索.md)和
[本地 CUDA Embedding](docs/04_本地CUDA_Embedding.md)。

## MCP 客户端配置

```json
{
  "mcpServers": {
    "ue58-rag": {
      "command": "C:/path/to/ue58-rag/.venv/Scripts/python.exe",
      "args": ["C:/path/to/ue58-rag/scripts/mcp_server.py"]
    }
  }
}
```

服务注册以下工具：

| 工具 | 作用 |
|---|---|
| `ue_search` | 使用 lexical、symbol、hybrid 或 rerank 模式统一检索 |
| `ue_find_symbol` | 精确查找 C++ 类、函数或属性符号 |
| `ue_search_docs` | 只检索 UE 文档 |
| `ue_search_source` | 只检索 UE 引擎源码，可按模块和插件过滤 |

## 项目结构

```text
config/          配置模板
data/            本地生成数据；大部分内容被 Git 忽略
docs/            中文教学与部署文档
scripts/         数据处理、建库、查询、评测和服务入口
src/ue_rag/      Python 核心实现
tests/           自动化测试
```

## 数据库配置

本项目有两个索引：SQLite FTS5 关键词索引和 Qdrant 向量索引。两者都属于本机生成
数据，不应提交到 GitHub。

### SQLite 关键词索引

配置文件为 `config/lexical.yaml`：

```yaml
index_path: "data/index/lexical.sqlite3"
batch_size: 512
symbol_top_k: 10
lexical_top_k: 10
```

从 chunks 重新构建：

```powershell
python scripts/build_lexical_index.py `
  --rebuild `
  --batch-size 2048 `
  --progress-every 100000
```

### Qdrant 本地模式（默认）

默认配置为 `config/qdrant.yaml`：

```yaml
collection: "ue58_global"
path: "data/qdrant"
url: null
batch_size: 64
distance: "cosine"
```

`url: null` 表示使用本地嵌入式 Qdrant，数据保存在 `data/qdrant`。向量数据准备好
后执行：

```powershell
python scripts/build_index.py `
  --input data/chunks/engine/chunks.jsonl `
  --vectors data/embeddings/engine/embeddings.npy `
  --ids data/embeddings/engine/embeddings.npy.ids.jsonl
```

首次建立或需要完全重建集合时，加上 `--recreate`：

```powershell
python scripts/build_index.py `
  --input data/chunks/engine/chunks.jsonl `
  --vectors data/embeddings/engine/embeddings.npy `
  --ids data/embeddings/engine/embeddings.npy.ids.jsonl `
  --recreate
```

### Qdrant Server 模式

仓库提供 `config/qdrant_server.yaml`，用于连接本机或远程 Qdrant Server：

```yaml
collection: "ue58_global"
path: null
url: "http://127.0.0.1:6333"
batch_size: 128
distance: "cosine"
timeout_seconds: 60
```

启动本地 Qdrant Server：

```powershell
.\scripts\start_qdrant.ps1
```

向 Server 导入向量：

```powershell
python scripts/build_index.py `
  --qdrant-config config/qdrant_server.yaml `
  --input data/chunks/engine/chunks.jsonl `
  --vectors data/embeddings/engine/embeddings.npy `
  --ids data/embeddings/engine/embeddings.npy.ids.jsonl `
  --recreate
```

启动 MCP 时也必须使用同一个 Qdrant 配置：

```powershell
python scripts/mcp_server.py `
  --qdrant-config config/qdrant_server.yaml `
  --enable-dense
```

如果 Qdrant Server 需要 API Key，可在配置中增加环境变量名称，并在启动前设置密钥：

```yaml
api_key_env: "QDRANT_API_KEY"
```

```powershell
$env:QDRANT_API_KEY = "your-qdrant-api-key"
```

不要把真实 API Key 写入 YAML、`.env.example` 或 Git。`config/*.local.yaml`、`.env`
和数据库文件均应留在本机。

### 从 Hugging Face 恢复向量

如果已经上传了兼容的 Dataset，可以双击仓库根目录的
`从HuggingFace下载并重建索引.bat`，或手动执行：

```powershell
hf download YOUR_USERNAME/ue58-rag-embeddings `
  --repo-type=dataset `
  --local-dir=hf_download `
  --include "engine/*"
```

然后按照上面的 Qdrant 配置执行 `build_index.py`。chunks、向量矩阵和
`embeddings.npy.ids.jsonl` 必须来自同一次构建，不能混用不同版本的数据。

## GitHub 提交边界

应该提交：

- `src/`、`scripts/`、`tests/`
- `config/*.yaml` 配置模板
- `README.md`、`README_EN.md`、`docs/`
- `.env.example`、`.gitignore`、`.gitattributes`
- 一键启动、下载和重建索引脚本

不应提交：

- Unreal Engine 源码、项目源码和下载的官方文档正文
- `data/raw/`、`data/parsed/`、`data/chunks/`
- `data/embeddings/`、`data/index/`、`data/qdrant/`
- `data/qdrant_server/`、`data/qdrant_server_incomplete_*/`
- `work/`、`outputs/`、`hf_package/`、`hf_download/`
- `*.sqlite3`、模型缓存、`__pycache__/`、`.venv/`
- `.env`、本地配置和任何 API Key

提交前检查：

```powershell
git status --short --ignored
git check-ignore -v data/embeddings/engine/embeddings.npy
git check-ignore -v data/qdrant/collection/ue58_global/storage.sqlite
git check-ignore -v hf_package
```

当前本地生成数据可以删除并重新生成，但删除前应确认不再需要：

```text
data/raw/       可重新爬取或扫描
data/parsed/    可由 raw 重新解析
data/chunks/    可由 parsed 重新切块
data/index/     可由 chunks 重新构建
data/embeddings/可由 chunks 重新向量化，或从 Hugging Face 恢复
data/qdrant/    可由 embeddings 重新导入
work/           临时文件和日志
outputs/        报告和仪表盘输出
hf_package/     Hugging Face 上传暂存包
hf_download/    Hugging Face 下载暂存目录
```

删除这些目录不会删除程序代码，但会导致后续重新生成或重新下载；不要删除
`src/`、`scripts/`、`tests/`、`config/` 和 `docs/`。

## 文档

- [中文文档目录](docs/README.md)
- [GitHub 拉取与使用](docs/09_GitHub使用指南.md)
- [文档与源码处理](docs/02_文档与源码处理.md)
- [索引与检索](docs/03_索引与检索.md)
- [MCP、项目和 Blueprint](docs/05_MCP项目与Blueprint.md)
- [测试、排错与上线](docs/06_测试排错与上线.md)
- [Benchmark 评测指南](docs/08_Benchmark评测指南.md)
- [完整英文说明](README_EN.md)

## 数据与授权

生成的 UE 源码语料、JSONL、SQLite 索引、Embedding 和 Qdrant 数据均不会提交。
每位使用者需要使用自己有权访问的 UE 5.8 安装在本机建库。公开或向第三方提供
检索服务前，请确认符合 Epic Games 的许可条款及所属项目的保密要求。

## 参与贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md)。提交前请确保：

- 没有 Unreal Engine 源码、项目私有源码或生成索引进入 Git。
- 没有 `.env`、API Key、访问令牌或本机绝对路径。
- `python -m pytest` 全部通过。
