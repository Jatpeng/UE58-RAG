# Unreal Engine 5.8 Developer RAG

## Project Goal

This project will provide high-quality Unreal Engine technical context to AI coding agents. The planned pipeline is:

```text
UE Documentation
UE Engine Source
Lyra
Project Source
Blueprint

→ Parse
→ Semantic Chunk
→ Dense + Sparse Retrieval
→ RRF
→ Reranker
→ MCP
```

The project currently implements the foundation, shared data contracts,
documentation ingestion/parser/chunker, Unreal Engine source inventory, and the
Unreal C++ semantic parser. Embedding, indexing, retrieval, reranking, and MCP
integration remain future tasks.

## Environment

- Python >= 3.11

## Install

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

## Configuration

Project settings live outside the Python source code:

- `config/ue58.yaml` — engine version, engine paths, and data paths
- `config/embedding.yaml` — future embedding provider and model settings
- `config/retrieval.yaml` — future retrieval and reranking limits
- `.env.example` — environment-specific values; copy it to `.env` when needed

Do not commit a local `.env` file or generated Unreal Engine data.

## Test

```bash
pytest
```

## Shared data contracts

All future ingestion pipelines use the same Pydantic models:

- `UEDocument` for parsed source units
- `UEChunk` for semantic chunks
- `RetrievalResult` for retrieval output
- `SourceType` and `SourceScope` for consistent provenance

Validated records can be persisted with `save_jsonl()` and loaded with
`load_jsonl()`. Every record requires an explicit Unreal Engine version; callers
must read that version from `config/ue58.yaml` rather than embedding it in code.

## Documentation crawler

T03 downloads raw Epic documentation HTML into `data/raw/docs/`. The crawler:

- supports Gameplay, Programming, C++, Blueprint, Networking, Rendering,
  Animation, AI, and UI topic allowlists;
- checks `robots.txt`, caches successful rules for 24 hours, and stops safely
  when no valid rules can be loaded;
- applies a configurable delay, retry policy, depth limit, and page limit;
- normalizes locale and the version read from `config/ue58.yaml` for URL
  deduplication;
- stores an append-only `manifest.jsonl` with source URL, fetch time, UE version,
  checksum, and local cache path;
- reuses successful cached pages and can continue link discovery after restart.

Inspect the seed plan without network or file writes:

```bash
python scripts/ingest_docs.py --topic cpp --dry-run
```

Run a deliberately small crawl before increasing the configured limits:

```bash
python scripts/ingest_docs.py --topic cpp --max-pages 10 --max-depth 1
```

Raw Epic documentation is intended for local indexing. Respect Epic's site
rules, terms, and copyright; do not redistribute the downloaded corpus.

## Documentation parser

T04 converts the cached HTML into validated `UEDocument` JSONL while preserving
page titles, heading paths, paragraphs, lists, links, callouts, tables, images,
and fenced code blocks. Navigation, footers, sidebars, cookie UI, table-of-content
menus, and code-copy controls are removed.

```bash
python scripts/parse_docs.py
```

The parser writes:

- `data/parsed/docs/documents.jsonl` — clean Markdown plus structured metadata
- `data/parsed/docs/issues.jsonl` — downloaded placeholders or malformed pages
  that could not safely become documents

Parsing is deterministic: rerunning it replaces the output with the same IDs and
ordering for an unchanged crawl manifest.

## Documentation semantic chunker

T05 splits parsed documentation by its H1/H2/H3 hierarchy. A section remains
whole until it exceeds the configured 1500-token maximum; oversized sections are
then split around an 800-token target with 125-token overlap. Every secondary
part repeats its complete heading path.

```bash
python scripts/chunk_docs.py
```

The output is written to `data/chunks/docs/chunks.jsonl`. Each `UEChunk` retains
the page title, section path, source URL, UE version, source type, source file,
and deterministic document/chunk IDs. Fenced code blocks are indivisible; a code
block larger than the hard limit is preserved intact and explicitly marked as an
oversized atomic block.

Token counts currently use the deterministic lightweight counter configured in
`config/docs_chunker.yaml`. The counter is replaceable so a later model-specific
tokenizer can be introduced without coupling chunking to an embedding provider.

## Unreal Engine source scanner

T06 inventories the configured Unreal Engine installation without parsing C++.
Set `engine.root` in `config/ue58.yaml` to the installation directory that
contains `Engine/`. Before scanning, the CLI validates `Engine/Build/Build.version`
against the configured major/minor version.

```bash
python scripts/ingest_engine.py --dry-run
python scripts/ingest_engine.py
```

The scanner covers `Runtime`, `Editor`, `Developer`, and `Programs` beneath
`Engine/Source`, plus source trees beneath `Engine/Plugins`. It includes `.h`,
`.cpp`, `.inl`, and `.Build.cs`, while excluding generated/cache directories,
ThirdParty source, plugin content, SDKs, resources, and shaders outside plugin
`Source` directories.

`data/parsed/engine/files.jsonl` records the engine-relative path, module,
plugin, file type, UE version, byte size, and SHA-256 for every file. File-system
errors are written to `data/parsed/engine/issues.jsonl`.

## Unreal C++ semantic parser

T07 parses inventoried `.h`, `.cpp`, and `.inl` files with Tree-sitter C++ and
a length-preserving Unreal macro scanner. It emits one `UEDocument` per class,
struct, enum, free function, method, constructor, or field instead of treating a
whole source file as one document. `UCLASS`, `USTRUCT`, `UENUM`, `UINTERFACE`,
`UFUNCTION`, `UPROPERTY`, and `UDELEGATE` annotations remain attached to their
symbols; export macros and parser-only UE annotations are masked without moving
source locations.

```bash
python scripts/parse_engine.py
```

The full output is written to `data/parsed/engine/documents.jsonl`, with
operational failures in `data/parsed/engine/parse_issues.jsonl`. Tree-sitter
recovery state is recorded separately at file and symbol level so partially
recoverable Unreal syntax is visible to later pipeline stages. Filtered smoke
runs must use another output path:

```bash
python scripts/parse_engine.py \
  --path-contains CharacterMovementComponent \
  --limit-files 20 \
  --output work/character_movement.jsonl
```

## C++ semantic chunker

T08 converts C++ symbol documents into retrieval-ready `UEChunk` records. Each
chunk starts with a semantic header containing UE version, module, class, symbol,
symbol type, and file. Functions normally remain whole; oversized functions are
split on source lines with overlap and the complete context header repeated in
every part. Classes are emitted as standalone type summaries, enums and structs
remain independent chunks, and fields in the same class/file are merged into a
`property_group` chunk with the original field symbols and document IDs retained
in metadata.

```bash
python scripts/chunk_engine.py
```

The output is written to `data/chunks/engine/chunks.jsonl`. For a smoke run on a
filtered parser output, override both paths:

```bash
python scripts/chunk_engine.py \
  --input work/t07_character_movement.jsonl \
  --output work/t08_character_movement.jsonl
```

## Embedding provider

T09 adds a replaceable `EmbeddingProvider` interface and a Qwen3 implementation
backed by Sentence Transformers. The provider supports document/query encoding,
configurable batches, automatic CPU/GPU selection, L2 normalization, and a
persistent SQLite text/vector cache. Models are loaded lazily, so tests and CLI
help do not download model weights.

```bash
python scripts/embed.py \
  --input data/chunks/engine/chunks.jsonl \
  --output data/embeddings/engine/embeddings.npy
```

The command writes a NumPy float32 matrix plus `.ids.jsonl` and
`.manifest.json` sidecars. The embedding model, device, batch size, cache, and
default output path are configured in `config/embedding.yaml`. The artifact
writer uses a two-pass JSONL stream and memory mapping so the full corpus does
not need to fit in RAM.

## Qdrant vector store

T10 adds the `QdrantVectorStore` integration. It creates the configured
collection, indexes searchable metadata fields, batch-upserts vectors, and
supports exact filters for engine version, source type, module, plugin, class,
and symbol. Upserts use deterministic UUIDs derived from chunk IDs and compare
content hashes, so unchanged repeats are skipped while changed chunks are
updated. A dependency-free in-memory mode is available for tests and smoke
runs.

```bash
python scripts/build_index.py \
  --input data/chunks/engine/chunks.jsonl \
  --vectors data/embeddings/engine/embeddings.npy \
  --ids data/embeddings/engine/embeddings.npy.ids.jsonl
```

Connection and collection settings live in `config/qdrant.yaml`. The command
prints total, added, updated, skipped, and failed counts. Use `--recreate` only
when intentionally rebuilding the collection.

## Keyword and symbol search

T11 adds a SQLite FTS5 lexical index for C++ chunks. It indexes symbols, class
and function names, UE macro names, file paths, and source text. Exact symbol,
class, function, and merged-property aliases are resolved first; lexical FTS
results then fill the remaining slots. The same `RetrievalResult` contract is
returned for symbol, lexical, and hybrid searches, with filters for engine
version, source type, module, plugin, class, and symbol.

```bash
python scripts/build_lexical_index.py \
  --input data/chunks/engine/chunks.jsonl \
  --index data/index/lexical.sqlite3

python scripts/query.py \
  "UCharacterMovementComponent::MaxWalkSpeed" \
  --mode symbol \
  --index data/index/lexical.sqlite3
```

The index builder streams JSONL in bounded transactions and reports added,
updated, skipped, and failed records. Generated SQLite indexes are local
artifacts and are excluded from Git.

## Hybrid retrieval

T12 combines the Qdrant dense results and SQLite lexical results with
Reciprocal Rank Fusion (RRF). The configured dense and sparse Top-K lists are
merged into a final `fusion_top_k`; each result retains both source ranks and
the computed fusion score. An embedding provider can generate the query vector,
or a precomputed vector can be supplied for deterministic/offline calls.

```bash
python scripts/hybrid_query.py \
  "UE5.8 角色移动网络预测" \
  --qdrant-config config/qdrant.yaml \
  --lexical-config config/lexical.yaml
```

RRF and Top-K parameters are configured in `config/retrieval.yaml`. T12 does
not require an LLM API; it only combines already indexed dense and lexical
results.

## Reranker

T13 adds a model-independent `Reranker` interface and a Qwen3-Reranker adapter.
The `RAGQueryPipeline` composes any retriever with any reranker, defaults to
retrieving the configured candidate set and returning `rerank_top_k` results,
and preserves the dense/sparse ranks in result metadata. Batch size, device,
model, and final top-k are configured in `config/reranker.yaml`.

```bash
python scripts/rerank_query.py \
  "UE5.8 角色移动网络预测" \
  --qdrant-config config/qdrant.yaml \
  --lexical-config config/lexical.yaml
```

The Qwen model is loaded only when the command is run; tests inject a local
fake CrossEncoder and do not download weights.

## Retrieval benchmark

T14 adds benchmark cases and ranking metrics: Hit@1/3/5/10, Recall@5/10, MRR,
and nDCG. Cases can be split across category JSONL files; reports contain both
overall macro averages and per-category scores. Optional baseline thresholds
make the CLI return exit code 1 when a change regresses a configured metric.

```bash
python scripts/evaluate.py \
  --input data/benchmark/symbol.jsonl \
  --index data/index/lexical.sqlite3
```

Results are written to `data/benchmark/benchmark_results.json` and
`benchmark_results.md`. Benchmark evaluation does not invent scores when an
index is unavailable; it fails with a clear input error instead.

## Unified query CLI

T15 makes `scripts/query.py` the single user-facing query entry point. It
supports symbol, lexical, hybrid, and rerank modes, optional metadata filters,
precomputed query vectors for offline use, and human-readable or JSON output.
Lexical mode is the default so the command can run without model weights; the
hybrid and rerank modes connect the configured Qdrant, embedding, and reranker
stages when those services/artifacts are available.

```bash
python scripts/query.py \
  "UCharacterMovementComponent::MaxWalkSpeed" \
  --mode symbol \
  --index data/index/lexical.sqlite3
```

Use `--mode hybrid` for dense + lexical RRF, or `--mode rerank` for the final
candidate reranking pipeline.

## MCP server

T16 exposes the retrieval pipeline through the official MCP Python SDK. The
server registers four tools: `ue_search`, `ue_find_symbol`, `ue_search_docs`,
and `ue_search_source`. It runs in lexical-only mode by default; pass
`--enable-dense` to enable Qdrant/embedding hybrid search and
`--enable-rerank` to add the reranker stage.

```bash
python scripts/mcp_server.py
```

The default transport is stdio for Cursor/Claude-style clients. Use
`--transport sse` or `--transport streamable-http` when an HTTP transport is
required. Install the optional server dependency with `pip install "mcp[cli]"`.

## Project RAG scanner

T17 adds a project-scoped scanner for the second-stage RAG corpus. It reads the
project root from `config/project_scanner.yaml` or `--project-root`, scans
`Source/`, `Plugins/`, `Config/`, and `Docs/`, and records project/engine source
provenance, module/plugin names, file type, size, and SHA-256. Generated folders
such as `Binaries`, `Intermediate`, `Saved`, and `DerivedDataCache` are ignored.

```bash
python scripts/ingest_project.py --project-root D:/Work/MyGame --dry-run
python scripts/ingest_project.py --project-root D:/Work/MyGame
```

The inventory is written to `data/parsed/project/files.jsonl`; filesystem issues
are written separately to `issues.jsonl`. The output is ready for the next
incremental project-indexing task and does not modify the project itself.

## Incremental Project RAG

T18 compares two project inventories by relative path and SHA-256. Added and
modified files are the only records sent through caller-provided
parse → chunk → embed → upsert stages; deleted paths are sent to a delete hook,
and unchanged files are skipped. The deterministic change plan is written as
JSONL for auditability and can be generated independently of the indexing
backend.

```bash
python scripts/incremental_project.py \
  --previous data/parsed/project/files.previous.jsonl \
  --current data/parsed/project/files.jsonl \
  --output data/parsed/project/changes.jsonl
```

The orchestration API is `IncrementalProjectIndexer`; it keeps project updates
bounded to changed files and leaves parser/chunker/model choices injectable.

## Blueprint exporter

T19 normalizes connector-neutral Blueprint JSON into the shared `UEDocument`
schema. Each asset produces separate summary, graph, function, variables, and
components documents with deterministic IDs, project scope, asset/package
provenance, graph node/link counts, and structured JSON content.

```bash
python scripts/export_blueprint.py \
  --input data/raw/project/blueprints.jsonl \
  --output data/parsed/project/blueprints.jsonl
```

The exporter methods mirror the future Unreal MCP operations:
`export_blueprint_summary`, `export_blueprint_graph`,
`export_blueprint_function`, `export_blueprint_variables`, and
`export_blueprint_components`.

## Blueprint RAG chunks

T20 converts the exported Blueprint documents into `UEChunk` records. Summary,
function, variable, and component views remain whole; oversized graph views are
split at node lines while repeating asset, graph, package, parent-class, and
symbol context in every part. The resulting chunks use `source_type=blueprint`
and can flow through the existing embedding, lexical, Qdrant, hybrid, reranker,
and MCP stages.

```bash
python scripts/chunk_blueprint.py \
  --input data/parsed/project/blueprints.jsonl \
  --output data/chunks/project/blueprints.jsonl
```

## CLI

Implemented commands expose their full options; later pipeline commands remain
help-only skeletons until their corresponding task is complete.

```bash
python scripts/ingest_docs.py --help
python scripts/parse_docs.py --help
python scripts/chunk_docs.py --help
python scripts/ingest_engine.py --help
python scripts/parse_engine.py --help
python scripts/chunk_engine.py --help
python scripts/embed.py --help
python scripts/build_index.py --help
python scripts/build_lexical_index.py --help
python scripts/hybrid_query.py --help
python scripts/rerank_query.py --help
python scripts/query.py --help
python scripts/evaluate.py --help
python scripts/mcp_server.py --help
python scripts/export_blueprint.py --help
python scripts/chunk_blueprint.py --help
```

## Current Scope

The T01–T20 implementation chain is complete. The next step is to evaluate the
full corpus with a real project, embeddings, Qdrant collection, and benchmark
cases before optimizing further.
