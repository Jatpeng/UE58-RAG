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
python scripts/query.py --help
python scripts/evaluate.py --help
```

## Current Scope

The next recommended task is **T11 — Keyword / Symbol Search**.
