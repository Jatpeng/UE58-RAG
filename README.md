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

The project currently contains the T01 foundation and T02 shared data contracts. It intentionally does not yet implement crawling, parsing, chunking, embedding, indexing, retrieval, reranking, or MCP integration.

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

## CLI

Implemented commands expose their full options; later pipeline commands remain
help-only skeletons until their corresponding task is complete.

```bash
python scripts/ingest_docs.py --help
python scripts/parse_docs.py --help
python scripts/chunk_docs.py --help
python scripts/ingest_engine.py --help
python scripts/build_index.py --help
python scripts/query.py --help
python scripts/evaluate.py --help
```

## Current Scope

The next recommended task is **T07 — Unreal C++ Semantic Parser**.
