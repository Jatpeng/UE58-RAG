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

T01 contains only the project skeleton, configuration files, CLI skeletons, smoke tests, and documentation. It intentionally does not implement crawling, parsing, chunking, embedding, indexing, retrieval, reranking, or MCP integration.

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

## CLI

The T01 commands expose help text only; their RAG behavior belongs to later tasks.

```bash
python scripts/ingest_docs.py --help
python scripts/ingest_engine.py --help
python scripts/build_index.py --help
python scripts/query.py --help
python scripts/evaluate.py --help
```

## Current Scope

The next recommended task is **T02 — Unified Document Schema**, which will define the shared data contracts before ingestion begins.
