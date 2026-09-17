---
pretty_name: UE5.8 RAG Embeddings
language:
  - en
  - zh
tags:
  - unreal-engine
  - unreal-engine-5
  - rag
  - embeddings
  - vector-search
  - qdrant
task_categories:
  - text-retrieval
license: other
viewer: false
---

# UE5.8 RAG Embeddings

Embedding artifacts for Unreal Engine 5.8 documentation and C++ source retrieval.
The dataset is intended for RAG pipelines and Qdrant index construction.

## Contents

```text
engine/
├── chunks.jsonl
├── embeddings.npy
├── embeddings.npy.ids.jsonl
└── embeddings.npy.manifest.json

docs/
├── chunks.jsonl
├── embeddings.npy
├── embeddings.npy.ids.jsonl
└── embeddings.npy.manifest.json

sha256.csv
```

The `engine` split contains 1,693,105 chunks and the `docs` split contains 2,055
chunks. All vectors have dimension 1024 and were generated with
`Qwen/Qwen3-Embedding-0.6B`, with normalization enabled and cosine distance.

## Why the Dataset Viewer is disabled

This repository contains both chunk records and an ID sidecar file. The sidecar
file is JSONL with a different schema from `chunks.jsonl`; automatic Dataset
Viewer conversion would therefore attempt to cast incompatible JSON schemas.
The files are raw RAG artifacts rather than a single tabular dataset. Use the
downloaded files directly with the project scripts.

## Restore the Qdrant index

```powershell
python scripts/build_index.py `
  --input data/chunks/engine/chunks.jsonl `
  --vectors data/embeddings/engine/embeddings.npy `
  --ids data/embeddings/engine/embeddings.npy.ids.jsonl
```

The chunk file, vector matrix, and ID mapping must come from the same build.
Do not reorder them independently.

## License and usage

This dataset may contain Unreal Engine documentation or source-code fragments.
Before public distribution, verify the applicable Epic Games license terms and
all third-party permissions. Do not include private source code, credentials, or
API keys. Keep this repository private if redistribution rights are uncertain.
