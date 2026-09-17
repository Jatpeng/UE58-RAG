---
title: UnrealEngine5.8RAG
emoji: 🛠️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.12"
suggested_hardware: zero-a10g
header: mini
tags:
  - unreal-engine
  - rag
  - semantic-search
  - qdrant
---

# Unreal Engine 5.8 RAG Web Test

This Space provides a web interface for testing semantic retrieval over the
pre-generated UE5.8 RAG embedding artifacts.

The application uses ZeroGPU for on-demand embedding inference and downloads
the required files from
[`Jatpeng/ue58-rag-embeddings`](https://huggingface.co/datasets/Jatpeng/ue58-rag-embeddings)
on demand. It does not upload or expose a Qdrant database.

## Notes

- The first query can take several minutes while the model and chunk offsets are loaded.
- The Space performs dense vector retrieval directly over the NumPy matrix.
- Results are for demonstration and evaluation; production use should rebuild a Qdrant index.
- The source data may be subject to Unreal Engine and third-party licensing restrictions.
