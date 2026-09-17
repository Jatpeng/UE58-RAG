"""Small Gradio demo for testing the UE5.8 RAG embedding artifacts."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download


DATASET_REPO = os.getenv("RAG_DATASET_REPO", "Jatpeng/ue58-rag-embeddings")
DATA_ROOT = Path(os.getenv("RAG_DATA_DIR", "/tmp/ue58-rag-data"))
MODEL_ID = os.getenv("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")


class ArtifactIndex:
    def __init__(self, split: str) -> None:
        self.split = split
        self.root = DATA_ROOT / split
        self.vectors: np.ndarray | None = None
        self.chunk_path: Path | None = None
        self.offsets: np.ndarray | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self.vectors is not None:
            return
        with self._lock:
            if self.vectors is not None:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            names = (
                "chunks.jsonl",
                "embeddings.npy",
                "embeddings.npy.ids.jsonl",
                "embeddings.npy.manifest.json",
            )
            local: dict[str, Path] = {}
            for name in names:
                local[name] = Path(
                    hf_hub_download(
                        repo_id=DATASET_REPO,
                        repo_type="dataset",
                        filename=f"{self.split}/{name}",
                        local_dir=str(DATA_ROOT),
                    )
                )
            self.chunk_path = local["chunks.jsonl"]
            self.vectors = np.load(local["embeddings.npy"], mmap_mode="r")
            if self.vectors.ndim != 2:
                raise ValueError(f"{self.split} embeddings must be a 2D matrix")
            manifest = json.loads(local["embeddings.npy.manifest.json"].read_text(encoding="utf-8"))
            expected = int(manifest["records"])
            if expected != self.vectors.shape[0]:
                raise ValueError("embedding manifest count does not match the vector matrix")

    def build_offsets(self, progress: Any = None) -> None:
        self.load()
        if self.offsets is not None:
            return
        with self._lock:
            if self.offsets is not None:
                return
            assert self.chunk_path is not None
            cached = self.chunk_path.with_suffix(".offsets.npy")
            if cached.is_file():
                self.offsets = np.load(cached, mmap_mode="r")
                return
            offsets: list[int] = []
            with self.chunk_path.open("rb") as stream:
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(position)
            self.offsets = np.asarray(offsets, dtype=np.int64)
            np.save(cached, self.offsets)
            if progress:
                progress(1.0, desc=f"Loaded {len(offsets):,} chunk offsets")

    def get_chunks(self, rows: np.ndarray) -> list[dict[str, Any]]:
        assert self.chunk_path is not None and self.offsets is not None
        result: list[dict[str, Any]] = []
        with self.chunk_path.open("rb") as stream:
            for row in rows.tolist():
                stream.seek(int(self.offsets[row]))
                result.append(json.loads(stream.readline().decode("utf-8")))
        return result


_indexes = {"engine": ArtifactIndex("engine"), "docs": ArtifactIndex("docs")}
_model: Any | None = None
_model_lock = threading.Lock()


def get_model() -> Any:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(MODEL_ID, device=os.getenv("RAG_DEVICE", "cpu"))
                _model.max_seq_length = 2048
    return _model


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    method = getattr(model, "encode_query", None) or model.encode
    try:
        vector = method(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    except TypeError:
        vector = method([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(vector, dtype=np.float32).reshape(-1)


def search(question: str, split: str, top_k: int, progress: Any = gr.Progress()) -> tuple[str, list[dict[str, Any]]]:
    question = question.strip()
    if not question:
        return "Please enter a question.", []
    index = _indexes[split]
    progress(0.05, desc="Loading embedding artifacts")
    index.load()
    progress(0.15, desc="Loading embedding model")
    query = embed_query(question)
    assert index.vectors is not None
    if query.size != index.vectors.shape[1]:
        raise ValueError(f"Query dimension {query.size} does not match index dimension {index.vectors.shape[1]}")
    progress(0.35, desc="Searching vectors")
    scores = np.asarray(index.vectors @ query).reshape(-1)
    count = min(max(int(top_k), 1), 20, scores.size)
    rows = np.argpartition(scores, -count)[-count:]
    rows = rows[np.argsort(scores[rows])[::-1]]
    progress(0.75, desc="Loading matching chunks")
    index.build_offsets(progress)
    chunks = index.get_chunks(rows)
    results = []
    for rank, (row, chunk) in enumerate(zip(rows.tolist(), chunks, strict=True), start=1):
        results.append(
            {
                "rank": rank,
                "score": round(float(scores[row]), 6),
                "row": row,
                "id": chunk.get("id"),
                "title": chunk.get("title"),
                "source_type": chunk.get("source_type"),
                "module": chunk.get("module"),
                "file_path": chunk.get("file_path"),
                "content": chunk.get("content", ""),
            }
        )
    return f"Found {len(results)} results from `{split}`.", results


with gr.Blocks(title="UE5.8 RAG Web Test") as demo:
    gr.Markdown(
        "# Unreal Engine 5.8 RAG\n"
        "Test semantic retrieval against the pre-generated Qwen3 embeddings. "
        "The first query may take longer while the model and index are loaded."
    )
    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="例如：How does Character Movement Component handle network prediction?",
            scale=5,
        )
        split = gr.Dropdown(["engine", "docs"], value="engine", label="Corpus", scale=1)
        top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K", scale=1)
    run = gr.Button("Search", variant="primary")
    status = gr.Markdown()
    results = gr.JSON(label="Retrieved chunks", open=False)
    run.click(search, inputs=[question, split, top_k], outputs=[status, results])
    question.submit(search, inputs=[question, split, top_k], outputs=[status, results])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
