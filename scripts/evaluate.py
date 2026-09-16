"""Evaluate lexical or hybrid retrieval against JSONL benchmark cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.embedding import QwenEmbeddingProvider, load_embedding_config
from ue_rag.eval import BenchmarkEvaluator, load_benchmark_config, load_cases
from ue_rag.index import QdrantVectorStore, load_qdrant_config
from ue_rag.retrieval import HybridRetriever, LexicalIndex, load_lexical_config, load_retrieval_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against benchmark cases.")
    parser.add_argument("--input", type=Path, action="append", required=True, help="Benchmark JSONL; repeat for categories.")
    parser.add_argument("--mode", choices=("lexical", "hybrid"), default="lexical")
    parser.add_argument("--index", type=Path, help="SQLite lexical index path.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--benchmark-config", type=Path, default=Path("config/benchmark.yaml"))
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    parser.add_argument("--qdrant-config", type=Path, default=Path("config/qdrant_server.yaml"))
    parser.add_argument("--embedding-config", type=Path, default=Path("config/embedding.yaml"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("config/retrieval.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lexical = None
    store = None
    provider = None
    try:
        benchmark_config = load_benchmark_config(args.ue_config, args.benchmark_config)
        output_updates = {
            key: value.resolve()
            for key, value in {
                "output_json": args.output_json,
                "output_markdown": args.output_markdown,
            }.items()
            if value is not None
        }
        if output_updates:
            benchmark_config = benchmark_config.model_copy(update=output_updates)
        lexical_config = load_lexical_config(args.ue_config, args.lexical_config)
        index_path = args.index or lexical_config.index_path
        cases = load_cases(args.input)
        lexical = LexicalIndex(index_path)
        search = lexical.search
        if args.mode == "hybrid":
            store = QdrantVectorStore.from_config(load_qdrant_config(args.qdrant_config))
            provider = QwenEmbeddingProvider(load_embedding_config(args.ue_config, args.embedding_config))
            hybrid = HybridRetriever(
                store,
                lexical,
                embedding_provider=provider,
                config=load_retrieval_config(args.ue_config, args.retrieval_config),
            )
            search = lambda query, limit=10, filters=None: hybrid.retrieve(
                query, fusion_top_k=limit, filters=filters
            )
        evaluator = BenchmarkEvaluator(search, benchmark_config)
        report = evaluator.evaluate(cases)
        evaluator.write(report)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2
    finally:
        if provider is not None:
            provider.close()
        if lexical is not None:
            lexical.close()
        if store is not None and getattr(store, "client", None) is not None:
            store.client.close()
    print(f"Mode: {args.mode}")
    print(f"JSON: {benchmark_config.output_json}")
    print(f"Markdown: {benchmark_config.output_markdown}")
    print(f"Cases: {report.cases}")
    print(f"MRR: {report.overall.mrr:.4f}")
    print(f"Recall@10: {report.overall.recall_at_10:.4f}")
    return 0 if report.baseline_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
