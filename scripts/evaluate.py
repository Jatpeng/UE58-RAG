"""Evaluate lexical retrieval against JSONL benchmark cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.eval import BenchmarkEvaluator, load_benchmark_config, load_cases
from ue_rag.retrieval import LexicalIndex, load_lexical_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against benchmark cases.")
    parser.add_argument("--input", type=Path, action="append", required=True, help="Benchmark JSONL; repeat for categories.")
    parser.add_argument("--index", type=Path, help="SQLite lexical index path.")
    parser.add_argument("--benchmark-config", type=Path, default=Path("config/benchmark.yaml"))
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--lexical-config", type=Path, default=Path("config/lexical.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        benchmark_config = load_benchmark_config(args.ue_config, args.benchmark_config)
        lexical_config = load_lexical_config(args.ue_config, args.lexical_config)
        index_path = args.index or lexical_config.index_path
        cases = load_cases(args.input)
        with LexicalIndex(index_path) as index:
            evaluator = BenchmarkEvaluator(index.search, benchmark_config)
            report = evaluator.evaluate(cases)
            evaluator.write(report)
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"JSON: {benchmark_config.output_json}")
    print(f"Markdown: {benchmark_config.output_markdown}")
    print(f"Cases: {report.cases}")
    print(f"MRR: {report.overall.mrr:.4f}")
    print(f"Recall@10: {report.overall.recall_at_10:.4f}")
    return 0 if report.baseline_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
