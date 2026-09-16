"""Generate a synthetic UE5.8 RAG evaluation testset with RAGAS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ue_rag.eval import (
    load_testset_config,
    sample_sources,
    write_generated_testset,
    write_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a bounded synthetic RAG testset with RAGAS.")
    parser.add_argument("--config", type=Path, default=Path("config/testset.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--documents", type=int)
    parser.add_argument("--testset-size", type=int)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--device")
    parser.add_argument("--prepare-only", action="store_true", help="Sample traceable sources without calling an LLM.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_testset_config(args.config)
        overrides: dict[str, object] = {}
        for name in ("input", "documents", "testset_size"):
            value = getattr(args, name)
            if value is not None:
                overrides[name] = value.resolve() if isinstance(value, Path) else value
        if overrides:
            config = config.model_copy(update=overrides)
        llm_updates = {
            key: value
            for key, value in {
                "model": args.model,
                "base_url": args.base_url,
                "api_key_env": args.api_key_env,
            }.items()
            if value is not None
        }
        if llm_updates:
            config = config.model_copy(update={"llm": config.llm.model_copy(update=llm_updates)})
        if args.device:
            config = config.model_copy(update={"embedding": config.embedding.model_copy(update={"device": args.device})})
        config = type(config).model_validate(config.model_dump())

        sources = sample_sources(config)
        write_sources(sources, config.sources_output)
        print(f"Prepared sources: {len(sources)}")
        print(f"Sources: {config.sources_output}")
        if args.prepare_only:
            return 0

        api_key = os.getenv(config.llm.api_key_env)
        if not api_key and config.llm.base_url:
            api_key = "local"
        if not api_key:
            raise RuntimeError(
                f"{config.llm.api_key_env} is not set; a generation LLM is required. "
                "Set the key or configure an OpenAI-compatible local base_url."
            )

        try:
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.run_config import RunConfig
            from ragas.testset import TestsetGenerator
        except ImportError as error:
            raise RuntimeError('install test generation dependencies with: pip install -e ".[eval]"') from error

        chat_args: dict[str, object] = {
            "model": config.llm.model,
            "api_key": api_key,
            "temperature": config.llm.temperature,
            "timeout": config.llm.timeout_seconds,
            "max_retries": config.llm.max_retries,
        }
        if config.llm.base_url:
            chat_args["base_url"] = config.llm.base_url
        generator_llm = LangchainLLMWrapper(ChatOpenAI(**chat_args))
        generator_embeddings = HuggingFaceEmbeddings(
            model=config.embedding.model,
            device=config.embedding.device,
            normalize_embeddings=True,
            batch_size=config.embedding.batch_size,
        )
        documents = [Document(page_content=source.page_content, metadata=source.metadata) for source in sources]
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            llm_context=config.llm.context,
        )
        run_config = RunConfig(
            timeout=config.llm.timeout_seconds,
            max_retries=config.llm.max_retries,
            max_workers=config.llm.max_workers,
            seed=config.seed,
        )
        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=config.testset_size,
            run_config=run_config,
            raise_exceptions=True,
        )
        generated, benchmark = write_generated_testset(
            dataset.to_list(),
            testset_output=config.testset_output,
            benchmark_output=config.benchmark_output,
        )
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Generated: {generated}")
    print(f"Benchmark cases: {benchmark}")
    print(f"RAGAS testset: {config.testset_output}")
    print(f"Retrieval benchmark: {config.benchmark_output}")
    return 0 if generated == benchmark == config.testset_size else 1


if __name__ == "__main__":
    raise SystemExit(main())
