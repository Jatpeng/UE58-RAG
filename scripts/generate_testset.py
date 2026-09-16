"""Generate a synthetic UE5.8 RAG evaluation testset with RAGAS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

from ue_rag.eval import (
    generation_request_size,
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
    parser.add_argument(
        "--rebuild-knowledge-graph",
        action="store_true",
        help="Ignore the cached RAGAS knowledge graph and rebuild all transforms.",
    )
    return parser


def _is_local_base_url(value: str | None) -> bool:
    if not value:
        return False
    hostname = urlparse(value).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _normalize_persona_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _install_tolerant_persona_lookup() -> None:
    """Allow RAGAS mappings that append a role description to a persona name."""

    from ragas.testset.persona import PersonaList

    if getattr(PersonaList, "_ue_rag_tolerant_lookup", False):
        return
    original = PersonaList.__getitem__

    def tolerant_getitem(persona_list, key: str):
        try:
            return original(persona_list, key)
        except KeyError:
            requested = _normalize_persona_name(key)
            matches = []
            for persona in persona_list.personas:
                candidate = _normalize_persona_name(persona.name)
                if candidate and (candidate in requested or requested in candidate):
                    matches.append(persona)
            if len(matches) == 1:
                return matches[0]
            raise

    PersonaList.__getitem__ = tolerant_getitem
    PersonaList._ue_rag_tolerant_lookup = True


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
        if not api_key and _is_local_base_url(config.llm.base_url):
            api_key = "local"
        if not api_key:
            raise RuntimeError(
                f"{config.llm.api_key_env} is not set in this PowerShell session; "
                "set the remote API key or use an OpenAI-compatible localhost base_url."
            )

        try:
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.run_config import RunConfig
            from ragas.testset import TestsetGenerator
            from ragas.testset.graph import KnowledgeGraph, Node, NodeType
            from ragas.testset.persona import Persona
            from ragas.testset.transforms import apply_transforms, default_transforms
            import rapidfuzz  # noqa: F401 - required by RAGAS relationship builders
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
        chat_model = ChatOpenAI(**chat_args)
        try:
            chat_model.invoke("Reply with exactly: OK")
        except Exception as error:
            raise RuntimeError(
                f"LLM preflight failed for {config.llm.model} at "
                f"{config.llm.base_url or 'the default endpoint'}: {error}"
            ) from error
        print(f"LLM preflight: OK ({config.llm.model})")
        generator_llm = LangchainLLMWrapper(chat_model)
        generator_embeddings = HuggingFaceEmbeddings(
            model=config.embedding.model,
            device=config.embedding.device,
            normalize_embeddings=True,
            batch_size=config.embedding.batch_size,
        )
        run_config = RunConfig(
            timeout=config.llm.timeout_seconds,
            max_retries=config.llm.max_retries,
            max_workers=config.llm.max_workers,
            seed=config.seed,
        )
        documents = [Document(page_content=source.page_content, metadata=source.metadata) for source in sources]
        _install_tolerant_persona_lookup()
        persona_list = [
            Persona(name=persona.name, role_description=persona.role_description)
            for persona in config.personas
        ]
        if config.knowledge_graph_output.exists() and not args.rebuild_knowledge_graph:
            knowledge_graph = KnowledgeGraph.load(config.knowledge_graph_output)
            print(f"Knowledge graph cache: {config.knowledge_graph_output}")
        else:
            knowledge_graph = KnowledgeGraph(
                nodes=[
                    Node(
                        type=NodeType.DOCUMENT,
                        properties={
                            "page_content": document.page_content,
                            "document_metadata": document.metadata,
                        },
                    )
                    for document in documents
                ]
            )
            transforms = default_transforms(
                documents=documents,
                llm=generator_llm,
                embedding_model=generator_embeddings,
            )
            apply_transforms(knowledge_graph, transforms, run_config=run_config)
            config.knowledge_graph_output.parent.mkdir(parents=True, exist_ok=True)
            knowledge_graph.save(config.knowledge_graph_output)
            print(f"Knowledge graph saved: {config.knowledge_graph_output}")
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=knowledge_graph,
            persona_list=persona_list,
            llm_context=config.llm.context,
        )
        requested_size = generation_request_size(config)
        print(f"RAGAS candidate request: {requested_size}")
        try:
            dataset = generator.generate(
                testset_size=requested_size,
                run_config=run_config,
                raise_exceptions=True,
            )
        except Exception as error:
            raise RuntimeError(f"RAGAS generation failed: {type(error).__name__}: {error}") from error
        generated, benchmark = write_generated_testset(
            dataset.to_list()[: config.testset_size],
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
