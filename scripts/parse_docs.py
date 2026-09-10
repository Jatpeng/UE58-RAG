"""Parse cached Epic documentation HTML into clean UEDocument JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.parser import DocumentationParser, load_docs_parser_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse cached UE documentation HTML into structured JSONL."
    )
    parser.add_argument(
        "--ue-config",
        type=Path,
        default=Path("config/ue58.yaml"),
        help="UE configuration containing version and data paths.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override the default raw documentation manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override data/parsed/docs/documents.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_docs_parser_config(args.ue_config)
        summary = DocumentationParser().parse_manifest(
            config,
            manifest_path=args.manifest,
            output_path=args.output,
        )
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Output: {summary.output_path}")
    print(f"Documents: {summary.documents}")
    print(f"Headings: {summary.headings}")
    print(f"Code blocks: {summary.code_blocks}")
    print(f"Skipped: {summary.skipped}")
    print(f"Issues: {summary.issues_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
