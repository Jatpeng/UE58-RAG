"""Parse inventoried Unreal Engine C++ files into symbol documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.parser import UnrealCPPParser, load_cpp_parser_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse UE C++ into class/function/property UEDocuments."
    )
    parser.add_argument(
        "--ue-config",
        type=Path,
        default=Path("config/ue58.yaml"),
        help="UE configuration containing engine root, version, and data paths.",
    )
    parser.add_argument(
        "--parser-config",
        type=Path,
        default=Path("config/cpp_parser.yaml"),
        help="C++ file types and Unreal macro preprocessing rules.",
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Parse only this module; repeat to select several.",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        dest="plugins",
        help="Parse only this plugin; repeat to select several.",
    )
    parser.add_argument(
        "--path-contains",
        help="Parse only inventory paths containing this text.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        help="Limit selected source files for a smoke run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output JSONL (recommended for filtered smoke runs).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_cpp_parser_config(args.ue_config, args.parser_config)
        if (args.modules or args.plugins or args.path_contains or args.limit_files) and not args.output:
            raise ValueError("filtered runs require --output to protect the full corpus")
        summary = UnrealCPPParser(config).parse_inventory(
            modules=set(args.modules) if args.modules else None,
            plugins=set(args.plugins) if args.plugins else None,
            path_contains=args.path_contains,
            limit_files=args.limit_files,
            output_path=args.output,
        )
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Output: {summary.output_path}")
    print(f"Inventory files: {summary.inventory_files}")
    print(f"Selected files: {summary.selected_files}")
    print(f"Parsed files: {summary.parsed_files}")
    print(f"Files with recoverable AST errors: {summary.syntax_error_files}")
    print(f"Documents: {summary.documents}")
    for symbol_type, count in summary.by_symbol_type.items():
        print(f"{symbol_type.value}: {count}")
    print(f"Issues: {summary.issues} ({summary.issues_path})")
    return 1 if summary.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
