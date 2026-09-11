"""Inventory a user's Unreal project for Project RAG."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.crawler import ProjectSourceScanner, load_project_scanner_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan project Source, Plugins, Config, and Docs.")
    parser.add_argument("--project-root", type=Path, help="Project root; overrides project_scanner.yaml.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--scanner-config", type=Path, default=Path("config/project_scanner.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate roots without scanning.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_project_scanner_config(args.ue_config, args.scanner_config)
        if args.project_root:
            config.project_root = args.project_root
        scanner = ProjectSourceScanner(config)
        if args.dry_run:
            root = scanner.validate()
            print(f"Project root: {root}")
            print(f"Output: {config.output_path}")
            return 0
        summary = scanner.scan()
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Output: {summary.output_path}")
    print(f"Files: {summary.files}")
    print(f"Modules: {summary.modules}")
    print(f"Plugins: {summary.plugins}")
    for file_type, count in summary.by_type.items():
        print(f"{file_type.value}: {count}")
    print(f"Issues: {summary.issues} ({summary.issues_path})")
    return 1 if summary.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
