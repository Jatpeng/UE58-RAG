"""Inventory configured Unreal Engine source files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.crawler import EngineSourceScanner, load_engine_scanner_config


def build_parser() -> argparse.ArgumentParser:
    """Create the engine source scanner command-line interface."""

    parser = argparse.ArgumentParser(
        description="Scan UE Engine/Source and Engine/Plugins into JSONL."
    )
    parser.add_argument(
        "--ue-config",
        type=Path,
        default=Path("config/ue58.yaml"),
        help="UE configuration containing engine root, version, and data paths.",
    )
    parser.add_argument(
        "--scanner-config",
        type=Path,
        default=Path("config/engine_scanner.yaml"),
        help="Scanner suffix, ignore-list, and hash settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print configured roots without scanning or writing files.",
    )
    return parser


def main() -> int:
    """Validate configuration, scan source files, and report inventory counts."""

    args = build_parser().parse_args()
    try:
        config = load_engine_scanner_config(args.ue_config, args.scanner_config)
        scanner = EngineSourceScanner(config)
        if args.dry_run:
            scanner.validate()
            print(f"Engine root: {config.engine_root}")
            print(f"Engine version: {config.engine_version}")
            print(f"Source root: {config.source_root}")
            print(f"Plugins root: {config.plugins_root}")
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
    print(f"Total bytes: {summary.total_bytes}")
    for file_type, count in summary.by_type.items():
        print(f"{file_type.value}: {count}")
    print(f"Issues: {summary.issues} ({summary.issues_path})")
    return 1 if summary.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
