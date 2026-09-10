"""CLI skeleton for future Unreal Engine documentation ingestion."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without implementing ingestion."""
    return argparse.ArgumentParser(
        description="Ingest Unreal Engine documentation (not implemented in T01)."
    )


def main() -> int:
    """Parse CLI arguments for the T01 placeholder command."""
    build_parser().parse_args()
    print("Documentation ingestion is not implemented in T01.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
