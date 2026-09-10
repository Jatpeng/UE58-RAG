"""CLI skeleton for future retrieval evaluation."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without implementing evaluation."""
    return argparse.ArgumentParser(
        description="Evaluate retrieval quality (not implemented in T01)."
    )


def main() -> int:
    """Parse CLI arguments for the T01 placeholder command."""
    build_parser().parse_args()
    print("Evaluation is not implemented in T01.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
