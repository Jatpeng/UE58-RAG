"""Export connector-neutral Blueprint JSON into UEDocument JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from ue_rag.blueprint import JsonBlueprintExporter, load_blueprint_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Blueprint summaries, graphs, functions, variables, and components.")
    parser.add_argument("--input", type=Path, help="Blueprint JSON or JSONL input.")
    parser.add_argument("--output", type=Path, help="UEDocument JSONL output.")
    parser.add_argument("--ue-config", type=Path, default=Path("config/ue58.yaml"))
    parser.add_argument("--blueprint-config", type=Path, default=Path("config/blueprint.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_blueprint_config(args.ue_config, args.blueprint_config)
        input_path = args.input or config.input_path
        output_path = args.output or config.output_path
        summary = JsonBlueprintExporter(config.engine_version).export_jsonl(input_path, output_path)
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: {error}")
        return 2
    print(f"Output: {summary.output_path}")
    print(f"Assets: {summary.assets}")
    print(f"Documents: {summary.documents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
