"""JSON Lines persistence helpers for Pydantic data contracts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


def save_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    """Write Pydantic records as UTF-8 JSONL, creating parent folders as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(record.model_dump_json())
            output_file.write("\n")


def load_jsonl(path: str | Path, model_type: type[ModelT]) -> list[ModelT]:
    """Load and validate UTF-8 JSONL records using ``model_type``."""

    input_path = Path(path)
    records: list[ModelT] = []

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
                records.append(model_type.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as error:
                raise ValueError(
                    f"Invalid JSONL record in {input_path} at line {line_number}"
                ) from error

    return records
