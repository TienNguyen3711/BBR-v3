"""Small explicit JSONL interchange for field-run records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def read_jsonl_rows(path: str | Path) -> list[dict[str, object]]:
    """Read records strictly; blank lines are allowed, malformed lines are not."""

    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL item must be an object")
            rows.append(value)
    return rows


def write_jsonl_rows(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Persist a collector output in a portable, append-friendly representation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")
