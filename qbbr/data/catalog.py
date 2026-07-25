"""Build a manifest of every run file in the tcp-cc-starlink dataset.

Directory depth is not uniform across categories: sequential-downlink logs
have an extra `iperf3-downlink-sequential-logs/` nesting level that the
other three categories don't. A recursive glob per category root handles
this without hardcoding depth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

_FILENAME_RE = re.compile(
    r"^(?P<cca>[a-z0-9]+)_(?P<location>[A-Za-z]+)__(?P<direction_code>FWD|REV)_run(?P<run>\d+)\.json$"
)

_CATEGORY_DIRS = {
    "downlink-sequential-logs": ("sequential", "downlink"),
    "uplink-sequential-logs": ("sequential", "uplink"),
    "downlink-competitive-logs": ("competitive", "downlink"),
    "uplink-competitive-logs": ("competitive", "uplink"),
}

_DIRECTION_CODE = {"REV": "downlink", "FWD": "uplink"}


@dataclass(frozen=True)
class FileRecord:
    path: Path
    category: str
    direction: str
    location: str
    cca: str
    run: int


def _parse_filename(path: Path) -> tuple[str, str, int] | None:
    m = _FILENAME_RE.match(path.name)
    if m is None:
        return None
    return m.group("cca"), m.group("location"), int(m.group("run"))


def build_catalog(dataset_root: str | Path) -> pd.DataFrame:
    """Scan dataset_root and return one row per run file."""
    dataset_root = Path(dataset_root)
    records: list[FileRecord] = []

    for dir_name, (category, expected_direction) in _CATEGORY_DIRS.items():
        category_root = dataset_root / dir_name
        if not category_root.is_dir():
            continue
        for path in sorted(category_root.glob("**/*.json")):
            parsed = _parse_filename(path)
            if parsed is None:
                continue
            cca, location, run = parsed
            direction_code = _FILENAME_RE.match(path.name).group("direction_code")
            direction = _DIRECTION_CODE[direction_code]
            if direction != expected_direction:
                raise ValueError(
                    f"{path}: filename direction {direction!r} does not match "
                    f"category dir {dir_name!r} (expected {expected_direction!r})"
                )
            records.append(
                FileRecord(
                    path=path,
                    category=category,
                    direction=direction,
                    location=location,
                    cca=cca,
                    run=run,
                )
            )

    if not records:
        raise FileNotFoundError(f"no run files found under {dataset_root}")

    return pd.DataFrame(
        {
            "path": [str(r.path) for r in records],
            "category": [r.category for r in records],
            "direction": [r.direction for r in records],
            "location": [r.location for r in records],
            "cca": [r.cca for r in records],
            "run": [r.run for r in records],
        }
    )


def iter_file_records(catalog: pd.DataFrame) -> Iterator[FileRecord]:
    """Yield a typed FileRecord for every row of a build_catalog() DataFrame."""
    for row in catalog.itertuples(index=False):
        yield FileRecord(
            path=Path(row.path),
            category=row.category,
            direction=row.direction,
            location=row.location,
            cca=row.cca,
            run=row.run,
        )
