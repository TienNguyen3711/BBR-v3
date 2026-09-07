"""Report whether a JSONL field dataset is ready for the active study.

Usage:
    python -m qbbr.scripts.assess_collection path/to/field_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qbbr.data.collection.jsonl import read_jsonl_rows
from qbbr.data.collection.schema import validate_collection_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSONL rows produced by the field collector")
    parser.add_argument("--require-ready", action="store_true", help="exit non-zero if the collection gate fails")
    args = parser.parse_args()
    report = validate_collection_rows(read_jsonl_rows(args.records))
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    if args.require_ready and not report.is_minimally_ready:
        raise SystemExit("Collection is not ready: resolve reported coverage gaps before training.")


if __name__ == "__main__":
    main()
