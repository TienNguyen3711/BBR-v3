"""Score BBR phase-transition predictions against observed field telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qbbr.data.collection.jsonl import read_jsonl_rows
from qbbr.validation.transitions import transition_fit_by_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--predicted-state-key", default="predicted_bbr_state")
    parser.add_argument("--observed-state-key", default="bbr_state")
    args = parser.parse_args()
    rows = read_jsonl_rows(args.records)
    report = [
        result.as_dict()
        for result in transition_fit_by_scenario(rows, args.predicted_state_key, args.observed_state_key)
    ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
