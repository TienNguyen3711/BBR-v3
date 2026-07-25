#!/usr/bin/env python3
"""CLI: compute per-location/direction calibration constants from raw traces.

Thin wrapper around qbbr.env.calibration.compute_calibration. See
qbbr/scripts/run_eda.py for the fuller Phase-1 pipeline (parse diagnostics +
full state/reward export), which this will fold into once qbbr.env /
qbbr.train / qbbr.eval exist.

Usage:
    .venv/bin/python qbbr/scripts/calibrate.py [--dataset-root PATH] [--cca bbr] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.env.calibration import compute_calibration, save_calibration

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cca", default="bbr", help="CCA subset to calibrate against")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    calib = compute_calibration(args.dataset_root, cca=args.cca)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calib, args.out)
    print(f"saved calibration for {len(calib)} locations -> {args.out}")


if __name__ == "__main__":
    main()
