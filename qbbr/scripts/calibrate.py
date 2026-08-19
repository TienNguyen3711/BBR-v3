from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.env.calibration import (
    calibrate_drawdown_activate_mult,
    calibrate_dwn_retransmit_rate,
    compute_calibration,
    save_calibration,
)

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cca", default="bbr", help="CCA subset to calibrate against")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--skip-drawdown-activate-mult", action="store_true",
        help="skip the (slower, ~5-7min/location) RTT-targeted iterative fit -- for quick iteration only",
    )
    parser.add_argument(
        "--skip-dwn-retransmit-rate", action="store_true",
        help="skip the (~1min/location) retransmit-rate calibration -- for quick iteration only",
    )
    args = parser.parse_args()

    calib = compute_calibration(args.dataset_root, cca=args.cca)

    # Order matters: drawdown_activate_mult changes i_dwn's dynamics, which
    # dwn_retransmit_rate_pps's calibration depends on (its slope is i_dwn's
    # average) -- so it must be fit and locked in FIRST, or c_loc drifts once
    # this lands. There is no dependency in the other direction: retransmit
    # generation is a pure output with no feedback into v_bytes/i_dwn/i_crs.
    if not args.skip_drawdown_activate_mult:
        print("calibrating drawdown_activate_mult (iterative RTT-targeted fit per location)...")
        activate_mults = calibrate_drawdown_activate_mult(args.dataset_root, calib, cca=args.cca)
        for location, directions in activate_mults.items():
            for direction, values in directions.items():
                calib[location][direction].update(values)

    if not args.skip_dwn_retransmit_rate:
        print("calibrating dwn_retransmit_rate_pps (runs the reference simulator per location)...")
        dwn_rates = calibrate_dwn_retransmit_rate(args.dataset_root, calib, cca=args.cca)
        for location, directions in dwn_rates.items():
            for direction, values in directions.items():
                calib[location][direction].update(values)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calib, args.out)
    print(f"saved calibration for {len(calib)} locations -> {args.out}")


if __name__ == "__main__":
    main()
