"""Regenerate `per_location_constants_v7c.json` (gitignored, so this recipe is
the reproducible artefact).

v7c calibration = downlink constants from BBRv1 (`cca=bbr`, the existing
`per_location_constants.json`) + uplink constants refit from BBRv2
(`cca=bbr2`).

Why the uplink is different: `utilization_fraction` floors at 0.05 on four of
the six uplinks, and that is FAITHFUL to BBRv1 -- real BBRv1 uplink on Starlink
stalls to a median throughput of 0.0 Mbps. BBRv2 does not (it sustains ~40-50%
of peak). BBR-v3 inherits BBRv2's inflight_hi / loss machinery, so BBRv2 is the
better stand-in for what a BBR-v3 uplink would deliver. This is a MODELLING
CHOICE, not a bug fix, and it is supervisor-flagged.

    python -m qbbr.scripts.make_v7c_calibration            # writes the file
    python -m qbbr.scripts.make_v7c_calibration --show     # print the diff only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
CALIB_DIR = PKG / "data" / "calibrated"
_UPLINK_KEYS = ("B_max_mbps", "utilization_fraction", "RTT_min_ms", "RTT_max_ms",
                "drawdown_activate_mult", "dwn_retransmit_rate_pps",
                "base_retransmit_rate_pps", "steady_inflight_bdp_frac")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=CALIB_DIR / "per_location_constants.json",
                    help="BBRv1 calibration providing the downlink constants")
    ap.add_argument("--bbr2", type=Path, default=None,
                    help="existing bbr2 calibration; recomputed via calibrate.py when omitted")
    ap.add_argument("--out", type=Path, default=CALIB_DIR / "per_location_constants_v7c.json")
    ap.add_argument("--show", action="store_true", help="print the uplink diff, write nothing")
    args = ap.parse_args()

    if args.bbr2 is None:
        tmp = Path(tempfile.mkdtemp()) / "calib_bbr2.json"
        print(f"running calibrate.py --cca bbr2 (slow: iterative per-location fits) -> {tmp}")
        subprocess.run([sys.executable, "-m", "qbbr.scripts.calibrate", "--cca", "bbr2",
                        "--out", str(tmp)], check=True)
        args.bbr2 = tmp

    base = json.loads(args.base.read_text())
    bbr2 = json.loads(args.bbr2.read_text())
    mixed = json.loads(json.dumps(base))
    for location in mixed:
        uplink, source = mixed[location]["uplink"], bbr2[location]["uplink"]
        uplink["_orig_bbr"] = {k: uplink.get(k) for k in _UPLINK_KEYS}
        for key in _UPLINK_KEYS:
            if key in source:
                uplink[key] = source[key]
        uplink["_uplink_calibration_cca"] = "bbr2"
        o, n = uplink["_orig_bbr"], uplink
        print(f"  {location:9} uplink: util {o['utilization_fraction']:.3f} -> "
              f"{n['utilization_fraction']:.3f}   B_max {o['B_max_mbps']:.1f} -> {n['B_max_mbps']:.1f} Mbps")

    if args.show:
        print("\n--show: nothing written")
        return
    args.out.write_text(json.dumps(mixed, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
