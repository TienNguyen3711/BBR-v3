"""Calibrate probe_max_queue_delay_ms from the real Starlink iperf3 logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qbbr.data.catalog import build_catalog
from qbbr.data.loader import load_trace

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_CCA = {"uplink": ("bbr2",), "downlink": ("bbr",)}
BBR_CCAS = ("bbr", "bbr1", "bbr2")  # legacy pooled bucket, reported for comparison only


def run_elevations(intervals: pd.DataFrame, window_s: float) -> np.ndarray:
    """Per-interval RTT above the local (rolling-minimum) propagation floor."""
    frame = intervals[intervals.get("omitted") != True]  # noqa: E712
    rtt = pd.to_numeric(frame.get("rtt_ms"), errors="coerce").dropna()
    if len(rtt) < 5:
        return np.asarray([])
    # Intervals are nominally 1 Hz; use the count directly as the window width.
    window = max(3, int(round(window_s)))
    floor = rtt.rolling(window=window, center=True, min_periods=2).min()
    elevation = (rtt - floor).to_numpy(dtype=float)
    return elevation[np.isfinite(elevation) & (elevation >= 0.0)]


def summarise(catalog: pd.DataFrame, location: str, direction: str, ccas, window_s: float):
    selection = catalog[(catalog["location"] == location)
                        & (catalog["direction"] == direction)
                        & (catalog["category"].str.contains("sequential"))
                        & (catalog["cca"].isin(ccas))]
    pooled, per_run_p90, runs = [], [], 0
    for _, row in selection.iterrows():
        try:
            trace = load_trace(_record(row))
        except Exception:
            continue
        elevation = run_elevations(trace.intervals, window_s)
        if elevation.size == 0:
            continue
        runs += 1
        pooled.append(elevation)
        per_run_p90.append(float(np.percentile(elevation, 90)))
    if not pooled:
        return None
    allv = np.concatenate(pooled)
    return {
        "runs": runs, "samples": int(allv.size),
        "median_ms": float(np.median(allv)),
        "p90_ms": float(np.percentile(allv, 90)),
        "p95_ms": float(np.percentile(allv, 95)),
        "p99_ms": float(np.percentile(allv, 99)),
        "median_of_per_run_p90_ms": float(np.median(per_run_p90)),
    }


def _record(row):
    from qbbr.data.catalog import FileRecord
    return FileRecord(path=Path(row["path"]), category=row["category"], direction=row["direction"],
                      location=row["location"], cca=row["cca"], run=int(row["run"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PACKAGE_ROOT / "data" / "raw")
    parser.add_argument("--locations", nargs="+", default=["London", "Sydney"])
    parser.add_argument("--window-s", type=float, default=5.0,
                        help="Rolling-minimum window: the timescale over which the "
                             "propagation floor is treated as constant.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    catalog = build_catalog(args.dataset_root)
    print(f"self-inflicted RTT elevation above a {args.window_s:.0f}s rolling floor "
          f"(sequential runs only)\n")
    print(f"{'cell':<20}{'cca':<8}{'runs':>5}{'median':>9}{'p90':>8}{'p95':>8}{'p99':>8}")
    results = {}
    for location in args.locations:
        for direction in ("downlink", "uplink"):
            for label, ccas in ((f"ref:{REFERENCE_CCA[direction][0]}", REFERENCE_CCA[direction]),
                                ("bbr*(pooled)", BBR_CCAS), ("cubic", ("cubic",))):
                summary = summarise(catalog, location, direction, ccas, args.window_s)
                if summary is None:
                    print(f"{location + ' ' + direction:<20}{label:<8}{'--':>5}  no usable rtt")
                    continue
                results[f"{location}/{direction}/{label}"] = summary
                print(f"{location + ' ' + direction:<20}{label:<8}{summary['runs']:>5}"
                      f"{summary['median_ms']:>9.2f}{summary['p90_ms']:>8.2f}"
                      f"{summary['p95_ms']:>8.2f}{summary['p99_ms']:>8.2f}")

    print()
    print("Suggested probe_max_queue_delay_ms per cell (p90 of BBR self-inflicted elevation):")
    for location in args.locations:
        for direction in ("downlink", "uplink"):
            key = f"{location}/{direction}/ref:{REFERENCE_CCA[direction][0]}"
            control = f"{location}/{direction}/cubic"
            if key not in results:
                continue
            bbr_p90 = results[key]["p90_ms"]
            note = ""
            if control in results:
                ratio = bbr_p90 / max(results[control]["p90_ms"], 1e-9)
                note = (f"   (cubic p90 {results[control]['p90_ms']:.2f} ms; BBR/CUBIC {ratio:.2f}x"
                        + ("  <- indistinguishable, treat with caution)" if 0.8 <= ratio <= 1.25 else ")"))
            print(f"  {location:<8}{direction:<10}{bbr_p90:>7.2f} ms{note}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"window_s": args.window_s, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
