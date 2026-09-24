"""Calibrate probe_max_queue_delay_ms from the real Starlink iperf3 logs.

WHY THIS EXISTS. `probe_max_queue_delay_ms` bounds the queueing delay a ProbeBW
pulse may add in the fluid simulator. It was introduced in v7b and set to 15 ms
uniformly across every path -- a modelling assumption, never measured. It turns
out to dominate every RTT claim the simulator makes: on London downlink, sweeping
it from 15 ms to 6 ms leaves the throughput of gain 1.25 unchanged (+4.01% ->
+4.04%) while its RTT p90 cost collapses from +9.44 ms to +1.07 ms. That is the
difference between "no action can gain throughput inside the 5 ms RTT budget"
and "the highest gain passes comfortably" -- so the value cannot be chosen by
what makes the agent look good. It has to come from the data.

WHAT IS MEASURED. The parameter is a bound on SELF-INFLICTED queueing delay: the
standing delay a flow adds to its own path by probing. In an iperf3 log the RTT
is the sender's smoothed RTT, which mixes that self-inflicted component with the
path's own drift -- and on Starlink the exogenous component is enormous (London
RTT_min 258 ms vs RTT_max 347 ms). Subtracting a per-RUN minimum would therefore
attribute ~90 ms of orbital and handover variation to the congestion controller.

So the estimator uses a ROLLING minimum: within a window of a few seconds the
propagation floor is effectively constant, so

    elevation(t) = rtt(t) - min(rtt over a window centred on t)

isolates delay the flow imposed on itself over the timescale a probe acts on,
while slow exogenous drift moves the floor with it and cancels. A high
percentile of that elevation over BBR runs is the observed ceiling on
probe-induced queueing -- which is what the parameter bounds.

CONTROL. The same statistic is computed for CUBIC, which is loss-based and fills
the buffer rather than probing in BBR's bounded way. If BBR's elevation were
indistinguishable from CUBIC's, the estimator would be measuring generic
queueing rather than anything probe-specific, and the calibration would not be
credible. The comparison is reported so that can be judged rather than assumed.

Sequential (isolated, single-CCA) runs only: competitive runs add cross-traffic
queueing that is not the flow's own probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qbbr.data.catalog import build_catalog
from qbbr.data.loader import load_trace

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
# CORRECTION (2026-09-12): this used to pool ("bbr", "bbr1", "bbr2") into one
# "bbr*" bucket. That was wrong -- the three variants behave completely
# differently on these paths, and the pooled number was dominated by bbr1's
# uplink STALL pathology (uplink self-inflicted p90: bbr1 44.6-85.5 ms, bbr
# 14.3-45.0 ms, bbr2 8.6-20.4 ms). Every constant derived from the pooled
# bucket was therefore contaminated on the uplink.
#
# The reference is now chosen PER DIRECTION, on a stated criterion: use the
# variant that does not exhibit a pathology BBR-v3 is known to have fixed.
#   uplink   -> bbr2. Real BBRv1/bbr1 deliver ZERO throughput in 47-57% of
#               1-second samples on London/Mumbai/Ohio/SaoPaulo/Tokyo. So do
#               CUBIC (16-62%), Vegas and Hybla -- it is a LINK property, not a
#               BBR bug -- but bbr2 escapes it (2-7%), and BBR-v3 inherits
#               BBRv2's inflight_hi/loss machinery.
#   downlink -> bbr. BBRv1 does not stall there (0.0-1.5% zero samples) and
#               reaches the higher peak; bbr2's downlink p99 is up to 2.8x
#               lower (London 94 vs 264 Mbps), which would understate the link.
# The downlink half of this choice is a JUDGEMENT CALL, not forced by a
# pathology, and is flagged as such.
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
