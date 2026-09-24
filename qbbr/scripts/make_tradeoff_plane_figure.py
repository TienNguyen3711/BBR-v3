"""Where does the QRL policy sit among real congestion controls on Starlink?

THE ONLY VALID WAY TO PUT SIMULATED AND MEASURED RESULTS ON ONE AXIS.

Absolute throughput and RTT cannot be compared across the simulator boundary:
this simulator reproduces only ~21% of the self-inflicted RTT elevation real BBR
shows, and its absolute throughput sits ~28% off. Plotting a simulated arm
beside a measured one invites a reading that is mostly simulator error.

What both sides CAN express in the same units is a DELTA AGAINST BBR measured
on the same link:

  real CCAs   mean(cca) vs mean(bbr), over the same 10 sequential runs per city
  QRL policy  agent vs stock in the simulator, on capacity forcing replayed
              from those same bbr runs, paired trace by trace

Both are ratios against a BBR baseline on the same path, so the parts of the
model that are systematically off -- absolute rate level, absolute RTT floor --
divide out. What does NOT divide out is any error in how the model responds to
a CHANGE in pacing gain, so this positions the policy among the alternatives
rather than measuring it against them. The two evidence classes are drawn
differently for that reason.

Reading the plane: right is more throughput, down is lower RTT, so the
bottom-right quadrant is unambiguously better than BBR and the top-left is
unambiguously worse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qbbr.data.catalog import build_catalog, FileRecord
from qbbr.data.loader import load_trace

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CITIES = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
REAL_CCAS = ["leocc", "pcc", "bbr2", "hybla", "cubic", "vegas", "ccp"]
BASELINE = "bbr"


def _cca_stats(catalog, location, cca):
    rows = catalog[(catalog["location"] == location) & (catalog["direction"] == "downlink")
                   & (catalog["category"].str.contains("sequential")) & (catalog["cca"] == cca)]
    throughput, rtt = [], []
    for _, row in rows.iterrows():
        try:
            trace = load_trace(FileRecord(path=Path(row["path"]), category=row["category"],
                                          direction=row["direction"], location=row["location"],
                                          cca=row["cca"], run=int(row["run"])))
        except Exception:
            continue
        frame = trace.intervals[trace.intervals.get("omitted") != True]  # noqa: E712
        bps = pd.to_numeric(frame["bits_per_second"], errors="coerce").dropna()
        rtt_ms = pd.to_numeric(frame["rtt_ms"], errors="coerce").dropna()
        if len(bps):
            throughput.append(bps.mean() / 1e6)
        if len(rtt_ms):
            rtt.append(float(np.percentile(rtt_ms, 90)))
    return (float(np.mean(throughput)) if throughput else np.nan,
            float(np.mean(rtt)) if rtt else np.nan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=PACKAGE_ROOT.parent / "figures" / "tradeoff_plane_v14.png")
    args = parser.parse_args()

    catalog = build_catalog(PACKAGE_ROOT / "data" / "raw")
    measured = {}
    for cca in REAL_CCAS:
        dthr, drtt = [], []
        for city in CITIES:
            base_t, base_r = _cca_stats(catalog, city, BASELINE)
            t, r = _cca_stats(catalog, city, cca)
            if np.isfinite(t) and np.isfinite(base_t) and base_t > 0:
                dthr.append(100.0 * (t / base_t - 1.0))
            if np.isfinite(r) and np.isfinite(base_r):
                drtt.append(r - base_r)
        measured[cca] = (np.mean(dthr), np.mean(drtt), np.std(dthr), np.std(drtt))

    rows = []
    for path in args.replays:
        rows += json.loads(Path(path).read_text())["rows"]
    simulated = {}
    for core in ("quantum", "classical"):
        group = [r for r in rows if r["core"] == core]
        if not group:
            continue
        dthr = [r["throughput_delta_vs_stock_pct"] for r in group]
        drtt = [r["rtt_p90_delta_vs_stock_ms"] for r in group]
        simulated[core] = (np.median(dthr), np.median(drtt), np.std(dthr), np.std(drtt))

    fig, axis = plt.subplots(figsize=(11.5, 7.6))

    axis.axhline(0, color="#9AA3AE", linewidth=1, zorder=1)
    axis.axvline(0, color="#9AA3AE", linewidth=1, zorder=1)
    axis.text(0.985, 0.03, "better than BBR\n(more throughput, lower RTT)", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=9, color="#2F6B4F", style="italic")
    axis.text(0.015, 0.97, "worse than BBR", transform=axis.transAxes,
              ha="left", va="top", fontsize=9, color="#9B3B2F", style="italic")

    for cca, (x, y, ex, ey) in measured.items():
        axis.errorbar(x, y, xerr=ex / np.sqrt(len(CITIES)), yerr=ey / np.sqrt(len(CITIES)),
                      fmt="o", ms=9, color="#5A6472", ecolor="#B6BDC7",
                      elinewidth=1.2, capsize=3, zorder=3)
        axis.annotate(cca, (x, y), textcoords="offset points", xytext=(11, -3),
                      fontsize=10, color="#3A424E")

    style = {"quantum": ("#2E6DA8", "QA2C (quantum)"), "classical": ("#D07A1E", "A2C (classical)")}
    for core, (x, y, ex, ey) in simulated.items():
        color, label = style[core]
        axis.errorbar(x, y, xerr=ex / np.sqrt(len(CITIES)), yerr=ey / np.sqrt(len(CITIES)),
                      fmt="*", ms=22, color=color, ecolor=color, alpha=0.9,
                      elinewidth=1.4, capsize=3, zorder=4, label=label)

    axis.set_xlabel("Δ mean throughput vs BBR  (%)", fontsize=11)
    axis.set_ylabel("Δ RTT p90 vs BBR  (ms)", fontsize=11)
    axis.set_title("Starlink downlink: where the learned policy sits among real congestion controls",
                   fontsize=13)
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=9, color="#5A6472",
                      label="real CCA — measured on Starlink (vs bbr)")]
    handles += [Line2D([], [], marker="*", ls="", ms=16, color=style[c][0], label=style[c][1] + " — simulated (vs stock)")
                for c in simulated]
    axis.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)

    fig.text(0.5, 0.012,
             "Both axes are deltas against a BBR baseline on the same path, so the simulator's absolute "
             "offsets divide out — but its response to a pacing-gain change does not. "
             "Bars are the standard error across the six cities.",
             ha="center", fontsize=8.5, style="italic", color="#6B7280")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
