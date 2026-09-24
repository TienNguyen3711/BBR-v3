"""Paired agent-vs-stock delta: synthetic training condition vs trace replay.

The absolute-throughput grid cannot show this effect -- a 1% difference beside
a 244 Mbps baseline is invisible, and zooming that axis until it looks large
would misrepresent the magnitude. So plot the quantity actually of interest
instead: the PAIRED delta against a stock run under identical conditions.

Two conditions, same policies, same seeds, one axis:

  synthetic   the periodic-handover model the agent was trained in
  replay      bottleneck capacity forcing taken from the real qbbr/data/raw
              runs, with the stock arm validated against each trace's own
              realised throughput

Points are drawn per seed rather than boxed: two of the five seeds sit at
exactly 0.00% (they collapse to constant stock), and that bimodality is part
of the result -- a box would hide it.

Simulator-proxy only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CORES = [("quantum", "QA2C", "#e45756"), ("classical", "A2C", "#8172b3")]


def synthetic_deltas(report: Path) -> dict:
    """(city, direction, core) -> {seed: delta%} from the 6-city screen."""
    out = {}
    for r in json.loads(report.read_text())["records"]:
        key = (r["location"], r["direction"], r["core"])
        out.setdefault(key, {})[r["seed"]] = r["evaluation"]["throughput_delta_vs_stock_pct"]
    return out


def replay_deltas(files: list[Path]) -> dict:
    """(city, direction, core) -> {seed: median delta% across replay traces}.

    Rows carry their own location/direction/core, so any mix of replay JSONs
    can be pooled -- one per (cell, core), or the whole replay6 directory.
    """
    grouped: dict = {}
    for f in files:
        if not f.exists():
            print(f"  missing {f}")
            continue
        for r in json.loads(f.read_text()).get("rows", []):
            key = (r["location"], r["direction"], r.get("core", "quantum"))
            grouped.setdefault(key, {}).setdefault(r["seed"], []).append(
                r["throughput_delta_vs_stock_pct"])
    return {k: {s: float(np.median(v)) for s, v in seeds.items()} for k, seeds in grouped.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=Path, default=ROOT / "outputs" / "sixcity_screen_v7c_report.json")
    ap.add_argument("--replay-dir", type=Path, default=ROOT / "outputs" / "replay6")
    ap.add_argument("--replay-files", nargs="+", type=Path, default=None,
                    help="explicit replay JSONs instead of --replay-dir")
    ap.add_argument("--cities", nargs="+", default=None)
    ap.add_argument("--caption", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "delta_synthetic_vs_replay.png")
    args = ap.parse_args()
    global CITIES
    if args.cities:
        CITIES = args.cities

    files = args.replay_files or [args.replay_dir / f"replay_{c}_{d}_{k}.json"
                                  for c in CITIES for d in ("downlink", "uplink")
                                  for k, *_ in CORES]
    syn, rep = synthetic_deltas(args.synthetic), replay_deltas(files)
    # Group offsets scale with the city count so a 2-city figure does not leave
    # its clusters marooned at the panel edges.
    span = 0.42 if len(CITIES) <= 2 else 0.34
    # Width floor is set by the caption, not the data: the footnote lines are
    # long and matplotlib will not wrap a suptitle.
    fig, axes = plt.subplots(2, 1, figsize=(max(14.0, 2.3 * len(CITIES)), 9), sharex=True)

    for ax, direction in zip(axes, ("downlink", "uplink")):
        for i, city in enumerate(CITIES):
            for c, (core, core_label, colour) in enumerate(CORES):
                for k, (cond, source, marker, alpha) in enumerate(
                        (("synthetic", syn, "o", 1.0), ("replay", rep, "^", 0.55))):
                    vals = source.get((city, direction, core), {})
                    if not vals:
                        continue
                    x = i + (c - 0.5) * span + (k - 0.5) * span * 0.42
                    ys = [vals[s] for s in sorted(vals)]
                    ax.scatter([x] * len(ys), ys, s=46, marker=marker, color=colour,
                               alpha=alpha, edgecolor="black", linewidth=0.5, zorder=3,
                               label=f"{core_label} ({cond})" if i == 0 else None)
                    ax.plot([x - span * 0.16, x + span * 0.16], [np.median(ys)] * 2, color="black",
                            lw=1.6, zorder=4)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_ylabel("throughput vs stock (%)")
        ax.set_title(f"{direction}", loc="left", fontsize=11)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xticks(range(len(CITIES)))
        ax.set_xticklabels(CITIES)
        ax.set_xlim(-0.62, len(CITIES) - 0.38)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.998))
    fig.suptitle(
        "Paired agent-vs-stock throughput delta: synthetic training condition (circles) vs "
        "trace replay on real capacity forcing (triangles)\n"
        "One point per training seed; black bar = median. Checkpoints: 6-city screen (v7c, "
        "stock_init_bias=0.0) -- its two weak seeds do NOT sit at stock, they converge to a "
        "losing policy, which real forcing then amplifies.\n"
        "The advantage learned in the synthetic environment does not survive real forcing; under "
        "v7c weights the replayed agents are net harmful. The reported v7-final config "
        "(stock_init_bias=0.3) freezes those seeds at stock and fares better.\n"
        "Simulator-proxy only.",
        fontsize=9, y=0.095) if args.caption is None else fig.suptitle(args.caption, fontsize=9, y=0.095)
    fig.tight_layout(rect=[0, 0.21, 1, 0.945])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    for direction in ("downlink", "uplink"):
        print(f"\n{direction}  median delta % (seeds 0-4)")
        print(f"{'city':10}{'core':10}{'synthetic':>26}{'replay':>26}")
        for city in CITIES:
            for core, core_label, _ in CORES:
                s = syn.get((city, direction, core), {})
                r = rep.get((city, direction, core), {})
                fmt = lambda d: "[" + " ".join(f"{d[k]:+5.2f}" for k in sorted(d)) + "]" if d else "-"
                print(f"{city:10}{core_label:10}{fmt(s):>26}{fmt(r):>26}")


if __name__ == "__main__":
    main()
