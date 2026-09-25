"""Is the policy's effect larger than the simulator's own error?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Validated for light surface with scripts/validate_palette.js: all checks pass
# (CVD dE 22.8 protan, normal-vision dE 29.1, contrast >= 3:1).
EFFECT = {"quantum": "#2E6DA8", "classical": "#D07A1E"}
LABEL = {"quantum": "QA2C (quantum)", "classical": "A2C (classical)"}
BAND = "#D6DBE3"        # noise floor: recessive by design, carries direct labels
BAND_EDGE = "#AEB6C2"
INK, MUTED = "#12161C", "#6B7280"
RESOLVED = "#2F6B4F"


def gather(testbed: Path, replays: list[Path]):
    kernel_rows = json.loads(testbed.read_text())["rows"]
    replay_rows = []
    for path in replays:
        replay_rows += json.loads(path.read_text())["rows"]

    cities = []
    for city in {r["location"] for r in kernel_rows}:
        kernel = [r for r in kernel_rows if r["location"] == city]
        runs = {r["run"] for r in kernel}
        stock = [r["stock_sim"] for r in replay_rows
                 if r["location"] == city and r["run"] in runs and r["core"] == "quantum"]
        if not kernel or not stock:
            continue
        kernel_thr = np.mean([r["kernel"]["throughput_mbps_mean"] for r in kernel])
        kernel_rtt = np.mean([r["kernel"]["rtt_p90_ms"] for r in kernel])
        sim_thr = np.mean([r["throughput_mbps_mean"] for r in stock])
        sim_rtt = np.mean([r["rtt_p90_ms"] for r in stock])
        entry = {
            "city": city,
            "path_rtt": float(np.mean([r["trace_realised_rtt_p90_ms"] for r in kernel])),
            "err_thr": abs(100.0 * (sim_thr / kernel_thr - 1.0)),
            "err_rtt": abs(sim_rtt - kernel_rtt),
        }
        for core in EFFECT:
            group = [r for r in replay_rows if r["location"] == city and r["core"] == core]
            if group:
                entry[f"eff_thr_{core}"] = abs(np.median([r["throughput_delta_vs_stock_pct"] for r in group]))
                entry[f"eff_rtt_{core}"] = abs(np.median([r["rtt_p90_delta_vs_stock_ms"] for r in group]))
        cities.append(entry)
    return sorted(cities, key=lambda e: e["path_rtt"])


def panel(axis, cities, err_key, eff_prefix, title, unit):
    y = np.arange(len(cities))
    floor = 10 ** (np.floor(np.log10(min(
        [c[err_key] for c in cities] +
        [c[f"{eff_prefix}_{k}"] for c in cities for k in EFFECT if f"{eff_prefix}_{k}" in c and c[f"{eff_prefix}_{k}"] > 0]
    ))) - 0.3)

    for index, city in enumerate(cities):
        axis.barh(index, city[err_key] - floor, left=floor, height=0.62,
                  color=BAND, edgecolor=BAND_EDGE, linewidth=0.8, zorder=2)

    for offset, core in enumerate(EFFECT):
        key = f"{eff_prefix}_{core}"
        values = [max(c.get(key, np.nan), floor * 1.02) for c in cities]
        shift = (offset - 0.5) * 0.26
        # 2px surface ring so overlapping markers stay separable.
        axis.scatter(values, y + shift, s=95, color=EFFECT[core], zorder=4,
                     edgecolor="white", linewidth=1.6,
                     label=LABEL[core] if title.startswith("Throughput") else None)

    for index, city in enumerate(cities):
        effect = city.get(f"{eff_prefix}_quantum", np.nan)
        error = city[err_key]
        ratio = effect / error if error else np.inf
        resolved = ratio > 1.0
        axis.text(axis.get_xlim()[1] if False else max(effect, error) * 1.45, index,
                  f"{ratio:.2f}x", va="center", fontsize=9.5,
                  color=RESOLVED if resolved else MUTED,
                  fontweight="600" if resolved else "normal")

    axis.set_yticks(y)
    axis.set_yticklabels([f"{c['city']}\n{c['path_rtt']:.0f} ms path" for c in cities], fontsize=9.5)
    axis.set_xscale("log")
    axis.set_xlabel(unit, fontsize=10)
    axis.set_title(title, fontsize=12, pad=10)
    axis.grid(axis="x", alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testbed", type=Path,
                        default=PACKAGE_ROOT.parent / "reports" / "kernel_testbed_6city.json")
    parser.add_argument("--replays", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=PACKAGE_ROOT.parent / "figures" / "resolvability_v14.png")
    args = parser.parse_args()

    cities = gather(args.testbed, args.replays)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    fig.suptitle("Can the simulator resolve the policy's effect? Grey band = the model's own error",
                 fontsize=13.5, y=0.975)

    panel(axes[0], cities, "err_thr", "eff_thr", "Throughput", "|effect| and |model error|  (%, log scale)")
    panel(axes[1], cities, "err_rtt", "eff_rtt", "RTT p90", "|effect| and |model error|  (ms, log scale)")
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(facecolor=BAND, edgecolor=BAND_EDGE,
                     label="simulator's own error (unresolvable below this)")]
    handles += [Line2D([], [], marker="o", ls="", ms=9, color=EFFECT[c],
                       markeredgecolor="white", markeredgewidth=1.4, label=LABEL[c]) for c in EFFECT]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.935), fontsize=10)

    fig.text(0.5, 0.015,
             "Ratio labels are effect ÷ model error for the quantum arm; green marks the one path where the "
             "effect exceeds the error. Model error measured against real Linux BBR on the same capacity schedule.",
             ha="center", fontsize=8.7, style="italic", color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.905))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
