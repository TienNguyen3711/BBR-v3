from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CITY_LABEL = {"SaoPaulo": "Sao Paulo"}

# Simulated arms share a palette; the measured arm is grey and hatched so it
# never reads as one more comparable box.
COLOR = {"quantum": "#5B8FC9", "classical": "#E8973C", "stock": "#5AA469", "real": "#9AA3AE"}
LABEL = {"quantum": "QA2C (quantum)", "classical": "A2C (classical, matched)",
         "stock": "stock BBR — simulated", "real": "real trace — measured"}

# Top row: absolute levels, comparable to a conventional per-CCA figure.
# Bottom row: the PAIRED delta against stock on the same replayed trace, which
# is what the experiment actually measures. The absolute panels cannot show the
# result -- a 4 ms RTT change is invisible on an axis that spans 30-420 ms
# across cities, and every arm sits on top of the others in throughput.
ABSOLUTE_PANELS = [
    ("throughput_mbps_mean", "Throughput", "Mbps"),
    ("rtt_p90_ms", "RTT (p90)", "ms"),
    ("rtt_std_ms", "RTT variability (sd)", "ms"),
]
DELTA_PANELS = [
    ("throughput_delta_vs_stock_pct", "\u0394 Throughput vs stock", "%"),
    ("rtt_p90_delta_vs_stock_ms", "\u0394 RTT p90 vs stock", "ms"),
    ("retransmits_delta_vs_stock_per_s", "\u0394 Retransmissions vs stock", "per second"),
]


def load(paths):
    rows = []
    for path in paths:
        rows += json.loads(Path(path).read_text())["rows"]
    return rows


def delta_series(rows, city, arm, field):
    """Paired agent-minus-stock on the same trace and seed (common random numbers)."""
    return [r[field] for r in rows if r["location"] == city and r["core"] == arm]


def series(rows, city, arm, field):
    group = [r for r in rows if r["location"] == city]
    if arm == "real":
        key = {"throughput_mbps_mean": "trace_realised_mbps_mean",
               "rtt_p90_ms": "trace_realised_rtt_p90_ms"}.get(field)
        if key is None:
            return []
        # One value per replayed trace, not per (trace, seed).
        return [v for _run, v in sorted({(r["run"], r[key]) for r in group})]
    if arm == "stock":
        return [v for _run, v in sorted({(r["run"], r["stock_sim"][field]) for r in group})]
    return [r["agent_sim"][field] for r in group if r["core"] == arm]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=PACKAGE_ROOT.parent / "figures" / "replay_comparison_v14.png")
    parser.add_argument("--title", default="Replayed real Starlink downlink forcing — policy vs stock")
    args = parser.parse_args()

    rows = load(args.replays)
    arms = ["quantum", "classical", "stock", "real"]

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))
    fig.suptitle(args.title, fontsize=14, y=0.985)

    width = 0.19
    for axis, (field, title, unit) in zip(axes[0], ABSOLUTE_PANELS):
        for offset, arm in enumerate(arms):
            data, positions = [], []
            for index, city in enumerate(CITIES):
                values = series(rows, city, arm, field)
                if values:
                    data.append(values)
                    positions.append(index + (offset - 1.5) * width)
            if not data:
                continue
            box = axis.boxplot(data, positions=positions, widths=width * 0.86,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color="#C8500A", linewidth=1.3),
                               whiskerprops=dict(color="#5A6472"),
                               capprops=dict(color="#5A6472"))
            for patch in box["boxes"]:
                patch.set_facecolor(COLOR[arm])
                patch.set_alpha(0.85)
                patch.set_edgecolor("#3A424E")
                if arm == "real":
                    patch.set_hatch("///")
                    patch.set_alpha(0.55)
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(unit)
        axis.set_xticks(range(len(CITIES)))
        axis.set_xticklabels([CITY_LABEL.get(c, c) for c in CITIES], rotation=20)
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)

    # Bottom row: paired deltas. Only the two policy arms exist here -- stock is
    # the baseline they are differenced against, so it is the zero line.
    dwidth = 0.3
    for axis, (field, title, unit) in zip(axes[1], DELTA_PANELS):
        for offset, arm in enumerate(["quantum", "classical"]):
            data, positions = [], []
            for index, city in enumerate(CITIES):
                values = delta_series(rows, city, arm, field)
                if values:
                    data.append(values)
                    positions.append(index + (offset - 0.5) * dwidth)
            if not data:
                continue
            box = axis.boxplot(data, positions=positions, widths=dwidth * 0.84,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color="#C8500A", linewidth=1.4),
                               whiskerprops=dict(color="#5A6472"),
                               capprops=dict(color="#5A6472"))
            for patch in box["boxes"]:
                patch.set_facecolor(COLOR[arm])
                patch.set_alpha(0.85)
                patch.set_edgecolor("#3A424E")
        axis.axhline(0.0, color="#2F6B4F", linewidth=1.4, zorder=1)
        axis.set_title(title, fontsize=12)
        axis.set_ylabel(unit)
        axis.set_xticks(range(len(CITIES)))
        axis.set_xticklabels([CITY_LABEL.get(c, c) for c in CITIES], rotation=20)
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.set_axisbelow(True)

    # Fidelity is the caveat that decides whether any of this is readable, so it
    # goes on the figure rather than in a caption someone can lose.
    notes = []
    for city in CITIES:
        group = [r for r in rows if r["location"] == city]
        if group:
            notes.append(f"{CITY_LABEL.get(city, city)} {np.median([r['replay_fidelity_stock_over_real'] for r in group]):.2f}x")
    fig.text(0.5, 0.028, "replay fidelity (stock-sim throughput / real trace; 1.00 = exact):  "
             + "   ".join(notes), ha="center", fontsize=9, color="#48505C")
    fig.text(0.5, 0.006,
             "Top row: absolute levels. Bottom row: paired delta against stock on the SAME trace "
             "(green line = stock). The measured arm is a different kind of evidence — "
             "a gap to it mixes policy effect with simulator error.",
             ha="center", fontsize=8.5, style="italic", color="#6B7280")

    fig.legend(handles=[Patch(facecolor=COLOR[a], edgecolor="#3A424E", alpha=0.85,
                              hatch="///" if a == "real" else None, label=LABEL[a]) for a in arms],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955), fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.925))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
