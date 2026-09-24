"""Plot all three policy objectives from the Sydney semantics audit report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    groups = report["summaries"]["evaluation"]
    labels = {"stock": "Stock", "highest_permitted": "Highest gain",
              "gain110": "Gain 1.10", "balanced110": "1.10 / drain",
              "balanced125": "1.25 / drain", "quantum": "QA2C", "classical": "A2C"}
    values = [s["medians"]["retransmits_delta_per_s"] for g in groups.values() for s in g.values()]
    bound = max(max(abs(x) for x in values), .01)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, layout="constrained")
    for ax, (mode, summaries) in zip(axes, groups.items()):
        for name, summary in summaries.items():
            m = summary["medians"]
            x, y, c = m["rtt_p90_delta_ms"], m["throughput_delta_pct"], m["retransmits_delta_per_s"]
            points = ax.scatter(x, y, c=[c], norm=norm, cmap="coolwarm", s=65,
                                marker="s" if name == "stock" else "o", edgecolors="#333333", linewidths=.6)
            offset = (7, 7) if name != "classical" else (7, -14)
            crowded = mode == "legacy_v14" and name in ("balanced110", "balanced125", "quantum", "classical")
            if crowded:
                offset = {"balanced110": (-25, -30), "balanced125": (10, 18),
                          "quantum": (35, -8), "classical": (35, -35)}[name]
            ax.annotate(labels[name], (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                        ha="right" if crowded and name == "balanced110" else "left",
                        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": .6} if crowded else None)
        ax.axhline(0, color="#666666", linewidth=.8)
        ax.axvline(0, color="#666666", linewidth=.8)
        ax.grid(alpha=.15)
        ax.set_xlabel("RTT p90 change vs own stock (ms); left is better")
        ax.set_title("Legacy v14" if mode == "legacy_v14" else "CRUISE-only proxy with stock probing")
        ax.margins(x=.3, y=.25)
    axes[0].set_ylabel("Throughput change vs own stock (%); up is better")
    fig.colorbar(points, ax=axes, shrink=.8, label="Retransmissions change (/s); blue is lower")
    fig.suptitle("Sydney downlink: reused evaluation runs 8–10\nMedians across traces, then all five learner seeds; exploratory fluid-model audit", fontsize=12)
    output = args.report.parent / "tradeoffs"
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
