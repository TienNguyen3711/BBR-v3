"""Paired policy-minus-stock deltas per city, synthetic (Tier 1) against real capacity (Tier 2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CITY_LABEL = {"SaoPaulo": "Sao Paulo"}
CORES = [("a2c", "Classical A2C-BBR"), ("qa2c", "Hybrid QA2C-BBR")]
COLOR = {"a2c": "#4f8fcf", "qa2c": "#f0a04b", "stock": "#35a893"}
EDGE = "#4d4d4d"
# Transport-model error of the simulator against real Linux BBR on identical
# capacity (Tier 3, main.tex Table "fidelity"): |simulated stock - real BBR|.
MODEL_ERROR = {"throughput_delta_pct": {"Sydney": 1.5, "Tokyo": 7.0, "Mumbai": 23.6, "Ohio": 11.4,
                                        "London": 9.2, "SaoPaulo": 7.6},
               "rtt_p90_delta_ms": {"Sydney": 2.8, "Tokyo": 6.7, "Mumbai": 14.5, "Ohio": 5.9,
                                    "London": 8.0, "SaoPaulo": 5.4}}
METRICS = [("throughput_delta_pct", "Throughput vs stock (%)"), ("rtt_p90_delta_ms", "RTT p90 vs stock (ms)")]
TIERS = [("result.json", "Tier 1: synthetic capacity"), ("transfer.json", "Tier 2: replayed real capacity")]


def deltas(study: Path, city: str, core: str, fname: str, key: str) -> np.ndarray:
    return np.asarray([e[key] for p in sorted(study.glob(f"*__{city}__*__{core}__*/{fname}"))
                       for e in json.loads(p.read_text())["evaluation"]])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs" / "rq_study" / "final-queue-delta0.16")
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "tier_deltas_queue_reward.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
    width = 0.36
    for row, (key, ylabel) in enumerate(METRICS):
        for col, (fname, title) in enumerate(TIERS):
            ax = axes[row, col]
            if col == 1:  # the band the policy effect must clear to be resolvable
                for c, city in enumerate(CITIES):
                    err = MODEL_ERROR[key][city]
                    ax.add_patch(plt.Rectangle((c - 0.45, -err), 0.9, 2 * err, color="#e6e6e3", zorder=0))
            for i, (core, _) in enumerate(CORES):
                positions = [c + (i - 0.5) * (width + 0.02) for c in range(len(CITIES))]
                ax.boxplot([deltas(args.study, city, core, fname, key) for city in CITIES],
                           positions=positions, widths=width, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=COLOR[core], edgecolor=EDGE, linewidth=0.8),
                           medianprops=dict(color="#1a1a19", linewidth=1.4),
                           whiskerprops=dict(color=EDGE, linewidth=0.8), capprops=dict(color=EDGE, linewidth=0.8))
            ax.axhline(0, color=COLOR["stock"], linewidth=1.6, zorder=1)
            ax.set_xticks(range(len(CITIES)))
            ax.set_xticklabels([CITY_LABEL.get(c, c) for c in CITIES], rotation=25)
            ax.grid(axis="y", color="#e4e4e0", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=13)
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[0, 1].set_ylim(-6, 10)
    axes[1, 1].set_ylim(-16, 16)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[c], edgecolor=EDGE) for c, _ in CORES]
    handles += [plt.Line2D([0], [0], color=COLOR["stock"], linewidth=1.6),
                plt.Rectangle((0, 0), 1, 1, facecolor="#e6e6e3")]
    labels = [label for _, label in CORES] + ["Stock BBR-v3 (zero)", "Simulator error vs real BBR (Tier 3)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
