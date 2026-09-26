"""Action-share and training-curve figures for the final-v3 main study (Tier 1)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CITIES = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CITY_LABEL = {"SaoPaulo": "Sao Paulo"}
DIRECTIONS = ["downlink", "uplink"]
GAINS = ["0.75", "0.90", "1.00", "1.10", "1.25"]
# Diverging around the stock gain: two blues below, neutral grey at 1.00, two oranges above.
GAIN_COLOR = ["#2f6fb3", "#8fb8e0", "#c9c9c4", "#f3b77a", "#dd7a1f"]
CORE_COLOR = {"a2c": "#4f8fcf", "qa2c": "#f0a04b"}
CORE_LABEL = {"a2c": "Classical A2C", "qa2c": "Hybrid QA2C"}
EDGE = "#4d4d4d"


def results(study: Path):
    for path in (study / "1a_full").glob("*/result.json"):
        yield json.loads(path.read_text())


def action_shares(study: Path) -> dict:
    counts = defaultdict(lambda: np.zeros(5))
    for r in results(study):
        j = r["job"]
        for e in r["evaluation"]:
            for k, v in e["policy"]["action_counts"].items():
                counts[(j["location"], j["direction"], j["core"])][int(k)] += v
    return {key: c / c.sum() for key, c in counts.items()}


def fig_actions(study: Path, out: Path) -> dict:
    import matplotlib.pyplot as plt
    shares = action_shares(study)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = 0.38
    for ax, direction in zip(axes, DIRECTIONS):
        for i, core in enumerate(("a2c", "qa2c")):
            xs = np.arange(len(CITIES)) + (i - 0.5) * (width + 0.03)
            bottom = np.zeros(len(CITIES))
            for g in range(5):
                vals = np.array([shares[(c, direction, core)][g] for c in CITIES])
                ax.bar(xs, vals, width, bottom=bottom, color=GAIN_COLOR[g], edgecolor="white", linewidth=0.6,
                       label=f"gain {GAINS[g]}" if (i == 0 and direction == "downlink") else None)
                bottom += vals
            for x in xs:
                ax.text(x, 1.01, "A" if core == "a2c" else "Q", ha="center", va="bottom", fontsize=8, color="#555555")
        ax.set_xticks(range(len(CITIES)))
        ax.set_xticklabels([CITY_LABEL.get(c, c) for c in CITIES], rotation=25)
        ax.set_title(f"Tier 1, {direction}", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", color="#e4e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Share of decisions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    pooled = {}
    for direction in DIRECTIONS:
        for core in ("a2c", "qa2c"):
            pooled[(direction, core)] = np.mean([shares[(c, direction, core)] for c in CITIES], axis=0)
    return pooled


def fig_training(study: Path, out: Path) -> dict:
    import matplotlib.pyplot as plt
    curves = defaultdict(list)
    for r in results(study):
        j = r["job"]
        rewards = [t["mean_reward"] for t in sorted(r["training"], key=lambda t: t["episode"])]
        curves[(j["direction"], j["core"])].append(rewards)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    summary = {}
    for ax, direction in zip(axes, DIRECTIONS):
        for core in ("a2c", "qa2c"):
            arr = np.array(curves[(direction, core)])
            ep = np.arange(1, arr.shape[1] + 1)
            med = np.median(arr, axis=0)
            lo, hi = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
            ax.fill_between(ep, lo, hi, color=CORE_COLOR[core], alpha=0.15, linewidth=0)
            ax.plot(ep, med, color=CORE_COLOR[core], linewidth=0.9, alpha=0.55)
            smooth = np.convolve(np.pad(med, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
            ax.plot(ep, smooth, color=CORE_COLOR[core], linewidth=2.4, label=f"{CORE_LABEL[core]} (5-episode mean)")
            summary[(direction, core)] = dict(first5=float(np.median(arr[:, :5])), last5=float(np.median(arr[:, -5:])),
                                              change_last10=float(np.median(arr[:, -5:]) - np.median(arr[:, -10:-5])))
        ax.axhline(0, color="#35a893", linewidth=1.2)
        ax.set_title(f"Training, {direction}", fontsize=12)
        ax.set_xlabel("Episode")
        ax.grid(color="#eeeeea", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Mean reward per decision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return summary


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "outputs" / "rq_study" / "final-v3")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    pooled = fig_actions(args.study, args.figures / "v3_action_shares.png")
    for (direction, core), share in pooled.items():
        print(f"actions {direction:8s} {core:5s} " + "  ".join(f"{g}:{s:5.1%}" for g, s in zip(GAINS, share)))
    for key, s in fig_training(args.study, args.figures / "v3_training_curves.png").items():
        print("training", key, {k: round(v, 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
