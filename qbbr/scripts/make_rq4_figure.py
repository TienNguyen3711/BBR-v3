"""RQ4 figure: quantum vs. parameter-matched classical policy core.

Top row -- one zoomed panel per location, the 10 per-seed final-episode
rewards for each core as a jittered strip with a median bar, Mann-Whitney p
annotated. The two clouds overlap at every location: the point is that the
difference is smaller than either core's own seed-to-seed spread.

Bottom panel -- median(quantum) - median(classical) per location for the
current seven-input run and for the archived six-input run, against a shaded
band of +/- a typical seven-input core IQR. Both straddle zero and the sign
of the sub-IQR gap flips between the two architectures.

Source: outputs/rq4_full_report.json (train_rq4_checkpoints.py). The
six-input medians are the archived tab:qvc values (primary-module match,
109 quantum vs 74 classical full parameters).

    python -m qbbr.scripts.make_rq4_figure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORT = PROJECT_ROOT / "outputs" / "rq4_full_report.json"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_CLASS, C_QUANT = "#1f77b4", "#ff7f0e"

# Archived six-input tab:qvc medians (primary-module match, 109q vs 74c).
SIX_INPUT_MEDIAN = {
    "London": (4.8439, 4.8442), "Mumbai": (4.5763, 4.5772), "Ohio": (4.9563, 4.9568),
    "SaoPaulo": (4.5934, 4.5943), "Sydney": (5.1203, 5.1202), "Tokyo": (4.9646, 4.9652),
}  # (classical, quantum)
PRETTY = {"SaoPaulo": "São Paulo"}


def _pstr(p: float) -> str:
    return f"p={p:.3f}" if p >= 0.001 else "p<0.001"


def main() -> None:
    d = json.load(open(REPORT))
    recs = d["records"]
    per_loc = {r["location"]: r for r in d["per_location"]}
    locations = list(d["meta"]["locations"])

    def samples(loc: str, core: str) -> list[float]:
        return [r["final_mean_reward"] for r in recs if r["location"] == loc and r["core"] == core]

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.62], hspace=0.42, wspace=0.5)

    rng = np.random.default_rng(0)
    for i, loc in enumerate(locations):
        ax = fig.add_subplot(gs[0, i])
        c, q = samples(loc, "classical"), samples(loc, "quantum")
        pooled = np.array(c + q)
        # Zoom to the central 70% of the pooled seeds with generous padding;
        # a few worse-converging seeds fall outside and are counted in an
        # annotation rather than allowed to compress the informative range.
        p15, p85 = np.percentile(pooled, [15, 85])
        pad = max((p85 - p15) * 1.9, 0.003)
        lo, hi = p15 - pad, p85 + pad
        for j, (vals, colour) in enumerate([(c, C_CLASS), (q, C_QUANT)]):
            v = np.array(vals)
            xj = j + rng.uniform(-0.11, 0.11, size=len(v))
            ax.scatter(xj, np.clip(v, lo, hi), s=15, color=colour, alpha=0.85,
                       zorder=3, edgecolor="none")
            ax.plot([j - 0.26, j + 0.26], [np.median(v)] * 2, color=colour, lw=2.6, zorder=4)
            n_lo = int(np.sum(v < lo))
            n_hi = int(np.sum(v > hi))
            if n_lo:
                ax.text(j, 0.02, f"{n_lo}↓", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color=colour)
            if n_hi:
                ax.text(j, 0.98, f"{n_hi}↑", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color=colour)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["C", "Q"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(PRETTY.get(loc, loc), fontsize=10)
        p = per_loc[loc].get("mann_whitney", {}).get("p_value", float("nan"))
        ax.text(0.5, 0.93, _pstr(p), transform=ax.transAxes, ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", lw=0.5, alpha=0.9))
        ax.tick_params(labelsize=7.5)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        if i == 0:
            ax.set_ylabel("final-episode reward", fontsize=9)

    axb = fig.add_subplot(gs[1, :])
    x = np.arange(len(locations))
    seven = np.array([per_loc[l]["quantum"]["median"] - per_loc[l]["classical"]["median"] for l in locations]) * 1e3
    six = np.array([SIX_INPUT_MEDIAN[l][1] - SIX_INPUT_MEDIAN[l][0] for l in locations]) * 1e3
    band = np.median([per_loc[l]["classical"]["iqr"] for l in locations]) * 1e3
    axb.axhspan(-band, band, color="0.88", zorder=0,
                label=f"$\\pm$ median classical seed IQR ($\\approx {band:.1f}\\times10^{{-3}}$)")
    axb.axhline(0, color="0.4", lw=1, zorder=1)
    axb.scatter(x - 0.09, seven, s=75, color=C_QUANT, zorder=3, label="seven-input (126p full-agent match)")
    axb.scatter(x + 0.09, six, s=62, facecolor="none", edgecolor="0.35", linewidths=1.4, zorder=3,
                label="six-input archived (109q vs 74c)")
    axb.set_xticks(x)
    axb.set_xticklabels([PRETTY.get(l, l) for l in locations])
    axb.set_ylabel(r"median$(Q)-$median$(C)$" "\n" r"($\times10^{-3}$ reward)", fontsize=9)
    axb.set_ylim(-3.0, 3.0)
    axb.tick_params(labelsize=8)
    axb.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), framealpha=0.92, ncol=3)
    axb.grid(axis="y", alpha=0.3)

    fig.suptitle("RQ4: Quantum vs. Parameter-Matched Classical Core "
                 "(10 seeds/location, $\\alpha{=}1$, $L{=}2$) — null at every location", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "rq4_quantum_vs_classical.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
