"""
Empirical CDF comparison figures, built ONLY from raw per-seed / per-grid-point arrays that are
actually still on disk under outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # .../Codebase
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"  # consolidated with all other paper figures
FIG_DIR.mkdir(exist_ok=True)

LOCATIONS_ALL = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
COLORS = {"a": "#1f77b4", "b": "#ff7f0e", "c": "#2ca02c", "d": "#9467bd"}


def _ecdf_ax(ax, values, label, color, linestyle="-"):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label, color=color, linestyle=linestyle, linewidth=1.6)


def _grid(n, ncols=3):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
    return fig, axes.flatten()


def _finish(fig, axes, n_used, title, out_name, xlabel):
    for ax in axes[n_used:]:
        ax.axis("off")
    for ax in axes[:n_used]:
        ax.set_ylabel("CDF")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    n_lines = title.count("\n") + 1
    fig.suptitle(title, fontsize=13 if n_lines > 1 else 14)
    fig.tight_layout(rect=(0, 0, 1, 0.96 - 0.05 * (n_lines - 1)))
    out_path = FIG_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path.relative_to(PROJECT_ROOT)}")


def fig_boundary_freeze():
    d = json.load(open(OUT_DIR / "boundary_freeze_perseed.json"))
    rows = d["rows"]
    fig, axes = _grid(len(LOCATIONS_ALL))
    for i, loc in enumerate(LOCATIONS_ALL):
        ax = axes[i]
        for arm, color in [("baseline", COLORS["a"]), ("freeze", COLORS["b"])]:
            vals = [r["retransmits_per_s_median"] for r in rows if r["location"] == loc and r["arm"] == arm]
            _ecdf_ax(ax, vals, f"{arm} (n={len(vals)})", color)
        ax.set_title(loc)
    _finish(fig, axes, len(LOCATIONS_ALL),
            "Boundary-Freeze Mechanism Check: Retransmit Rate CDF, Baseline vs. Freeze (10 seeds/location)",
            "cdf_boundary_freeze.png", "retransmits / s")


def fig_beta_scaleup():
    d = json.load(open(OUT_DIR / "scaleup_10seed_results.json"))
    fig, axes = _grid(len(LOCATIONS_ALL))
    for i, loc in enumerate(LOCATIONS_ALL):
        ax = axes[i]
        entry = d[loc]
        for arm, color in [("beta_default", COLORS["a"]), ("beta12", COLORS["b"])]:
            vals = entry[arm]["rtx"]
            _ecdf_ax(ax, vals, f"{arm} (n={len(vals)})", color)
        stock_rtx = entry["stock"]["rtx"]
        ax.axvline(stock_rtx, color="gray", linestyle=":", linewidth=1.3, label=f"stock BBR-v3 ({stock_rtx:.1f})")
        ax.set_title(loc)
    _finish(fig, axes, len(LOCATIONS_ALL),
            "Reward-Weight Robustness: Retransmit Rate CDF, $\\beta$ default vs. $4\\times\\beta$ (10 seeds/location)",
            "cdf_beta_scaleup.png", "retransmits / s")


def fig_s7_ablation():
    d = json.load(open(OUT_DIR / "point2_s7_ablation.json"))
    locs = [loc for loc in LOCATIONS_ALL if loc in d]
    fig, axes = _grid(len(locs))
    for i, loc in enumerate(locs):
        ax = axes[i]
        entry = d[loc]
        for arm, key, color in [("s7 live (ref)", "ref", COLORS["a"]), ("s7 ablated (abl)", "abl", COLORS["b"])]:
            vals = entry[key]
            _ecdf_ax(ax, vals, f"{arm} (n={len(vals)})", color)
        ax.set_title(f"{loc}  (median drop: {entry['drop_pp']:+.1f}pp, p={entry['p']:.2f})", fontsize=9)
    _finish(fig, axes, len(locs),
            "$s_7$ (reconfig-phase feature) Ablation: Retransmit-Reduction-vs-Stock CDF (10 seeds/location)",
            "cdf_s7_ablation.png", "retransmit reduction vs. stock BBR-v3 (%)")


def fig_rq4_grid():
    d = json.load(open(OUT_DIR / "point5_grid_comparison.json"))
    locs = list(d.keys())
    fig, axes = _grid(len(locs), ncols=2)
    fig.set_size_inches(11.0, 4.6)
    for i, loc in enumerate(locs):
        ax = axes[i]
        entry = d[loc]
        _ecdf_ax(ax, entry["ref_classical"], f"classical (n={len(entry['ref_classical'])})", COLORS["a"])
        _ecdf_ax(ax, entry["ref_quantum_l2"], f"quantum L=2 (n={len(entry['ref_quantum_l2'])})", COLORS["b"])
        ax.set_title(loc)
    _finish(fig, axes, len(locs),
            "RQ4 Reference Grid Point: Retransmit-Reduction CDF, Classical vs. Quantum\n(screening subset -- Mumbai/Ohio only)",
            "cdf_rq4_grid.png", "retransmit reduction vs. stock BBR-v3 (%)")


def fig_oracle_freeze_vs_nofreeze():
    d_freeze = json.load(open(OUT_DIR / "oracle_action_sensitivity_multihead.json"))
    d_nofreeze = json.load(open(OUT_DIR / "oracle_action_sensitivity_multihead_no_freeze.json"))
    fig, axes = _grid(len(LOCATIONS_ALL))
    for i, loc in enumerate(LOCATIONS_ALL):
        ax = axes[i]
        vals_freeze = [a["mean_delta_retransmits_per_decision"] for a in d_freeze[loc]["actions"]]
        vals_nofreeze = [a["mean_delta_retransmits_per_decision"] for a in d_nofreeze[loc]["actions"]]
        _ecdf_ax(ax, vals_freeze, f"with freeze (n={len(vals_freeze)})", COLORS["a"])
        _ecdf_ax(ax, vals_nofreeze, f"without freeze (n={len(vals_nofreeze)})", COLORS["b"])
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.2)
        ax.set_title(loc)
    _finish(fig, axes, len(LOCATIONS_ALL),
            "One-Step Oracle Action Grid (125 actions): Retransmit-Delta-vs-Stock CDF, With vs. Without Boundary Freeze",
            "cdf_oracle_freeze_vs_nofreeze.png", "mean $\\Delta$retransmits / decision vs. stock (negative = fewer)")


def main():
    fig_boundary_freeze()
    fig_beta_scaleup()
    fig_s7_ablation()
    fig_rq4_grid()
    fig_oracle_freeze_vs_nofreeze()


if __name__ == "__main__":
    main()
