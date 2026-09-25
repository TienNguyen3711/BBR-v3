"""RQ2 and RQ3 publication figures, drawn at IEEEtran's physical sizes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qbbr.eval.successor_protocol import bootstrap_median_ci

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
SCREEN = ROOT / "outputs/rq_study/native-bbr-rq-screen-v2-sixcity"
CONVERGENCE = ROOT / "outputs/rq_study/native-bbr-rq3-convergence-50ep"
TEXT_WIDTH, COLUMN_WIDTH = 7.16, 3.5
INK, MUTED, GRID, ZERO = "#1a1a19", "#5b5a56", "#e3e2de", "#8a8986"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "#9c9b97", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _style(axis, grid_axis="both"):
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def load(run_dir: Path):
    rows = []
    for path in sorted(run_dir.glob("*/result.json")):
        result = json.loads(path.read_text())
        job = result["job"]
        for entry in result["evaluation"]:
            rows.append({**{k: job[k] for k in ("location", "core", "variant", "seed")},
                         "holdout_seed": entry["holdout_seed"],
                         "throughput_delta_pct": entry["throughput_delta_pct"]})
        for point in result["learning_curve"]:
            rows.append({**{k: job[k] for k in ("location", "core", "variant", "seed")},
                         "episode": point["episode"],
                         "curve": median(e["throughput_delta_pct"] for e in point["evaluation"])})
    return rows


def paired(rows, core, treatment, control):
    """Per-(location, seed) difference at matched holdout seeds."""
    left, right = defaultdict(dict), defaultdict(dict)
    for row in rows:
        if "throughput_delta_pct" not in row or row["core"] != core:
            continue
        target = left if row["variant"] == treatment else right if row["variant"] == control else None
        if target is not None:
            target[(row["location"], row["seed"])][row["holdout_seed"]] = row["throughput_delta_pct"]
    out = {}
    for key, values in left.items():
        shared = values.keys() & right.get(key, {}).keys()
        if shared:
            out[key] = median(values[s] - right[key][s] for s in shared)
    return out


def paired_cores(rows, treatment, control, variant="full"):
    left, right = defaultdict(dict), defaultdict(dict)
    for row in rows:
        if "throughput_delta_pct" not in row or row["variant"] != variant:
            continue
        target = left if row["core"] == treatment else right if row["core"] == control else None
        if target is not None:
            target[(row["location"], row["seed"])][row["holdout_seed"]] = row["throughput_delta_pct"]
    out = {}
    for key, values in left.items():
        shared = values.keys() & right.get(key, {}).keys()
        if shared:
            out[key] = median(values[s] - right[key][s] for s in shared)
    return out


def _effect_row(axis, y, values, colour, marker):
    """One contrast: its cells as dots, its median and bootstrap interval."""
    jitter = np.linspace(-0.22, 0.22, len(values))
    axis.scatter(values, y + jitter, s=5.5, facecolor="none", edgecolor=colour,
                 linewidth=0.6, marker=marker, zorder=3, clip_on=False)
    interval = bootstrap_median_ci(values, seed=20260921)
    axis.plot([interval["lower"], interval["upper"]], [y - 0.33, y - 0.33],
              color=colour, linewidth=1.6, solid_capstyle="butt", zorder=4)
    axis.plot([median(values)], [y - 0.33], marker="|", color=INK,
              markersize=7, markeredgewidth=1.1, zorder=5)
    positive = sum(1 for v in values if v > 0)
    axis.annotate(f"{positive}/{len(values)}", (1.02, y), clip_on=False,
                  xycoords=("axes fraction", "data"), ha="left", va="center",
                  fontsize=6.4, color=MUTED)


def rq2_figure(rows, out: Path):
    contrasts = [
        ("QA2C: full $-$ telemetry", paired(rows, "qa2c", "full", "telemetry"), BLUE, "o"),
        ("QA2C: full $-$ telemetry+queue", paired(rows, "qa2c", "full", "telemetry_queue"), BLUE, "s"),
        ("A2C: full $-$ telemetry", paired(rows, "a2c", "full", "telemetry"), ORANGE, "o"),
        ("A2C: full $-$ telemetry+queue", paired(rows, "a2c", "full", "telemetry_queue"), ORANGE, "s"),
    ]
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.05))
    for index, (_, values, colour, marker) in enumerate(contrasts):
        _effect_row(axis, len(contrasts) - 1 - index, list(values.values()), colour, marker)
    axis.axvline(0.0, color=ZERO, linewidth=0.8, zorder=1)
    axis.set_yticks(range(len(contrasts)))
    axis.set_yticklabels([label for label, *_ in contrasts][::-1])
    axis.set_ylim(-0.75, len(contrasts) - 0.45)
    axis.set_xlabel("throughput difference (percentage points)")
    axis.annotate("cells\n$>0$", (1.02, len(contrasts) - 0.35), clip_on=False,
                  xycoords=("axes fraction", "data"), ha="left", va="center",
                  fontsize=6.4, color=MUTED)
    _style(axis, grid_axis="x")
    axis.tick_params(axis="y", length=0)
    fig.savefig(out)
    plt.close(fig)


def rq3_figure(screen_rows, curve_rows, out: Path):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.25), gridspec_kw={"width_ratios": [1.35, 1.0]})

    series = [("QA2C", "qa2c", BLUE, "-", "o"), ("A2C", "a2c", ORANGE, "--", "s"),
              ("QDQN", "qdqn", AQUA, ":", "^")]
    episodes = sorted({row["episode"] for row in curve_rows if "episode" in row})
    for label, core, colour, style, marker in series:
        values = []
        for episode in episodes:
            cell = [row["curve"] for row in curve_rows
                    if row.get("episode") == episode and row["core"] == core]
            values.append(median(cell))
        left.plot(episodes, values, color=colour, linestyle=style, marker=marker,
                  markersize=3.2, linewidth=1.3, label=label, zorder=3)
        left.annotate(label, (episodes[-1], values[-1]), xytext=(4, 0),
                      textcoords="offset points", fontsize=7, color=colour, va="center")
    left.axvline(30, color=ZERO, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    left.annotate("30-episode budget", (30, left.get_ylim()[0]), xytext=(3, 3),
                  textcoords="offset points", fontsize=6.2, color=MUTED, va="bottom")
    left.set_xlabel("training episodes")
    left.set_ylabel("held-out throughput vs stock (\\%)")
    left.set_xlim(episodes[0] - 2, episodes[-1] + 9)
    _style(left)

    budgets = [("30 episodes", paired_cores(screen_rows, "qa2c", "a2c"), BLUE, "o"),
               ("50 episodes", paired_cores(curve_rows, "qa2c", "a2c"), ORANGE, "s")]
    for index, (_, values, colour, marker) in enumerate(budgets):
        _effect_row(right, len(budgets) - 1 - index, list(values.values()), colour, marker)
    right.axvline(0.0, color=ZERO, linewidth=0.8, zorder=1)
    right.set_yticks(range(len(budgets)))
    right.set_yticklabels([label for label, *_ in budgets][::-1])
    right.set_ylim(-0.75, len(budgets) - 0.45)
    right.set_xlabel("QA2C $-$ A2C (percentage points)")
    right.annotate("cells\n$>0$", (1.02, len(budgets) - 0.35), clip_on=False,
                   xycoords=("axes fraction", "data"), ha="left", va="center",
                   fontsize=6.4, color=MUTED)
    _style(right, grid_axis="x")
    right.tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(out)
    plt.close(fig)


def coexistence_figure(report: Path, out: Path):
    """Isolated to contended, per algorithm, on a shared reference."""
    cells = json.loads(report.read_text())["cells"]
    downlink = [k for k in cells if k.endswith("downlink") and cells[k].get("usable_runs")]
    names = sorted(cells[downlink[0]]["per_cca"])
    rows = []
    for name in names:
        isolated = median(cells[k]["per_cca"][name]["isolated_mbps"] for k in downlink)
        contended = median(cells[k]["per_cca"][name]["contended_mbps"] for k in downlink)
        share = median(cells[k]["per_cca"][name]["share_of_total_pct"] for k in downlink)
        rows.append((share, name, isolated, contended))
    rows.sort()

    labels = {"bbr": "BBR", "bbr1": "BBRv1", "bbr2": "BBRv2", "ccp": "Copa",
              "cubic": "CUBIC", "hybla": "Hybla", "leocc": "LEOcc", "pcc": "PCC",
              "vegas": "Vegas"}
    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    equal_share = 100.0 / len(rows)
    for index, (share, name, isolated, contended) in enumerate(rows):
        highlight = name == "bbr"
        colour = BLUE if highlight else MUTED
        axis.plot([contended, isolated], [index, index], color=colour,
                  linewidth=1.0, alpha=0.9 if highlight else 0.45, zorder=2)
        axis.scatter([isolated], [index], s=13, facecolor="none", edgecolor=colour,
                     linewidth=0.8, zorder=3)
        axis.scatter([contended], [index], s=15, color=colour, zorder=4)
        axis.annotate(f"{share / equal_share:.2f}$\\times$", (1.02, index), clip_on=False,
                      xycoords=("axes fraction", "data"), ha="left", va="center",
                      fontsize=6.4, color=INK if highlight else MUTED)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([labels.get(name, name) for _, name, _, _ in rows])
    axis.set_xscale("log")
    axis.set_xlabel("throughput (Mbit/s, log scale)")
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.annotate("share of\nequal", (1.02, len(rows) - 0.05), clip_on=False,
                  xycoords=("axes fraction", "data"), ha="left", va="center",
                  fontsize=6.4, color=MUTED)
    # Anchored on the row with the widest gap and pushed outward: on a log axis
    # the two ends of the top row sit close enough to collide if both centre.
    widest = max(rows, key=lambda row: row[2] / max(row[3], 1e-9))
    axis.annotate("isolated", (widest[2], rows.index(widest)), xytext=(6, 8),
                  textcoords="offset points", ha="left", fontsize=6.4, color=MUTED)
    axis.annotate("contended", (widest[3], rows.index(widest)), xytext=(-6, 8),
                  textcoords="offset points", ha="right", fontsize=6.4, color=MUTED)
    _style(axis, grid_axis="x")
    axis.tick_params(axis="y", length=0)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    screen_rows, curve_rows = load(SCREEN), load(CONVERGENCE)
    rq2_figure(screen_rows, args.out / "rq2_ablation_ieee.pdf")
    rq3_figure(screen_rows, curve_rows, args.out / "rq3_convergence_ieee.pdf")
    coexistence = ROOT / "reports/measured_coexistence.json"
    if coexistence.exists():
        coexistence_figure(coexistence, args.out / "coexistence_ieee.pdf")
    print(f"wrote {args.out}/rq2_ablation_ieee.pdf, rq3_convergence_ieee.pdf, "
          f"coexistence_ieee.pdf")


if __name__ == "__main__":
    main()
