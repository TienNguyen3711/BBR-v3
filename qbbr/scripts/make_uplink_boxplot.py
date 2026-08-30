"""Uplink counterpart of make_comparison_figures.py's fig_rq1_boxplot_vs_bbr:
stock BBR-v3 (real per-trace-file uplink data) vs. qbbr (pacing_gain-only,
newly trained under uplink -- see train_uplink_parallel.py +
eval_rq1_uplink_raw.py). Only one qbbr arm exists for uplink so far (no
extended-action-space uplink checkpoints yet).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
C1 = "#1f77b4"


def main():
    d = json.load(open(OUT_DIR / "rq1_uplink_raw.json"))
    metrics = [("throughput_mbps", "Throughput (Mbps)"), ("rtt_ms", "RTT (ms)"),
               ("retransmits_per_s", "Retransmissions (count/s)")]
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    group_width = 3
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        all_positions, all_data, all_colors = [], [], []
        for i, loc in enumerate(LOCATIONS):
            bbr_vals = d["real_cca"][f"{loc}|bbr"][metric_key]
            qbbr_vals = d["qbbr"][loc][metric_key]
            all_data.append(bbr_vals)
            all_positions.append(i * group_width)
            all_colors.append("#7f7f7f")
            all_data.append(qbbr_vals)
            all_positions.append(i * group_width + 1)
            all_colors.append(C1)
        bp = ax.boxplot(all_data, positions=all_positions, widths=0.8, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_xticks([i * group_width + 0.5 for i in range(len(LOCATIONS))])
        ax.set_xticklabels(LOCATIONS)
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.85) for c in ["#7f7f7f", C1]]
    axes[0].legend(handles, ["stock BBR-v3 (uplink, real traces)", "qbbr (pacing_gain only, uplink, newly trained)"],
                   fontsize=9, loc="upper right")
    fig.suptitle("UPLINK: qbbr vs. Stock BBR-v3 -- Throughput / RTT / Retransmissions "
                  "(10 seeds vs. 10 trace-files per box)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG_DIR / "boxplot_rq1_uplink_vs_bbr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
