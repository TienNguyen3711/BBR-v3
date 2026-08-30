"""Empirical-CDF comparison of qbbr vs. stock BBR-v3, UPLINK vs. DOWNLINK,
per location.

Reads the two raw per-seed / per-trace-file dumps that are actually on disk:
  * outputs/rq1_raw_for_boxplot.json   (direction=downlink; qbbr key "{loc}|pacing_only")
  * outputs/rq1_uplink_raw.json        (direction=uplink;   qbbr key "{loc}")

Each metric -> one 2x3 figure (one panel per location). Every panel overlays
four ECDFs:
    qbbr   downlink   (blue, solid)      BBR-v3 downlink   (gray, solid)
    qbbr   uplink     (blue, dashed)     BBR-v3 uplink     (gray, dashed)

n = 10 per curve (10 seeds for qbbr, 10 trace-files for the real CCAs) -- these
are small-sample ECDFs, read them as distribution shape, not smooth estimates.

"Improvement" for retransmits/RTT = the blue (qbbr) curve sits LEFT of the gray
(BBR-v3) curve of the SAME linestyle (lower is better). For throughput, left =
worse. The throughput panel uses a log x-axis because uplink (~0-44 Mbps) and
downlink (~100-240 Mbps) live on different scales; exact-zero throughput runs
(BBR-v3 uplink stalls) are clipped to 0.1 Mbps so they still plot.

Output -> figures/cdf_ul_dl_{retransmits,throughput,rtt}.png
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # .../Codebase
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
C_QBBR = "#1f77b4"
C_BBR = "#7f7f7f"
ZERO_FLOOR = 0.1  # Mbps, for log-x throughput panel only


def _load():
    dn = json.load(open(OUT_DIR / "rq1_raw_for_boxplot.json"))
    up = json.load(open(OUT_DIR / "rq1_uplink_raw.json"))
    data = {}
    for loc in LOCATIONS:
        data[loc] = {
            ("qbbr", "downlink"): dn["qbbr"][f"{loc}|pacing_only"],
            ("bbr", "downlink"): dn["real_cca"][f"{loc}|bbr"],
            ("qbbr", "uplink"): up["qbbr"][loc],
            ("bbr", "uplink"): up["real_cca"][f"{loc}|bbr"],
        }
    return data


def _ecdf(ax, values, label, color, linestyle, floor=None, alpha=1.0):
    x = np.sort(np.asarray(values, dtype=float))
    if floor is not None:
        x = np.clip(x, floor, None)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label, color=color, linestyle=linestyle,
            linewidth=1.8, marker=".", markersize=5, alpha=alpha)


def _reduction(data, loc, direction, metric):
    """Median % reduction of qbbr vs BBR-v3 for a lower-is-better metric."""
    q = statistics.median(data[loc][("qbbr", direction)][metric])
    b = statistics.median(data[loc][("bbr", direction)][metric])
    if b <= 0:
        return float("nan"), q, b
    return 1.0 - q / b, q, b


def _make_metric_figure(data, metric, xlabel, title, out_name, logx=False,
                        lower_is_better=True):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    axes = axes.flatten()
    arms = [
        (("qbbr", "downlink"), C_QBBR, "-", "qbbr downlink", 1.0),
        (("bbr", "downlink"), C_BBR, "-", "BBR-v3 downlink", 0.75),
        (("qbbr", "uplink"), C_QBBR, "--", "qbbr uplink", 1.0),
        (("bbr", "uplink"), C_BBR, "--", "BBR-v3 uplink", 0.75),
    ]
    for i, loc in enumerate(LOCATIONS):
        ax = axes[i]
        for (arm, direction), color, ls, label, alpha in arms:
            floor = ZERO_FLOOR if logx else None
            _ecdf(ax, data[loc][(arm, direction)][metric], label, color, ls,
                  floor=floor, alpha=alpha)
        if lower_is_better:
            rd_dl, _, _ = _reduction(data, loc, "downlink", metric)
            rd_ul, _, _ = _reduction(data, loc, "uplink", metric)
            def _fmt(v):
                return "n/a" if v != v else f"{v:+.0%}"
            sub = f"qbbr vs BBR-v3 median:  DL {_fmt(rd_dl)}   UL {_fmt(rd_ul)}"
            ax.set_title(f"{loc}\n{sub}", fontsize=10)
        else:
            ax.set_title(loc, fontsize=11)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out.relative_to(PROJECT_ROOT)}")


def _write_summary(data):
    """Per-location numeric backing for the CDF panels: qbbr-vs-BBR-v3 median
    shift + distribution tests (Mann-Whitney, Kolmogorov-Smirnov), each
    direction separately."""
    from scipy import stats as ss

    lines = [
        "# Uplink vs. Downlink: qbbr vs. Stock BBR-v3 (CDF backing numbers)",
        "",
        "Per curve n=10 (10 seeds for qbbr, 10 trace-file medians for BBR-v3). "
        "`red%` = 1 - median(qbbr)/median(BBR-v3) for retransmits/RTT "
        "(positive = qbbr lower); for throughput it is the raw median ratio "
        "qbbr/BBR-v3 (`ret%`, 100% = parity). p_MWU / p_KS test qbbr vs "
        "BBR-v3 within that direction.",
        "",
    ]
    for metric, tag, better_low in [
        ("retransmits_per_s", "Retransmits/s", True),
        ("throughput_mbps", "Throughput Mbps", False),
        ("rtt_ms", "RTT ms", True),
    ]:
        lines.append(f"## {tag}")
        lines.append("")
        lines.append("| location | dir | qbbr med | BBR-v3 med | "
                     + ("red%" if better_low else "ret%") + " | p_MWU | p_KS |")
        lines.append("|---|---|---|---|---|---|---|")
        for loc in LOCATIONS:
            for direction in ("downlink", "uplink"):
                q = data[loc][("qbbr", direction)][metric]
                b = data[loc][("bbr", direction)][metric]
                qm, bm = statistics.median(q), statistics.median(b)
                if better_low:
                    val = "n/a" if bm <= 0 else f"{1 - qm / bm:+.0%}"
                else:
                    val = "n/a" if bm <= 0 else f"{qm / bm:.0%}"
                try:
                    p_mwu = ss.mannwhitneyu(q, b, alternative="two-sided").pvalue
                except ValueError:
                    p_mwu = float("nan")
                p_ks = ss.ks_2samp(q, b).pvalue
                lines.append(f"| {loc} | {direction[:2].upper()} | {qm:.2f} | "
                             f"{bm:.2f} | {val} | {p_mwu:.3f} | {p_ks:.3f} |")
        lines.append("")
    out = OUT_DIR / "uplink_downlink_cdf_summary.md"
    out.write_text("\n".join(lines))
    print(f"saved -> {out.relative_to(PROJECT_ROOT)}")


def main():
    data = _load()
    _write_summary(data)
    _make_metric_figure(
        data, "retransmits_per_s", "retransmissions / s",
        "Retransmit-Rate CDF: qbbr vs. Stock BBR-v3, Uplink vs. Downlink "
        "(10 seeds / 10 trace-files per curve; lower = better)",
        "cdf_ul_dl_retransmits.png", logx=False, lower_is_better=True)
    _make_metric_figure(
        data, "throughput_mbps", "throughput (Mbps, log scale; 0 clipped to 0.1)",
        "Throughput CDF: qbbr vs. Stock BBR-v3, Uplink vs. Downlink "
        "(10 seeds / 10 trace-files per curve; higher = better)",
        "cdf_ul_dl_throughput.png", logx=True, lower_is_better=False)
    _make_metric_figure(
        data, "rtt_ms", "RTT (ms)",
        "RTT CDF: qbbr vs. Stock BBR-v3, Uplink vs. Downlink "
        "(10 seeds / 10 trace-files per curve; lower = better)",
        "cdf_ul_dl_rtt.png", logx=False, lower_is_better=True)


if __name__ == "__main__":
    main()
