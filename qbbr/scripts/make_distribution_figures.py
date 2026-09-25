"""CDF and box-plot views of fidelity and policy effect, per city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ORDER = ["Sydney", "Tokyo", "Mumbai", "Ohio", "London", "SaoPaulo"]
LABEL_CITY = {"SaoPaulo": "São Paulo"}
INK, MUTED, GRID = "#12161C", "#6B7280", "#E4E8EE"

COLOR = {
    "sim_quantum": "#2a78d6",    # slot 1
    "sim_classical": "#eb6834",  # slot 2
    "sim_stock": "#1baf7a",      # slot 3
    "real_kernel": "#4a3aa7",    # slot 7
    "real_trace": "#52514e",     # text-secondary neutral: a different evidence class
}
NAME = {
    "real_trace": "real trace (Starlink, measured)",
    "real_kernel": "real Linux BBR (emulated capacity)",
    "sim_stock": "simulator — stock BBR",
    "sim_quantum": "simulator — QA2C",
    "sim_classical": "simulator — A2C",
}


def load(path: Path):
    return json.loads(path.read_text())["series"]


def pooled(series, city, source, metric):
    values = []
    for record in series:
        if record["location"] == city and source in record:
            values += record[source][metric]
    return np.asarray(values, dtype=float)


def _ecdf(values):
    ordered = np.sort(values)
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def _style(axis):
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#B8C0CC")
    axis.tick_params(colors=MUTED, labelsize=9)


def fidelity_cdf(series, out: Path) -> None:
    cities = [c for c in ORDER if any(r["location"] == c for r in series)]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), sharey=True)
    fig.suptitle("Per-second RTT: is the simulator faithful to real BBR on the same capacity?",
                 fontsize=14, color=INK, y=0.985)

    for axis, city in zip(axes.ravel(), cities):
        stats = {}
        for source, style in (("real_trace", dict(ls=(0, (5, 3)), lw=1.8)),
                              ("real_kernel", dict(ls="-", lw=2.2)),
                              ("sim_stock", dict(ls="-", lw=2.2))):
            values = pooled(series, city, source, "rtt_ms")
            if values.size == 0:
                continue
            x, y = _ecdf(values)
            axis.plot(x, y, color=COLOR[source], label=NAME[source], **style)
            stats[source] = values
        _style(axis)
        path = np.median(stats["real_trace"]) if "real_trace" in stats else 0
        axis.set_title(f"{LABEL_CITY.get(city, city)}  ·  {path:.0f} ms path median",
                       fontsize=11.5, color=INK, loc="left")
        axis.set_xlabel("RTT (ms)", fontsize=9.5, color=MUTED)

        if "real_kernel" in stats and "sim_stock" in stats:
            def gap(a, b, q):
                return np.percentile(stats[a], q) - np.percentile(stats[b], q)
            lines = ["                         p50       p90",
                     f"model error (sim − kernel)  {gap('sim_stock', 'real_kernel', 50):+6.1f}  {gap('sim_stock', 'real_kernel', 90):+6.1f} ms"]
            if "real_trace" in stats:
                lines.append(f"emulation gap (kernel − trace)  {gap('real_kernel', 'real_trace', 50):+6.1f}  {gap('real_kernel', 'real_trace', 90):+6.1f} ms")
            axis.text(0.98, 0.06, "\n".join(lines), transform=axis.transAxes, ha="right",
                      va="bottom", fontsize=8.2, color=INK, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                edgecolor=GRID, linewidth=0.8))

    for axis in axes[:, 0]:
        axis.set_ylabel("fraction of seconds", fontsize=9.5, color=MUTED)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.945), fontsize=10)
    fig.text(0.5, 0.012,
             "p50 and p90 differences. Model error = simulated stock minus real Linux BBR on identical capacity "
             "(clean testbed: symmetric RTT, receiver-side throughput). Emulation gap = what tc capacity "
             "emulation cannot reproduce about the satellite link: its jitter and loss.",
             ha="center", fontsize=8.6, style="italic", color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.915))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"saved -> {out}")


def policy_boxes(series, out: Path, calibration: dict | None = None) -> None:
    cities = [c for c in ORDER if any(r["location"] == c for r in series)]
    sources = ["real_kernel", "sim_stock", "sim_quantum", "sim_classical"]
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.4), sharex=True)
    fig.suptitle("Per-second throughput and RTT — policy vs stock on replayed Starlink capacity",
                 fontsize=14, color=INK, y=0.985)

    width = 0.19
    floors = {c: calibration[c]["downlink"]["RTT_min_ms"] for c in cities} if calibration else {c: 0.0 for c in cities}
    for axis, metric, unit in ((axes[0], "throughput_mbps", "throughput (Mbps)"),
                               (axes[1], "rtt_ms", "RTT above propagation floor (ms)")):
        for offset, source in enumerate(sources):
            data, positions = [], []
            for index, city in enumerate(cities):
                values = pooled(series, city, source, metric)
                if metric == "rtt_ms" and values.size:
                    values = values - floors[city]
                if values.size:
                    data.append(values)
                    positions.append(index + (offset - 1.5) * width)
            if not data:
                continue
            box = axis.boxplot(
                data, positions=positions, widths=width * 0.82, patch_artist=True,
                showfliers=False, whis=(5, 95),
                medianprops=dict(color="white", linewidth=1.8),
                whiskerprops=dict(color=COLOR[source], linewidth=1.2),
                capprops=dict(color=COLOR[source], linewidth=1.2),
                boxprops=dict(linewidth=0),
            )
            for patch in box["boxes"]:
                patch.set_facecolor(COLOR[source])
                patch.set_alpha(0.92 if source != "real_kernel" else 0.55)
                if source == "real_kernel":
                    patch.set_hatch("////")
                    patch.set_edgecolor("white")
        _style(axis)
        axis.set_ylabel(unit, fontsize=10, color=MUTED)
    axes[1].set_xticks(range(len(cities)))
    axes[1].set_xticklabels([LABEL_CITY.get(c, c) for c in cities], fontsize=10.5, color=INK)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR[s], alpha=0.92 if s != "real_kernel" else 0.55,
                     hatch="////" if s == "real_kernel" else None, edgecolor="white",
                     label=NAME[s]) for s in sources]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.95), fontsize=10)
    fig.text(0.5, 0.012,
             "IQR box, white = median, whiskers = p5–p95 of 1-s samples over five replayed traces (policy seed 0). "
             "RTT above each path's propagation floor. Hatched = real Linux BBR.",
             ha="center", fontsize=8.6, style="italic", color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"saved -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", type=Path,
                        default=PACKAGE_ROOT.parent / "reports" / "per_second_series.json")
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_ROOT.parent / "figures")
    args = parser.parse_args()
    series = load(args.series)
    fidelity_cdf(series, args.out_dir / "fidelity_rtt_cdf_v14.png")
    from qbbr.env.calibration import load_calibration
    calibration = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants_v14.json")
    policy_boxes(series, args.out_dir / "policy_boxes_v14.png", calibration)
    write_table(series, args.out_dir / "distribution_table_v14.csv")
    return 0


def write_table(series, out: Path) -> None:
    """
    Table view: the relief the palette validator requires, and the numbers a caption can quote
    without anyone reading them off a plot.
    """
    import csv
    sources = ["real_trace", "real_kernel", "sim_stock", "sim_quantum", "sim_classical"]
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["city", "source", "metric", "n_seconds", "p5", "p25", "median", "p75", "p95"])
        for city in ORDER:
            for source in sources:
                for metric in ("throughput_mbps", "rtt_ms"):
                    values = pooled(series, city, source, metric)
                    if values.size == 0:
                        continue
                    q = np.percentile(values, [5, 25, 50, 75, 95])
                    writer.writerow([city, source, metric, values.size, *[round(float(v), 2) for v in q]])
    print(f"saved -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
