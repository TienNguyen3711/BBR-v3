from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Matches the reference figure's scope and location ordering exactly.
LOCATIONS = ["Tokyo", "SaoPaulo", "Ohio", "London", "Mumbai", "Sydney"]
CCAS = [("cubic", "Cubic"), ("hybla", "Hybla"), ("vegas", "Vegas"), ("bbr", "BBR-v3")]
CCA_COLORS = {"cubic": "#66c2a5", "hybla": "#fc8d62", "vegas": "#e5c494", "bbr": "#8da0cb"}
DIRECTION = "uplink"  # "dedicated upload"
CATEGORY = "sequential"  # "dedicated" = single isolated flow, not shared/competitive

# (result column, panel title, y-axis unit, raw-interval -> metric transform)
METRICS = [
    ("throughput_mbps", "Throughput", "Mbps", lambda df: df["bits_per_second"] / 1e6),
    ("retransmits", "Retransmissions", "Count", lambda df: df["retransmits"]),
    ("cwnd_mb", "Congestion Window", "MB", lambda df: df["snd_cwnd"] / 1e6),
    ("rwnd_mb", "Receiver Advertised Window", "MB", lambda df: df["snd_wnd"] / 1e6),
    ("rtt_ms", "RTT", "ms", lambda df: df["rtt_ms"]),
    ("rttvar_ms", "RTT Variance", "ms", lambda df: df["rttvar_ms"]),
]


def load_long_form(dataset_root: Path) -> pd.DataFrame:
    catalog = build_catalog(dataset_root)
    rows = []
    for location in LOCATIONS:
        for cca, cca_label in CCAS:
            subset = catalog[
                (catalog["location"] == location)
                & (catalog["cca"] == cca)
                & (catalog["direction"] == DIRECTION)
                & (catalog["category"] == CATEGORY)
            ]
            if subset.empty:
                print(f"  WARNING: no data for {location}/{cca}, skipping")
                continue
            parts = [load_trace(record).intervals for record in iter_file_records(subset)]
            df = pd.concat(parts, ignore_index=True)
            row = {"location": location, "cca": cca, "cca_label": cca_label}
            for key, _label, _unit, extractor in METRICS:
                row[key] = extractor(df).dropna().to_numpy()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(long_form: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    n_cca = len(CCAS)
    group_width = 0.8
    box_width = group_width / n_cca

    for ax, (key, label, unit, _extractor) in zip(axes.flat, METRICS):
        positions, data, colors = [], [], []
        for loc_idx, location in enumerate(LOCATIONS):
            for cca_idx, (cca, _cca_label) in enumerate(CCAS):
                match = long_form[(long_form["location"] == location) & (long_form["cca"] == cca)]
                if match.empty:
                    continue
                positions.append(loc_idx + (cca_idx - (n_cca - 1) / 2) * box_width)
                data.append(match.iloc[0][key])
                colors.append(CCA_COLORS[cca])

        # Standard Tukey whiskers (1.5x IQR), not min/max: a handful of
        # genuine extreme samples exist in this real satellite dataset
        # (e.g. one interval in Tokyo/hybla run1 reads an 11.9s RTT --
        # almost certainly a single corrupted TCP_INFO read, not a real
        # network event) and dominate the axis under min/max whiskers.
        # Outliers are suppressed as points, not excluded from the data.
        bp = ax.boxplot(
            data, positions=positions, widths=box_width * 0.9,
            patch_artist=True, showfliers=False, whis=1.5,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_xticks(range(len(LOCATIONS)))
        ax.set_xticklabels(LOCATIONS, rotation=30, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(label)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CCA_COLORS[cca]) for cca, _ in CCAS]
    labels = [cca_label for _, cca_label in CCAS]
    fig.legend(handles, labels, loc="upper center", ncol=len(CCAS), bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Our-dataset reproduction of base-paper Fig. 8 (dedicated upload, sequential category)",
        y=1.06, fontsize=10, style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print(f"loading {DIRECTION}/{CATEGORY} traces for {len(CCAS)} CCAs x {len(LOCATIONS)} locations...")
    long_form = load_long_form(DEFAULT_DATASET_ROOT)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_comparison(long_form, FIGURES_DIR / "cca_comparison_uplink.png")


if __name__ == "__main__":
    main()
