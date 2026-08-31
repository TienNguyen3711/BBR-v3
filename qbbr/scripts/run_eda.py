from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.data.parse import ParseDiagnostics
from qbbr.env.calibration import compute_calibration, save_calibration
from qbbr.features.state_builder import compute_state_vector
from qbbr.risk.ptot import compute_risk_features
from qbbr.features.telemetry import extract_telemetry_features
from qbbr.reward.alpha_fair import compute_reward

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SERIES_PATH = OUTPUT_DIR / "state_reward_series.csv.gz"


def report_parse_diagnostics(dataset_root: Path) -> pd.DataFrame:
    print("== Stage 0: parse diagnostics over full corpus ==")
    catalog = build_catalog(dataset_root)
    rows = []
    for record in iter_file_records(catalog):
        from qbbr.data.parse import parse_iperf3_file

        diag: ParseDiagnostics = parse_iperf3_file(record.path).diagnostics
        rows.append(
            {
                "leading_junk": diag.had_leading_junk,
                "trailing_junk": diag.had_trailing_junk,
                "error_field": diag.had_error_field,
                "n_intervals": diag.n_intervals,
            }
        )
    diag_df = pd.DataFrame(rows)
    clean = ~(diag_df["leading_junk"] | diag_df["trailing_junk"] | diag_df["error_field"])
    print(f"  total files:    {len(diag_df)}")
    print(f"  leading junk:   {diag_df['leading_junk'].sum()}")
    print(f"  trailing junk:  {diag_df['trailing_junk'].sum()}")
    print(f"  error field:    {diag_df['error_field'].sum()}")
    print(f"  fully clean:    {clean.sum()}")
    print(f"  interval range: [{diag_df['n_intervals'].min()}, {diag_df['n_intervals'].max()}]")
    return catalog


def run_calibration(dataset_root: Path) -> dict:
    print("\n== Stage 2 calibration: per-location B_max / RTT_min / RTT_max ==")
    calib = compute_calibration(dataset_root)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(calib, CALIBRATION_PATH)
    for location in sorted(calib):
        d = calib[location]["downlink"]
        print(
            f"  {location:10s} downlink: B_max={d['B_max_mbps']:7.1f} Mbps  "
            f"RTT_min={d['RTT_min_ms']:6.1f} ms  RTT_max={d['RTT_max_ms']:6.1f} ms"
        )
    print(f"  saved -> {CALIBRATION_PATH.relative_to(PROJECT_ROOT)}")
    return calib


def run_state_reward_extraction(
    catalog: pd.DataFrame, calibration: dict, limit: int | None, risk_mode: str
) -> pd.DataFrame:
    print(f"\n== Stage 1/2/5: extracting state vectors + reward (risk_mode={risk_mode!r}) ==")
    rows_by_file = []
    records = list(iter_file_records(catalog))
    if limit is not None:
        records = records[:limit]

    t0 = time.time()
    n_skipped = 0
    for i, record in enumerate(records, start=1):
        loc_calib = calibration.get(record.location, {}).get(record.direction)
        if loc_calib is None:
            n_skipped += 1
            continue

        trace = load_trace(record)
        telemetry = extract_telemetry_features(trace.intervals)
        risk = compute_risk_features(telemetry, mode=risk_mode)
        state = compute_state_vector(telemetry, risk, loc_calib)
        reward = compute_reward(telemetry)

        combined = state.copy()
        combined["reward"] = reward.values
        combined["category"] = record.category
        combined["direction"] = record.direction
        combined["location"] = record.location
        combined["cca"] = record.cca
        combined["run"] = record.run
        rows_by_file.append(combined)

        if i % 500 == 0:
            print(f"  ... {i}/{len(records)} files processed")

    series = pd.concat(rows_by_file, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  processed {len(records) - n_skipped} files ({n_skipped} skipped, no calibration) in {elapsed:.1f}s")
    print(f"  total rows (1Hz samples across all runs): {len(series)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series.to_csv(SERIES_PATH, index=False, compression="gzip")
    print(f"  saved -> {SERIES_PATH.relative_to(PROJECT_ROOT)}")
    return series


def report_summary(series: pd.DataFrame) -> None:
    print("\n== Summary: mean reward by location, stock BBR-v3 vs. other CCAs ==")
    bbr = series[series["cca"] == "bbr"]
    others = series[series["cca"] != "bbr"]
    summary = pd.DataFrame(
        {
            "bbr_mean_reward": bbr.groupby("location")["reward"].mean(),
            "other_cca_mean_reward": others.groupby("location")["reward"].mean(),
        }
    )
    print(summary.round(3))

    print("\n== State vector column ranges (sanity check: all must be in [0,1]) ==")
    state_cols = [c for c in series.columns if c.startswith("s") and "_" in c]
    print(series[state_cols].agg(["min", "max"]).round(4))


def maybe_plot(series: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bbr = series[series["cca"] == "bbr"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bbr.boxplot(column="s1_bhat", by="location", ax=axes[0], rot=45)
    axes[0].set_title("s1: normalized B_hat_theta (stock BBR-v3)")
    axes[0].set_xlabel("")
    bbr.boxplot(column="reward", by="location", ax=axes[1], rot=45)
    axes[1].set_title("reward (stock BBR-v3)")
    axes[1].set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    out_path = fig_dir / "bbr_state_reward_by_location.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  saved plot -> {out_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="cap number of files (for fast iteration)")
    parser.add_argument(
        "--risk-mode", default="stub_constant", choices=["stub_constant", "empirical_proxy", "closed_form", "closed_form_dynamic"]
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    catalog = report_parse_diagnostics(args.dataset_root)
    calibration = run_calibration(args.dataset_root)
    series = run_state_reward_extraction(catalog, calibration, args.limit, args.risk_mode)
    report_summary(series)
    if args.plot:
        maybe_plot(series)


if __name__ == "__main__":
    main()
