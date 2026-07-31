from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.risk.handover import estimate_handover_cadence_s

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
MAX_LAG_S = 290
NOISE_FLOOR_LAG_RANGE = (99, 200)  # lags 100-200s: no plausible periodic mechanism here
CANDIDATE_LAGS_S = [15, 30, 60, 120, 180, 240]


def pooled_autocorrelation(dataset_root: Path, column: str) -> np.ndarray:
    catalog = build_catalog(dataset_root)
    bbr = catalog[catalog["cca"] == "bbr"]

    autocorrs = []
    for record in iter_file_records(bbr):
        trace = load_trace(record)
        series = trace.intervals[column].astype(float).values
        n = len(series)
        if n < MAX_LAG_S + 10:
            continue
        x = series - series.mean()
        var = x.var()
        if var < 1e-9:
            continue
        ac = np.array([np.mean(x[: n - lag] * x[lag:]) / var for lag in range(1, MAX_LAG_S + 1)])
        autocorrs.append(ac)

    return np.array(autocorrs).mean(axis=0)


def report_signal_strength(mean_ac: np.ndarray, label: str, geometric_cadence_s: float) -> None:
    lo, hi = NOISE_FLOOR_LAG_RANGE
    noise_std = mean_ac[lo:hi].std()

    print(f"\n{label}: noise floor std (lag {lo+1}-{hi}s) = {noise_std:.4f}")
    for lag in CANDIDATE_LAGS_S:
        std_units = mean_ac[lag - 1] / noise_std
        print(f"  lag={lag:4d}s  ac={mean_ac[lag-1]:+.4f}  ({std_units:+.2f} std above noise floor)")

    geo_lag = int(round(geometric_cadence_s))
    if geo_lag <= MAX_LAG_S:
        std_units = mean_ac[geo_lag - 1] / noise_std
        print(f"  lag={geo_lag:4d}s (geometric estimate)  ac={mean_ac[geo_lag-1]:+.4f}  ({std_units:+.2f} std)")


def main() -> None:
    geometric = estimate_handover_cadence_s()
    geo_median_s = float(np.median([e.median_pass_s for e in geometric]))
    print(f"Geometric single-satellite visibility cadence (median across 5 shells): {geo_median_s:.0f}s")

    for column in ("retransmits", "rtt_ms"):
        mean_ac = pooled_autocorrelation(DEFAULT_DATASET_ROOT, column)
        report_signal_strength(mean_ac, column, geo_median_s)

    print(
        "\nInterpretation: a candidate lag with |std units| >> 1 shows real periodicity at that "
        "cadence; near 0 shows none. This is a report for main.tex's validation gate, not a "
        "pass/fail assertion -- read the printed numbers."
    )


if __name__ == "__main__":
    main()
