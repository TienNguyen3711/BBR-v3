from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_sim import probe_bw_interval_s

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
LOCATIONS = ["Sydney", "London", "Mumbai", "Ohio", "SaoPaulo", "Tokyo"]
MSS_BYTES = 1500.0
BBR_HI_GAIN_FRACTION = 0.25  # BBR's 5/4 ProbeBW_UP gain overshoots the BDP by this fraction
N_SHIFT_NULLS = 2000
RNG_SEED = 0


@dataclass
class RunData:
    location: str
    rtx_rate: float  # median retransmits/s for this run
    rtt_min_s: float  # this location's calibrated RTT_min
    b_bps: float  # this run's own median throughput, bytes/s
    t_start: np.ndarray  # seconds since THIS run's flow started
    rtx_series: np.ndarray  # per-second retransmit counts


def _load_runs(dataset_root: Path, calibration: dict) -> list[RunData]:
    catalog = build_catalog(dataset_root)
    bbr = catalog[
        (catalog["cca"] == "bbr") & (catalog["category"] == "sequential") & (catalog["direction"] == "downlink")
    ]
    runs = []
    for record in iter_file_records(bbr):
        trace = load_trace(record)
        rtx = trace.intervals["retransmits"].astype(float).values
        bps = trace.intervals["bits_per_second"].astype(float).values
        t_start = trace.intervals["t_start"].astype(float).values
        if len(rtx) < 30:
            continue
        rtt_min_s = calibration[record.location]["downlink"]["RTT_min_ms"] / 1000.0
        runs.append(
            RunData(
                location=record.location,
                rtx_rate=float(np.median(rtx)),
                rtt_min_s=rtt_min_s,
                b_bps=float(np.median(bps)) / 8.0,
                t_start=t_start,
                rtx_series=rtx,
            )
        )
    return runs


def decisive_check(runs: list[RunData]) -> None:
    print("=" * 70)
    print("Decisive check: locations sharing the SAME predicted probe interval")
    print("=" * 70)
    by_loc: dict[str, list[RunData]] = {}
    for r in runs:
        by_loc.setdefault(r.location, []).append(r)
    for loc in LOCATIONS:
        rs = by_loc.get(loc, [])
        if not rs:
            continue
        interval = probe_bw_interval_s(rs[0].rtt_min_s)
        rates = [r.rtx_rate for r in rs]
        print(f"  {loc:10s} predicted_interval={interval:.3f}s  "
              f"rtx_rate median={np.median(rates):6.2f}/s  range=[{min(rates):.2f}, {max(rates):.2f}]")
    print("  -> Locations at the SAME predicted interval (the saturated 3.0s cap) should show")
    print("     near-identical rates if probing frequency alone drove the spread.")


def variance_decomposition(runs: list[RunData]) -> None:
    print(f"\n{'=' * 70}\nStep 1: variance decomposition (within- vs. between-location)\n{'=' * 70}")
    by_loc: dict[str, list[float]] = {}
    for r in runs:
        by_loc.setdefault(r.location, []).append(r.rtx_rate)

    grand_mean = np.mean([v for vs in by_loc.values() for v in vs])
    loc_means = {loc: np.mean(vs) for loc, vs in by_loc.items()}
    n_per_loc = {loc: len(vs) for loc, vs in by_loc.items()}

    ss_between = sum(n_per_loc[loc] * (loc_means[loc] - grand_mean) ** 2 for loc in by_loc)
    ss_within = sum((v - loc_means[loc]) ** 2 for loc, vs in by_loc.items() for v in vs)
    df_between = len(by_loc) - 1
    df_within = sum(n_per_loc.values()) - len(by_loc)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    icc = ms_between / (ms_between + ms_within) if (ms_between + ms_within) > 0 else float("nan")

    print(f"  MS_between={ms_between:.2f}  MS_within={ms_within:.2f}  ICC (location)={icc:.3f}")
    print(f"  F({df_between},{df_within}) = {ms_between / ms_within:.2f}")
    print("  -> ICC close to 1: variance is almost all between-location (consistent with")
    print("     *some* location-level driver, though not necessarily probing frequency).")
    print("     ICC well below 1: substantial run-to-run variance WITHIN a location, which")
    print("     probing frequency (constant within a location) structurally cannot explain.")


def cluster_robust_regression(runs: list[RunData]) -> None:
    print(f"\n{'=' * 70}\nStep 2: cluster-robust regression, retrans_rate ~ 1/t_pbw + overshoot\n{'=' * 70}")
    y = np.array([r.rtx_rate for r in runs])
    inv_t_pbw = np.array([1.0 / probe_bw_interval_s(r.rtt_min_s) for r in runs])
    overshoot_bytes = np.array([BBR_HI_GAIN_FRACTION * r.b_bps * r.rtt_min_s for r in runs])
    locations = np.array([r.location for r in runs])

    x = np.column_stack([np.ones(len(y)), inv_t_pbw, overshoot_bytes / MSS_BYTES])  # overshoot in packets
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta

    # Cluster-robust ("sandwich") covariance, clustered by location.
    xtx_inv = np.linalg.inv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for loc in np.unique(locations):
        mask = locations == loc
        xg = x[mask]
        ug = resid[mask]
        score = xg.T @ ug
        meat += np.outer(score, score)
    cluster_cov = xtx_inv @ meat @ xtx_inv
    cluster_se = np.sqrt(np.diag(cluster_cov))

    naive_cov = xtx_inv * (resid @ resid) / (len(y) - x.shape[1])
    naive_se = np.sqrt(np.diag(naive_cov))

    names = ["intercept", "1/t_pbw", "overshoot_packets"]
    print(f"  {'term':18s} {'coef':>10s} {'naive_SE':>10s} {'cluster_SE':>12s} {'naive_t':>9s} {'cluster_t':>10s}")
    for name, b, nse, cse in zip(names, beta, naive_se, cluster_se):
        print(f"  {name:18s} {b:10.4f} {nse:10.4f} {cse:12.4f} {b / nse:9.2f} {b / cse:10.2f}")
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    print(f"  R^2={r2:.3f}, n={len(y)}, n_clusters={len(np.unique(locations))}")
    print("  -> Compare naive vs. cluster t-stats: a naive t that looks significant but")
    print("     collapses under clustering confirms the effective-N problem (6 locations,")
    print("     not 60 runs, for these location-constant predictors).")


def _resultant(phases: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total_w = weights.sum()
    if total_w < 1e-9:
        return 0.0, 0.0
    c = float(np.sum(weights * np.cos(phases)))
    s = float(np.sum(weights * np.sin(phases)))
    return np.sqrt(c**2 + s**2) / total_w, np.arctan2(s, c)


def rayleigh_test(angles: np.ndarray) -> dict[str, float]:
    n = len(angles)
    c, s = np.sum(np.cos(angles)), np.sum(np.sin(angles))
    r_bar = np.sqrt(c**2 + s**2) / n
    z = n * r_bar**2
    p = np.exp(-z) * (
        1 + (2 * z - z**2) / (4 * n) - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    )
    return {"n": n, "r_bar": r_bar, "z": z, "p_value": float(np.clip(p, 0.0, 1.0))}


def probing_phase_lock_test(runs: list[RunData]) -> None:
    print(f"\n{'=' * 70}\nStep 3: does retransmit timing phase-lock to the predicted probe interval?\n{'=' * 70}")
    print("(using time relative to flow start -- BBR's ProbeBW timer is per-connection, not")
    print(" wall-clock-anchored like the satellite handover process)")

    groups = {"Sydney (unique interval)": ["Sydney"], "other 5 (shared 3.0s cap)": [l for l in LOCATIONS if l != "Sydney"]}
    for label, locs in groups.items():
        group_runs = [r for r in runs if r.location in locs]
        angles = []
        for r in group_runs:
            interval = probe_bw_interval_s(r.rtt_min_s)
            phase = (r.t_start % interval) / interval * 2 * np.pi
            _rr, angle = _resultant(phase, np.clip(r.rtx_series, 0.0, None))
            if np.clip(r.rtx_series, 0.0, None).sum() > 1e-9:
                angles.append(angle % (2 * np.pi))
        if len(angles) < 2:
            print(f"  {label}: insufficient runs with signal, skipping")
            continue
        result = rayleigh_test(np.array(angles))
        sig = "  <-- significant" if result["p_value"] < 0.05 else "  (not significant)"
        print(
            f"  {label}: n_runs={result['n']}, R_bar={result['r_bar']:.3f}, "
            f"Z={result['z']:.2f}, p={result['p_value']:.4f}{sig}"
        )


def main() -> None:
    calibration = load_calibration(DEFAULT_CALIBRATION_PATH)
    runs = _load_runs(DEFAULT_DATASET_ROOT, calibration)
    print(f"Loaded {len(runs)} sequential downlink BBR runs across {len(LOCATIONS)} locations.\n")

    decisive_check(runs)
    variance_decomposition(runs)
    cluster_robust_regression(runs)
    probing_phase_lock_test(runs)


if __name__ == "__main__":
    main()
