from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import detrend

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.data.parse import parse_iperf3_file
from qbbr.risk.handover import estimate_handover_cadence_s

DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"
MAX_LAG_S = 290
CANDIDATE_LAGS_S = [15, 30, 60, 120, 180, 240]
HANDOVER_CYCLE_S = 15.0
N_SURROGATES = 300
N_SHIFT_NULLS = 2000
SWEEP_PERIODS_S = np.arange(5.0, 30.01, 0.5)
RNG_SEED = 0


@dataclass
class RunSeries:
    wall_s: np.ndarray
    raw: np.ndarray
    detrended: np.ndarray


def _load_run(record, column: str) -> tuple[np.ndarray, np.ndarray] | None:
    trace = load_trace(record)
    series = trace.intervals[column].astype(float).values
    if len(series) < MAX_LAG_S + 10:
        return None
    wall_start = parse_iperf3_file(record.path).raw["start"]["timestamp"]["timesecs"]
    t_start = trace.intervals["t_start"].astype(float).values
    return wall_start + t_start, series


def _load_all_runs(dataset_root: Path, category: str, column: str, direction: str = "downlink") -> list[RunSeries]:
    catalog = build_catalog(dataset_root)
    bbr = catalog[(catalog["cca"] == "bbr") & (catalog["category"] == category) & (catalog["direction"] == direction)]
    runs = []
    for record in iter_file_records(bbr):
        loaded = _load_run(record, column)
        if loaded is None:
            continue
        wall_s, raw = loaded
        runs.append(RunSeries(wall_s=wall_s, raw=raw, detrended=detrend(raw, type="linear")))
    return runs


# ── Step 1 (kept for transparency; established as non-diagnostic under LRD) ─


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray | None:
    n = len(x)
    x = x - x.mean()
    var = x.var()
    if var < 1e-9:
        return None
    return np.array([np.mean(x[: n - lag] * x[lag:]) / var for lag in range(1, max_lag + 1)])


def detrended_acf_with_surrogates(runs: list[RunSeries], max_lag: int = MAX_LAG_S) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(RNG_SEED)
    observed = []
    surrogate_pool = [[] for _ in range(N_SURROGATES)]
    for run in runs:
        ac = _acf(run.detrended, max_lag)
        if ac is None:
            continue
        observed.append(ac)
        for k in range(N_SURROGATES):
            shuffled = rng.permutation(run.detrended)
            ac_s = _acf(shuffled, max_lag)
            if ac_s is not None:
                surrogate_pool[k].append(ac_s)
    observed_mean = np.array(observed).mean(axis=0)
    surrogate_means = np.array([np.array(pool).mean(axis=0) for pool in surrogate_pool if pool])
    return {"observed": observed_mean, "surrogate_hi": np.percentile(surrogate_means, 97.5, axis=0)}


def report_acf(result: dict[str, np.ndarray], label: str) -> None:
    print(f"\n{label}: detrended ACF (established non-diagnostic under LRD -- shown for transparency only)")
    for lag in CANDIDATE_LAGS_S:
        obs, hi = result["observed"][lag - 1], result["surrogate_hi"][lag - 1]
        print(f"  lag={lag:4d}s  ac={obs:+.4f}  surrogate_97.5={hi:.4f}")


# ── Step (a): sample-level resultant vector, CIRCULAR-SHIFT null ──────────


def _resultant(phases: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total_w = weights.sum()
    c = float(np.sum(weights * np.cos(phases)))
    s = float(np.sum(weights * np.sin(phases)))
    return np.sqrt(c**2 + s**2) / total_w, np.arctan2(s, c)


def phase_fold_circular_shift(
    runs: list[RunSeries], cycle_s: float = HANDOVER_CYCLE_S, n_nulls: int = N_SHIFT_NULLS, seed: int = RNG_SEED
) -> dict[str, float]:
    phases_per_run = [(run.wall_s % cycle_s) / cycle_s * 2 * np.pi for run in runs]
    weights_per_run = [np.clip(run.raw, 0.0, None) for run in runs]

    observed_R, observed_angle = _resultant(np.concatenate(phases_per_run), np.concatenate(weights_per_run))
    preferred_phase_s = (observed_angle % (2 * np.pi)) / (2 * np.pi) * cycle_s

    rng = np.random.default_rng(seed)
    null_R = np.empty(n_nulls)
    for k in range(n_nulls):
        shifted = [np.roll(w, rng.integers(0, len(w))) for w in weights_per_run]
        null_R[k], _ = _resultant(np.concatenate(phases_per_run), np.concatenate(shifted))
    p_value = (np.sum(null_R >= observed_R) + 1) / (n_nulls + 1)

    return {
        "resultant_length": observed_R,
        "preferred_phase_s": preferred_phase_s,
        "p_value": p_value,
        "null_95pct": float(np.percentile(null_R, 95)),
    }


def report_phase_fold(result: dict[str, float], label: str) -> bool:
    sig = result["p_value"] < 0.05
    flag = "  <-- significant" if sig else ""
    print(
        f"{label}: R={result['resultant_length']:.4f} (circular-shift null 95th pct={result['null_95pct']:.4f}), "
        f"preferred phase={result['preferred_phase_s']:.1f}s, p={result['p_value']:.4f}{flag}"
    )
    return sig


# ── Step (b): per-run phase estimate + closed-form Rayleigh test ──────────


def per_run_phases(runs: list[RunSeries], cycle_s: float = HANDOVER_CYCLE_S) -> np.ndarray:
    angles = []
    for run in runs:
        phase = (run.wall_s % cycle_s) / cycle_s * 2 * np.pi
        w = np.clip(run.raw, 0.0, None)
        if w.sum() < 1e-9:
            continue  # no signal in this run to estimate a phase from
        _r, angle = _resultant(phase, w)
        angles.append(angle % (2 * np.pi))
    return np.array(angles)


def rayleigh_test(angles: np.ndarray) -> dict[str, float]:
    n = len(angles)
    c, s = np.sum(np.cos(angles)), np.sum(np.sin(angles))
    r_bar = np.sqrt(c**2 + s**2) / n
    z = n * r_bar**2
    # Zar (1999) higher-order approximation to the Rayleigh p-value.
    p = np.exp(-z) * (
        1
        + (2 * z - z**2) / (4 * n)
        - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    )
    mean_phase_s = (np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * HANDOVER_CYCLE_S
    return {"n": n, "r_bar": r_bar, "z": z, "p_value": float(np.clip(p, 0.0, 1.0)), "mean_phase_s": mean_phase_s}


def report_rayleigh(result: dict[str, float], label: str) -> bool:
    sig = result["p_value"] < 0.05
    flag = "  <-- significant" if sig else ""
    print(
        f"{label}: n_runs={result['n']}, R_bar={result['r_bar']:.4f}, Z={result['z']:.2f}, "
        f"mean phase={result['mean_phase_s']:.1f}s, p={result['p_value']:.5f}{flag}"
    )
    return sig


# ── Step (c): period sweep using the run-level Rayleigh statistic ─────────


def period_sweep(runs: list[RunSeries], periods_s: np.ndarray) -> np.ndarray:
    z_values = []
    for period in periods_s:
        angles = per_run_phases(runs, cycle_s=period)
        z_values.append(rayleigh_test(angles)["z"] if len(angles) >= 2 else 0.0)
    return np.array(z_values)


def plot_phase_circle(run_phases_by_regime: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    colors = {"sequential": "tab:blue", "competitive": "tab:orange"}
    for regime, angles in run_phases_by_regime.items():
        ax.scatter(angles, np.ones_like(angles), c=colors.get(regime, "gray"), label=regime, alpha=0.7, s=40)
    ax.set_theta_zero_location("N")
    ax.set_rticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 15, endpoint=False))
    ax.set_xticklabels([f"{t:.0f}s" for t in np.linspace(0, HANDOVER_CYCLE_S, 15, endpoint=False)])
    ax.set_title("Per-run preferred phase within the 15s cycle (retransmits)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved -> {out_path.relative_to(PROJECT_ROOT)}")


def plot_period_sweep(sweep_by_regime: dict[str, np.ndarray], periods_s: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for regime, z_values in sweep_by_regime.items():
        ax.plot(periods_s, z_values, marker="o", markersize=2, label=regime)
    ax.axvline(15.0, color="red", linestyle="--", alpha=0.6, label="15s (hypothesized)")
    ax.axvline(7.5, color="gray", linestyle=":", alpha=0.4, label="7.5s (sub-harmonic)")
    ax.axvline(30.0, color="gray", linestyle=":", alpha=0.4, label="30s (harmonic)")
    ax.set_xlabel("Assumed cycle period (s)")
    ax.set_ylabel("Rayleigh Z (run-level phase concentration)")
    ax.set_title("Period sweep: retransmits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved -> {out_path.relative_to(PROJECT_ROOT)}")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    geometric = estimate_handover_cadence_s()
    geo_median_s = float(np.median([e.median_pass_s for e in geometric]))
    print(f"Geometric single-satellite visibility cadence (median across 5 shells): {geo_median_s:.0f}s (reference only)")

    grid_results = {}
    run_phases_by_regime = {}
    sweep_by_regime = {}
    for category in ("sequential", "competitive", "pooled"):
        for column in ("retransmits", "rtt_ms"):
            print(f"\n{'=' * 70}\n{category} / {column}\n{'=' * 70}")
            if category == "pooled":
                runs = _load_all_runs(DEFAULT_DATASET_ROOT, "sequential", column)
                runs += _load_all_runs(DEFAULT_DATASET_ROOT, "competitive", column)
            else:
                runs = _load_all_runs(DEFAULT_DATASET_ROOT, category, column)
            if not runs:
                print("  no usable runs, skipping")
                continue

            report_acf(detrended_acf_with_surrogates(runs), f"{category}/{column} ACF")
            fold_sig = report_phase_fold(phase_fold_circular_shift(runs), f"{category}/{column} phase-fold (circular-shift null)")

            angles = per_run_phases(runs)
            rayleigh_sig = report_rayleigh(rayleigh_test(angles), f"{category}/{column} per-run Rayleigh")
            grid_results[(category, column)] = {"phase_fold_significant": fold_sig, "rayleigh_significant": rayleigh_sig}

            if category in ("sequential", "competitive") and column == "retransmits":
                run_phases_by_regime[category] = angles
                sweep_by_regime[category] = period_sweep(runs, SWEEP_PERIODS_S)

    print(f"\n{'=' * 70}\nAdjudication summary (circular-shift null + run-level Rayleigh, the trustworthy tests)\n{'=' * 70}")
    print(f"{'subset':12s} {'metric':12s} {'phase-fold sig':>16s} {'Rayleigh sig':>14s}")
    for (category, column), r in grid_results.items():
        print(f"{category:12s} {column:12s} {str(r['phase_fold_significant']):>16s} {str(r['rayleigh_significant']):>14s}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if run_phases_by_regime:
        plot_phase_circle(run_phases_by_regime, FIGURES_DIR / "handover_phase_circle.png")
    if sweep_by_regime:
        plot_period_sweep(sweep_by_regime, SWEEP_PERIODS_S, FIGURES_DIR / "handover_period_sweep.png")


if __name__ == "__main__":
    main()
