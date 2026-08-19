from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace

_LOW_PCT = 1.0
_HIGH_PCT = 99.0
_MIN_UTILIZATION_FRACTION = 0.05  # safety floor: see utilization_fraction docstring note
_REFERENCE_DWN_RATE_PPS = 1.0  # arbitrary; retransmits scale exactly linearly in this parameter
_CALIBRATION_EPISODE_S = 300.0
_CALIBRATION_N_EPISODES = 3  # cheap redundancy check; the fluid model is deterministic given phase_offset_s,
_STOCK_ACTION_INDEX = 2  # configs/action_pacing_gain.yaml: levels[2] == 1.0 (matches validate_simulator.py)

_QUEUE_DELAY_SEEDS = (7001, 7002, 7003)  # FIXED across iterations: unlike calibrate_dwn_retransmit_rate's
_QUEUE_DELAY_BRACKET = (0.1, 4.0)  # empirically confirmed to bracket every location's real RTT target
_QUEUE_DELAY_REL_TOL = 0.02  # |median_sim - median_trace| / median_trace
_QUEUE_DELAY_MAX_ITERS = 5


def compute_calibration(
    dataset_root: str | Path, cca: str = "bbr", category: str = "sequential"
) -> dict[str, dict[str, dict[str, float]]]:
    catalog = build_catalog(dataset_root)
    subset = catalog[(catalog["cca"] == cca) & (catalog["category"] == category)]
    if subset.empty:
        raise ValueError(f"no traces found for cca={cca!r}, category={category!r} under {dataset_root}")

    result: dict[str, dict[str, dict[str, float]]] = {}
    for (location, direction), group in subset.groupby(["location", "direction"]):
        bps_parts, rtt_parts = [], []
        for record in iter_file_records(group):
            trace = load_trace(record)
            bps_parts.append(trace.intervals["bits_per_second"].dropna())
            rtt_parts.append(trace.intervals["rtt_ms"].dropna())
        bps = pd.concat(bps_parts)
        rtt = pd.concat(rtt_parts)

        b_max = np.percentile(bps, _HIGH_PCT)
        utilization_fraction = max(float(np.median(bps) / b_max), _MIN_UTILIZATION_FRACTION)

        result.setdefault(location, {})[direction] = {
            "B_max_mbps": float(b_max / 1e6),
            "RTT_min_ms": float(np.percentile(rtt, _LOW_PCT)),
            "RTT_max_ms": float(np.percentile(rtt, _HIGH_PCT)),
            "utilization_fraction": utilization_fraction,
        }
    return result


def calibrate_dwn_retransmit_rate(
    dataset_root: str | Path,
    base_calibration: dict[str, dict[str, dict[str, float]]],
    cca: str = "bbr",
    category: str = "sequential",
) -> dict[str, dict[str, dict[str, float]]]:

    from qbbr.env.fluid_env import FluidSimEnv
    from qbbr.eval.metrics import real_cca_distribution_stats

    def _median_rate(calibration: dict, dwn_rate: float) -> float:
        run_calibration = {
            location: {direction: {**calibration[location][direction], "dwn_retransmit_rate_pps": dwn_rate}}
        }
        rates = []
        for _episode in range(_CALIBRATION_N_EPISODES):
            env = FluidSimEnv(location, direction, run_calibration, episode_s=_CALIBRATION_EPISODE_S)
            env.reset()
            done = False
            while not done:
                _state, _reward, done, info = env.step(_STOCK_ACTION_INDEX)
                rates.append(info["retransmits"] / info["t_dec_s"])
        return float(np.median(rates))

    result: dict[str, dict[str, dict[str, float]]] = {}
    for location, directions in base_calibration.items():
        for direction in directions:
            real_rate = real_cca_distribution_stats(dataset_root, location, direction, cca, category=category)[
                "retransmits_per_s_median"
            ]

            baseline_rate = _median_rate(base_calibration, 0.0)
            reference_rate = _median_rate(base_calibration, _REFERENCE_DWN_RATE_PPS)
            slope = reference_rate - baseline_rate  # d(total_rate)/d(dwn_retransmit_rate_pps)

            if slope > 1e-9:
                calibrated_rate = max(0.0, (real_rate - baseline_rate) / slope)
            else:
                calibrated_rate = _REFERENCE_DWN_RATE_PPS
            result.setdefault(location, {})[direction] = {"dwn_retransmit_rate_pps": calibrated_rate}
    return result


def calibrate_drawdown_activate_mult(
    dataset_root: str | Path,
    base_calibration: dict[str, dict[str, dict[str, float]]],
    cca: str = "bbr",
    category: str = "sequential",
) -> dict[str, dict[str, dict[str, float]]]:
    from qbbr.env.fluid_env import FluidSimEnv
    from qbbr.eval.metrics import real_cca_distribution_stats

    def _median_rtt(calibration: dict, activate_mult: float) -> float:
        run_calibration = {
            location: {direction: {**calibration[location][direction], "drawdown_activate_mult": activate_mult}}
        }
        rtts = []
        for seed in _QUEUE_DELAY_SEEDS:
            env = FluidSimEnv(location, direction, run_calibration, episode_s=_CALIBRATION_EPISODE_S)
            env.reset(seed=seed)
            done = False
            while not done:
                _state, _reward, done, info = env.step(_STOCK_ACTION_INDEX)
                rtts.append(info["rtt_ms"])
        return float(np.median(rtts))

    result: dict[str, dict[str, dict[str, float]]] = {}
    for location, directions in base_calibration.items():
        for direction in directions:
            real_rtt = real_cca_distribution_stats(dataset_root, location, direction, cca, category=category)[
                "rtt_ms_median"
            ]

            x_lo, x_hi = _QUEUE_DELAY_BRACKET
            f_lo = _median_rtt(base_calibration, x_lo) - real_rtt
            f_hi = _median_rtt(base_calibration, x_hi) - real_rtt
            if f_lo > 0.0 or f_hi < 0.0:
                raise ValueError(
                    f"{location}/{direction}: drawdown_activate_mult bracket {_QUEUE_DELAY_BRACKET} does not "
                    f"contain the real RTT target ({real_rtt:.1f} ms); sim range at the bracket ends is "
                    f"[{f_lo + real_rtt:.1f}, {f_hi + real_rtt:.1f}] ms -- widen the bracket."
                )

            x_new = x_hi
            for _iteration in range(_QUEUE_DELAY_MAX_ITERS):
                x_new = x_hi - f_hi * (x_hi - x_lo) / (f_hi - f_lo)
                f_new = _median_rtt(base_calibration, x_new) - real_rtt
                if abs(f_new) / real_rtt < _QUEUE_DELAY_REL_TOL:
                    break
                if f_new > 0.0:
                    x_hi, f_hi = x_new, f_new
                else:
                    x_lo, f_lo = x_new, f_new

            result.setdefault(location, {})[direction] = {"drawdown_activate_mult": x_new}
    return result


def save_calibration(calibration: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(calibration, indent=2, sort_keys=True))


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
