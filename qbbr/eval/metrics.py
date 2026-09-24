from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from qbbr.reward.alpha_fair import EPSILON, alpha_fair_utility


def alpha_fair_optimal_allocation(total_capacity: float, n: int) -> np.ndarray:
    return np.full(n, total_capacity / n)


def alpha_fair_utility_sum(x: Sequence[float], alpha: float, eps: float = EPSILON) -> float:

    return float(alpha_fair_utility(pd.Series(x, dtype=float), alpha=alpha, eps=eps).sum())


def min_per_flow_throughput(x_achieved: Sequence[float]) -> float:
    return float(np.min(x_achieved))


def alpha_fair_efficiency_ratio(x_achieved: Sequence[float], alpha: float, eps: float = EPSILON) -> float:
    """WARNING: mathematically unbounded for ANY alpha>1, not just alpha=2."""
    x_achieved = np.asarray(x_achieved, dtype=float)
    n = len(x_achieved)
    total_capacity = float(x_achieved.sum())
    x_star = alpha_fair_optimal_allocation(total_capacity, n)

    if math.isinf(alpha):
        return min_per_flow_throughput(x_achieved) / min_per_flow_throughput(x_star)

    phi_achieved = alpha_fair_utility_sum(x_achieved, alpha, eps)
    phi_star = alpha_fair_utility_sum(x_star, alpha, eps)
    return phi_achieved / phi_star


def real_cca_distribution_stats(
    dataset_root: str | Path, location: str, direction: str, cca: str, category: str = "sequential"
) -> dict[str, float]:
    """Real per-CCA throughput/RTT/retransmit distribution from the raw dataset."""
    from qbbr.data.catalog import build_catalog, iter_file_records
    from qbbr.data.loader import load_trace

    catalog = build_catalog(dataset_root)
    subset = catalog[
        (catalog["cca"] == cca)
        & (catalog["location"] == location)
        & (catalog["direction"] == direction)
        & (catalog["category"] == category)
    ]
    if subset.empty:
        raise ValueError(
            f"no traces found for cca={cca!r}, location={location!r}, "
            f"direction={direction!r}, category={category!r}"
        )

    bps_all, rtt_all, rtx_all = [], [], []
    for record in iter_file_records(subset):
        trace = load_trace(record)
        bps_all.append(trace.intervals["bits_per_second"])
        rtt_all.append(trace.intervals["rtt_ms"])
        rtx_all.append(trace.intervals["retransmits"])  # ~1s intervals -> already a per-second count
    bps = pd.concat(bps_all)
    rtt = pd.concat(rtt_all)
    rtx = pd.concat(rtx_all)
    return {
        "throughput_mbps_median": float(bps.median() / 1e6),
        "throughput_mbps_iqr": float(bps.quantile(0.75) / 1e6 - bps.quantile(0.25) / 1e6),
        "rtt_ms_median": float(rtt.median()),
        "rtt_ms_iqr": float(rtt.quantile(0.75) - rtt.quantile(0.25)),
        "rtt_ms_p95": float(rtt.quantile(0.95)),
        "retransmits_per_s_median": float(rtx.median()),
        "retransmits_per_s_iqr": float(rtx.quantile(0.75) - rtx.quantile(0.25)),
    }


def real_cca_per_run_medians(
    dataset_root: str | Path, location: str, direction: str, cca: str, category: str = "sequential"
) -> dict[str, list[float]]:
    """
    One median per TRACE FILE (not pooled across files, unlike real_cca_distribution_stats
    above).
    """
    from qbbr.data.catalog import build_catalog, iter_file_records
    from qbbr.data.loader import load_trace

    catalog = build_catalog(dataset_root)
    subset = catalog[
        (catalog["cca"] == cca)
        & (catalog["location"] == location)
        & (catalog["direction"] == direction)
        & (catalog["category"] == category)
    ]
    if subset.empty:
        raise ValueError(
            f"no traces found for cca={cca!r}, location={location!r}, "
            f"direction={direction!r}, category={category!r}"
        )

    medians: dict[str, list[float]] = {"throughput_mbps": [], "rtt_ms": [], "retransmits_per_s": []}
    for record in iter_file_records(subset):
        trace = load_trace(record)
        medians["throughput_mbps"].append(float(trace.intervals["bits_per_second"].median() / 1e6))
        medians["rtt_ms"].append(float(trace.intervals["rtt_ms"].median()))
        medians["retransmits_per_s"].append(float(trace.intervals["retransmits"].median()))
    return medians
