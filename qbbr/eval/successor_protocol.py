"""Evidence gates shared by primary QA2C and recurrent-QDQN ablation protocols."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def mean_js_divergence(action_shares: list[list[float]]) -> float:
    distributions = np.asarray(action_shares, dtype=float)
    centre = np.mean(distributions, axis=0)
    epsilon = np.finfo(float).tiny
    values = []
    for distribution in distributions:
        midpoint = (distribution + centre) / 2.0
        left = np.sum(np.where(distribution > 0, distribution * np.log((distribution + epsilon) / (midpoint + epsilon)), 0.0))
        right = np.sum(np.where(centre > 0, centre * np.log((centre + epsilon) / (midpoint + epsilon)), 0.0))
        values.append((left + right) / 2.0)
    return float(np.mean(values))


def bootstrap_median_ci(
    values: list[float], resamples: int = 10_000, confidence: float = 0.95, seed: int = 20260905,
) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("Bootstrap confidence intervals require at least two independent training seeds.")
    if resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration.")
    source = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    sample_indices = rng.randint(0, len(source), size=(resamples, len(source)))
    medians = np.median(source[sample_indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point_estimate": float(np.median(source)),
        "lower": float(np.quantile(medians, alpha)),
        "upper": float(np.quantile(medians, 1.0 - alpha)),
        "confidence": float(confidence),
        "resamples": int(resamples),
    }


def assess_full_successor_records(
    records: list[dict], criteria: dict, bootstrap: dict,
) -> list[dict]:
    """Assess independent training seeds; reporting metrics never alter reward."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["location"], record["direction"], record["core"])].append(record)
    assessments = []
    for (location, direction, core), group in sorted(groups.items()):
        deltas = [row["evaluation"]["throughput_delta_vs_stock_pct"] for row in group]
        shares = [[row["evaluation"]["action_shares"][str(action)] for action in range(5)] for row in group]
        retransmit_deltas = [row["evaluation"]["retransmits_delta_vs_stock_per_s"] for row in group]
        ci = bootstrap_median_ci(
            deltas, resamples=bootstrap["resamples"], confidence=bootstrap["confidence"],
            seed=bootstrap["seed"],
        )
        positive_fraction = float(np.mean(np.asarray(deltas) > 0.0))
        mean_shares = np.mean(np.asarray(shares), axis=0)
        low_gain_share = float(mean_shares[0] + mean_shares[1])
        action_jsd = mean_js_divergence(shares)
        passes = {
            "median_throughput_vs_stock": ci["point_estimate"] >= criteria["min_median_throughput_delta_vs_stock_pct"],
            "bootstrap_lower_bound": ci["lower"] >= criteria["min_bootstrap_ci_lower_pct"],
            "positive_training_seed_fraction": positive_fraction >= criteria["min_positive_training_seed_fraction"],
            "avoid_systematic_low_gain": low_gain_share <= criteria["max_low_gain_action_share"],
            "stable_action_distribution": action_jsd <= criteria["max_mean_action_js_divergence"],
        }
        for metric in ("rtt_p90_delta_vs_stock_ms", "rtt_p95_delta_vs_stock_ms", "retransmission_ratio_delta_vs_stock", "retransmits_delta_vs_stock_per_s"):
            limit = criteria.get("max_" + metric)
            if limit is not None:
                passes[metric] = all(np.isfinite(row["evaluation"].get(metric, float("nan"))) and row["evaluation"][metric] <= limit for row in group)
        assessments.append(
            {
                "location": location,
                "direction": direction,
                "core": core,
                "independent_training_seed_count": len(group),
                "per_seed_throughput_delta_vs_stock_pct": deltas,
                "positive_training_seed_fraction": positive_fraction,
                "bootstrap_median_throughput_delta_vs_stock_pct": ci,
                "mean_action_shares": {str(i): float(mean_shares[i]) for i in range(5)},
                "low_gain_action_075_share": float(mean_shares[0]),
                "low_gain_action_share": low_gain_share,
                "mean_action_js_divergence": action_jsd,
                "mean_retransmits_delta_vs_stock_per_s": float(np.mean(retransmit_deltas)),
                "criteria_pass": passes,
                "qualified_simulator_proxy_result": bool(all(passes.values())),
            }
        )
    return assessments
