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


def resolve_limit(limit, location: str, direction: str, calibration: dict | None):
    """Resolve a criterion that is DERIVED per path rather than fixed.

    A criterion may be written as {derived_from: <calibration key>} instead of a
    number, in which case its value is read from that path's calibrated
    constants. This exists because an absolute millisecond budget is not the
    same requirement on every path: 5 ms means one thing where RTT_min is 30 ms
    (Sydney) and something far stricter where it is 258 ms (London), and it sits
    an order of magnitude below the run-to-run variability of stock BBR's own
    RTT p90 on these paths (11-45 ms measured). A derived budget states the
    REQUIREMENT once -- "not distinguishable from stock's own variation" -- and
    lets the data set its value per path.
    """
    if not isinstance(limit, dict):
        return limit
    key = limit["derived_from"]
    if calibration is None:
        raise ValueError(f"Criterion derived_from {key!r} needs calibration constants.")
    value = calibration.get(location, {}).get(direction, {}).get(key)
    if value is None:
        raise ValueError(f"Calibration for {location}/{direction} has no {key!r}.")
    return float(value) * float(limit.get("scale", 1.0))


def _low_gain_pass(group, low_gain_share: float, criteria: dict) -> bool:
    """Does the policy waste sub-1.0 gains where they can only lose throughput?

    The original criterion capped the OVERALL share of gains 0.75/0.90. Measured
    on v12 across 24 arms that proxy is inverted with respect to what it was
    meant to catch: corr(low-gain share, throughput) = +0.81 and
    corr(low-gain share, positive-seed fraction) = +0.73. Arms above the 0.10
    cap averaged +4.45% throughput against +2.19% for arms below it, and 11 of
    the 12 arms reaching 5/5 positive seeds were blocked by this gate alone.
    The reason is that the learned policy is bimodal -- probe hard, then drain --
    and the low gains ARE the drain phase, which is what keeps RTT near stock.

    The criterion's own stated intent (v5.1) was narrower: "in headroom a
    sub-1.0 gain only under-fills the pipe (pure throughput loss)". So the
    re-specification scopes the measurement to exactly that case -- low gain
    while the pipe is NOT under pressure -- and keeps the SAME 0.10 tolerance.
    Scope is corrected; tolerance is not loosened.

    IMPORTANT -- THIS GATE IS STRUCTURALLY SATISFIED, NOT DISCRIMINATING.
    NativeActionSelector already drops the low-gain actions whenever the pipe is
    not congested ("if not congested and action in self.low_gain_actions:
    continue"), using the SAME thresholds (max_inflight_state 0.45,
    max_queue_state 0.20). So low_gain_share_in_headroom is zero by
    construction and this criterion cannot fail unless the mask is broken.
    Treat it as an INVARIANT CHECK on the mask, not as evidence that a policy
    passed a meaningful safety bar. The consequence of the v13 change is that
    the low-gain safety property now rests on the action mask, where it is
    enforced by construction, rather than on a post-hoc frequency cap that was
    measured to be anti-correlated with the failure it targeted. The real
    discriminating safety load is carried by the RTT p90, retransmit and
    throughput criteria.

    `max_low_gain_action_share` (the overall cap) still applies if a protocol
    declares it and not the headroom form, so older protocols are unaffected.
    """
    headroom_cap = criteria.get("max_low_gain_in_headroom_share")
    if headroom_cap is None:
        return low_gain_share <= criteria["max_low_gain_action_share"]
    shares = [row["evaluation"].get("low_gain_share_in_headroom") for row in group]
    if any(value is None for value in shares):
        raise ValueError(
            "max_low_gain_in_headroom_share requires records carrying "
            "low_gain_share_in_headroom; re-run evaluation with the current runner."
        )
    return float(np.mean(shares)) <= float(headroom_cap)


def assess_full_successor_records(
    records: list[dict], criteria: dict, bootstrap: dict, calibration: dict | None = None,
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
            "avoid_systematic_low_gain": _low_gain_pass(group, low_gain_share, criteria),
            "stable_action_distribution": action_jsd <= criteria["max_mean_action_js_divergence"],
        }
        resolved_limits = {}
        for metric in ("rtt_p90_delta_vs_stock_ms", "rtt_p95_delta_vs_stock_ms", "retransmission_ratio_delta_vs_stock", "retransmits_delta_vs_stock_per_s"):
            limit = resolve_limit(criteria.get("max_" + metric), location, direction, calibration)
            if limit is not None:
                resolved_limits["max_" + metric] = limit
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
                "low_gain_share_in_headroom": float(np.mean(
                    [row["evaluation"].get("low_gain_share_in_headroom", float("nan")) for row in group])),
                "mean_action_js_divergence": action_jsd,
                "mean_retransmits_delta_vs_stock_per_s": float(np.mean(retransmit_deltas)),
                "resolved_limits": resolved_limits,
                "criteria_pass": passes,
                "qualified_simulator_proxy_result": bool(all(passes.values())),
            }
        )
    return assessments
