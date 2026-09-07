import numpy as np

from qbbr.eval.successor_protocol import assess_full_successor_records, bootstrap_median_ci


def _record(seed: int, delta: float, low_gain_share: float = 0.0) -> dict:
    return {
        "location": "London",
        "direction": "downlink",
        "core": "quantum",
        "seed": seed,
        "evaluation": {
            "throughput_delta_vs_stock_pct": delta,
            "retransmits_delta_vs_stock_per_s": 0.1,
            "action_shares": {"0": low_gain_share, "1": 0.0, "2": 1.0 - low_gain_share, "3": 0.0, "4": 0.0},
        },
    }


def test_bootstrap_interval_is_reproducible() -> None:
    first = bootstrap_median_ci([1.0, 2.0, 3.0], resamples=100, seed=3)
    second = bootstrap_median_ci([1.0, 2.0, 3.0], resamples=100, seed=3)
    assert first == second
    assert first["lower"] <= first["point_estimate"] <= first["upper"]


def test_full_gate_requires_majority_of_independent_training_seeds() -> None:
    criteria = {
        "min_median_throughput_delta_vs_stock_pct": 0.0,
        "min_bootstrap_ci_lower_pct": 0.0,
        "min_positive_training_seed_fraction": 0.70,
        "max_low_gain_action_share": 0.10,
        "max_mean_action_js_divergence": 0.20,
    }
    bootstrap = {"resamples": 100, "confidence": 0.95, "seed": 3}
    assessment = assess_full_successor_records(
        [_record(0, 0.0), _record(1, 0.0), _record(2, 5.0)], criteria, bootstrap,
    )[0]
    assert not assessment["criteria_pass"]["positive_training_seed_fraction"]
    assert not assessment["qualified_simulator_proxy_result"]


def test_full_gate_passes_consistently_positive_policy() -> None:
    criteria = {
        "min_median_throughput_delta_vs_stock_pct": 0.0,
        "min_bootstrap_ci_lower_pct": 0.0,
        "min_positive_training_seed_fraction": 0.70,
        "max_low_gain_action_share": 0.10,
        "max_mean_action_js_divergence": 0.20,
    }
    bootstrap = {"resamples": 100, "confidence": 0.95, "seed": 3}
    assessment = assess_full_successor_records(
        [_record(0, 1.0), _record(1, 2.0), _record(2, 3.0)], criteria, bootstrap,
    )[0]
    assert np.isclose(assessment["positive_training_seed_fraction"], 1.0)
    assert assessment["qualified_simulator_proxy_result"]
