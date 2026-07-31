from __future__ import annotations

import pytest

from qbbr.risk.atmospheric import (
    compute_atmospheric_failure_probability,
    elevation_time_weights,
    p_at_threshold_single_elevation,
)

# Coarse/short propagation purely for test speed.
_FAST_KWARGS = dict(n_sats_per_shell=3, duration_hours=3.0, step_s=15.0)


def test_elevation_time_weights_sum_to_one():
    weights = elevation_time_weights(**_FAST_KWARGS)
    assert len(weights) > 0
    total = sum(w for _el, w in weights)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_elevation_time_weights_favor_low_elevation():
    # grazing passes near the visibility cutoff are geometrically far more
    # common than directly-overhead ones (Eq. 8's "nonuniform distribution").
    weights = elevation_time_weights(**_FAST_KWARGS)
    lowest_bin_weight = weights[0][1]
    highest_bin_weight = weights[-1][1]
    assert lowest_bin_weight > highest_bin_weight


def test_p_at_threshold_decreases_as_threshold_increases():
    p_low = p_at_threshold_single_elevation(1.0, elevation_deg=40.0)
    p_high = p_at_threshold_single_elevation(10.0, elevation_deg=40.0)
    assert 0.0 <= p_high < p_low <= 50.0


def test_p_at_threshold_is_higher_at_lower_elevation():
    p_grazing = p_at_threshold_single_elevation(3.0, elevation_deg=27.5)
    p_high = p_at_threshold_single_elevation(3.0, elevation_deg=70.0)
    assert p_grazing > p_high


def test_compute_atmospheric_failure_probability_in_unit_interval():
    weights = elevation_time_weights(**_FAST_KWARGS)
    for M in [1.0, 3.0, 10.0]:
        p = compute_atmospheric_failure_probability(M, elevation_weights=weights)
        assert 0.0 <= p <= 1.0


def test_compute_atmospheric_failure_probability_decreases_with_threshold():
    weights = elevation_time_weights(**_FAST_KWARGS)
    p_1db = compute_atmospheric_failure_probability(1.0, elevation_weights=weights)
    p_10db = compute_atmospheric_failure_probability(10.0, elevation_weights=weights)
    assert p_10db < p_1db
