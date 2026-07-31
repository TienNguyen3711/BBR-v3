from __future__ import annotations

import math

import numpy as np
import pytest

from qbbr.eval.metrics import (
    alpha_fair_efficiency_ratio,
    alpha_fair_optimal_allocation,
    alpha_fair_utility_sum,
    min_per_flow_throughput,
)


def test_optimal_allocation_is_equal_split_summing_to_capacity():
    x_star = alpha_fair_optimal_allocation(total_capacity=100.0, n=4)
    assert np.allclose(x_star, 25.0)
    assert x_star.sum() == pytest.approx(100.0)


def test_equal_achieved_allocation_gives_ratio_one_for_every_alpha():
    x_achieved = [25.0, 25.0, 25.0, 25.0]
    for alpha in (0.0, 0.5, 1.0, 2.0, math.inf):
        assert alpha_fair_efficiency_ratio(x_achieved, alpha) == pytest.approx(1.0, abs=1e-6)


def test_alpha_zero_is_indifferent_to_split_since_total_is_what_matters():
    # U_0(x) = x (linear): any split summing to the same capacity gives the
    # same Phi_0, so rho_0 = 1 regardless of how unfair the split is.
    assert alpha_fair_efficiency_ratio([90.0, 10.0], alpha=0.0) == pytest.approx(1.0, abs=1e-6)
    assert alpha_fair_efficiency_ratio([99.0, 1.0], alpha=0.0) == pytest.approx(1.0, abs=1e-6)


def test_alpha_one_penalizes_unfair_splits_and_stays_bounded():
    ratio = alpha_fair_efficiency_ratio([90.0, 10.0], alpha=1.0)
    assert 0.0 < ratio < 1.0


def test_alpha_infinity_is_governed_by_the_minimum_and_stays_bounded():
    # min(90,10)=10 vs the equal-split minimum (50,50)->50: rho_inf = 10/50.
    ratio = alpha_fair_efficiency_ratio([90.0, 10.0], alpha=math.inf)
    assert ratio == pytest.approx(10.0 / 50.0, rel=1e-6)
    assert 0.0 < ratio <= 1.0


def test_min_per_flow_throughput():
    assert min_per_flow_throughput([90.0, 10.0, 50.0]) == 10.0


def test_alpha_two_known_caveat_can_exceed_one():
    # Exact worked example from qbbr.eval.metrics' module docstring: this
    # documents a real property of main.tex's literal formula (Phi_alpha is
    # negative-valued for alpha>1), not a bug -- see that docstring.
    phi_star = alpha_fair_utility_sum([5.0, 5.0], alpha=2.0)
    phi_achieved = alpha_fair_utility_sum([9.0, 1.0], alpha=2.0)
    assert phi_star == pytest.approx(-0.4, abs=1e-4)
    assert phi_achieved == pytest.approx(-1.1111, abs=1e-3)

    ratio = alpha_fair_efficiency_ratio([9.0, 1.0], alpha=2.0)
    assert ratio == pytest.approx(2.7775, abs=1e-2)
    assert ratio > 1.0  # outside main.tex's stated (0, 1] range -- the documented caveat
