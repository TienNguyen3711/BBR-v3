from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbbr.features.telemetry import extract_telemetry_features
from qbbr.reward.alpha_fair import alpha_fair_utility, compute_multi_flow_reward, compute_reward


def test_alpha_fair_utility_log_at_alpha_one():
    x = pd.Series([1.0, 10.0, 100.0])
    u = alpha_fair_utility(x, alpha=1.0)
    assert np.allclose(u.values, np.log(x.values + 1e-6))


def test_alpha_fair_utility_monotonic_in_throughput():
    x = pd.Series([1.0, 10.0, 100.0])
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        u = alpha_fair_utility(x, alpha=alpha)
        assert u.is_monotonic_increasing, alpha


def test_reward_finite_and_bounded_on_real_trace(sample_trace):
    tel = extract_telemetry_features(sample_trace.intervals)
    reward = compute_reward(tel)

    assert np.isfinite(reward).all()
    # loose sanity bound: no single-interval reward should be able to swing
    # by more than this on real per-second Starlink telemetry.
    assert (reward.abs() < 1000).all()


def test_eps_rtx_prevents_blowup_from_zero_baseline():
    tel = pd.DataFrame(
        {
            "bits_per_second": [1e8, 1e8],
            "rtt_ms": [50.0, 50.0],
            "rtt_base_ms": [50.0, 50.0],
            "retransmits": [0.0, 100.0],  # 0 -> 100 retransmits between intervals
        }
    )
    reward = compute_reward(tel)
    assert np.isfinite(reward).all()
    assert reward.abs().max() < 1000


def test_default_gamma_zero_ignores_ecn_mark_fraction_column():
    # DEFAULT_GAMMA=0.0 must reproduce the exact pre-ECN reward even when an
    # ecn_mark_fraction column is present (e.g. from FluidSimEnv), so every
    # caller not explicitly opting into gamma>0 is unaffected.
    base = pd.DataFrame({
        "bits_per_second": [1e8, 1e8], "rtt_ms": [50.0, 50.0], "rtt_base_ms": [50.0, 50.0],
        "retransmits": [0.0, 1.0],
    })
    with_ecn_col = base.assign(ecn_mark_fraction=[0.0, 1.0])
    assert np.allclose(compute_reward(base).values, compute_reward(with_ecn_col).values)


def test_positive_gamma_penalizes_ecn_marked_intervals():
    tel = pd.DataFrame({
        "bits_per_second": [1e8, 1e8], "rtt_ms": [50.0, 50.0], "rtt_base_ms": [50.0, 50.0],
        "retransmits": [0.0, 0.0], "ecn_mark_fraction": [0.0, 1.0],
    })
    reward_off = compute_reward(tel, gamma=0.0)
    reward_on = compute_reward(tel, gamma=0.5)
    assert reward_on.iloc[0] == pytest.approx(reward_off.iloc[0])  # ecn_mark_fraction=0 -> no penalty
    assert reward_on.iloc[1] < reward_off.iloc[1]  # ecn_mark_fraction=1 -> penalized
    assert reward_off.iloc[1] - reward_on.iloc[1] == pytest.approx(0.5)


def _multi_flow_tel(bps=1e8, rtt=50.0, rtt_base=50.0, rtx_prev=0.0, rtx=0.0):
    return pd.DataFrame({
        "bits_per_second": [bps, bps], "rtt_ms": [rtt, rtt], "rtt_base_ms": [rtt_base, rtt_base],
        "retransmits": [rtx_prev, rtx],
    })


def test_multi_flow_reward_matches_single_flow_reward_with_one_flow():
    # a single-element flow list must reduce to exactly compute_reward's own
    # single-flow utility term, since compute_multi_flow_reward is meant as
    # its multi-flow generalization, not a different formula.
    tel = _multi_flow_tel(bps=1e8, rtx_prev=0.0, rtx=5.0)
    own_mbps = tel["bits_per_second"].iloc[-1] / 1e6
    single = compute_reward(tel).iloc[-1]
    multi = compute_multi_flow_reward(tel, [own_mbps])
    assert multi == pytest.approx(single)


def test_multi_flow_reward_sums_utility_across_flows():
    tel = _multi_flow_tel(bps=1e8)
    alpha = 1.0
    reward_one = compute_multi_flow_reward(tel, [50.0], alpha=alpha, delta=0.0, beta=0.0)
    reward_two = compute_multi_flow_reward(tel, [50.0, 20.0], alpha=alpha, delta=0.0, beta=0.0)
    # with delay/loss terms zeroed out, the only difference is the added
    # flow's own alpha-fair utility term.
    expected_added_utility = float(alpha_fair_utility(pd.Series([20.0]), alpha=alpha).iloc[0])
    assert reward_two - reward_one == pytest.approx(expected_added_utility, abs=1e-4)


def test_multi_flow_reward_finite_and_bounded():
    tel = _multi_flow_tel(bps=1e8, rtx_prev=0.0, rtx=100.0)
    reward = compute_multi_flow_reward(tel, [50.0, 0.1, 30.0, 1.0])
    assert np.isfinite(reward)
    assert abs(reward) < 1000
