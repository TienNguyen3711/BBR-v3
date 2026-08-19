from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qbbr.features.telemetry import extract_telemetry_features
from qbbr.reward.alpha_fair import alpha_fair_utility, compute_reward


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
