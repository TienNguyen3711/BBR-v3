"""Contract tests for the counterfactual difference reward."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.reward.difference import compute_difference_reward

CALIBRATION = Path(__file__).resolve().parent.parent / "data" / "calibrated" / "per_location_constants.json"
STOCK_ACTION = 2


def _env(**kwargs) -> FluidSimEnv:
    defaults = dict(
        location="London", direction="downlink", calibration=load_calibration(CALIBRATION),
        episode_s=60.0, reward_mode="difference", reward_kwargs={"delta": 2.0, "beta": 0.5},
    )
    defaults.update(kwargs)
    return FluidSimEnv(**defaults)


def test_stock_policy_scores_exactly_zero():
    """The load-bearing invariant: differencing must cancel the exogenous term.

    An agent that plays stock every step traces the counterfactual exactly, so
    every per-decision reward must be identically zero -- not merely small. Any
    drift here means the twin is not seeing the same forcing, and the reward is
    silently measuring weather again (the v5-v9 failure this reward exists to
    remove).
    """
    env = _env()
    env.reset(seed=7)
    rewards, done = [], False
    while not done:
        _state, reward, done, _info = env.step(STOCK_ACTION)
        rewards.append(reward)
    assert rewards, "episode produced no decisions"
    assert np.max(np.abs(rewards)) == 0.0


def test_non_stock_policy_scores_non_zero():
    """A departure from stock must actually register, or the test above is vacuous."""
    env = _env()
    env.reset(seed=7)
    rewards, done = [], False
    while not done:
        _state, reward, done, _info = env.step(4)  # pacing_gain 1.25
        rewards.append(reward)
    assert np.max(np.abs(rewards)) > 0.0


def test_difference_mode_requires_a_seed():
    """Without a seed the twin draws a different phase offset, so the exogenous
    term would not cancel and the reward would be quietly wrong."""
    env = _env()
    with pytest.raises(ValueError, match="requires reset"):
        env.reset(seed=None)


def test_baseline_is_cached_per_seed():
    env = _env()
    env.reset(seed=7)
    cached = env._difference_baselines[7]
    env.reset(seed=7)
    assert env._difference_baselines[7] is cached, "baseline recomputed for a seen seed"
    env.reset(seed=8)
    assert set(env._difference_baselines) == {7, 8}


def test_penalties_have_the_declared_sign():
    baseline = {"bits_per_second": [10e6], "rtt_ms": [50.0], "retransmits": [4.0]}
    neutral = {"bits_per_second": 10e6, "rtt_ms": 50.0, "retransmits": 4.0}
    assert compute_difference_reward(neutral, baseline, 0) == pytest.approx(0.0)

    faster = dict(neutral, bits_per_second=12e6)
    assert compute_difference_reward(faster, baseline, 0) > 0.0
    slower_rtt = dict(neutral, rtt_ms=70.0)
    assert compute_difference_reward(slower_rtt, baseline, 0) < 0.0
    more_loss = dict(neutral, retransmits=9.0)
    assert compute_difference_reward(more_loss, baseline, 0) < 0.0

    # delta and beta must actually scale their terms.
    assert (compute_difference_reward(slower_rtt, baseline, 0, delta=4.0)
            < compute_difference_reward(slower_rtt, baseline, 0, delta=1.0))
    assert (compute_difference_reward(more_loss, baseline, 0, beta=1.0)
            < compute_difference_reward(more_loss, baseline, 0, beta=0.5))


def test_index_past_baseline_end_is_clamped():
    """A trajectory may outrun the cached stock rollout by a decision; the tail
    is compared against stock's last decision rather than crashing."""
    baseline = {"bits_per_second": [10e6], "rtt_ms": [50.0], "retransmits": [4.0]}
    neutral = {"bits_per_second": 10e6, "rtt_ms": 50.0, "retransmits": 4.0}
    assert compute_difference_reward(neutral, baseline, 99) == pytest.approx(0.0)


def test_queue_delay_form_prices_queue_not_path_length():
    from qbbr.reward.difference import compute_difference_reward
    base = {"bits_per_second": [100e6], "rtt_ms": [258.0], "retransmits": [0.0]}
    row = {"bits_per_second": 100e6, "rtt_ms": 261.0, "rtt_base_ms": 258.0, "retransmits": 0.0}
    total = compute_difference_reward(row, base, 0, delta=1.0)
    queue = compute_difference_reward(row, base, 0, delta=1.0, delay_form="queue", queue_floor_ms=5.0)
    assert abs(total + math.log(261.0 / 258.0)) < 1e-9
    assert abs(queue + math.log(8.0 / 5.0)) < 1e-9
    # The same 3 ms of queue costs the same on a 30 ms path.
    short = dict(row, rtt_ms=33.0, rtt_base_ms=30.0)
    short_base = dict(base, rtt_ms=[30.0])
    assert abs(compute_difference_reward(short, short_base, 0, delta=1.0, delay_form="queue") - queue) < 1e-9
