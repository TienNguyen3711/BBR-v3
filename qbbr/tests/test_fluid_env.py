from __future__ import annotations

import math

import numpy as np
import pytest

from qbbr.env.fluid_env import FluidSimEnv, minrtt_100_decision_interval_s


def test_minrtt_100_rule_matches_main_tex_eq1():
    # minRTT <= 100ms -> 2*minRTT
    assert minrtt_100_decision_interval_s(50.0) == pytest.approx(0.1)
    # minRTT > 100ms -> 100ms + minRTT
    assert minrtt_100_decision_interval_s(250.0) == pytest.approx(0.35)
    # continuous at the boundary
    assert minrtt_100_decision_interval_s(100.0) == pytest.approx(0.2)


def test_reset_returns_valid_6dim_state(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    s0 = env.reset()
    assert s0.shape == (6,)
    assert np.isfinite(s0).all()
    assert (s0 >= 0.0).all() and (s0 <= 1.0).all()


def test_action_space_size_matches_pacing_gain_config(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    assert env.action_space_size == 5
    assert env.observation_dim == 6


def test_step_returns_well_formed_transition(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env.reset()
    s, r, done, info = env.step(2)  # index 2 -> pacing_gain 1.0
    assert s.shape == (6,)
    assert (s >= 0.0).all() and (s <= 1.0).all()
    assert math.isfinite(r)
    assert isinstance(done, bool) or isinstance(done, np.bool_)
    assert info["pacing_gain"] == 1.0
    assert info["t_dec_s"] > 0.0


def test_episode_terminates_at_episode_length(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.5)
    env.reset()
    done = False
    n_steps = 0
    while not done and n_steps < 10_000:
        _, _, done, _ = env.step(2)
        n_steps += 1
    assert done
    assert n_steps > 0


def test_out_of_range_action_raises(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env.reset()
    with pytest.raises(IndexError):
        env.step(99)


def test_reward_kwargs_reach_compute_reward(sample_calibration):
    # beta=0 should always give an equal-or-higher reward than a heavy loss
    # penalty, since it removes the only term that can ever subtract for
    # loss-rate increases (delta/alpha unaffected).
    env_default = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.5)
    env_no_loss_penalty = FluidSimEnv(
        "Sydney", "downlink", sample_calibration, episode_s=0.5, reward_kwargs={"beta": 0.0}
    )
    env_default.reset()
    env_no_loss_penalty.reset()

    _, r_default, _, _ = env_default.step(2)
    _, r_no_penalty, _, _ = env_no_loss_penalty.step(2)
    assert r_no_penalty >= r_default


def test_runs_for_every_calibrated_location_direction(sample_calibration):
    for location, directions in sample_calibration.items():
        for direction in directions:
            env = FluidSimEnv(location, direction, sample_calibration, episode_s=0.2)
            s0 = env.reset()
            assert np.isfinite(s0).all(), (location, direction)
            s, r, done, info = env.step(2)
            assert np.isfinite(s).all(), (location, direction)
            assert math.isfinite(r), (location, direction)
