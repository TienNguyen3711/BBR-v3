from __future__ import annotations

import numpy as np
import pytest

from qbbr.env.multi_flow_env import MultiFlowFluidEnv


def test_reset_returns_valid_8dim_state(sample_calibration):
    env = MultiFlowFluidEnv("Sydney", "downlink", sample_calibration)
    s0 = env.reset()
    assert s0.shape == (8,)
    assert np.isfinite(s0).all()
    assert (s0 >= 0.0).all() and (s0 <= 1.0).all()


def test_step_returns_well_formed_transition_with_per_flow_throughput(sample_calibration):
    env = MultiFlowFluidEnv("Sydney", "downlink", sample_calibration, episode_s=30.0)
    env.reset()
    s, r, done, info = env.step(2)
    assert s.shape == (8,)
    assert (s >= 0.0).all() and (s <= 1.0).all()
    assert np.isfinite(r)
    assert set(info["flow_throughput_bps"].keys()) == {"qbbr", "cubic", "vegas", "hybla"}
    for v in info["flow_throughput_bps"].values():
        assert v >= 0.0
        assert np.isfinite(v)


def test_episode_terminates_at_episode_length(sample_calibration):
    env = MultiFlowFluidEnv("Sydney", "downlink", sample_calibration, episode_s=0.5)
    env.reset()
    done = False
    n_steps = 0
    while not done and n_steps < 10_000:
        _, _, done, _ = env.step(2)
        n_steps += 1
    assert done
    assert n_steps > 0


def test_step_before_reset_raises(sample_calibration):
    env = MultiFlowFluidEnv("Sydney", "downlink", sample_calibration)
    with pytest.raises(RuntimeError):
        env.step(2)


def test_custom_competing_cca_subset_is_respected(sample_calibration):
    env = MultiFlowFluidEnv("Sydney", "downlink", sample_calibration, episode_s=5.0, competing_ccas=("cubic",))
    env.reset()
    _, _, _, info = env.step(2)
    assert set(info["flow_throughput_bps"].keys()) == {"qbbr", "cubic"}


def test_agent_flow_dominates_over_a_short_run(sample_calibration):
    env = MultiFlowFluidEnv("Sydney", "downlink", sample_calibration, episode_s=10.0)
    env.reset()
    done = False
    last_info = None
    while not done:
        _, _, done, last_info = env.step(2)
    assert last_info["flow_throughput_bps"]["qbbr"] > 0.0


def test_runs_for_every_calibrated_location_direction(sample_calibration):
    for location, directions in sample_calibration.items():
        for direction in directions:
            env = MultiFlowFluidEnv(location, direction, sample_calibration, episode_s=0.2)
            s0 = env.reset()
            assert np.isfinite(s0).all(), (location, direction)
            s, r, done, info = env.step(2)
            assert np.isfinite(s).all(), (location, direction)
            assert np.isfinite(r), (location, direction)
