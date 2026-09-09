from __future__ import annotations

import copy
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


def test_reset_returns_valid_7dim_state(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    s0 = env.reset()
    assert s0.shape == (7,)
    assert np.isfinite(s0).all()
    assert (s0 >= 0.0).all() and (s0 <= 1.0).all()


def test_action_space_size_matches_pacing_gain_config(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    assert env.action_space_size == 5
    assert env.observation_dim == 7


def test_step_returns_well_formed_transition(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env.reset()
    s, r, done, info = env.step(2)  # index 2 -> pacing_gain 1.0
    assert s.shape == (7,)
    assert (s >= 0.0).all() and (s <= 1.0).all()
    assert math.isfinite(r)
    assert isinstance(done, bool) or isinstance(done, np.bool_)
    assert info["pacing_gain"] == 1.0
    assert info["t_dec_s"] > 0.0


def test_reconfig_freeze_window_overrides_aggressive_action(sample_calibration):
    # Force the env's clock to sit exactly on the calibrated retransmit
    # concentration phase (mean_phase_s=10.5, see fluid_env._HANDOVER_PHASE_PROFILE)
    # with zero phase offset, so the whole decision interval falls inside
    # the freeze window. An aggressive pacing_gain=1.25 (index 4) should
    # then produce an identical transition to the neutral pacing_gain=1.0
    # (index 2), since fluid_env.py substitutes pacing_gain=1.0 for every
    # substep inside the freeze window regardless of the chosen action.
    from qbbr.env.fluid_sim import FluidState

    env_neutral = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env_neutral.reset()
    env_neutral._phase_offset_s = 0.0
    env_neutral._state = FluidState(t_s=10.5, v_bytes=env_neutral.params.bdp_bytes, i_dwn=0.0, i_crs=1.0)

    env_aggressive = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env_aggressive.reset()
    env_aggressive._phase_offset_s = 0.0
    env_aggressive._state = FluidState(t_s=10.5, v_bytes=env_aggressive.params.bdp_bytes, i_dwn=0.0, i_crs=1.0)

    _s_n, _r_n, _d_n, info_neutral = env_neutral.step(2)  # pacing_gain=1.0
    _s_a, _r_a, _d_a, info_aggressive = env_aggressive.step(4)  # pacing_gain=1.25, should be frozen to 1.0

    assert info_neutral["delivered_bytes"] == pytest.approx(info_aggressive["delivered_bytes"])
    assert info_neutral["retransmits"] == pytest.approx(info_aggressive["retransmits"])


def test_reconfig_freeze_window_does_not_affect_actions_outside_it(sample_calibration):
    # Same setup but starting far from mean_phase_s (t_s=3.0, >1s half-width
    # away): an aggressive pacing_gain should now actually differ from
    # neutral, confirming the freeze is localized rather than a global
    # no-op bug that always forces pacing_gain=1.0.
    from qbbr.env.fluid_sim import FluidState

    env_neutral = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env_neutral.reset()
    env_neutral._phase_offset_s = 0.0
    env_neutral._state = FluidState(t_s=3.0, v_bytes=env_neutral.params.bdp_bytes, i_dwn=0.0, i_crs=1.0)

    env_aggressive = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env_aggressive.reset()
    env_aggressive._phase_offset_s = 0.0
    env_aggressive._state = FluidState(t_s=3.0, v_bytes=env_aggressive.params.bdp_bytes, i_dwn=0.0, i_crs=1.0)

    env_neutral.step(2)  # pacing_gain=1.0
    env_aggressive.step(4)  # pacing_gain=1.25, not frozen here

    # delivered_bytes alone can saturate at capacity regardless of
    # pacing_gain (Sydney's short RTT means the queue is already full at
    # v_bytes=bdp), masking the difference -- v_bytes (queue occupancy)
    # is the more sensitive signal: a higher pacing_gain injects more
    # bytes than capacity can drain, growing the queue even when delivered
    # throughput is already capacity-capped.
    assert env_aggressive._state.v_bytes != pytest.approx(env_neutral._state.v_bytes)


def test_ema_smoothing_damps_first_step_then_converges(sample_calibration):
    # First step choosing an aggressive pacing_gain=1.25 (action index 4)
    # should apply something between the stock start (1.0) and 1.25, not
    # 1.25 outright (_EMA_ALPHA=0.5: effective = 0.5*1.25 + 0.5*1.0 =
    # 1.125). Repeating the same action should move the effective value
    # closer to 1.25 each step, converging rather than jumping.
    from qbbr.env.fluid_sim import FluidState

    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    env.reset(seed=0)
    env._phase_offset_s = 0.0  # t_s stays well outside the freeze window (mean_phase_s=10.5, +-1s) for these 2 steps
    env._state = FluidState(t_s=0.0, v_bytes=env.params.bdp_bytes, i_dwn=0.0, i_crs=1.0)

    _s1, _r1, _d1, info1 = env.step(4)  # pacing_gain=1.25
    assert info1["pacing_gain"] == pytest.approx(1.125)
    _s2, _r2, _d2, info2 = env.step(4)
    assert info2["pacing_gain"] == pytest.approx(0.5 * 1.25 + 0.5 * 1.125)
    assert info2["pacing_gain"] > info1["pacing_gain"]


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


def test_throughput_only_reward_ignores_legacy_penalty_kwargs(sample_calibration):
    env = FluidSimEnv(
        "Sydney",
        "downlink",
        sample_calibration,
        episode_s=0.5,
        reward_mode="throughput_only",
        reward_kwargs={"alpha": 2.0, "delta": 100.0, "beta": 100.0},
    )
    env.reset()
    _state, reward, _done, info = env.step(2)
    assert reward == pytest.approx(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)


def test_bbr_action_mask_keeps_only_stock_choice_during_startup(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.5)
    env.reset()
    assert env.allowed_action_indices() == (2,)


def test_ablate_s7_freezes_the_phase_feature_at_its_neutral_midpoint(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration, ablate_s7=True)
    s0 = env.reset()
    assert s0[6] == pytest.approx(0.5)  # s7_reconfig_phase is _STATE_COLS[6]
    s, _r, _done, _info = env.step(2)
    assert s[6] == pytest.approx(0.5)


def test_steady_inflight_bdp_frac_raises_reported_inflight_in_cruise(sample_calibration):
    # Stage 1b: with the calibrated pipe term the reported inflight/BDP carries
    # a persistent ~frac component instead of tracking only the (near-zero
    # under the Tier-1 dynamics) queue backlog. Test the pipe contribution.
    overrides = {"bandwidth_estimate_recovery_s": 3.0, "drain_throughput_penalty": 0.15}

    def cruise_vob(cal):
        env = FluidSimEnv("Sydney", "downlink", cal, episode_s=60.0,
                          dynamics_overrides=overrides, probe_bw_phase_gate=True)
        env.reset(seed=0)
        vob, done = [], False
        while not done:
            _s, _r, done, _info = env.step(2)  # stock gain
            vob.append(float(env._history[-1]["v_over_bdp"]))
        return float(np.median(vob[len(vob) // 2:]))  # past STARTUP

    legacy = cruise_vob(sample_calibration)  # steady_inflight_bdp_frac defaults to 0.0
    cal = copy.deepcopy(sample_calibration)
    cal["Sydney"]["downlink"]["steady_inflight_bdp_frac"] = 0.8
    fitted = cruise_vob(cal)

    assert legacy < 0.2                       # legacy: near-empty pipe under Tier-1 dynamics
    assert fitted - legacy == pytest.approx(0.8, abs=0.25)  # pipe term (shrunk while draining)


def test_ablate_s7_defaults_to_off_preserving_live_phase_tracking(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration)
    assert env.ablate_s7 is False


def test_runs_for_every_calibrated_location_direction(sample_calibration):
    for location, directions in sample_calibration.items():
        for direction in directions:
            env = FluidSimEnv(location, direction, sample_calibration, episode_s=0.2)
            s0 = env.reset()
            assert np.isfinite(s0).all(), (location, direction)
            s, r, done, info = env.step(2)
            assert np.isfinite(s).all(), (location, direction)
            assert math.isfinite(r), (location, direction)


def test_probe_bw_phase_gate_offers_only_stock_outside_cruise(sample_calibration):
    from dataclasses import replace

    env = FluidSimEnv("London", "downlink", sample_calibration, probe_bw_phase_gate=True)
    env.reset(seed=0)
    stock = 2
    env._state = replace(env._state, startup_done=True, i_crs=0.2)
    assert env.allowed_action_indices() == (stock,)
    env._state = replace(env._state, i_crs=0.9)
    assert env.allowed_action_indices() == tuple(range(env.action_space_size))


def test_probe_bw_phase_gate_defaults_off(sample_calibration):
    from dataclasses import replace

    env = FluidSimEnv("London", "downlink", sample_calibration)  # gate not set
    env.reset(seed=0)
    env._state = replace(env._state, startup_done=True, i_crs=0.1)
    assert env.allowed_action_indices() == tuple(range(env.action_space_size))
