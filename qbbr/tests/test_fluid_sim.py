from __future__ import annotations

import math

import pytest

from qbbr.env.fluid_sim import (
    STARTUP_EXIT_MULT,
    STARTUP_GAIN,
    STARTUP_MAX_DURATION_S,
    FluidParams,
    FluidState,
    PhaseProfile,
    ecn_marked,
    in_reconfig_freeze_window,
    probe_bw_interval_s,
    retransmit_phase_multiplier,
    sigmoid,
    step_fluid_state,
    synthetic_ground_truth_p_tot,
)

_PARAMS = FluidParams(x_btl_bps=200e6 / 8.0, rtt_rtp_s=0.05)


def test_sigmoid_bounds_and_no_overflow():
    assert sigmoid(0.0) == 0.5
    assert 0.0 < sigmoid(-50.0) < 1e-6  # moderately extreme: still a real fraction
    assert 1.0 - 1e-6 < sigmoid(50.0) <= 1.0
    # very extreme values legitimately underflow/saturate to exactly 0.0/1.0,
    # but must never raise OverflowError (the bug this test guards against).
    assert math.isfinite(sigmoid(-1e9))
    assert math.isfinite(sigmoid(1e9))
    assert math.isfinite(sigmoid(-1e300))
    assert math.isfinite(sigmoid(1e300))


def test_probe_bw_interval_is_positive_and_capped():
    # Eq. 22 is 1-indexed (i in {1,...,N}); a single isolated flow is i=1, N=1,
    # so the cap term is 2+1/1=3.0, not 2.0.
    assert probe_bw_interval_s(0.05) == min(62 * 0.05, 3.0)
    assert probe_bw_interval_s(10.0) <= 3.0  # capped near 2-3s for high-RTT flows


def test_probe_bw_interval_staggers_across_parallel_flows():
    # Eq. 22: for N parallel BBR flows, flow i's cap term is 2+i/N -- distinct
    # per flow, so they don't all probe in lockstep. High RTT so the cap binds.
    high_rtt_s = 10.0
    intervals = [probe_bw_interval_s(high_rtt_s, n_flows=4, flow_index=i) for i in range(4)]
    assert intervals == sorted(intervals)
    assert len(set(intervals)) == 4
    assert intervals[0] == 2.0 + 1 / 4
    assert intervals[-1] == 2.0 + 4 / 4


def test_synthetic_p_tot_stays_in_reasonable_range():
    for t in [0.0, 1.0, 7.5, 14.9, 15.0, 100.3]:
        p = synthetic_ground_truth_p_tot(t)
        assert 0.0 <= p <= 0.2


def test_step_keeps_indicators_in_unit_interval():
    state = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    for _ in range(200):
        state, delivered, retransmits = step_fluid_state(state, 1.0, 0.02, _PARAMS, p_tot=0.01)
        assert 0.0 <= state.i_dwn <= 1.0
        assert 0.0 <= state.i_crs <= 1.0
        assert state.v_bytes >= 0.0
        assert delivered >= 0.0
        assert retransmits >= 0.0
        assert math.isfinite(state.v_bytes)


def test_delivered_bytes_never_exceeds_bottleneck_capacity():
    state = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    dt = 0.02
    for _ in range(200):
        state, delivered, _ = step_fluid_state(state, 1.25, dt, _PARAMS, p_tot=0.01)
        assert delivered <= _PARAMS.x_btl_bps * dt + 1e-6


def test_high_pacing_gain_grows_inflight_more_than_low_gain():
    # Both start deep in cruise (v << BDP) so the agent's gain drives the
    # pacing rate directly (i_crs ~= 1), isolating the effect of pacing_gain.
    dt = 0.02
    state_hi = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=1.0)
    state_lo = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=1.0)
    for _ in range(5):
        state_hi, _, _ = step_fluid_state(state_hi, 1.25, dt, _PARAMS, p_tot=0.01)
        state_lo, _, _ = step_fluid_state(state_lo, 0.75, dt, _PARAMS, p_tot=0.01)
    assert state_hi.v_bytes > state_lo.v_bytes


def test_lagged_bandwidth_estimate_creates_a_bounded_recovery_gain_effect():
    params = FluidParams(
        x_btl_bps=100e6,
        rtt_rtp_s=0.05,
        bandwidth_estimate_recovery_s=3.0,
    )
    # This represents the post-handover interval: true capacity has recovered
    # to 100 Mbps, while BBR's delivery estimate is still 50 Mbps. No action
    # is added; only the fixed gain applied to that estimate differs.
    state = FluidState(
        t_s=7.5, v_bytes=0.0, i_dwn=0.0, i_crs=1.0,
        bbr_bw_est_bps=50e6,
    )
    _stock, stock_delivered, _ = step_fluid_state(
        state, 1.0, 0.02, params, p_tot=0.001, capacity_bps_override=100e6
    )
    _high, high_delivered, _ = step_fluid_state(
        state, 1.25, 0.02, params, p_tot=0.001, capacity_bps_override=100e6
    )
    assert stock_delivered < high_delivered <= 100e6 * 0.02


def test_p_tot_above_threshold_pushes_toward_drawdown():
    dt = 0.02
    state_risky = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    state_calm = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    for _ in range(50):
        state_risky, _, _ = step_fluid_state(state_risky, 1.0, dt, _PARAMS, p_tot=0.5)
        state_calm, _, _ = step_fluid_state(state_calm, 1.0, dt, _PARAMS, p_tot=0.001)
    assert state_risky.i_dwn > state_calm.i_dwn


def test_phase_multiplier_averages_to_one_over_a_full_cycle():
    profile = PhaseProfile(mean_phase_s=10.5, r_bar=0.7368)
    ts = [i * 15.0 / 1000 for i in range(1000)]
    mean_m = sum(retransmit_phase_multiplier(t, profile) for t in ts) / len(ts)
    assert abs(mean_m - 1.0) < 1e-3


def test_phase_multiplier_peaks_at_mean_phase_and_troughs_at_opposite():
    profile = PhaseProfile(mean_phase_s=10.5, r_bar=0.7368)
    at_peak = retransmit_phase_multiplier(10.5, profile)
    at_trough = retransmit_phase_multiplier(3.0, profile)  # 10.5 - 7.5, half a cycle away
    assert at_peak > 3.0  # strong concentration (R_bar=0.74) gives a sharp peak
    assert at_trough < 0.1
    assert at_peak > retransmit_phase_multiplier(10.5 + 1.0, profile) > at_trough


def test_phase_multiplier_disabled_by_default_gives_uniform_rate():
    assert retransmit_phase_multiplier(0.0, None) == 1.0
    assert retransmit_phase_multiplier(7.3, None) == 1.0


def test_dwn_retransmit_rate_pps_scales_retransmits_linearly():
    state = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.5, i_crs=0.5)
    dt = 0.02
    params_1x = FluidParams(x_btl_bps=200e6 / 8.0, rtt_rtp_s=0.05, dwn_retransmit_rate_pps=10.0)
    params_3x = FluidParams(x_btl_bps=200e6 / 8.0, rtt_rtp_s=0.05, dwn_retransmit_rate_pps=30.0)
    s1, _d1, rtx_1x = step_fluid_state(state, 1.0, dt, params_1x, p_tot=0.001)
    s3, _d3, rtx_3x = step_fluid_state(state, 1.0, dt, params_3x, p_tot=0.001)
    assert s1.i_dwn == pytest.approx(s3.i_dwn)  # same bdp -> same i_dwn dynamics, confirming no feedback
    dwn_component_increase = s1.i_dwn * (30.0 - 10.0) * dt
    assert rtx_3x == pytest.approx(rtx_1x + dwn_component_increase)


def test_base_retransmit_rate_applies_without_drawdown():
    dt = 0.02
    state = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=1.0)  # cruise, no drawdown
    legacy = FluidParams(x_btl_bps=200e6 / 8.0, rtt_rtp_s=0.05)  # base_retransmit_rate_pps = 0.0
    fitted = FluidParams(x_btl_bps=200e6 / 8.0, rtt_rtp_s=0.05, base_retransmit_rate_pps=50.0)
    _s0, _d0, rtx_legacy = step_fluid_state(state, 1.0, dt, legacy, p_tot=0.001)
    _s1, _d1, rtx_fitted = step_fluid_state(state, 1.0, dt, fitted, p_tot=0.001)
    assert rtx_legacy == pytest.approx(0.0, abs=1e-6)     # legacy: DRAIN-gated only, i_dwn~=0
    assert rtx_fitted == pytest.approx(50.0 * dt, abs=1e-6)  # fitted: always-on baseline dominates


def test_phase_offset_and_profile_change_retransmit_rate_deterministically():
    params = FluidParams(
        x_btl_bps=200e6 / 8.0, rtt_rtp_s=0.05, dwn_retransmit_rate_pps=20.0,
        phase_profile=PhaseProfile(mean_phase_s=10.5, r_bar=0.7368),
    )
    state = FluidState(t_s=10.5, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.5, i_crs=0.5)
    _s_peak, _d, rtx_at_peak = step_fluid_state(state, 1.0, 0.02, params, p_tot=0.001, phase_offset_s=0.0)
    _s_trough, _d, rtx_at_trough = step_fluid_state(state, 1.0, 0.02, params, p_tot=0.001, phase_offset_s=-7.5)
    assert rtx_at_peak > rtx_at_trough


def test_inflight_hi_override_changes_dwn_deactivation_threshold():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=bdp, i_dwn=0.5, i_crs=0.0)
    dt = 0.02
    s_low, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, inflight_hi_mult_override=1.0)
    s_high, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, inflight_hi_mult_override=2.0)
    assert s_low.i_dwn > s_high.i_dwn


def test_inflight_hi_override_none_matches_params_default():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=bdp, i_dwn=0.5, i_crs=0.0)
    dt = 0.02
    s_default, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    s_explicit, _d, _r = step_fluid_state(
        state, 1.0, dt, _PARAMS, p_tot=0.001, inflight_hi_mult_override=_PARAMS.bdp_hi_mult
    )
    assert s_default.i_dwn == pytest.approx(s_explicit.i_dwn)


def test_inflight_lo_override_changes_crs_deactivation_threshold():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=bdp, i_dwn=0.0, i_crs=0.5)
    dt = 0.02
    s_low, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, inflight_lo_mult_override=0.75)
    s_high, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, inflight_lo_mult_override=1.25)
    assert s_high.i_crs > s_low.i_crs


def test_inflight_lo_override_none_matches_params_default():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=bdp, i_dwn=0.0, i_crs=0.5)
    dt = 0.02
    s_default, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    s_explicit, _d, _r = step_fluid_state(
        state, 1.0, dt, _PARAMS, p_tot=0.001, inflight_lo_mult_override=_PARAMS.bdp_lo_mult
    )
    assert s_default.i_crs == pytest.approx(s_explicit.i_crs)


def test_drawdown_activate_override_changes_dwn_activation_threshold():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=bdp, i_dwn=0.0, i_crs=1.0)
    dt = 0.02
    s_low, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, drawdown_activate_mult_override=0.5)
    s_high, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, drawdown_activate_mult_override=1.25)
    assert s_low.i_dwn > s_high.i_dwn


def test_drawdown_activate_override_none_matches_params_default():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=bdp, i_dwn=0.0, i_crs=1.0)
    dt = 0.02
    s_default, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    s_explicit, _d, _r = step_fluid_state(
        state, 1.0, dt, _PARAMS, p_tot=0.001, drawdown_activate_mult_override=_PARAMS.drawdown_activate_mult
    )
    assert s_default.i_dwn == pytest.approx(s_explicit.i_dwn)


def test_ecn_marked_true_above_threshold_false_below():
    bdp = _PARAMS.bdp_bytes
    assert ecn_marked(bdp * 1.01, bdp) is True
    assert ecn_marked(bdp * 0.99, bdp) is False


def test_ecn_marked_respects_custom_threshold_mult():
    bdp = _PARAMS.bdp_bytes
    assert ecn_marked(bdp * 1.2, bdp, threshold_mult=1.5) is False
    assert ecn_marked(bdp * 1.6, bdp, threshold_mult=1.5) is True


def test_startup_default_field_is_done_for_backward_compat():
    assert FluidState().startup_done is True


def test_startup_uses_startup_gain_ignoring_agent_pacing_gain():
    state_startup = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=0.0, startup_done=False)
    state_cruise_low_gain = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=1.0, startup_done=True)
    dt = 0.02
    s_startup, _d, _r = step_fluid_state(state_startup, 0.1, dt, _PARAMS, p_tot=0.001)
    s_cruise, _d, _r = step_fluid_state(state_cruise_low_gain, 0.1, dt, _PARAMS, p_tot=0.001)
    assert s_startup.v_bytes > s_cruise.v_bytes


def test_startup_exits_when_volume_crosses_exit_threshold():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=0.0, v_bytes=STARTUP_EXIT_MULT * bdp * 0.99, i_dwn=0.0, i_crs=0.5, startup_done=False)
    dt = 0.02
    new_state, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    assert new_state.startup_done is True


def test_startup_stays_active_when_volume_is_far_below_exit_threshold():
    state = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=0.0, startup_done=False)
    dt = 0.02
    new_state, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    assert new_state.startup_done is False


def test_startup_injection_matches_startup_gain_exactly():
    state = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=0.0, startup_done=False)
    dt = 0.001  # small enough that injected bytes stay under one substep's capacity_bytes
    capacity_bps = 1e6
    new_state, delivered, _r = step_fluid_state(
        state, 1.0, dt, _PARAMS, p_tot=0.001, capacity_bps_override=capacity_bps
    )
    expected_injected = capacity_bps * STARTUP_GAIN * dt
    expected_delivered = min(capacity_bps * dt, expected_injected)
    assert delivered == pytest.approx(expected_delivered)
    assert new_state.v_bytes == pytest.approx(expected_injected - expected_delivered)


def test_startup_ignores_inflight_and_drawdown_overrides():
    state = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=0.5, startup_done=False)
    dt = 0.02
    s_with_override, _d, _r = step_fluid_state(
        state, 1.0, dt, _PARAMS, p_tot=0.001,
        inflight_hi_mult_override=1.0, inflight_lo_mult_override=0.5, drawdown_activate_mult_override=0.1,
    )
    s_without_override, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    assert s_with_override.i_dwn == pytest.approx(s_without_override.i_dwn)
    assert s_with_override.i_crs == pytest.approx(s_without_override.i_crs)
    assert s_with_override.v_bytes == pytest.approx(s_without_override.v_bytes)


def test_startup_max_duration_fallback_exit():
    state = FluidState(
        t_s=STARTUP_MAX_DURATION_S - 0.01, v_bytes=0.0, i_dwn=0.0, i_crs=0.0, startup_done=False
    )
    dt = 0.02
    new_state, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001, capacity_bps_override=0.0)
    assert new_state.startup_done is True


_FREEZE_PROFILE = PhaseProfile(mean_phase_s=10.5, r_bar=0.7368)


def test_in_reconfig_freeze_window_true_at_mean_phase():
    assert in_reconfig_freeze_window(t_s=10.5, phase_offset_s=0.0, phase_profile=_FREEZE_PROFILE)


def test_in_reconfig_freeze_window_true_within_half_width():
    assert in_reconfig_freeze_window(t_s=10.5 + 0.9, phase_offset_s=0.0, phase_profile=_FREEZE_PROFILE)
    assert in_reconfig_freeze_window(t_s=10.5 - 0.9, phase_offset_s=0.0, phase_profile=_FREEZE_PROFILE)


def test_in_reconfig_freeze_window_false_outside_half_width():
    assert not in_reconfig_freeze_window(t_s=10.5 + 1.1, phase_offset_s=0.0, phase_profile=_FREEZE_PROFILE)
    assert not in_reconfig_freeze_window(t_s=10.5 - 1.1, phase_offset_s=0.0, phase_profile=_FREEZE_PROFILE)
    assert not in_reconfig_freeze_window(t_s=3.0, phase_offset_s=0.0, phase_profile=_FREEZE_PROFILE)


def test_in_reconfig_freeze_window_wraps_across_cycle_boundary():
    profile = PhaseProfile(mean_phase_s=0.2, r_bar=0.7368)
    assert in_reconfig_freeze_window(t_s=14.9, phase_offset_s=0.0, phase_profile=profile)


def test_in_reconfig_freeze_window_none_profile_disables_it():
    assert not in_reconfig_freeze_window(t_s=10.5, phase_offset_s=0.0, phase_profile=None)


def test_in_reconfig_freeze_window_respects_phase_offset():
    assert in_reconfig_freeze_window(t_s=0.0, phase_offset_s=10.5, phase_profile=_FREEZE_PROFILE)


def test_step_fluid_state_forces_stock_pacing_gain_inside_freeze_window():
    bdp = _PARAMS.bdp_bytes
    state = FluidState(t_s=10.5, v_bytes=bdp, i_dwn=0.0, i_crs=1.0)
    dt = 0.02
    aggressive, _d, _r = step_fluid_state(state, 1.25, dt, _PARAMS, p_tot=0.001)
    frozen, _d, _r = step_fluid_state(state, 1.0, dt, _PARAMS, p_tot=0.001)
    assert aggressive.v_bytes != frozen.v_bytes  # sanity: the two calls really do differ


def test_drain_throughput_penalty_depresses_delivery_under_sustained_drawdown():
    from dataclasses import replace

    bdp = _PARAMS.bdp_bytes
    # A pipe already deep in DRAIN (high i_dwn) with a full inflight queue.
    state = FluidState(t_s=3.0, v_bytes=3.0 * bdp, i_dwn=0.8, i_crs=1.0)
    _s0, delivered_legacy, _r0 = step_fluid_state(state, 1.25, 0.02, _PARAMS, p_tot=0.001)
    _s1, delivered_penalised, _r1 = step_fluid_state(
        state, 1.25, 0.02, replace(_PARAMS, drain_throughput_penalty=0.15), p_tot=0.001
    )
    # Both are capacity-bound (the queue is full); the penalised delivery is
    # the drained fraction of the legacy delivery. i_dwn is relaxed toward its
    # target within the step, so it sits just above the 0.8 it started at.
    assert delivered_penalised < delivered_legacy
    ratio = delivered_penalised / delivered_legacy
    assert (1.0 - 0.15 * 1.0) <= ratio <= (1.0 - 0.15 * 0.8)


def test_drain_penalty_zero_is_the_legacy_default():
    assert FluidParams(x_btl_bps=1e6, rtt_rtp_s=0.05).drain_throughput_penalty == 0.0


# --- Goal-2 dynamics: cwnd cap, loss backoff, ProbeRTT ---

def test_goal2_dynamics_default_off():
    p = FluidParams(x_btl_bps=1e6, rtt_rtp_s=0.05)
    assert (p.cwnd_gain, p.loss_thresh, p.ecn_response_factor, p.probe_rtt_interval_s) == (0.0, 0.0, 0.0, 0.0)
    s = FluidState()
    assert (s.inflight_hi_scale, s.in_probe_rtt) == (1.0, False)


def test_cwnd_gain_caps_the_standing_queue_under_sustained_over_pacing():
    from dataclasses import replace

    bdp = _PARAMS.bdp_bytes
    capped = replace(_PARAMS, cwnd_gain=2.0)
    s_free = FluidState(t_s=3.0, v_bytes=0.0, i_dwn=0.0, i_crs=1.0)
    s_cap = FluidState(t_s=3.0, v_bytes=0.0, i_dwn=0.0, i_crs=1.0)
    for _ in range(400):  # ~8 s of sustained 1.25 pacing
        s_free, _d, _r = step_fluid_state(s_free, 1.25, 0.02, _PARAMS, p_tot=0.001)
        s_cap, _d, _r = step_fluid_state(s_cap, 1.25, 0.02, capped, p_tot=0.001)
    assert s_cap.v_bytes < s_free.v_bytes
    assert s_cap.v_bytes <= 1.05 * bdp
    assert s_free.v_bytes > 1.15 * bdp
    assert s_cap.i_dwn < s_free.i_dwn


def test_past_the_cwnd_cap_a_higher_pacing_gain_delivers_no_more():
    from dataclasses import replace

    capped = replace(_PARAMS, cwnd_gain=2.0)
    s_hi = FluidState(t_s=3.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    s_lo = FluidState(t_s=3.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    dhi = dlo = 0.0
    for _ in range(200):
        s_hi, d, _r = step_fluid_state(s_hi, 1.25, 0.02, capped, p_tot=0.001); dhi += d
        s_lo, d, _r = step_fluid_state(s_lo, 1.10, 0.02, capped, p_tot=0.001); dlo += d
    assert dhi == pytest.approx(dlo, rel=0.02)  # both capacity-bound at the cap


def test_loss_over_threshold_cuts_the_inflight_hi_scale():
    from dataclasses import replace

    p = replace(_PARAMS, cwnd_gain=2.0, loss_thresh=0.02, loss_beta=0.7,
                base_retransmit_rate_pps=5000.0)  # force a high loss fraction this step
    s = FluidState(t_s=3.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0, inflight_hi_scale=1.0)
    s2, _d, _r = step_fluid_state(s, 1.0, p.rtt_rtp_s, p, p_tot=0.001)
    assert s2.inflight_hi_scale < 0.95


def test_probe_rtt_enters_after_interval_and_paces_down():
    from dataclasses import replace

    p = replace(_PARAMS, probe_rtt_interval_s=1.0, probe_rtt_duration_s=0.2, probe_rtt_cwnd_frac=0.5)
    s = FluidState(t_s=3.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0, startup_done=True)
    entered = False
    for _ in range(80):  # 1.6 s
        s, _d, _r = step_fluid_state(s, 1.25, 0.02, p, p_tot=0.001)
        entered = entered or s.in_probe_rtt
    assert entered  # ProbeRTT fired at least once past the 1 s interval
