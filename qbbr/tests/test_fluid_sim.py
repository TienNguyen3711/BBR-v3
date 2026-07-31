from __future__ import annotations

import math

from qbbr.env.fluid_sim import (
    FluidParams,
    FluidState,
    probe_bw_interval_s,
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
    assert probe_bw_interval_s(0.05) == min(62 * 0.05, 2.0)
    assert probe_bw_interval_s(10.0) <= 2.0 + 1.0  # capped near 2-3s for high-RTT flows


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


def test_p_tot_above_threshold_pushes_toward_drawdown():
    dt = 0.02
    state_risky = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    state_calm = FluidState(t_s=0.0, v_bytes=_PARAMS.bdp_bytes, i_dwn=0.0, i_crs=1.0)
    for _ in range(50):
        state_risky, _, _ = step_fluid_state(state_risky, 1.0, dt, _PARAMS, p_tot=0.5)
        state_calm, _, _ = step_fluid_state(state_calm, 1.0, dt, _PARAMS, p_tot=0.001)
    assert state_risky.i_dwn > state_calm.i_dwn
