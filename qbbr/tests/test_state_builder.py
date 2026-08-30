from __future__ import annotations

import pandas as pd
import pytest

from qbbr.features.state_builder import T_ORBIT_MIN, compute_state_vector
from qbbr.risk.ptot import compute_risk_features
from qbbr.features.telemetry import extract_telemetry_features

_STATE_COLS = [
    "s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue",
    "s5_handover_eta", "s6_p_tot", "s7_reconfig_phase",
]


def test_state_vector_in_unit_interval(sample_trace, sample_calibration):
    tel = extract_telemetry_features(sample_trace.intervals)
    risk = compute_risk_features(tel, mode="stub_constant")
    calib = sample_calibration["Sydney"]["downlink"]
    state = compute_state_vector(tel, risk, calib)

    for col in _STATE_COLS:
        assert (state[col] >= 0.0).all(), col
        assert (state[col] <= 1.0).all(), col


def test_stub_risk_features_normalize_to_half(sample_trace, sample_calibration):
    tel = extract_telemetry_features(sample_trace.intervals)
    risk = compute_risk_features(tel, mode="stub_constant")
    calib = sample_calibration["Sydney"]["downlink"]
    state = compute_state_vector(tel, risk, calib)

    assert (state["s5_handover_eta"] == 0.5).all()
    assert (state["s6_p_tot"] == 0.5).all()


def test_t_orbit_constant_is_positive():
    assert T_ORBIT_MIN > 0


def _minimal_state(t_start_values, cycle_s=15.0, mean_phase_s=10.5):
    telemetry = pd.DataFrame({
        "t_start": t_start_values,
        "b_hat_mbps": [10.0] * len(t_start_values),
        "rtt_ms": [50.0] * len(t_start_values),
        "v_over_bdp": [1.0] * len(t_start_values),
        "q_packets": [5.0] * len(t_start_values),
    })
    risk = pd.DataFrame({
        "delta_t_ho_min": [1.0] * len(t_start_values),
        "p_tot": [0.01] * len(t_start_values),
    })
    calibration = {"B_max_mbps": 100.0, "RTT_min_ms": 20.0, "RTT_max_ms": 120.0}
    return compute_state_vector(telemetry, risk, calibration, cycle_s, mean_phase_s)


def test_s7_reconfig_phase_peaks_at_the_freeze_center():
    cycle_s, mean_phase_s = 15.0, 10.5
    state = _minimal_state([mean_phase_s], cycle_s, mean_phase_s)
    assert state["s7_reconfig_phase"].iloc[0] == pytest.approx(1.0)


def test_s7_reconfig_phase_is_zero_at_maximum_distance():
    cycle_s, mean_phase_s = 15.0, 10.5
    state = _minimal_state([mean_phase_s + cycle_s / 2.0], cycle_s, mean_phase_s)
    assert state["s7_reconfig_phase"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_s7_reconfig_phase_is_symmetric_before_and_after_center():
    cycle_s, mean_phase_s = 15.0, 10.5
    state = _minimal_state([mean_phase_s - 2.0, mean_phase_s + 2.0], cycle_s, mean_phase_s)
    assert state["s7_reconfig_phase"].iloc[0] == pytest.approx(state["s7_reconfig_phase"].iloc[1])


def test_s7_reconfig_phase_wraps_around_the_cycle_boundary():
    cycle_s, mean_phase_s = 15.0, 10.5
    # t_start=0.5 is 1.0s before phase-0 wraps to phase-(cycle_s), i.e. distance
    # to mean_phase_s should be computed via the wraparound, not the raw diff.
    state = _minimal_state([0.5], cycle_s, mean_phase_s)
    unwrapped_state = _minimal_state([0.5 + cycle_s], cycle_s, mean_phase_s)
    assert state["s7_reconfig_phase"].iloc[0] == pytest.approx(unwrapped_state["s7_reconfig_phase"].iloc[0])


def test_fairness_ratio_omitted_by_default_preserving_7col_output():
    # single-flow callers (FluidSimEnv) pass no fairness_ratio and must keep
    # getting the exact original 7-column state, so already-trained
    # single-flow checkpoints' input dimensionality never silently changes.
    state = _minimal_state([10.5])
    assert "s8_fairness_ratio" not in state.columns
    assert list(state.columns) == ["t_start"] + _STATE_COLS


def test_fairness_ratio_included_and_clipped_when_supplied():
    telemetry = pd.DataFrame({
        "t_start": [10.5], "b_hat_mbps": [10.0], "rtt_ms": [50.0],
        "v_over_bdp": [1.0], "q_packets": [5.0],
    })
    risk = pd.DataFrame({"delta_t_ho_min": [1.0], "p_tot": [0.01]})
    calibration = {"B_max_mbps": 100.0, "RTT_min_ms": 20.0, "RTT_max_ms": 120.0}

    state = compute_state_vector(telemetry, risk, calibration, fairness_ratio=0.4)
    assert state["s8_fairness_ratio"].iloc[0] == pytest.approx(0.4)

    # out-of-range inputs still get clipped to [0, 1], same convention as s1-s7.
    over_state = compute_state_vector(telemetry, risk, calibration, fairness_ratio=1.5)
    assert over_state["s8_fairness_ratio"].iloc[0] == pytest.approx(1.0)
