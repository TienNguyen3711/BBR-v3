from __future__ import annotations

from qbbr.features.normalize import T_ORBIT_MIN, compute_state_vector
from qbbr.features.risk import compute_risk_features
from qbbr.features.telemetry import extract_telemetry_features

_STATE_COLS = [
    "s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue",
    "s5_handover_eta", "s6_p_tot",
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
