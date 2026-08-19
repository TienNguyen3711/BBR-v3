from __future__ import annotations

import pandas as pd

from qbbr.features.state_builder import HANDOVER_ETA_SCALE_MIN, compute_state_vector
from qbbr.risk.ptot import compute_risk_features
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


def test_handover_eta_scale_constant_is_positive():
    assert HANDOVER_ETA_SCALE_MIN > 0


def test_s5_spans_full_unit_range_across_a_handover_cycle(sample_calibration):
    # t_start=0.0 is right AFTER a reconfiguration boundary (the next one is
    # a full 15s away -- delta_t_ho_min is maximal, so s5=0, least urgent).
    # t_start=7.5 is exactly half a cycle from the next boundary (s5=0.5).
    # t_start=14.99 is a fraction of a second BEFORE the next boundary
    # (delta_t_ho_min near 0, s5 near 1, most urgent). Before the
    # HANDOVER_ETA_SCALE_MIN fix, s5 was normalized by the ~5.4min orbital
    # period instead of the 15s handover cycle, compressing all three of
    # these into ~[0.95, 1.0] regardless of actual timing.
    tel = pd.DataFrame({
        "t_start": [0.0, 7.5, 14.99], "retransmits": [0.0, 0.0, 0.0],
        "b_hat_mbps": [0.0, 0.0, 0.0], "rtt_ms": [0.0, 0.0, 0.0],
        "v_over_bdp": [0.0, 0.0, 0.0], "q_packets": [0.0, 0.0, 0.0],
    })
    risk = compute_risk_features(tel, mode="empirical_proxy")
    calib = sample_calibration["Sydney"]["downlink"]
    state = compute_state_vector(tel, risk, calib)

    assert state["s5_handover_eta"].iloc[0] == 0.0
    assert state["s5_handover_eta"].iloc[1] == 0.5
    assert state["s5_handover_eta"].iloc[2] > 0.99
