from __future__ import annotations

import numpy as np

from qbbr.features.telemetry import extract_telemetry_features


def test_telemetry_shape_and_no_nans(sample_trace):
    tel = extract_telemetry_features(sample_trace.intervals)

    assert len(tel) == len(sample_trace.intervals)
    for col in ["b_hat_mbps", "v_over_bdp", "q_packets", "rtt_ms", "rtt_base_ms"]:
        assert not tel[col].isna().any(), col
        assert np.isfinite(tel[col]).all(), col


def test_b_hat_is_windowed_max_and_never_below_instantaneous(sample_trace):
    tel = extract_telemetry_features(sample_trace.intervals)
    instantaneous_mbps = sample_trace.intervals["bits_per_second"] / 1e6
    # a windowed max can only be >= the instantaneous sample it's built from
    assert (tel["b_hat_mbps"].values >= instantaneous_mbps.values - 1e-9).all()


def test_queue_and_inflight_are_nonnegative(sample_trace):
    tel = extract_telemetry_features(sample_trace.intervals)
    assert (tel["q_packets"] >= 0).all()
    assert (tel["v_over_bdp"] >= 0).all()
