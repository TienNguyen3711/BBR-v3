from __future__ import annotations

import numpy as np
import pandas as pd

from qbbr.features.bbr_internals import compute_bhat_mbps, compute_v_over_bdp


def test_bhat_is_windowed_max_and_never_below_instantaneous(sample_trace):
    bps = sample_trace.intervals["bits_per_second"].astype(float)
    b_hat_mbps = compute_bhat_mbps(bps)
    instantaneous_mbps = bps / 1e6
    assert (b_hat_mbps.values >= instantaneous_mbps.values - 1e-9).all()


def test_v_over_bdp_is_nonnegative_and_finite(sample_trace):
    tel = sample_trace.intervals
    bps = tel["bits_per_second"].astype(float)
    rtt_ms = tel["rtt_ms"].astype(float)
    b_hat_mbps = compute_bhat_mbps(bps)
    # rtt_base_ms is qbbr.features.telemetry's concern; a constant stand-in
    # here keeps this test scoped to compute_v_over_bdp itself.
    rtt_base_ms = pd.Series([rtt_ms.min()] * len(tel), index=tel.index)

    v_over_bdp = compute_v_over_bdp(tel["snd_cwnd"], b_hat_mbps, rtt_base_ms)
    assert (v_over_bdp >= 0).all()
    assert np.isfinite(v_over_bdp).all()


def test_v_over_bdp_zero_bdp_yields_zero_not_nan():
    snd_cwnd = pd.Series([1000.0, 2000.0])
    b_hat_mbps = pd.Series([0.0, 0.0])  # forces bdp_hat_bytes == 0
    rtt_base_ms = pd.Series([50.0, 50.0])

    v_over_bdp = compute_v_over_bdp(snd_cwnd, b_hat_mbps, rtt_base_ms)
    assert (v_over_bdp == 0.0).all()
