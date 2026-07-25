"""Stage 1 BBR-internal features (s1, s3): bandwidth estimate and inflight/BDP.

iperf3 JSON has no tcp_probe-level delivery-rate samples or inflight_hi/lo,
so s1 and s3 are necessarily proxies rather than literal kernel reads:

- s1 (B_hat_theta): BBR's own BtlBw filter is itself a windowed max of
  delivery-rate samples, so a rolling max of the per-interval
  `bits_per_second` is a reasonable stand-in for the same quantity.
- s3 (v_t / BDP): `snd_cwnd` (bytes) is used as the inflight-volume proxy,
  divided by an estimated BDP = B_hat_theta (bytes/s) * RTT baseline (s).
  RTT baseline is qbbr.features.telemetry's rolling 5th-percentile RTT_base,
  passed in rather than recomputed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BHAT_WINDOW_SAMPLES = 10


def compute_bhat_mbps(bps: pd.Series, window: int = BHAT_WINDOW_SAMPLES) -> pd.Series:
    b_hat_bps = bps.rolling(window=window, min_periods=1).max()
    return b_hat_bps / 1e6


def compute_v_over_bdp(
    snd_cwnd_bytes: pd.Series, b_hat_mbps: pd.Series, rtt_base_ms: pd.Series
) -> pd.Series:
    b_hat_bps = b_hat_mbps * 1e6
    bdp_hat_bytes = (b_hat_bps / 8.0) * (rtt_base_ms / 1000.0)
    v_over_bdp = snd_cwnd_bytes.astype(float) / bdp_hat_bytes.replace(0.0, np.nan)
    return v_over_bdp.fillna(0.0)
