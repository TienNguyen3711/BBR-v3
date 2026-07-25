"""Stage 1 raw features: BBR internals (s1, s3) + live telemetry (s2, s4).

iperf3 JSON has no tcp_probe-level delivery-rate samples or inflight_hi/lo,
so s1 and s3 are necessarily proxies rather than literal kernel reads:

- s1 (B_hat_theta): BBR's own BtlBw filter is itself a windowed max of
  delivery-rate samples, so a rolling max of the per-interval
  `bits_per_second` is a reasonable stand-in for the same quantity.
- s3 (v_t / BDP): `snd_cwnd` (bytes) is used as the inflight-volume proxy,
  divided by an estimated BDP = B_hat_theta (bytes/s) * RTT baseline (s).

s4 (queue occupancy, in packets) follows the base paper's M/G/1
approximation (Eq. 30-32): a queuing-delay estimate q = max(0, RTT -
RTT_base) drives a busy/idle indicator, which gives an M/G/1 arrival rate,
a Pollaczek-Khinchine mean waiting time, and finally an average queue size
via Little's Law (Q = lambda * W_q). RTT_base is a rolling 5th-percentile
over a 120s window, per the base paper's definition -- reused here as the
same running baseline for the reward's RTT_min term, rather than a second,
separately-defined quantity.

The M/G/1 sample window `w` (Eq. 31) is not given a specific value in the
base paper text available to us; a 10-sample rolling window is used here
as a documented default, not a value taken from the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MSS_BYTES = 1500
RTT_BASE_WINDOW_S = 120.0
RTT_BASE_PERCENTILE = 5.0
BHAT_WINDOW_SAMPLES = 10
QUEUE_OCCUPANCY_WINDOW_SAMPLES = 10
_MAX_RHO = 0.999  # numerical guard: Pollaczek-Khinchine W_q diverges as rho -> 1


def extract_telemetry_features(intervals: pd.DataFrame) -> pd.DataFrame:
    """Compute raw (un-normalized) Stage-1 features from one trace's intervals.

    intervals is expected 1 Hz (one row per second), as produced by
    qbbr.data.loader.load_trace. RTT_BASE_WINDOW_S is therefore treated as a
    sample count, not a wall-clock resample.
    """
    n = len(intervals)
    rtt_ms = intervals["rtt_ms"].astype(float)
    bps = intervals["bits_per_second"].astype(float)

    window = min(int(RTT_BASE_WINDOW_S), max(n, 1))
    rtt_base_ms = rtt_ms.rolling(window=window, min_periods=1).quantile(
        RTT_BASE_PERCENTILE / 100.0
    )

    b_hat_bps = bps.rolling(window=BHAT_WINDOW_SAMPLES, min_periods=1).max()
    b_hat_mbps = b_hat_bps / 1e6

    q_delay_ms = (rtt_ms - rtt_base_ms).clip(lower=0.0)

    mu_pps = (bps / 8.0) / MSS_BYTES  # service rate, packets/s
    busy = (q_delay_ms > 0.0).astype(float)
    busy_frac = busy.rolling(window=QUEUE_OCCUPANCY_WINDOW_SAMPLES, min_periods=1).mean()

    lambda_pps = mu_pps * busy_frac
    rho = busy_frac.clip(upper=_MAX_RHO)  # rho = lambda/mu = busy_frac algebraically

    mean_service_s = 1.0 / mu_pps.replace(0.0, np.nan)
    mean_service_sq = 2.0 * mean_service_s**2  # E[S^2] with c_s^2 = 1 (exponential)
    wait_s = (lambda_pps * mean_service_sq) / (2.0 * (1.0 - rho))
    q_packets = (lambda_pps * wait_s).fillna(0.0)  # Little's Law: Q = lambda * W_q

    bdp_hat_bytes = (b_hat_bps / 8.0) * (rtt_base_ms / 1000.0)
    v_t_bytes = intervals["snd_cwnd"].astype(float)
    v_over_bdp = v_t_bytes / bdp_hat_bytes.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "t_start": intervals["t_start"],
            "b_hat_mbps": b_hat_mbps,
            "v_over_bdp": v_over_bdp.fillna(0.0),
            "q_packets": q_packets,
            "rtt_ms": rtt_ms,
            "rtt_base_ms": rtt_base_ms,
            "bits_per_second": bps,
            "retransmits": intervals["retransmits"].astype(float),
        }
    )
