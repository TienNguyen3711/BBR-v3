from __future__ import annotations

import numpy as np
import pandas as pd

T_ORBIT_MIN = 5.4  # approx. Starlink orbital period used for s5's ETA scale
V_OVER_BDP_CAP = 2.5  # covers BBR's 5/4 probe; values > 1 already mean queue buildup
Q_CAP_PACKETS = 50.0  # saturation cap observed in the base paper's Fig. 12-13


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def compute_state_vector(
    telemetry: pd.DataFrame,
    risk: pd.DataFrame,
    calibration: dict[str, float],
) -> pd.DataFrame:
    b_max = calibration["B_max_mbps"]
    rtt_min = calibration["RTT_min_ms"]
    rtt_max = calibration["RTT_max_ms"]
    rtt_span = rtt_max - rtt_min

    s1 = telemetry["b_hat_mbps"] / b_max
    s2 = (telemetry["rtt_ms"] - rtt_min) / rtt_span if rtt_span > 0 else telemetry["rtt_ms"] * 0.0
    s3 = telemetry["v_over_bdp"] / V_OVER_BDP_CAP
    s4 = telemetry["q_packets"] / Q_CAP_PACKETS
    s5 = 1.0 - (risk["delta_t_ho_min"] / T_ORBIT_MIN)
    s6 = risk["p_tot"]

    return pd.DataFrame(
        {
            "t_start": telemetry["t_start"].values,
            "s1_bhat": _clip01(s1),
            "s2_rtt_ratio": _clip01(s2),
            "s3_inflight_bdp": _clip01(s3),
            "s4_queue": _clip01(s4),
            "s5_handover_eta": _clip01(s5).values,
            "s6_p_tot": _clip01(s6).values,
        }
    )
