from __future__ import annotations

import numpy as np
import pandas as pd

# The handover proxy is intentionally slower than the 15-second reconfiguration
# clock used by s7.  It is a configurable modelling timescale, not Starlink's
# physical orbital period.
HANDOVER_CYCLE_MIN = 5.4
T_ORBIT_MIN = HANDOVER_CYCLE_MIN  # backwards-compatible name for existing callers
V_OVER_BDP_CAP = 2.5  # covers BBR's 5/4 probe; values > 1 already mean queue buildup
Q_CAP_PACKETS = 50.0  # saturation cap observed in the base paper's Fig. 12-13


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


# Defaults match qbbr.env.fluid_sim's _HANDOVER_CYCLE_S and the project's
# calibrated reconfig-phase profile (fluid_env._HANDOVER_PHASE_PROFILE:
# mean_phase_s=10.5, Sec. VI-B). Real callers (FluidSimEnv, MultiFlowFluidEnv)
# pass their own params.phase_profile values explicitly; these defaults only
# matter for standalone/real-trace callers that don't track a reconfig cycle.
_RECONFIG_CYCLE_S = 15.0
_RECONFIG_MEAN_PHASE_S = 10.5


def compute_state_vector(
    telemetry: pd.DataFrame,
    risk: pd.DataFrame,
    calibration: dict[str, float],
    reconfig_cycle_s: float = _RECONFIG_CYCLE_S,
    reconfig_mean_phase_s: float = _RECONFIG_MEAN_PHASE_S,
    fairness_ratio: float | None = None,
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

    # s7: how close t_start is to the nearest reconfig-phase center, on a
    # symmetric [0, 1] scale (1 = at the freeze-window peak, 0 = maximally
    # far, cycle_s/2 away) -- lets a policy anticipate the reconfig-freeze
    # window (fluid_env.in_reconfig_freeze_window) instead of only reacting
    # to it, unlike s5/s6 which encode the unrelated, much slower orbital
    # handover cycle.
    phase = telemetry["t_start"] % reconfig_cycle_s
    raw_dist = (phase - reconfig_mean_phase_s).abs()
    dist_to_freeze = np.minimum(raw_dist, reconfig_cycle_s - raw_dist)
    s7 = 1.0 - dist_to_freeze / (reconfig_cycle_s / 2.0)

    columns = {
        "t_start": telemetry["t_start"].values,
        "s1_bhat": _clip01(s1),
        "s2_rtt_ratio": _clip01(s2),
        "s3_inflight_bdp": _clip01(s3),
        "s4_queue": _clip01(s4),
        "s5_handover_eta": _clip01(s5).values,
        "s6_p_tot": _clip01(s6).values,
        "s7_reconfig_phase": _clip01(s7).values,
    }
    # s8: only added when the caller (MultiFlowFluidEnv) supplies a fairness
    # ratio -- single-flow callers (FluidSimEnv) pass None and get the exact
    # original 7-column output, so this doesn't disturb the already-trained
    # single-flow checkpoints' state dimensionality. min(all flows' throughput,
    # including the agent's own) / agent's own throughput: 1.0 when the agent
    # IS the worst-off flow (or tied), lower when some other flow is worse off
    # than the agent -- naturally bounded to (0, 1] since the agent's own
    # throughput is always >= the min over a set that includes it.
    if fairness_ratio is not None:
        columns["s8_fairness_ratio"] = _clip01(pd.Series(fairness_ratio, index=telemetry.index))

    return pd.DataFrame(columns)
