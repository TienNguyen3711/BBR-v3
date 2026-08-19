from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 1.0
DEFAULT_DELTA = 1.0
DEFAULT_BETA = 0.5
DEFAULT_GAMMA = 0.0  # ECN-style level congestion penalty weight. 0.0 preserves the exact
# pre-existing reward for any caller not supplying an ecn_mark_fraction column or explicitly
# opting in with gamma>0 -- unlike beta's delta-only l_t, this is a LEVEL term (Sec.~sec:rq1b's
# extension-attempt discussion): it can safely penalize a persistently elevated but stable
# congestion signal, since ecn_mark_fraction (qbbr.env.fluid_sim.ecn_marked) is purely a
# function of physical queue occupancy against a fixed threshold, not retransmit counts, so it
# does not conflate Starlink's non-congestive losses with congestion the way an absolute
# retransmit-level term would.
EPSILON = 1e-6
EPSILON_RTX = 1.0


def alpha_fair_utility(x: pd.Series, alpha: float = DEFAULT_ALPHA, eps: float = EPSILON) -> pd.Series:
    x = x.astype(float)
    if alpha == 1.0:
        return np.log(x + eps)
    return (x + eps) ** (1.0 - alpha) / (1.0 - alpha)


def compute_reward(
    telemetry: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    delta: float = DEFAULT_DELTA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    eps: float = EPSILON,
    eps_rtx: float = EPSILON_RTX,
) -> pd.Series:
    x_t_mbps = telemetry["bits_per_second"] / 1e6
    rtt_t = telemetry["rtt_ms"].astype(float)
    rtt_min = telemetry["rtt_base_ms"].astype(float)

    rtx = telemetry["retransmits"].astype(float)
    rtx_prev = rtx.shift(1)
    l_t = ((rtx - rtx_prev) / (rtx_prev + eps_rtx)).fillna(0.0)
    ecn_mark_fraction = telemetry["ecn_mark_fraction"].astype(float) if "ecn_mark_fraction" in telemetry else 0.0

    utility = alpha_fair_utility(x_t_mbps, alpha, eps)
    delay_term = delta * np.log((rtt_t + eps) / (rtt_min + eps))

    return utility - delay_term - beta * l_t - gamma * ecn_mark_fraction
