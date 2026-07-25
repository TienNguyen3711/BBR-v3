"""Stage 5 reward: alpha-fair utility minus delay and loss-change penalties.

r_t = U_alpha(x_t) - delta * log(RTT_t / RTT_min) - beta * l_t

U_alpha(x) = (x+eps)^(1-alpha)/(1-alpha) for alpha != 1, log(x+eps) for alpha=1
l_t = (rtx_t - rtx_{t-1}) / (rtx_{t-1} + eps_rtx)

RTT_min here reuses the same rolling RTT_base (Eq. 30's 120s/5th-percentile
baseline) already computed in qbbr.features.telemetry, rather than a
separately-defined global minimum -- one running baseline, used for both
the queue-delay estimate and this reward term.

Note on eps_rtx vs eps: the design doc uses a single epsilon=1e-6 (chosen
because "throughput is genuinely zero during handover gaps") for both the
utility term and the loss-change denominator. Running this against real
per-interval retransmit *counts* (integers, frequently exactly 0) showed
that 1e-6 in the l_t denominator is the wrong scale: any interval following
a zero-retransmit interval produces l_t ~ rtx_t/1e-6, spiking the reward by
several orders of magnitude. eps_rtx defaults to 1.0 instead ("one
retransmit" worth of smoothing) so l_t stays bounded on real data; eps
(1e-6) is kept only for the throughput utility term, where it is
appropriate.

x_t is throughput in the decision window; for this 1 Hz trace-derived
series, x_t is each interval's own bits_per_second sample, in Mbps. This
produces a 1 Hz reward *series* for calibration/inspection of what the
reward signal looks like on real stock-BBR data -- not literal
per-decision-interval training transitions (those come from the Phase 2
fluid-model simulator, whose decision interval is sub-second at several
locations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 1.0
DEFAULT_DELTA = 1.0
DEFAULT_BETA = 0.5
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
    eps: float = EPSILON,
    eps_rtx: float = EPSILON_RTX,
) -> pd.Series:
    """telemetry: output of qbbr.features.telemetry.extract_telemetry_features."""
    x_t_mbps = telemetry["bits_per_second"] / 1e6
    rtt_t = telemetry["rtt_ms"].astype(float)
    rtt_min = telemetry["rtt_base_ms"].astype(float)

    rtx = telemetry["retransmits"].astype(float)
    rtx_prev = rtx.shift(1)
    l_t = ((rtx - rtx_prev) / (rtx_prev + eps_rtx)).fillna(0.0)

    utility = alpha_fair_utility(x_t_mbps, alpha, eps)
    delay_term = delta * np.log((rtt_t + eps) / (rtt_min + eps))

    return utility - delay_term - beta * l_t
