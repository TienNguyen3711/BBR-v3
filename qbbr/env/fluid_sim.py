from __future__ import annotations

import math
from dataclasses import dataclass

MSS_BYTES = 1500.0  # matches qbbr.features.telemetry.MSS_BYTES
_DWN_RETRANSMIT_RATE_PPS = 20.0  # packets/s of "self-inflicted" loss while draining (i_dwn active)
_RISK_RETRANSMIT_RATE_PPS = 100.0  # packets/s of loss during an active risk event (p_tot > 0.02)


@dataclass(frozen=True)
class FluidParams:
    """Per-location/direction fluid-model parameters, derived from qbbr.env.calibration."""

    x_btl_bps: float  # peak (p99) estimated bottleneck bandwidth, bytes/s
    rtt_rtp_s: float  # estimated minimum propagation RTT, s
    utilization_fraction: float = 1.0  # median/p99 sustained-capacity fraction (see module docstring)
    capacity_dip_fraction: float = 0.6  # fraction of sustained capacity lost during a handover-linked dip
    bdp_hi_mult: float = 2.0  # inflight_hi bound, as a multiple of B_bar_DP (documented default)
    bdp_lo_mult: float = 1.0  # inflight_lo bound, as a multiple of B_bar_DP (documented default)
    sigmoid_k: float = 8.0  # soft-indicator steepness (higher = closer to a hard 0/1 switch)
    relax_rate_hz: float = 4.0  # how fast i_dwn/i_crs relax toward their sigmoid targets

    @property
    def sustained_x_btl_bps(self) -> float:
        """Average sustained capacity (utilization_fraction of the p99 peak x_btl_bps)."""
        return self.x_btl_bps * self.utilization_fraction

    @property
    def bdp_bytes(self) -> float:
        """B_bar_DP: bandwidth-delay product, based on sustained (not peak) capacity."""
        return self.sustained_x_btl_bps * self.rtt_rtp_s

    @property
    def bdp_hi_bytes(self) -> float:
        return self.bdp_hi_mult * self.bdp_bytes

    @property
    def bdp_hat_bytes(self) -> float:
        """B_hat_DP: conservative drain target, min(B_bar_DP, 0.85 * BDP_hi)."""
        return min(self.bdp_bytes, 0.85 * self.bdp_hi_bytes)


@dataclass(frozen=True)
class FluidState:
    t_s: float = 0.0
    v_bytes: float = 0.0  # inflight volume
    i_dwn: float = 0.0  # drawdown indicator, continuous in [0, 1]
    i_crs: float = 1.0  # cruise indicator, continuous in [0, 1]
    t_since_probe_s: float = 0.0


def sigmoid(x: float, k: float = 8.0) -> float:
    z = k * x
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def probe_bw_interval_s(rtt_rtp_s: float, n_flows: int = 1, flow_index: int = 0) -> float:
    return min(62.0 * rtt_rtp_s, 2.0 + flow_index / max(n_flows, 1))


def synthetic_ground_truth_p_tot(
    t_s: float,
    base_p: float = 0.01,
    handover_period_s: float = 15.0,
    handover_width_s: float = 1.0,
    handover_spike_p: float = 0.05,
) -> float:

    phase = t_s % handover_period_s
    dist_to_handover = min(phase, handover_period_s - phase)
    spike = handover_spike_p * math.exp(-0.5 * (dist_to_handover / handover_width_s) ** 2)
    return base_p + spike


def synthetic_capacity_fraction(
    t_s: float,
    dip_fraction: float = 0.6,
    handover_period_s: float = 15.0,
    handover_width_s: float = 1.0,
) -> float:

    phase = t_s % handover_period_s
    dist_to_handover = min(phase, handover_period_s - phase)
    dip = dip_fraction * math.exp(-0.5 * (dist_to_handover / handover_width_s) ** 2)
    return 1.0 - dip


def _dwn_target(state: FluidState, p_tot: float, params: FluidParams) -> float:
    bdp = params.bdp_bytes
    k = params.sigmoid_k

    def _vol_sigmoid(delta_bytes: float) -> float:
        return sigmoid(delta_bytes / bdp, k)

    dwn_activate = min(1.0, _vol_sigmoid(state.v_bytes - 1.25 * bdp) + sigmoid(p_tot - 0.02, k))
    dwn_deactivate = _vol_sigmoid(params.bdp_hat_bytes - state.v_bytes)
    return dwn_activate * (1.0 - dwn_deactivate)


def bbr_offered_rate_bps(
    state: FluidState, pacing_gain: float, p_tot: float, params: FluidParams
) -> float:
    dwn_target = _dwn_target(state, p_tot, params)
    stock_multiplier = 1.25 - 0.5 * dwn_target
    multiplier = state.i_crs * pacing_gain + (1.0 - state.i_crs) * stock_multiplier
    full_capacity_bps = params.sustained_x_btl_bps * synthetic_capacity_fraction(state.t_s)
    return full_capacity_bps * multiplier


def step_fluid_state(
    state: FluidState,
    pacing_gain: float,
    dt_s: float,
    params: FluidParams,
    p_tot: float,
    capacity_bps_override: float | None = None,
) -> tuple[FluidState, float, float]:

    bdp = params.bdp_bytes
    k = params.sigmoid_k

    def _vol_sigmoid(delta_bytes: float) -> float:
        return sigmoid(delta_bytes / bdp, k)

    dwn_target = _dwn_target(state, p_tot, params)

    # Eq. 24: I^crs activates when v <= B_bar_DP; Eq. 27: disabled while I^dwn is active.
    crs_target = _vol_sigmoid(bdp - state.v_bytes) * (1.0 - dwn_target)

    relax = min(1.0, params.relax_rate_hz * dt_s)
    i_dwn = min(max(state.i_dwn + (dwn_target - state.i_dwn) * relax, 0.0), 1.0)
    i_crs = min(max(state.i_crs + (crs_target - state.i_crs) * relax, 0.0), 1.0)

    # Eq. 25: pacing rate, using *this step's* freshly-relaxed i_crs (not
    # bbr_offered_rate_bps's one-step-stale approximation above).
    stock_multiplier = 1.25 - 0.5 * dwn_target
    multiplier = i_crs * pacing_gain + (1.0 - i_crs) * stock_multiplier

    if capacity_bps_override is not None:
        capacity_bps_now = capacity_bps_override
    else:
        capacity_bps_now = params.sustained_x_btl_bps * synthetic_capacity_fraction(state.t_s)
    pacing_rate_bps = capacity_bps_now * multiplier

    # Fluid queue: injection at the pacing rate, drainage at bottleneck capacity.
    injected_bytes = pacing_rate_bps * dt_s
    capacity_bytes = capacity_bps_now * dt_s
    delivered_bytes = min(capacity_bytes, state.v_bytes + injected_bytes)
    v_bytes = max(state.v_bytes + injected_bytes - delivered_bytes, 0.0)

    retransmits = (i_dwn * _DWN_RETRANSMIT_RATE_PPS + sigmoid(p_tot - 0.02, k) * _RISK_RETRANSMIT_RATE_PPS) * dt_s

    t_since_probe_s = state.t_since_probe_s + dt_s
    if t_since_probe_s >= probe_bw_interval_s(params.rtt_rtp_s):
        t_since_probe_s = 0.0

    new_state = FluidState(
        t_s=state.t_s + dt_s,
        v_bytes=v_bytes,
        i_dwn=i_dwn,
        i_crs=i_crs,
        t_since_probe_s=t_since_probe_s,
    )
    return new_state, delivered_bytes, retransmits
