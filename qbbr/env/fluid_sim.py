from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import i0 as _bessel_i0

MSS_BYTES = 1500.0  # matches qbbr.features.telemetry.MSS_BYTES
_DWN_RETRANSMIT_RATE_PPS = 20.0  # location-agnostic placeholder default; calibrated per-location in practice
_P_TOT_RISK_THRESHOLD = 0.02  # p_tot level considered "active risk" (was inlined as a magic 0.02 in two places)
_HANDOVER_CYCLE_S = 15.0  # matches qbbr.risk.ptot._HANDOVER_CYCLE_S; the validated real cadence (Sec. VI-B)

ECN_MARK_THRESHOLD_MULT = 1.0  # v_bytes/bdp ratio above which a substep counts as ECN-marked
# (RFC 3168/DCTCP-style: any queueing beyond the pure bandwidth-delay product is a congestion
# signal). A LEVEL-based, congestion-only companion to the reward's delta-only l_t term
# (qbbr.reward.alpha_fair) -- purely a function of v_bytes/bdp (physical queue occupancy), with
# a FIXED, non-agent-controllable threshold, so unlike drawdown_activate_mult there is no lever
# an agent can move to shrink the *measured* signal without genuinely reducing queueing.


def ecn_marked(v_bytes: float, bdp: float, threshold_mult: float = ECN_MARK_THRESHOLD_MULT) -> bool:
    return v_bytes > threshold_mult * bdp


STARTUP_GAIN = 2.89  # real BBR STARTUP's cwnd_gain/pacing_gain constant (2/ln(2), historically
# approximated as 2.89 in BBR's own implementation and confirmed by Gomez et al.'s measurement).
# Applied as the ENTIRE injection multiplier during STARTUP (bypassing the normal
# i_crs*pacing_gain + (1-i_crs)*stock_multiplier blend below) -- the agent does not control
# pacing during STARTUP in real BBR-v3 either, matching main.tex's design (overrides apply only
# in ProbeBW_CRUISE).
STARTUP_EXIT_MULT = 2.5  # v_bytes/bdp level at which STARTUP is considered to have filled the
# pipe and exits -- matches this project's existing V_OVER_BDP_CAP convention
# (qbbr.features.state_builder) and the ~2-2.9x overshoot range Scherrer et al. report during
# real BBR-v2/v3 STARTUP.
STARTUP_MAX_DURATION_S = 10.0  # fallback exit if the exit-volume condition is never reached
# (e.g. a severely capacity-limited multi-flow scenario) -- a small fraction of a 300s episode.


@dataclass(frozen=True)
class PhaseProfile:
    """Empirical handover-phase concentration of the retransmit process
    (Sec. VI-B: run-level Rayleigh test, sequential downlink traces), encoded
    as a von Mises density so retransmit_phase_multiplier() has a closed form.
    Not a fitted/assumed shape: mean_phase_s and r_bar are the measured
    values (mean phase, mean resultant length) from validate_handover_cadence.py.
    """

    mean_phase_s: float
    r_bar: float
    cycle_s: float = _HANDOVER_CYCLE_S

    @property
    def kappa(self) -> float:
        """Standard R-bar -> von Mises concentration approximation (Fisher 1993,
        Sec. 5.3), valid for R_bar in [0, 1)."""
        r = self.r_bar
        if r < 0.53:
            return 2 * r + r**3 + 5 * r**5 / 6
        if r < 0.85:
            return -0.4 + 1.39 * r + 0.43 / (1 - r)
        return 1 / (r**3 - 4 * r**2 + 3 * r)


def retransmit_phase_multiplier(t_s: float, phase_profile: PhaseProfile | None) -> float:
    """m(phi(t)): phase-dependent multiplier on the per-location retransmit
    rate, normalized so its average over one full cycle is 1.0 -- it
    redistributes a location's calibrated rate IN TIME to match the measured
    handover-phase-lock (Sec. VI-B), without changing the calibrated total.
    phase_profile=None disables modulation (uniform rate, m=1 everywhere),
    e.g. for tests that don't care about phase structure.
    """
    if phase_profile is None:
        return 1.0
    phase = (t_s % phase_profile.cycle_s) / phase_profile.cycle_s * 2 * math.pi
    mu = phase_profile.mean_phase_s / phase_profile.cycle_s * 2 * math.pi
    kappa = phase_profile.kappa
    return float(math.exp(kappa * math.cos(phase - mu)) / _bessel_i0(kappa))


@dataclass(frozen=True)
class FluidParams:
    """Per-location/direction fluid-model parameters, derived from qbbr.env.calibration."""

    x_btl_bps: float  # peak (p99) estimated bottleneck bandwidth, bytes/s
    rtt_rtp_s: float  # estimated minimum propagation RTT, s
    utilization_fraction: float = 1.0  # median/p99 sustained-capacity fraction (see module docstring)
    capacity_dip_fraction: float = 0.6  # fraction of sustained capacity lost during a handover-linked dip
    bdp_hi_mult: float = 2.0  # inflight_hi bound, as a multiple of B_bar_DP (documented default)
    bdp_lo_mult: float = 1.0  # inflight_lo bound, as a multiple of B_bar_DP (documented default)
    drawdown_activate_mult: float = 1.25  # v_bytes/bdp level that starts triggering I^dwn (Eq. 26 default);
    # calibrated per-location in practice -- see env/calibration.py. Empirically confirmed monotonic in and
    # near-orthogonal to simulated throughput (throughput is capacity-capped; RTT, via queue overshoot, is not).
    sigmoid_k: float = 8.0  # soft-indicator steepness (higher = closer to a hard 0/1 switch)
    relax_rate_hz: float = 4.0  # how fast i_dwn/i_crs relax toward their sigmoid targets
    dwn_retransmit_rate_pps: float = _DWN_RETRANSMIT_RATE_PPS  # calibrated per-location; see env/calibration.py
    phase_profile: PhaseProfile | None = None  # shared across locations (Sec. VI-B pooled measurement)

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
    startup_done: bool = True  # True = skip STARTUP modeling (pre-STARTUP-prefix default
    # behavior: every existing caller that constructs FluidState without this field, e.g. every
    # unit test written before STARTUP was added, is unaffected). Only FluidSimEnv.reset() sets
    # this False, opting a fresh episode into the STARTUP-phase prefix (see step_fluid_state).


def sigmoid(x: float, k: float = 8.0) -> float:
    z = k * x
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _risk_sigmoid(p_tot: float, k: float) -> float:
    """Same soft-indicator convention as the volume-based terms below (see
    step_fluid_state's _vol_sigmoid): normalize the argument to O(1) before
    handing it to the shared steepness k. p_tot deviates from
    _P_TOT_RISK_THRESHOLD on a raw-probability scale (this simulator's
    realistic range is ~7.5e-4 baseline to ~5e-2 at a handover-spike peak),
    about 1000x smaller than delta_bytes/bdp's O(1) scale -- without this
    normalization sigmoid(p_tot - _P_TOT_RISK_THRESHOLD, k) stays stuck at
    ~0.46-0.56 for any p_tot in that range, a location-invariant "always on"
    floor rather than a real 0/1 switch. Feeds only _dwn_target's
    dwn_activate (p_tot's effect on drawdown/throughput dynamics); no longer
    used to add a separate p_tot-driven retransmit term (see
    step_fluid_state's retransmits line and env/calibration.py's docstring
    for why that term was removed rather than rescaled).
    """
    return sigmoid((p_tot - _P_TOT_RISK_THRESHOLD) / _P_TOT_RISK_THRESHOLD, k)


RECONFIG_FREEZE_HALF_WIDTH_S = 1.0  # matches synthetic_ground_truth_p_tot's handover_width_s convention


def in_reconfig_freeze_window(
    t_s: float,
    phase_offset_s: float,
    phase_profile: PhaseProfile | None,
    half_width_s: float = RECONFIG_FREEZE_HALF_WIDTH_S,
) -> bool:
    """True within half_width_s of phase_profile.mean_phase_s on the
    validated 15s reconfiguration cycle (Sec. VI-B) -- the phase where the
    model's own calibrated retransmit generation concentrates (see
    step_fluid_state's dwn_rate_now). Two independent published Starlink
    studies (a BBR bandwidth/RTT-filter reset at known reconfiguration
    boundaries, and StarQUIC's CC-reaction freeze in a +-100ms window
    around them) report large gains from reacting to this fixed external
    clock rather than to noisy loss/RTT signals -- and because the clock is
    external (not a function of the agent's own actions), it cannot be
    gamed the way the inflight_hi/lo action's exploit gamed the reward's
    RTT term. phase_profile=None (uniform-rate mode, e.g. some tests)
    disables the freeze entirely, matching the pre-freeze default.
    """
    if phase_profile is None:
        return False
    cycle_s = phase_profile.cycle_s
    phase = (t_s + phase_offset_s) % cycle_s
    raw_dist = abs(phase - phase_profile.mean_phase_s)
    dist = min(raw_dist, cycle_s - raw_dist)
    return dist <= half_width_s


def probe_bw_interval_s(rtt_rtp_s: float, n_flows: int = 1, flow_index: int = 0) -> float:
    """Eq. 22: t^pbw_i = min(62*RTT_min, 2 + i/N) for flow i in a parallel set
    of N BBR flows. The base paper indexes i in {1,...,N} (1-indexed); flow_index
    here is the ordinary 0-indexed Python convention, converted internally
    (i = flow_index + 1) so a single isolated flow (n_flows=1, flow_index=0)
    correctly gives i=1, N=1 -> cap of 3.0s, not 2.0s.
    """
    i = flow_index + 1
    return min(62.0 * rtt_rtp_s, 2.0 + i / max(n_flows, 1))


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


def _dwn_target(
    state: FluidState, p_tot: float, params: FluidParams, bdp_hat_bytes: float,
    drawdown_activate_mult: float | None = None,
) -> float:
    """bdp_hat_bytes and drawdown_activate_mult are passed explicitly (not
    read from params) so the caller can substitute agent-controlled
    overrides for this step -- see step_fluid_state's inflight_hi_mult_override
    and drawdown_activate_mult_override. drawdown_activate_mult=None falls
    back to params.drawdown_activate_mult (the calibrated per-location
    default), matching pre-override behavior exactly."""
    bdp = params.bdp_bytes
    k = params.sigmoid_k
    activate_mult = drawdown_activate_mult if drawdown_activate_mult is not None else params.drawdown_activate_mult

    def _vol_sigmoid(delta_bytes: float) -> float:
        return sigmoid(delta_bytes / bdp, k)

    dwn_activate = min(1.0, _vol_sigmoid(state.v_bytes - activate_mult * bdp) + _risk_sigmoid(p_tot, k))
    dwn_deactivate = _vol_sigmoid(bdp_hat_bytes - state.v_bytes)
    return dwn_activate * (1.0 - dwn_deactivate)


def bbr_offered_rate_bps(
    state: FluidState, pacing_gain: float, p_tot: float, params: FluidParams
) -> float:
    # One-substep-stale approximation (see multi_flow_env.py's caller comment)
    # -- uses params' own (unoverridden) inflight_hi/drawdown-activate bounds,
    # not this step's agent-chosen overrides, since it is only a rough
    # pre-allocation estimate.
    dwn_target = _dwn_target(state, p_tot, params, params.bdp_hat_bytes)
    stock_multiplier = 1.25 - 0.5 * dwn_target
    # See step_fluid_state's identical STARTUP bypass -- kept consistent so this
    # pre-allocation estimate doesn't understate the agent's own offered rate
    # during its STARTUP ramp (i_crs starts at 0, which would otherwise dilute
    # towards stock_multiplier here too).
    multiplier = (
        STARTUP_GAIN if not state.startup_done
        else state.i_crs * pacing_gain + (1.0 - state.i_crs) * stock_multiplier
    )
    full_capacity_bps = params.sustained_x_btl_bps * synthetic_capacity_fraction(state.t_s)
    return full_capacity_bps * multiplier


def step_fluid_state(
    state: FluidState,
    pacing_gain: float,
    dt_s: float,
    params: FluidParams,
    p_tot: float,
    capacity_bps_override: float | None = None,
    n_flows: int = 1,
    flow_index: int = 0,
    phase_offset_s: float = 0.0,
    inflight_hi_mult_override: float | None = None,
    inflight_lo_mult_override: float | None = None,
    drawdown_activate_mult_override: float | None = None,
) -> tuple[FluidState, float, float]:
    """inflight_hi_mult_override / inflight_lo_mult_override: this step's
    agent-chosen multiples of B_bar_DP (TMC paper Eq. 28's BDP^hi/BDP^lo),
    substituting for params.bdp_hi_mult/bdp_lo_mult when given. inflight_hi
    sets I^dwn's deactivation threshold (bdp_hat_bytes, was already wired to
    params.bdp_hi_mult before this override existed); inflight_lo sets
    I^crs's deactivation threshold (newly wired here -- bdp_lo_mult was
    previously defined on FluidParams but not used anywhere in this file).

    drawdown_activate_mult_override: this step's agent-chosen multiple of
    B_bar_DP for I^dwn's *activation* threshold (real BBR-v3's PROBE_BW:UP
    exit criterion -- inflight volume crossing ~1.25xBDP -- confirmed by
    Gomez et al. and Scherrer et al., independent of this project's own
    fluid model), substituting for params.drawdown_activate_mult when given.
    Unlike inflight_hi/lo, this directly gates the retransmit-generating
    term itself (see dwn_rate_now/retransmits below), not just a secondary
    threshold -- so it is a more causally direct lever on RQ1's target
    metric, and correspondingly a more direct route to gaming it (see
    action_multihead.yaml's comment on level-range choice)."""

    bdp = params.bdp_bytes
    k = params.sigmoid_k
    in_startup = not state.startup_done

    def _vol_sigmoid(delta_bytes: float) -> float:
        return sigmoid(delta_bytes / bdp, k)

    # During STARTUP the agent controls nothing (matches real BBR-v3 and this
    # project's own design, Sec. "Action Space and Protocol Hook": overrides
    # apply only in ProbeBW_CRUISE) -- every override falls back to its stock
    # default, same as an action space that doesn't control that dimension.
    bdp_hi_mult_now = params.bdp_hi_mult if in_startup else (
        inflight_hi_mult_override if inflight_hi_mult_override is not None else params.bdp_hi_mult
    )
    bdp_lo_mult_now = params.bdp_lo_mult if in_startup else (
        inflight_lo_mult_override if inflight_lo_mult_override is not None else params.bdp_lo_mult
    )
    drawdown_override_now = None if in_startup else drawdown_activate_mult_override
    bdp_hi_bytes_now = bdp_hi_mult_now * bdp
    bdp_hat_bytes_now = min(bdp, 0.85 * bdp_hi_bytes_now)

    dwn_target = _dwn_target(state, p_tot, params, bdp_hat_bytes_now, drawdown_override_now)

    crs_target = _vol_sigmoid(bdp_lo_mult_now * bdp - state.v_bytes) * (1.0 - dwn_target)

    relax = min(1.0, params.relax_rate_hz * dt_s)
    i_dwn = min(max(state.i_dwn + (dwn_target - state.i_dwn) * relax, 0.0), 1.0)
    i_crs = min(max(state.i_crs + (crs_target - state.i_crs) * relax, 0.0), 1.0)
    stock_multiplier = 1.25 - 0.5 * dwn_target
    # STARTUP bypasses the i_crs blend entirely (matching real BBR-v3 -- STARTUP is not
    # ProbeBW_CRUISE, so this project's cruise-gain blend doesn't apply): i_crs starts low
    # (0.0 by construction, see FluidSimEnv.reset()) and the blend would otherwise dilute
    # STARTUP_GAIN down to near stock_multiplier exactly when the aggressive ramp is needed.
    multiplier = STARTUP_GAIN if in_startup else (i_crs * pacing_gain + (1.0 - i_crs) * stock_multiplier)

    if capacity_bps_override is not None:
        capacity_bps_now = capacity_bps_override
    else:
        capacity_bps_now = params.sustained_x_btl_bps * synthetic_capacity_fraction(state.t_s)
    pacing_rate_bps = capacity_bps_now * multiplier

    injected_bytes = pacing_rate_bps * dt_s
    capacity_bytes = capacity_bps_now * dt_s
    delivered_bytes = min(capacity_bytes, state.v_bytes + injected_bytes)
    v_bytes = max(state.v_bytes + injected_bytes - delivered_bytes, 0.0)

    dwn_rate_now = params.dwn_retransmit_rate_pps * retransmit_phase_multiplier(
        state.t_s + phase_offset_s, params.phase_profile
    )
    # No separate p_tot-driven retransmit term: an earlier version added
    # _risk_sigmoid(p_tot, k) * a fixed amplitude here, but that mechanism was
    # never empirically tested (unlike dwn_retransmit_rate_pps, which IS
    # calibrated against real per-location rates -- see env/calibration.py).
    # p_tot still affects dynamics via _dwn_target's dwn_activate above.
    retransmits = i_dwn * dwn_rate_now * dt_s

    t_since_probe_s = state.t_since_probe_s + dt_s
    if t_since_probe_s >= probe_bw_interval_s(params.rtt_rtp_s, n_flows=n_flows, flow_index=flow_index):
        t_since_probe_s = 0.0

    new_t_s = state.t_s + dt_s
    startup_done_now = state.startup_done or (v_bytes >= STARTUP_EXIT_MULT * bdp) or (new_t_s >= STARTUP_MAX_DURATION_S)

    new_state = FluidState(
        t_s=new_t_s,
        v_bytes=v_bytes,
        i_dwn=i_dwn,
        i_crs=i_crs,
        t_since_probe_s=t_since_probe_s,
        startup_done=startup_done_now,
    )
    return new_state, delivered_bytes, retransmits
