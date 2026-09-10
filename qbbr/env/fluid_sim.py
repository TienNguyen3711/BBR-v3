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


STARTUP_GAIN = 2.77  # BBR-v3 STARTUP *pacing* gain. BBR-v1/v2 used 2/ln(2) ~= 2.89; BBR-v3
# recalibrated the pacing gain to 2.77 via analytic re-derivation (Gomez et al. Sec. 2.5.2).
# This is a fluid (rate) model, so the injection multiplier IS a pacing-rate gain -- the separate
# BBR-v3 cwnd_gain recalibration (2.89 -> 2.0) has no analogue here (no congestion window), and
# the revised loss-based STARTUP exit (6 vs 8 loss events per round) is likewise not modelled:
# STARTUP here exits on volume (STARTUP_EXIT_MULT) or STARTUP_MAX_DURATION_S, not on loss counting.
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
    # Disabled by default to preserve legacy results. When positive, this is
    # the recovery time constant of BBR's delivery-bandwidth estimate after a
    # handover capacity dip. It must be calibrated from observed telemetry
    # before it can be treated as a field-model parameter.
    bandwidth_estimate_recovery_s: float = 0.0
    # Disabled by default (legacy: delivered throughput was pinned at the
    # bottleneck for any pacing gain >= 1, so over-pacing was free on
    # throughput and only cost retransmits). When positive, sustained
    # over-pacing that drives BBR into DRAIN (high i_dwn) also depresses
    # delivered throughput -- effective capacity is scaled by
    # (1 - drain_throughput_penalty * i_dwn). This is BBR's own behaviour
    # (inflight clamps, the pipe partially drains); it makes "always
    # aggressive" strictly worse on throughput than a gain that is only
    # raised when it is needed. It is a modelling parameter, not a fitted
    # Starlink constant, until calibrated against observed DRAIN episodes.
    drain_throughput_penalty: float = 0.0
    # Stage 1b fit fixes (default 0.0 = legacy). The rate-balance delivery
    # model above lets v_bytes fall to ~0 under stock pacing, but real BBR-v3
    # holds ~1 BDP in flight (cwnd = 2 BDP, one-RTT delivery lag). When
    # positive, the *reported* inflight/BDP carries a persistent pipe term of
    # this many BDP (shrunk while draining); v_bytes itself still tracks only
    # the queue backlog, so i_crs/i_dwn, ECN, and q_packets are unchanged.
    # Calibrated per location from observed snd_cwnd/BDP (env/calibration.py).
    steady_inflight_bdp_frac: float = 0.0
    # Real BBR-v3 retransmits continuously over Starlink (aggressive probing +
    # LEO loss), not only during modelled DRAIN. When positive, this
    # per-location baseline rate (packets/s, phase-modulated) is added to the
    # DRAIN-gated term. Calibrated from the observed retransmit rate.
    base_retransmit_rate_pps: float = 0.0

    # --- Goal-2 dynamics (throughput vs retransmit/RTT). All default off (0.0)
    # => legacy behaviour. They make the *cost* of over-pacing faithful so a
    # throughput-seeking pacing_gain policy cannot inflate queue/loss for free.
    #
    # cwnd_gain: BBR-v3's inflight *cap* = cwnd_gain * BDP (real default 2.0).
    # When > 0 the queue backlog is clamped at (cwnd_gain-1)*BDP*inflight_hi_scale,
    # so past the cap a higher pacing_gain buys nothing (BBR stops sending at
    # cwnd) and RTT / buffer-loss are bounded rather than unbounded.
    consistent_transport: bool = False  # versioned queue/goodput proxy; requires recalibration
    cwnd_gain: float = 0.0
    # Loss response: if the per-round retransmission fraction (retransmits / total serviced
    # packets) exceeds loss_thresh (real BBR-v3 ~0.02), inflight_hi_scale is cut
    # by loss_beta (real ~0.7); it recovers toward 1.0 over inflight_hi_recover_s.
    loss_thresh: float = 0.0
    loss_beta: float = 0.7
    inflight_hi_recover_s: float = 3.0
    # ECN response: a continuous pull on inflight_hi_scale toward
    # (1 - ecn_response_factor) while modelled inflight sits above
    # ecn_response_thresh_bdp * BDP (a standing-queue signal the AQM would mark).
    ecn_response_factor: float = 0.0
    ecn_response_thresh_bdp: float = 1.0
    # ProbeRTT: every probe_rtt_interval_s (real ~5 s) drain inflight toward
    # probe_rtt_cwnd_frac * BDP (real ~0.5) for probe_rtt_duration_s (real
    # ~0.2 s), refreshing min-RTT and clearing the queue. off when interval = 0.
    probe_rtt_interval_s: float = 0.0
    probe_rtt_duration_s: float = 0.2
    probe_rtt_cwnd_frac: float = 0.5

    # --- v7 ProbeBW cycle (goal-2 fidelity). Default off => legacy behaviour.
    # When on, a chosen gain > 1.0 is executed as a real BBR-v3 ProbeBW_UP
    # pulse of probe_up_rounds round(s), then a ProbeBW_DOWN drain of
    # probe_down_rounds round(s) at probe_down_gain (0.75, a native action
    # value), then ProbeBW_CRUISE at <= 1.0 until the next probe is due
    # (probe_bw_interval_s, Eq. 22). The queue then sawtooths at ~(gain-1)*BDP
    # instead of holding a full cwnd of standing backlog, so a *sustained*
    # high gain no longer roughly doubles RTT -- only a well-timed probe pays.
    # A chosen gain <= 1.0 is pure cruise: stock 1.0 reproduces the pre-v7
    # baseline exactly (no autonomous probe). This changes how the frozen
    # action maps to queue/RTT dynamics, not the action set or the
    # throughput-only reward. See docs/transport_v6_review.md (v7 section);
    # supervisor-flagged reinterpretation of the ProbeBW hook.
    probe_bw_cycle: bool = False
    probe_up_rounds: float = 1.0
    probe_down_rounds: float = 1.0
    probe_down_gain: float = 0.75
    # v7b: hard ceiling on the standing queue a ProbeBW_UP/DOWN pulse may build,
    # expressed as queuing delay (ms) rather than a BDP fraction. On a
    # degenerate low-capacity path (e.g. the util=0.05 uplinks) 0.5 BDP is tens
    # of ms of RTT; this bounds the probe's RTT cost the same on every link.
    # 0.0 disables (v7 behaviour: only the cwnd cap / 0.5-BDP hold limit apply).
    probe_max_queue_delay_ms: float = 0.0
    # Queue-pressure retransmissions (goal-2). Default 0.0 => legacy. When > 0,
    # once the modelled inflight backlog exceeds the buffer ceiling
    # (cwnd_gain * BDP, or drawdown_activate_mult * BDP when the cwnd cap is
    # off) a fraction overflow_retransmit_frac of the packets serviced that
    # substep are also retransmitted, scaled by how far over the ceiling the
    # backlog sits. This restores a causal link from over-pacing to loss
    # without a reward term (the throughput-only reward is unchanged); it is
    # enforced only through the reported retransmission safety gate.
    overflow_retransmit_frac: float = 0.0

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
    v_bytes: float = 0.0  # queue backlog; pipe_bytes tracks propagation occupancy
    i_dwn: float = 0.0  # drawdown indicator, continuous in [0, 1]
    i_crs: float = 1.0  # cruise indicator, continuous in [0, 1]
    t_since_probe_s: float = 0.0
    startup_done: bool = True  # True = skip STARTUP modeling (pre-STARTUP-prefix default
    # behavior: every existing caller that constructs FluidState without this field, e.g. every
    # unit test written before STARTUP was added, is unaffected). Only FluidSimEnv.reset() sets
    # this False, opting a fresh episode into the STARTUP-phase prefix (see step_fluid_state).
    bbr_bw_est_bps: float = 0.0  # zero means use instantaneous capacity (legacy path)
    # Goal-2 dynamics state (defaults are the legacy no-op values):
    inflight_hi_scale: float = 1.0  # multiplicative backoff on the cwnd cap from loss / ECN
    t_since_probe_rtt_s: float = 0.0  # ProbeRTT phase timer
    round_elapsed_s: float = 0.0
    round_retransmits: float = 0.0
    round_delivered_bytes: float = 0.0
    startup_full_rounds: int = 0
    startup_best_rate: float = 0.0
    pipe_bytes: float = 0.0
    service_rate_bytes_s: float = 0.0
    in_probe_rtt: bool = False
    probe_phase: int = 0  # v7 ProbeBW state: 0 = CRUISE, 1 = UP, 2 = DOWN
    probe_phase_elapsed_s: float = 0.0


def sigmoid(x: float, k: float = 8.0) -> float:
    z = k * x
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _risk_sigmoid(p_tot: float, k: float) -> float:
    return sigmoid((p_tot - _P_TOT_RISK_THRESHOLD) / _P_TOT_RISK_THRESHOLD, k)


RECONFIG_FREEZE_HALF_WIDTH_S = 1.0  # matches synthetic_ground_truth_p_tot's handover_width_s convention


def in_reconfig_freeze_window(
    t_s: float,
    phase_offset_s: float,
    phase_profile: PhaseProfile | None,
    half_width_s: float = RECONFIG_FREEZE_HALF_WIDTH_S,
) -> bool:
    if phase_profile is None:
        return False
    cycle_s = phase_profile.cycle_s
    phase = (t_s + phase_offset_s) % cycle_s
    raw_dist = abs(phase - phase_profile.mean_phase_s)
    dist = min(raw_dist, cycle_s - raw_dist)
    return dist <= half_width_s


def probe_bw_interval_s(rtt_rtp_s: float, n_flows: int = 1, flow_index: int = 0) -> float:
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

    dwn_target = _dwn_target(state, p_tot, params, params.bdp_hat_bytes)
    stock_multiplier = 1.25 - 0.5 * dwn_target

    multiplier = (
        STARTUP_GAIN if not state.startup_done
        else state.i_crs * pacing_gain + (1.0 - state.i_crs) * stock_multiplier
    )
    full_capacity_bps = params.sustained_x_btl_bps * synthetic_capacity_fraction(state.t_s)
    bw_estimate_bps = (
        state.bbr_bw_est_bps if state.bbr_bw_est_bps > 0.0 else full_capacity_bps
    ) if params.bandwidth_estimate_recovery_s > 0.0 else full_capacity_bps
    return bw_estimate_bps * multiplier


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

    bdp = params.bdp_bytes
    k = params.sigmoid_k
    in_startup = not state.startup_done

    def _vol_sigmoid(delta_bytes: float) -> float:
        return sigmoid(delta_bytes / bdp, k)
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

    # consistent_transport (v6+) uses the exact exponential relaxation factor
    # so i_dwn / i_crs -- and everything they gate (retransmits, the loss
    # backoff, the RTT tail) -- do not depend on substep_s. The legacy path
    # keeps the original explicit factor to preserve calibrated outputs.
    relax = (
        -math.expm1(-params.relax_rate_hz * dt_s)
        if params.consistent_transport
        else min(1.0, params.relax_rate_hz * dt_s)
    )
    i_dwn = min(max(state.i_dwn + (dwn_target - state.i_dwn) * relax, 0.0), 1.0)
    i_crs = min(max(state.i_crs + (crs_target - state.i_crs) * relax, 0.0), 1.0)
    stock_multiplier = 1.25 - 0.5 * dwn_target

    if capacity_bps_override is not None:
        capacity_bps_now = capacity_bps_override
    else:
        capacity_bps_now = params.sustained_x_btl_bps * synthetic_capacity_fraction(state.t_s)
    recovery_enabled = params.bandwidth_estimate_recovery_s > 0.0
    bw_estimate_bps = (
        state.bbr_bw_est_bps if state.bbr_bw_est_bps > 0.0 else capacity_bps_now
    ) if recovery_enabled else capacity_bps_now

    probe_interval_s = probe_bw_interval_s(params.rtt_rtp_s, n_flows=n_flows, flow_index=flow_index)
    probe_phase = state.probe_phase
    probe_phase_elapsed_s = state.probe_phase_elapsed_s
    t_since_probe_s = state.t_since_probe_s + dt_s
    eff_gain = pacing_gain
    if params.probe_bw_cycle and not in_startup:
        # A pipe genuinely below its bandwidth-delay target: the post-handover
        # underfill window where pacing above 1.0 recovers throughput. Empty in
        # steady cruise (estimate tracked), so it never fires there.
        underfilled = (
            recovery_enabled
            and capacity_bps_now > bw_estimate_bps * 1.02
            and state.v_bytes < 0.50 * bdp
        )
        if probe_phase == 1:  # ProbeBW_UP: pace at the chosen amplitude
            eff_gain = max(pacing_gain, 1.0)
            probe_phase_elapsed_s += dt_s
            # One round by default; hold UP longer while still refilling a real
            # underfill (BBR-v3 stays in UP/REFILL until inflight hits target).
            if probe_phase_elapsed_s >= params.probe_up_rounds * params.rtt_rtp_s and not underfilled:
                probe_phase_elapsed_s = 0.0
                probe_phase = 2
        elif probe_phase == 2:  # ProbeBW_DOWN: drain at BBR's own down gain
            eff_gain = params.probe_down_gain
            probe_phase_elapsed_s += dt_s
            # Exit once drained (target-driven) or after the safety round cap.
            if state.v_bytes <= 0.05 * bdp or probe_phase_elapsed_s >= params.probe_down_rounds * params.rtt_rtp_s:
                probe_phase_elapsed_s = 0.0
                probe_phase = 0
        else:  # ProbeBW_CRUISE: a chosen gain > 1.0 waits here for the UP window
            eff_gain = min(pacing_gain, 1.0)
            # Leave CRUISE for ProbeBW_UP on the periodic timer (Eq. 22) OR
            # early when the pipe is underfull -- the latter is what lets a
            # *well-timed* high gain recover a post-handover hole a fixed
            # cadence would miss, without adding any RTT cost in steady cruise.
            if pacing_gain > 1.0 and (t_since_probe_s >= probe_interval_s or underfilled):
                probe_phase, probe_phase_elapsed_s = 1, 0.0
                t_since_probe_s = 0.0 if underfilled else t_since_probe_s - probe_interval_s
    elif t_since_probe_s >= probe_interval_s:
        t_since_probe_s = 0.0  # legacy reset (no cycle; value is vestigial)

    multiplier = STARTUP_GAIN if in_startup else (i_crs * eff_gain + (1.0 - i_crs) * stock_multiplier)

    # --- ProbeRTT phase (goal-2): a short periodic drain. off when interval == 0.
    probe_rtt_on = params.probe_rtt_interval_s > 0.0 and not in_startup
    t_since_probe_rtt_s = state.t_since_probe_rtt_s + dt_s
    in_probe_rtt = state.in_probe_rtt
    if probe_rtt_on:
        if in_probe_rtt and t_since_probe_rtt_s >= params.probe_rtt_duration_s:
            in_probe_rtt, t_since_probe_rtt_s = False, t_since_probe_rtt_s - params.probe_rtt_duration_s
        elif not in_probe_rtt and t_since_probe_rtt_s >= params.probe_rtt_interval_s:
            in_probe_rtt, t_since_probe_rtt_s = True, t_since_probe_rtt_s - params.probe_rtt_interval_s
    else:
        in_probe_rtt, t_since_probe_rtt_s = False, 0.0
    if in_probe_rtt:
        multiplier = min(multiplier, params.probe_rtt_cwnd_frac)  # pace down to drain

    pacing_rate_bps = bw_estimate_bps * multiplier

    injected_bytes = pacing_rate_bps * dt_s
    effective_capacity_bps = capacity_bps_now * max(
        1.0 - params.drain_throughput_penalty * i_dwn, 0.05
    )
    phase_mult = retransmit_phase_multiplier(state.t_s + phase_offset_s, params.phase_profile)
    retransmits = (params.base_retransmit_rate_pps + i_dwn * params.dwn_retransmit_rate_pps) * phase_mult * dt_s
    capacity_bytes = effective_capacity_bps * dt_s
    if params.overflow_retransmit_frac > 0.0:
        buffer_bytes = (params.cwnd_gain if params.cwnd_gain > 0.0 else params.drawdown_activate_mult) * bdp
        overshoot_frac = min(max(state.v_bytes / max(buffer_bytes, 1.0) - 1.0, 0.0), 1.0)
        retransmits += params.overflow_retransmit_frac * overshoot_frac * (capacity_bytes / MSS_BYTES)
    if params.consistent_transport:
        # Exogenous retransmission demand consumes wire service; this remains
        # a calibrated loss proxy, not a packet-level loss/recovery model.
        retransmits = min(retransmits, min(capacity_bytes, state.v_bytes + injected_bytes) / MSS_BYTES)
        capacity_bytes = max(capacity_bytes - retransmits * MSS_BYTES, 0.0)
    if params.cwnd_gain > 0.0:
        cap_frac = params.probe_rtt_cwnd_frac if in_probe_rtt else params.cwnd_gain
        max_backlog = max((cap_frac * state.inflight_hi_scale - 1.0) * bdp, 0.0)
        # v7b: cap the standing queue at a fixed queuing-delay budget (drained
        # at the current service rate) so a ProbeBW pulse -- and any residual
        # it leaves into cruise or a handover dip -- adds a bounded RTT cost,
        # the same on every path (fixes the +50 ms the probe cost on the
        # util=0.05 uplinks where a cwnd of queue is hundreds of ms).
        if params.probe_bw_cycle and params.probe_max_queue_delay_ms > 0.0:
            max_backlog = min(max_backlog, params.probe_max_queue_delay_ms / 1000.0 * effective_capacity_bps)
        if params.consistent_transport:
            cap_bytes = cap_frac * state.inflight_hi_scale * bdp
            capacity_bytes = min(capacity_bytes, cap_bytes / params.rtt_rtp_s * dt_s)
        room = max(max_backlog - state.v_bytes + capacity_bytes, 0.0)
        injected_bytes = min(injected_bytes, room)
    delivered_bytes = min(capacity_bytes, state.v_bytes + injected_bytes)
    v_bytes = max(state.v_bytes + injected_bytes - delivered_bytes, 0.0)

    if recovery_enabled:
        if capacity_bps_now <= bw_estimate_bps:
            next_bw_estimate_bps = capacity_bps_now
        else:
            # Exact exponential approach to a piecewise-constant target (same
            # idiom as pipe_bytes / inflight_hi_scale below): substep-size
            # independent, unlike the former explicit dt/tau Euler step.
            recovery = -math.expm1(-dt_s / params.bandwidth_estimate_recovery_s)
            next_bw_estimate_bps = bw_estimate_bps + recovery * (capacity_bps_now - bw_estimate_bps)
    else:
        next_bw_estimate_bps = 0.0
    round_elapsed = state.round_elapsed_s + dt_s
    round_retransmits = state.round_retransmits + retransmits
    round_delivered = state.round_delivered_bytes + delivered_bytes
    round_complete = round_elapsed + 1e-12 >= params.rtt_rtp_s
    loss_frac = round_retransmits / max(round_delivered / MSS_BYTES + round_retransmits, 1.0)
    full_rounds, best_rate = state.startup_full_rounds, state.startup_best_rate
    if round_complete and in_startup:
        rate = round_delivered / round_elapsed
        if rate >= 1.25 * best_rate and rate > 0:
            best_rate, full_rounds = rate, 0
        else:
            full_rounds += 1
    next_scale = state.inflight_hi_scale
    if params.cwnd_gain > 0.0:
        rec = -math.expm1(-dt_s / max(params.inflight_hi_recover_s, 1e-6))
        target_scale = 1.0
        if params.ecn_response_factor > 0.0:
            ecn_ind = _vol_sigmoid(v_bytes - params.ecn_response_thresh_bdp * bdp)
            target_scale -= params.ecn_response_factor * ecn_ind
        next_scale = state.inflight_hi_scale + (target_scale - state.inflight_hi_scale) * rec
        if params.loss_thresh > 0.0 and round_complete:
            if loss_frac > params.loss_thresh:
                next_scale *= params.loss_beta
        next_scale = min(max(next_scale, 0.25), 1.0)


    new_t_s = state.t_s + dt_s
    startup_done_now = state.startup_done or (v_bytes >= STARTUP_EXIT_MULT * bdp) or (new_t_s >= STARTUP_MAX_DURATION_S) or (full_rounds >= 3) or (round_complete and params.loss_thresh > 0 and loss_frac > params.loss_thresh)

    new_state = FluidState(
        t_s=new_t_s,
        v_bytes=v_bytes,
        i_dwn=i_dwn,
        i_crs=i_crs,
        t_since_probe_s=t_since_probe_s,
        startup_done=startup_done_now,
        bbr_bw_est_bps=next_bw_estimate_bps,
        inflight_hi_scale=next_scale,
        t_since_probe_rtt_s=t_since_probe_rtt_s,
        in_probe_rtt=in_probe_rtt,
        round_elapsed_s=0.0 if round_complete else round_elapsed,
        round_retransmits=0.0 if round_complete else round_retransmits,
        round_delivered_bytes=0.0 if round_complete else round_delivered,
        startup_full_rounds=full_rounds,
        startup_best_rate=best_rate,
        pipe_bytes=state.pipe_bytes + (-math.expm1(-dt_s / params.rtt_rtp_s)) * (delivered_bytes / dt_s * params.rtt_rtp_s - state.pipe_bytes),
        service_rate_bytes_s=effective_capacity_bps,
        probe_phase=probe_phase,
        probe_phase_elapsed_s=probe_phase_elapsed_s,
    )
    return new_state, delivered_bytes, retransmits
