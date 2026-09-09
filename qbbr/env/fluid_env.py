from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from qbbr.action.registry import levels_for_action, load_action_space, n_actions
from qbbr.env.base_env import BaseEnv
from qbbr.env.fluid_sim import (
    MSS_BYTES,
    FluidParams,
    FluidState,
    PhaseProfile,
    ecn_marked,
    in_reconfig_freeze_window,
    step_fluid_state,
    synthetic_ground_truth_p_tot,
)
from qbbr.features.bbr_internals import compute_bhat_mbps
from qbbr.features.state_builder import compute_state_vector
from qbbr.reward.alpha_fair import compute_reward
from qbbr.reward.throughput_only import compute_throughput_only_reward
from qbbr.risk.ptot import closed_form_p_tot, compute_risk_features

_DEFAULT_ACTION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "action_pacing_gain.yaml"
_SUBSTEP_S = 0.02  # internal fluid_sim integration step, independent of T_dec
_BHAT_WINDOW_S = 10.0  # matches qbbr.features.bbr_internals' 10-sample window at real traces' 1Hz rate
_STATE_COLS = ["s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue", "s5_handover_eta", "s6_p_tot", "s7_reconfig_phase"]

_HANDOVER_PHASE_PROFILE = PhaseProfile(mean_phase_s=10.5, r_bar=0.7368)
_CRUISE_MASK_MIN = 0.5  # i_crs below this = not in ProbeBW_CRUISE; see probe_bw_phase_gate

_EMA_ALPHA = 0.5  # weight on the newly-chosen level each decision, vs. (1-alpha) on
# the running effective value. DNCCQ-PPO's EMA-smoothed action-magnitude technique
# (Yu et al., Eq. 4-5/9-11) applies compounding smoothing to damp abrupt action
# transitions, which can otherwise cause self-inflicted, oscillation-driven
# retransmits; adapted here to qbbr's per-decision discrete levels rather than
# their continuous direction+magnitude formulation, so 0.5 is a deliberately
# chosen moderate middle ground, not their paper's own alpha value (whose
# convention isn't directly comparable to this per-dimension formulation).


def minrtt_100_decision_interval_s(min_rtt_ms: float) -> float:
    if min_rtt_ms <= 100.0:
        return 2.0 * min_rtt_ms / 1000.0
    return (100.0 + min_rtt_ms) / 1000.0


class FluidSimEnv(BaseEnv):
    def __init__(
        self,
        location: str,
        direction: str,
        calibration: dict[str, dict[str, dict[str, float]]],
        action_config: dict[str, Any] | str | Path | None = None,
        risk_mode: str = "stub_constant",
        episode_s: float = 300.0,
        substep_s: float = _SUBSTEP_S,
        reward_kwargs: dict[str, float] | None = None,
        reward_mode: Literal["legacy_alpha_fair", "throughput_only"] = "legacy_alpha_fair",
        dynamics_overrides: dict[str, float] | None = None,
        ablate_s7: bool = False,
        probe_bw_phase_gate: bool = False,
    ) -> None:
        self.location = location
        self.direction = direction
        self.calibration = calibration[location][direction]
        # ablation switch for the Point-2 causal isolation study: when True,
        # s7_reconfig_phase is fixed at its neutral midpoint (0.5) instead of
        # tracking the real wall-clock reconfig phase, holding every other
        # dimension (incl. observation_dim / param_count) identical so the
        # only thing that changes is whether the agent can observe phase.
        self.ablate_s7 = ablate_s7
        # When True, non-stock native gains are only offered while BBR is
        # genuinely in ProbeBW_CRUISE (i_crs high). This matches the audited
        # contract (Table II: the non-1.0 gains are "PROBE_BW only") and stops
        # the agent from issuing a gain override every decision interval
        # regardless of BBR phase.
        self.probe_bw_phase_gate = probe_bw_phase_gate

        if action_config is None:
            action_config = _DEFAULT_ACTION_CONFIG_PATH
        if isinstance(action_config, (str, Path)):
            action_config = load_action_space(action_config)
        self._action_config = action_config

        self.risk_mode = risk_mode
        self.episode_s = episode_s
        self.substep_s = substep_s
        self.reward_kwargs = reward_kwargs or {}
        if reward_mode not in {"legacy_alpha_fair", "throughput_only"}:
            raise ValueError("reward_mode must be 'legacy_alpha_fair' or 'throughput_only'")
        # The historic simulator remains reproducible by default.  New QRL
        # pilots opt into the supervisor-approved throughput-only contract.
        self.reward_mode = reward_mode

        rtt_rtp_s = self.calibration["RTT_min_ms"] / 1000.0
        x_btl_bps = self.calibration["B_max_mbps"] * 1e6 / 8.0
        utilization_fraction = self.calibration.get("utilization_fraction", 1.0)

        dwn_retransmit_rate_pps = self.calibration.get("dwn_retransmit_rate_pps", FluidParams.__dataclass_fields__["dwn_retransmit_rate_pps"].default)
        drawdown_activate_mult = self.calibration.get("drawdown_activate_mult", FluidParams.__dataclass_fields__["drawdown_activate_mult"].default)
        steady_inflight_bdp_frac = self.calibration.get("steady_inflight_bdp_frac", 0.0)
        base_retransmit_rate_pps = self.calibration.get("base_retransmit_rate_pps", 0.0)
        self.params = FluidParams(
            x_btl_bps=x_btl_bps, rtt_rtp_s=rtt_rtp_s, utilization_fraction=utilization_fraction,
            dwn_retransmit_rate_pps=dwn_retransmit_rate_pps, drawdown_activate_mult=drawdown_activate_mult,
            steady_inflight_bdp_frac=steady_inflight_bdp_frac,
            base_retransmit_rate_pps=base_retransmit_rate_pps,
            phase_profile=_HANDOVER_PHASE_PROFILE,
        )
        if dynamics_overrides:
            allowed = set(FluidParams.__dataclass_fields__)
            unknown = set(dynamics_overrides) - allowed
            if unknown:
                raise ValueError(f"Unknown fluid dynamics overrides: {sorted(unknown)}")
            self.params = replace(self.params, **dynamics_overrides)
        self._phase_offset_s = 0.0  # redrawn per episode in reset()

        self._ground_truth_base_p = closed_form_p_tot()

        self._state: FluidState | None = None
        self._history: list[dict[str, float]] = []

    def reset(self, seed: int | None = None) -> Any:
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self._phase_offset_s = float(rng.uniform(0.0, 15.0))

        # EMA state for action smoothing, keyed the same way levels_for_action()
        # keys its dict; initialized to each dimension's stock/no-op value so the
        # first decision of the episode is smoothed toward "no override" rather
        # than an arbitrary starting point.
        self._ema_levels = {
            "pacing_gain": 1.0,
            "inflight_hi_mult": self.params.bdp_hi_mult,
            "inflight_lo_mult": self.params.bdp_lo_mult,
            "drawdown_activate_relative_mult": 1.0,
        }

        # v_bytes=0/i_crs=0/startup_done=False: a fresh episode is a true cold
        # start, opting into the STARTUP-phase prefix (see step_fluid_state) --
        # not an already-established connection at steady state (the pre-
        # STARTUP-prefix default, still used by FluidState()'s own defaults
        # and every direct construction elsewhere, e.g. unit tests).
        initial_bw_estimate = self.params.sustained_x_btl_bps if self.params.bandwidth_estimate_recovery_s > 0.0 else 0.0
        self._state = FluidState(
            t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=0.0, startup_done=False,
            bbr_bw_est_bps=initial_bw_estimate,
        )
        self._history = []
        row = self._telemetry_row(
            t_start=self._phase_offset_s, state=self._state, delivered_bytes=0.0, retransmits=0.0, t_dec_s=1.0
        )
        self._history.append(row)
        self._update_bhat(t_dec_s=1.0)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(
            window, risk, self.calibration,
            reconfig_cycle_s=self.params.phase_profile.cycle_s,
            reconfig_mean_phase_s=self.params.phase_profile.mean_phase_s,
        )
        return self._extract_state(state_df)

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")

        levels = levels_for_action(self._action_config, action)
        for dim, chosen in levels.items():
            if dim in self._ema_levels:
                self._ema_levels[dim] = _EMA_ALPHA * chosen + (1.0 - _EMA_ALPHA) * self._ema_levels[dim]

        pacing_gain = self._ema_levels["pacing_gain"] if "pacing_gain" in levels else 1.0
        inflight_hi_mult = self._ema_levels["inflight_hi_mult"] if "inflight_hi_mult" in levels else None
        inflight_lo_mult = self._ema_levels["inflight_lo_mult"] if "inflight_lo_mult" in levels else None
        drawdown_activate_mult = None
        if "drawdown_activate_relative_mult" in levels:
            drawdown_activate_mult = self._ema_levels["drawdown_activate_relative_mult"] * self.params.drawdown_activate_mult
        t_dec_s = minrtt_100_decision_interval_s(self.calibration["RTT_min_ms"])

        t_start = self._state.t_s + self._phase_offset_s
        state = self._state
        delivered_total = 0.0
        retransmits_total = 0.0
        ecn_marked_s = 0.0
        remaining = t_dec_s
        while remaining > 1e-9:
            dt = min(self.substep_s, remaining)
            p_tot = synthetic_ground_truth_p_tot(state.t_s, base_p=self._ground_truth_base_p)
            if in_reconfig_freeze_window(state.t_s, self._phase_offset_s, self.params.phase_profile):
                step_pacing_gain, step_inflight_hi, step_inflight_lo, step_drawdown = 1.0, None, None, None
            else:
                step_pacing_gain, step_inflight_hi, step_inflight_lo, step_drawdown = (
                    pacing_gain, inflight_hi_mult, inflight_lo_mult, drawdown_activate_mult,
                )
            # n_flows=1, flow_index=0 (defaults): this is a single isolated BBR
            # flow (Scenario A), i.e. i=1, N=1 in Eq. 22's 1-indexed i in {1,...,N}.
            state, delivered, retransmits = step_fluid_state(
                state, step_pacing_gain, dt, self.params, p_tot, phase_offset_s=self._phase_offset_s,
                inflight_hi_mult_override=step_inflight_hi, inflight_lo_mult_override=step_inflight_lo,
                drawdown_activate_mult_override=step_drawdown,
            )
            delivered_total += delivered
            retransmits_total += retransmits
            if ecn_marked(state.v_bytes, self.params.bdp_bytes):
                ecn_marked_s += dt
            remaining -= dt
        self._state = state

        row = self._telemetry_row(t_start, state, delivered_total, retransmits_total, t_dec_s, ecn_marked_s)
        self._history.append(row)
        self._update_bhat(t_dec_s)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(
            window, risk, self.calibration,
            reconfig_cycle_s=self.params.phase_profile.cycle_s,
            reconfig_mean_phase_s=self.params.phase_profile.mean_phase_s,
        )
        reward_series = (
            compute_throughput_only_reward(window)
            if self.reward_mode == "throughput_only"
            else compute_reward(window, **self.reward_kwargs)
        )

        s_t = self._extract_state(state_df)
        r_t = float(reward_series.iloc[-1])
        done = state.t_s >= self.episode_s
        info = {
            "pacing_gain": pacing_gain,
            "t_dec_s": t_dec_s,
            "t_start": t_start,
            "delivered_bytes": delivered_total,
            "retransmits": retransmits_total,
            "rtt_ms": row["rtt_ms"],
            "i_dwn": state.i_dwn,
            "i_crs": state.i_crs,
            "bbr_bw_est_bps": state.bbr_bw_est_bps,
        }
        return s_t, r_t, done, info

    def _telemetry_row(
        self, t_start: float, state: FluidState, delivered_bytes: float, retransmits: float, t_dec_s: float,
        ecn_marked_s: float = 0.0,
    ) -> dict[str, float]:
        bdp = self.params.bdp_bytes
        queue_delay_ms = max(state.v_bytes - bdp, 0.0) / self.params.sustained_x_btl_bps * 1000.0
        # Stage 1b: reported inflight/BDP carries a persistent pipe term (real
        # BBR-v3 holds ~1 BDP in flight); it shrinks as the flow drains. This
        # feeds s3_inflight_bdp only -- v_bytes, q_packets, RTT, and ECN stay
        # on the queue-backlog quantity. pipe_frac = 0 recovers legacy.
        pipe_frac = self.params.steady_inflight_bdp_frac * (1.0 - 0.5 * state.i_dwn)
        return {
            "t_start": t_start,
            "b_hat_mbps": 0.0,
            "rtt_ms": self.params.rtt_rtp_s * 1000.0 + queue_delay_ms,
            "rtt_base_ms": self.params.rtt_rtp_s * 1000.0,
            "v_over_bdp": pipe_frac + state.v_bytes / bdp,
            "q_packets": max(state.v_bytes - bdp, 0.0) / MSS_BYTES,
            "bits_per_second": delivered_bytes * 8.0 / t_dec_s,
            "retransmits": retransmits,
            "ecn_mark_fraction": ecn_marked_s / t_dec_s if t_dec_s > 0 else 0.0,
        }

    def _update_bhat(self, t_dec_s: float) -> None:
        bps_history = pd.Series([r["bits_per_second"] for r in self._history])
        window_samples = max(1, round(_BHAT_WINDOW_S / t_dec_s))
        self._history[-1]["b_hat_mbps"] = float(compute_bhat_mbps(bps_history, window=window_samples).iloc[-1])

    def _extract_state(self, state_df: pd.DataFrame):
        if self.ablate_s7:
            state_df = state_df.assign(s7_reconfig_phase=0.5)
        return state_df.iloc[-1][_STATE_COLS].to_numpy(dtype=float)

    @property
    def action_space_size(self) -> int:
        return n_actions(self._action_config)

    def allowed_action_indices(self) -> tuple[int, ...]:


        if self._state is None:
            raise RuntimeError("call reset() before querying available actions")
        stock = next(
            (
                index
                for index in range(self.action_space_size)
                if levels_for_action(self._action_config, index).get("pacing_gain") == 1.0
            ),
            0,
        )
        frozen = in_reconfig_freeze_window(
            self._state.t_s, self._phase_offset_s, self.params.phase_profile
        )
        off_cruise = self.probe_bw_phase_gate and self._state.i_crs < _CRUISE_MASK_MIN
        if not self._state.startup_done or frozen or off_cruise:
            return (stock,)
        return tuple(range(self.action_space_size))

    @property
    def observation_dim(self) -> int:
        return len(_STATE_COLS)
