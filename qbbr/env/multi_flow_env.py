from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qbbr.action.registry import levels_for_action, load_action_space, n_actions
from qbbr.env.base_env import BaseEnv
from qbbr.env.competing_ccas import CompetingCCAState, offered_rate_bps, update_competing_cca
from qbbr.env.fluid_env import _EMA_ALPHA, minrtt_100_decision_interval_s
from qbbr.env.fluid_sim import (
    MSS_BYTES,
    FluidParams,
    FluidState,
    PhaseProfile,
    bbr_offered_rate_bps,
    ecn_marked,
    in_reconfig_freeze_window,
    sigmoid,
    step_fluid_state,
    synthetic_capacity_fraction,
    synthetic_ground_truth_p_tot,
)
from qbbr.features.bbr_internals import compute_bhat_mbps
from qbbr.features.state_builder import compute_state_vector
from qbbr.reward.alpha_fair import compute_reward
from qbbr.risk.ptot import closed_form_p_tot, compute_risk_features

_DEFAULT_ACTION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "action_pacing_gain.yaml"
_SUBSTEP_S = 0.02
_BHAT_WINDOW_S = 10.0
_STATE_COLS = ["s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue", "s5_handover_eta", "s6_p_tot"]
_DEFAULT_COMPETING_CCAS = ("cubic", "vegas", "hybla")
_AGENT_KEY = "__agent__"
_QUEUE_DELAY_SCALE_S = 0.05  # simple linear proxy: shared queueing delay per unit oversubscription

_HANDOVER_PHASE_PROFILE = PhaseProfile(mean_phase_s=10.5, r_bar=0.7368)


class MultiFlowFluidEnv(BaseEnv):
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
        competing_ccas: tuple[str, ...] = _DEFAULT_COMPETING_CCAS,
    ) -> None:
        self.location = location
        self.direction = direction
        self.calibration = calibration[location][direction]

        if action_config is None:
            action_config = _DEFAULT_ACTION_CONFIG_PATH
        if isinstance(action_config, (str, Path)):
            action_config = load_action_space(action_config)
        self._action_config = action_config

        self.risk_mode = risk_mode
        self.episode_s = episode_s
        self.substep_s = substep_s
        self.reward_kwargs = reward_kwargs or {}
        self.competing_ccas = competing_ccas

        rtt_rtp_s = self.calibration["RTT_min_ms"] / 1000.0
        x_btl_bps = self.calibration["B_max_mbps"] * 1e6 / 8.0
        utilization_fraction = self.calibration.get("utilization_fraction", 1.0)
        # see FluidSimEnv's identical comment on dwn_retransmit_rate_pps.
        dwn_retransmit_rate_pps = self.calibration.get(
            "dwn_retransmit_rate_pps", FluidParams.__dataclass_fields__["dwn_retransmit_rate_pps"].default
        )
        drawdown_activate_mult = self.calibration.get(
            "drawdown_activate_mult", FluidParams.__dataclass_fields__["drawdown_activate_mult"].default
        )
        self.params = FluidParams(
            x_btl_bps=x_btl_bps, rtt_rtp_s=rtt_rtp_s, utilization_fraction=utilization_fraction,
            dwn_retransmit_rate_pps=dwn_retransmit_rate_pps, drawdown_activate_mult=drawdown_activate_mult,
            phase_profile=_HANDOVER_PHASE_PROFILE,
        )
        self._phase_offset_s = 0.0  # redrawn per episode in reset()

        # see FluidSimEnv's identical comment: ground truth is always the
        # real baseline, independent of risk_mode.
        self._ground_truth_base_p = closed_form_p_tot()

        self._agent_state: FluidState | None = None
        self._competing_states: dict[str, CompetingCCAState] = {}
        self._shared_rtt_s: float = rtt_rtp_s
        self._history: list[dict[str, float]] = []  # agent's own telemetry, for state/reward

    def reset(self, seed: int | None = None) -> Any:
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self._phase_offset_s = float(rng.uniform(0.0, 15.0))

        # see FluidSimEnv.reset()'s identical comment on EMA action smoothing.
        self._ema_levels = {
            "pacing_gain": 1.0,
            "inflight_hi_mult": self.params.bdp_hi_mult,
            "inflight_lo_mult": self.params.bdp_lo_mult,
            "drawdown_activate_relative_mult": 1.0,
        }

        # see FluidSimEnv.reset()'s identical comment on the STARTUP-phase cold start.
        self._agent_state = FluidState(t_s=0.0, v_bytes=0.0, i_dwn=0.0, i_crs=0.0, startup_done=False)
        self._competing_states = {
            name: CompetingCCAState(window_bytes=self.params.bdp_bytes) for name in self.competing_ccas
        }
        self._shared_rtt_s = self.params.rtt_rtp_s
        self._history = []

        row = self._telemetry_row(
            t_start=self._phase_offset_s, state=self._agent_state, delivered_bytes=0.0, retransmits=0.0, t_dec_s=1.0
        )
        self._history.append(row)
        self._update_bhat(t_dec_s=1.0)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        return self._extract_state(state_df)

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        if self._agent_state is None:
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

        t_start = self._agent_state.t_s + self._phase_offset_s
        agent_state = self._agent_state
        competing_states = dict(self._competing_states)
        shared_rtt_s = self._shared_rtt_s

        agent_delivered_total = 0.0
        agent_retransmits_total = 0.0
        agent_ecn_marked_s = 0.0
        flow_delivered_totals = {name: 0.0 for name in self.competing_ccas}

        remaining = t_dec_s
        while remaining > 1e-9:
            dt = min(self.substep_s, remaining)
            p_tot = synthetic_ground_truth_p_tot(agent_state.t_s, base_p=self._ground_truth_base_p)
            if in_reconfig_freeze_window(agent_state.t_s, self._phase_offset_s, self.params.phase_profile):
                step_pacing_gain, step_inflight_hi, step_inflight_lo, step_drawdown = 1.0, None, None, None
            else:
                step_pacing_gain, step_inflight_hi, step_inflight_lo, step_drawdown = (
                    pacing_gain, inflight_hi_mult, inflight_lo_mult, drawdown_activate_mult,
                )

            offered = {_AGENT_KEY: bbr_offered_rate_bps(agent_state, step_pacing_gain, p_tot, self.params)}
            for name, cca_state in competing_states.items():
                offered[name] = offered_rate_bps(cca_state, shared_rtt_s)
            total_offered = sum(offered.values())

            shared_capacity_now = self.params.sustained_x_btl_bps * synthetic_capacity_fraction(agent_state.t_s)

            # 2. proportional allocation under oversubscription.
            if total_offered <= shared_capacity_now or total_offered <= 0.0:
                delivered = dict(offered)
                oversubscription = 0.0
            else:
                delivered = {k: shared_capacity_now * (v / total_offered) for k, v in offered.items()}
                oversubscription = total_offered / shared_capacity_now - 1.0

            congestion_signal = min(
                1.0,
                sigmoid(oversubscription, self.params.sigmoid_k) + sigmoid(p_tot - 0.02, self.params.sigmoid_k),
            )
            queue_delay_s = _QUEUE_DELAY_SCALE_S * oversubscription
            shared_rtt_s = self.params.rtt_rtp_s + queue_delay_s
            agent_state, agent_delivered, agent_retransmits = step_fluid_state(
                agent_state, step_pacing_gain, dt, self.params, p_tot, capacity_bps_override=delivered[_AGENT_KEY],
                n_flows=1, flow_index=0, phase_offset_s=self._phase_offset_s,
                inflight_hi_mult_override=step_inflight_hi, inflight_lo_mult_override=step_inflight_lo,
                drawdown_activate_mult_override=step_drawdown,
            )
            agent_delivered_total += agent_delivered
            agent_retransmits_total += agent_retransmits
            if ecn_marked(agent_state.v_bytes, self.params.bdp_bytes):
                agent_ecn_marked_s += dt

            # 4. advance each competing flow.
            ctx = {
                "congestion_signal": congestion_signal,
                "rtt_base_s": self.params.rtt_rtp_s,
                "shared_rtt_s": shared_rtt_s,
                "dt_s": dt,
            }
            for name in self.competing_ccas:
                flow_delivered_totals[name] += delivered[name] * dt
                competing_states[name] = update_competing_cca(name, competing_states[name], ctx)

            remaining -= dt

        self._agent_state = agent_state
        self._competing_states = competing_states
        self._shared_rtt_s = shared_rtt_s

        row = self._telemetry_row(
            t_start, agent_state, agent_delivered_total, agent_retransmits_total, t_dec_s, agent_ecn_marked_s
        )
        self._history.append(row)
        self._update_bhat(t_dec_s)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        reward_series = compute_reward(window, **self.reward_kwargs)

        s_t = self._extract_state(state_df)
        r_t = float(reward_series.iloc[-1])
        done = agent_state.t_s >= self.episode_s

        flow_throughput_bps = {"qbbr": agent_delivered_total * 8.0 / t_dec_s}
        for name in self.competing_ccas:
            flow_throughput_bps[name] = flow_delivered_totals[name] * 8.0 / t_dec_s

        info = {
            "pacing_gain": pacing_gain,
            "t_dec_s": t_dec_s,
            "delivered_bytes": agent_delivered_total,
            "retransmits": agent_retransmits_total,
            "rtt_ms": row["rtt_ms"],
            "shared_rtt_ms": shared_rtt_s * 1000.0,
            "flow_throughput_bps": flow_throughput_bps,
        }
        return s_t, r_t, done, info

    def _telemetry_row(
        self, t_start: float, state: FluidState, delivered_bytes: float, retransmits: float, t_dec_s: float,
        ecn_marked_s: float = 0.0,
    ) -> dict[str, float]:
        bdp = self.params.bdp_bytes
        queue_delay_ms = max(state.v_bytes - bdp, 0.0) / self.params.sustained_x_btl_bps * 1000.0
        return {
            "t_start": t_start,
            "b_hat_mbps": 0.0,
            "rtt_ms": self.params.rtt_rtp_s * 1000.0 + queue_delay_ms,
            "rtt_base_ms": self.params.rtt_rtp_s * 1000.0,
            "v_over_bdp": state.v_bytes / bdp,
            "q_packets": max(state.v_bytes - bdp, 0.0) / MSS_BYTES,
            "bits_per_second": delivered_bytes * 8.0 / t_dec_s,
            "retransmits": retransmits,
            "ecn_mark_fraction": ecn_marked_s / t_dec_s if t_dec_s > 0 else 0.0,
        }

    def _update_bhat(self, t_dec_s: float) -> None:
        bps_history = pd.Series([r["bits_per_second"] for r in self._history])
        window_samples = max(1, round(_BHAT_WINDOW_S / t_dec_s))
        self._history[-1]["b_hat_mbps"] = float(compute_bhat_mbps(bps_history, window=window_samples).iloc[-1])

    @staticmethod
    def _extract_state(state_df: pd.DataFrame):
        return state_df.iloc[-1][_STATE_COLS].to_numpy(dtype=float)

    @property
    def action_space_size(self) -> int:
        return n_actions(self._action_config)

    @property
    def observation_dim(self) -> int:
        return 6
