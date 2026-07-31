from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qbbr.action.registry import level_for_action, load_action_space, n_actions
from qbbr.env.base_env import BaseEnv
from qbbr.env.fluid_sim import (
    MSS_BYTES,
    FluidParams,
    FluidState,
    step_fluid_state,
    synthetic_ground_truth_p_tot,
)
from qbbr.features.bbr_internals import compute_bhat_mbps
from qbbr.features.state_builder import compute_state_vector
from qbbr.reward.alpha_fair import compute_reward
from qbbr.risk.ptot import compute_risk_features

_DEFAULT_ACTION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "action_pacing_gain.yaml"
_SUBSTEP_S = 0.02  # internal fluid_sim integration step, independent of T_dec
_BHAT_WINDOW_S = 10.0  # matches qbbr.features.bbr_internals' 10-sample window at real traces' 1Hz rate
_STATE_COLS = ["s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue", "s5_handover_eta", "s6_p_tot"]


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

        rtt_rtp_s = self.calibration["RTT_min_ms"] / 1000.0
        x_btl_bps = self.calibration["B_max_mbps"] * 1e6 / 8.0
        utilization_fraction = self.calibration.get("utilization_fraction", 1.0)
        self.params = FluidParams(
            x_btl_bps=x_btl_bps, rtt_rtp_s=rtt_rtp_s, utilization_fraction=utilization_fraction
        )

        self._state: FluidState | None = None
        self._history: list[dict[str, float]] = []

    def reset(self, seed: int | None = None) -> Any:
        self._state = FluidState(t_s=0.0, v_bytes=self.params.bdp_bytes, i_dwn=0.0, i_crs=1.0)
        self._history = []
        row = self._telemetry_row(t_start=0.0, state=self._state, delivered_bytes=0.0, retransmits=0.0, t_dec_s=1.0)
        self._history.append(row)
        self._update_bhat(t_dec_s=1.0)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        return self._extract_state(state_df)

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        if self._state is None:
            raise RuntimeError("call reset() before step()")

        pacing_gain = level_for_action(self._action_config, action)
        t_dec_s = minrtt_100_decision_interval_s(self.calibration["RTT_min_ms"])

        t_start = self._state.t_s
        state = self._state
        delivered_total = 0.0
        retransmits_total = 0.0
        remaining = t_dec_s
        while remaining > 1e-9:
            dt = min(self.substep_s, remaining)
            p_tot = synthetic_ground_truth_p_tot(state.t_s)
            state, delivered, retransmits = step_fluid_state(state, pacing_gain, dt, self.params, p_tot)
            delivered_total += delivered
            retransmits_total += retransmits
            remaining -= dt
        self._state = state

        row = self._telemetry_row(t_start, state, delivered_total, retransmits_total, t_dec_s)
        self._history.append(row)
        self._update_bhat(t_dec_s)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        reward_series = compute_reward(window, **self.reward_kwargs)

        s_t = self._extract_state(state_df)
        r_t = float(reward_series.iloc[-1])
        done = state.t_s >= self.episode_s
        info = {
            "pacing_gain": pacing_gain,
            "t_dec_s": t_dec_s,
            "delivered_bytes": delivered_total,
            "retransmits": retransmits_total,
            "rtt_ms": row["rtt_ms"],
            "i_dwn": state.i_dwn,
            "i_crs": state.i_crs,
        }
        return s_t, r_t, done, info

    def _telemetry_row(
        self, t_start: float, state: FluidState, delivered_bytes: float, retransmits: float, t_dec_s: float
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
