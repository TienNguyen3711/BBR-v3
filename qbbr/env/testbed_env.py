from __future__ import annotations

import socket
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from qbbr.action import bbr_hook
from qbbr.action.registry import level_for_action, load_action_space, n_actions
from qbbr.env.base_env import BaseEnv
from qbbr.env.fluid_env import minrtt_100_decision_interval_s
from qbbr.features.state_builder import compute_state_vector
from qbbr.reward.alpha_fair import compute_reward
from qbbr.risk.ptot import compute_risk_features

_DEFAULT_ACTION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "action_pacing_gain.yaml"
_STATE_COLS = ["s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue", "s5_handover_eta", "s6_p_tot"]

from qbbr.field.tcp_info import (
    read_tcp_info, require_measurement_fields, retransmit_delta,
    _TCP_INFO_FORMAT, _TCP_INFO_SIZE, _TCP_INFO_FIELDS,
)
from qbbr.action.heartbeat import CommandHeartbeat


class TestbedEnv(BaseEnv):
    __test__ = False  # not a pytest test class; name coincidentally starts with "Test"

    def __init__(
        self,
        sock: socket.socket,
        calibration: dict[str, Any],
        action_config: dict[str, Any] | str | Path | None = None,
        risk_mode: str = "stub_constant",
        episode_s: float = 300.0,
        decision_interval_s: float | None = None,
        max_consecutive_high_loss_intervals: int = 5,  # main.tex's k=5 fallback guardrail
    ) -> None:

        if not math.isfinite(episode_s) or episode_s <= 0:
            raise ValueError("episode_s must be finite and positive")
        if decision_interval_s is not None and (not math.isfinite(decision_interval_s) or decision_interval_s <= 0):
            raise ValueError("decision_interval_s must be finite and positive")
        self.sock = sock
        self.calibration = calibration
        self.risk_mode = risk_mode
        self.episode_s = episode_s
        self._fixed_decision_interval_s = decision_interval_s
        self._max_consecutive_high_loss = max_consecutive_high_loss_intervals

        if action_config is None:
            action_config = _DEFAULT_ACTION_CONFIG_PATH
        if isinstance(action_config, (str, Path)):
            action_config = load_action_space(action_config)
        self._action_config = action_config

        self._episode_start_s: float | None = None
        self._history: list[dict[str, float]] = []
        self._consecutive_high_loss = 0
        self._heartbeat = None
        self._previous_info = None
        self._previous_time = None

    def _decision_interval_s(self) -> float:
        if self._fixed_decision_interval_s is not None:
            return self._fixed_decision_interval_s
        return minrtt_100_decision_interval_s(self.calibration["RTT_min_ms"])

    def _telemetry_row(self, now: float, info: dict[str, int]) -> dict[str, float]:
        require_measurement_fields(info)
        previous = self._previous_info
        elapsed = 0.0 if self._previous_time is None else now - self._previous_time
        if previous is not None and elapsed <= 0:
            raise RuntimeError("Non-positive telemetry interval")
        delivered = 0 if previous is None else info["bytes_acked"] - previous["bytes_acked"]
        if delivered < 0:
            raise RuntimeError("Acknowledged-byte counter decreased; socket changed")
        retrans = 0 if previous is None else retransmit_delta(info["total_retrans"], previous["total_retrans"])
        throughput = delivered * 8 / elapsed if elapsed else 0.0
        baseline_ms = info["min_rtt"] / 1000.0
        if info["min_rtt"] in (0, 0xffffffff):
            baseline_ms = self.calibration["RTT_min_ms"]
        bandwidth = info["delivery_rate"] * 8.0
        bdp = max(bandwidth / 8 * baseline_ms / 1000, 1.0)
        inflight = max(info["unacked"] - info["sacked"] - info["lost"] + info["retrans"], 0) * info["snd_mss"]
        # Queue occupancy is an estimate from measured inflight and BDP,
        # not the number of lost packets or a direct bottleneck measurement.
        queue_estimate = max(inflight - bdp, 0.0)
        self._previous_info, self._previous_time = info, now
        return {
            "t_start": now - self._episode_start_s,
            "b_hat_mbps": bandwidth / 1e6,
            "rtt_ms": info["rtt"] / 1000.0,
            "rtt_base_ms": baseline_ms,
            "v_over_bdp": inflight / bdp,
            "q_packets": queue_estimate / max(info["snd_mss"], 1),
            "bits_per_second": throughput,
            "retransmits": float(retrans),
            "retransmits_per_second": retrans / elapsed if elapsed else 0.0,
            "delivered_bytes": delivered,
            "interval_s": elapsed,
        }

    def reset(self, seed: int | None = None) -> Any:
        self.close()
        self._episode_start_s = time.monotonic()
        self._history = []
        self._consecutive_high_loss = 0
        self._previous_info = self._previous_time = None
        info = read_tcp_info(self.sock)
        row = self._telemetry_row(self._episode_start_s, info)
        self._history.append(row)
        self._heartbeat = CommandHeartbeat()
        return self._observation()[0]

    def _observation(self):
        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        return state_df.iloc[-1][_STATE_COLS].to_numpy(dtype=float), window

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        if self._episode_start_s is None or self._heartbeat is None:
            raise RuntimeError("call reset() before step()")
        try:
            remaining = self.episode_s - (time.monotonic() - self._episode_start_s)
            if remaining <= 0:
                raise RuntimeError("Episode has ended; reset before stepping")
            t_dec_s = min(self._decision_interval_s(), remaining)
            gain = level_for_action(self._action_config, action)
            if self._consecutive_high_loss >= self._max_consecutive_high_loss:
                gain = None
            # Queue requests before CRUISE entry; the kernel gates application.
            self._heartbeat.submit(gain, valid_for_s=t_dec_s + 1.0)
            time.sleep(t_dec_s)
            self._heartbeat.check()
            info = read_tcp_info(self.sock)
            row = self._telemetry_row(time.monotonic(), info)
            self._history.append(row)
            self._consecutive_high_loss = self._consecutive_high_loss + 1 if row["retransmits"] > 0 else 0
            if self._consecutive_high_loss >= self._max_consecutive_high_loss:
                gain = None
                self._heartbeat.submit(None, valid_for_s=1.0)
            state, window = self._observation()
            reward = float(compute_reward(window).iloc[-1])
            done = time.monotonic() - self._episode_start_s >= self.episode_s
            if done:
                self.close()
            return state, reward, done, {"tcp_info": info, "t_dec_s": t_dec_s,
                "telemetry": row, "requested_pacing_gain": gain,
                "throughput_source": "TCP_INFO acknowledged-byte delta",
                "queue_source": "inflight-minus-BDP estimate"}
        except BaseException:
            try:
                self.close()
            except OSError:
                pass  # preserve the original failure; kernel lease is the fallback
            raise

    def close(self):
        heartbeat, self._heartbeat = self._heartbeat, None
        if heartbeat is not None:
            heartbeat.close()
        else:
            bbr_hook.clear_pacing_gain_override()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def action_space_size(self) -> int:
        return n_actions(self._action_config)

    @property
    def observation_dim(self) -> int:
        return 6
