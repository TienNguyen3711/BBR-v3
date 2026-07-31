from __future__ import annotations

import socket
import struct
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

# struct tcp_info's historically-stable prefix (see module docstring); native
# byte order/alignment ("=") since this is read directly off a real local socket.
_TCP_INFO_FORMAT = "=BBBBBBBxIIIIIIIIIIIIIIIIIIIIIIII"  # 7 uint8 + 1 pad + 24 uint32
_TCP_INFO_SIZE = struct.calcsize(_TCP_INFO_FORMAT)
_TCP_INFO_FIELDS = [
    "state", "ca_state", "retransmits", "probes", "backoff", "options", "wscale",
    "rto", "ato", "snd_mss", "rcv_mss", "unacked", "sacked", "lost", "retrans",
    "fackets", "last_data_sent", "last_ack_sent", "last_data_recv", "last_ack_recv",
    "pmtu", "rcv_ssthresh", "rtt", "rttvar", "snd_ssthresh", "snd_cwnd", "advmss",
    "reordering", "rcv_rtt", "rcv_space", "total_retrans",
]
_SOL_TCP = getattr(socket, "IPPROTO_TCP", 6)
_TCP_INFO_OPT = getattr(socket, "TCP_INFO", 11)  # Linux-only; absent on macOS/BSD


def read_tcp_info(sock: socket.socket) -> dict[str, int]:
    raw = sock.getsockopt(_SOL_TCP, _TCP_INFO_OPT, _TCP_INFO_SIZE)
    values = struct.unpack(_TCP_INFO_FORMAT, raw[:_TCP_INFO_SIZE])
    return dict(zip(_TCP_INFO_FIELDS, values))


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

    def _decision_interval_s(self) -> float:
        if self._fixed_decision_interval_s is not None:
            return self._fixed_decision_interval_s
        return minrtt_100_decision_interval_s(self.calibration["RTT_min_ms"])

    def _telemetry_row(self, t_start: float, info: dict[str, int]) -> dict[str, float]:
        return {
            "t_start": t_start,
            "b_hat_mbps": info["snd_cwnd"] * info.get("snd_mss", 1460) * 8.0 / 1e6 / max(info["rtt"] / 1e6, 1e-6),
            "rtt_ms": info["rtt"] / 1000.0,  # tcpi_rtt is in microseconds
            "rtt_base_ms": info["rtt"] / 1000.0,  # no rolling baseline available yet from a single sample
            "v_over_bdp": 1.0,  # placeholder: real inflight-vs-BDP needs tcpi_notsent_bytes/unacked, not read here
            "q_packets": max(info["lost"], 0),
            "bits_per_second": info["snd_cwnd"] * info.get("snd_mss", 1460) * 8.0 / max(info["rtt"] / 1e6, 1e-6),
            "retransmits": float(info["total_retrans"]),
        }

    def reset(self, seed: int | None = None) -> Any:
        bbr_hook.clear_pacing_gain_override()
        self._episode_start_s = time.time()
        self._history = []
        self._consecutive_high_loss = 0

        info = read_tcp_info(self.sock)
        row = self._telemetry_row(0.0, info)
        self._history.append(row)

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        return state_df.iloc[-1][_STATE_COLS].to_numpy(dtype=float)

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        if self._episode_start_s is None:
            raise RuntimeError("call reset() before step()")

        t_dec_s = self._decision_interval_s()
        t_start = time.time() - self._episode_start_s

        # Guardrail (i): clamp restricting overrides to ProbeBW_CRUISE.
        if bbr_hook.in_probe_bw_cruise():
            if self._consecutive_high_loss >= self._max_consecutive_high_loss:
                bbr_hook.clear_pacing_gain_override()
            else:
                pacing_gain = level_for_action(self._action_config, action)
                bbr_hook.set_pacing_gain(pacing_gain)

        time.sleep(t_dec_s)  # let the real network run for one decision interval

        info = read_tcp_info(self.sock)
        row = self._telemetry_row(t_start, info)
        self._history.append(row)
        if row["retransmits"] > 0:
            self._consecutive_high_loss += 1
        else:
            self._consecutive_high_loss = 0

        window = pd.DataFrame(self._history[-2:])
        risk = compute_risk_features(window, mode=self.risk_mode)
        state_df = compute_state_vector(window, risk, self.calibration)
        reward_series = compute_reward(window)

        s_t = state_df.iloc[-1][_STATE_COLS].to_numpy(dtype=float)
        r_t = float(reward_series.iloc[-1])
        done = (time.time() - self._episode_start_s) >= self.episode_s
        return s_t, r_t, done, {"tcp_info": info, "t_dec_s": t_dec_s}

    @property
    def action_space_size(self) -> int:
        return n_actions(self._action_config)

    @property
    def observation_dim(self) -> int:
        return 6
