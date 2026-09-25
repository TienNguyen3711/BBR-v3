"""Shared-bottleneck coexistence under the frozen native BBR-v3 contract."""
from __future__ import annotations

from typing import Any

from qbbr.env.competing_ccas import CompetingCCAState, offered_rate_bps, update_competing_cca
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.env.fluid_sim import (
    bbr_offered_rate_bps, sigmoid, synthetic_capacity_fraction,
)

DEFAULT_COMPETING_CCAS = ("cubic", "vegas", "hybla")
AGENT_FLOW = "qbbr"
_QUEUE_DELAY_SCALE_S = 0.05  # carried over from MultiFlowFluidEnv unchanged


class NativeMultiFlowEnv(FluidSimEnv):
    def __init__(self, *args: Any, competing_ccas: tuple[str, ...] = DEFAULT_COMPETING_CCAS,
                 **kwargs: Any) -> None:
        if not competing_ccas:
            raise ValueError("Coexistence needs at least one competing flow")
        # Set before super().__init__ so that a capacity lookup during
        # construction or reset never sees a half-built competitor set.
        self.competing_ccas = tuple(competing_ccas)
        self._competing_states: dict[str, CompetingCCAState] = {}
        self._competing_delivered_bytes: dict[str, float] = {}
        self._shared_rtt_s = 0.0
        super().__init__(*args, **kwargs)
        self._reset_competitors()

    # ---------------------------------------------------------------- state

    def _reset_competitors(self) -> None:
        self._competing_states = {
            name: CompetingCCAState(window_bytes=self.params.bdp_bytes)
            for name in self.competing_ccas
        }
        self._competing_delivered_bytes = {name: 0.0 for name in self.competing_ccas}
        self._shared_rtt_s = self.params.rtt_rtp_s

    def reset(self, seed: int | None = None) -> Any:
        self._reset_competitors()
        return super().reset(seed=seed)

    # ------------------------------------------------------------ coupling

    def _total_capacity_at(self, t_s: float) -> float:
        """Absolute bottleneck capacity, resolving the base class's None."""
        replayed = super()._capacity_at(t_s)
        if replayed is not None:
            return float(replayed)
        return self.params.sustained_x_btl_bps * synthetic_capacity_fraction(t_s)

    def _capacity_at(self, t_s: float, **substep: Any) -> float:
        """The agent's share of a contended bottleneck for this substep."""
        dt_s = substep.get("dt_s")
        flow_state = substep.get("flow_state")
        if dt_s is None or flow_state is None:
            return self._total_capacity_at(t_s)

        p_tot = float(substep.get("p_tot") or 0.0)
        pacing_gain = substep.get("pacing_gain")
        offered = {AGENT_FLOW: bbr_offered_rate_bps(flow_state, pacing_gain, p_tot, self.params)}
        for name, state in self._competing_states.items():
            offered[name] = offered_rate_bps(state, self._shared_rtt_s)
        total_offered = sum(offered.values())

        capacity = self._total_capacity_at(t_s)
        if total_offered <= capacity or total_offered <= 0.0:
            delivered = dict(offered)
            oversubscription = 0.0
        else:
            delivered = {k: capacity * (v / total_offered) for k, v in offered.items()}
            oversubscription = total_offered / capacity - 1.0

        congestion_signal = min(1.0, sigmoid(oversubscription, self.params.sigmoid_k)
                                + sigmoid(p_tot - 0.02, self.params.sigmoid_k))
        self._shared_rtt_s = self.params.rtt_rtp_s + _QUEUE_DELAY_SCALE_S * oversubscription
        context = {"congestion_signal": congestion_signal, "rtt_base_s": self.params.rtt_rtp_s,
                   "shared_rtt_s": self._shared_rtt_s, "dt_s": dt_s}
        for name in self.competing_ccas:
            self._competing_delivered_bytes[name] += delivered[name] * dt_s
            self._competing_states[name] = update_competing_cca(
                name, self._competing_states[name], context)
        return delivered[AGENT_FLOW]

    # ----------------------------------------------------------------- step

    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        for name in self._competing_delivered_bytes:
            self._competing_delivered_bytes[name] = 0.0
        observation, reward, done, info = super().step(action)
        interval_s = float(info["t_dec_s"])
        throughput = {AGENT_FLOW: float(info["delivered_bytes"]) * 8.0 / interval_s}
        for name, delivered in self._competing_delivered_bytes.items():
            throughput[name] = delivered * 8.0 / interval_s
        info["flow_throughput_bps"] = throughput
        info["shared_rtt_s"] = self._shared_rtt_s
        info["competing_ccas"] = list(self.competing_ccas)
        return observation, reward, done, info
