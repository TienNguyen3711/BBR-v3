"""Real-control-loop environment for the constrained QRL--BBR brain.

No shell/debugfs implementation is hidden here.  A deployment must provide an
audited adapter for its exact BBR-v3 kernel build before a native semantic can
be enabled in the contract.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol

from qbbr.control.contracts import ActionSpec, MDPContract
from qbbr.control.safety import ActionDecision, enforce_action
from qbbr.control.space import NativeActionSpace, encode_observation
from qbbr.env.base_env import BaseEnv
from qbbr.reward.native_performance import NativePerformanceReward


class NativeBBRAdapter(Protocol):
    """The narrow, auditable boundary between QRL and a BBR-v3 testbed."""

    def snapshot(self) -> Mapping[str, object]:
        """Return one complete MDP observation from the live testbed."""

    def apply_native_action(self, action: ActionSpec) -> None:
        """Apply one contract-declared existing BBR action and its fixed values."""

    def advance(self, decision_interval_s: float) -> None:
        """Let the live transfer advance; implementation owns timing/telemetry."""


class NativeBBRControlEnv(BaseEnv):
    """Control loop whose action type is an index into named native semantics."""

    def __init__(
        self,
        adapter: NativeBBRAdapter,
        contract: MDPContract,
        reward: NativePerformanceReward | None = None,
        observation_transform: Callable[[Mapping[str, object]], object] | None = None,
        decision_interval_s: float = 1.0,
        episode_s: float = 300.0,
    ) -> None:
        if decision_interval_s <= 0 or episode_s <= 0:
            raise ValueError("decision_interval_s and episode_s must be positive")
        self.adapter = adapter
        self.contract = contract
        self.action_space = NativeActionSpace(contract)
        self.reward = reward or NativePerformanceReward()
        self.observation_transform = observation_transform
        self.decision_interval_s = decision_interval_s
        self.episode_s = episode_s
        self._elapsed_s = 0.0
        self._observation: Mapping[str, object] | None = None
        self._previous_retransmission_rate: float | None = None

    def reset(self, seed: int | None = None):
        del seed  # live testbed repeatability is represented by the run manifest, not a PRNG.
        self._elapsed_s = 0.0
        self._observation = self.adapter.snapshot()
        self._previous_retransmission_rate = None
        return self._encode(self._observation)

    def _encode(self, observation: Mapping[str, object]):
        if self.observation_transform is not None:
            return self.observation_transform(observation)
        return encode_observation(self.contract, observation)

    def allowed_action_indices(self) -> tuple[int, ...]:
        if self._observation is None:
            raise RuntimeError("call reset() before querying native actions")
        state = str(self._observation["bbr_state"])
        loss_rate = self._observation.get("retransmission_rate")
        return self.action_space.allowed_indices(
            state,
            self._observation,
            float(loss_rate) if loss_rate is not None else None,
        )

    def step(self, action: int):
        if self._observation is None:
            raise RuntimeError("call reset() before step()")
        requested = self.action_space.action_for_index(action)
        current = self._observation
        decision: ActionDecision = enforce_action(
            self.contract,
            requested,
            str(current["bbr_state"]),
            current,
            float(current["retransmission_rate"]),
        )
        applied_spec = self.action_space.spec_for_id(decision.applied_action)
        self.adapter.apply_native_action(applied_spec)
        self.adapter.advance(self.decision_interval_s)
        next_observation = self.adapter.snapshot()
        reward = self.reward(
            throughput_bps=float(next_observation["delivery_rate_bps"]),
            rtt_s=float(next_observation["rtt_s"]),
            min_rtt_s=float(next_observation["min_rtt_s"]),
            retransmission_rate=float(next_observation["retransmission_rate"]),
            previous_retransmission_rate=self._previous_retransmission_rate,
        )
        self._previous_retransmission_rate = float(next_observation["retransmission_rate"])
        self._observation = next_observation
        self._elapsed_s += self.decision_interval_s
        return (
            self._encode(next_observation),
            reward,
            self._elapsed_s >= self.episode_s,
            {
                "requested_action": requested,
                "applied_action": decision.applied_action,
                "applied_fixed_parameters": applied_spec.parameters,
                "used_stock_fallback": decision.used_stock_fallback,
                "safety_reason": decision.reason,
                "raw_observation": dict(next_observation),
            },
        )

    @property
    def action_space_size(self) -> int:
        return self.action_space.size

    @property
    def observation_dim(self) -> int:
        return len(self.contract.observation_space)
