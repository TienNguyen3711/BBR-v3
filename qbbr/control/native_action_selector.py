"""Safety and deployment selection over the frozen native BBR-v3 actions.

The selector never creates a transport action. Its hard mask retains only
native actions that are safe in the observed BBR state. It also retains the
previous soft stock-fold as an explicitly *deployment-audit* distribution;
that distribution is not used to compute A2C gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class NativeActionSelector:
    """State-aware, stock-anchored selection within existing BBR actions.

    The congestion guard blocks the high native gains when any of four
    canonical state features cross a configured threshold: inflight/BDP
    (``s3``), queue occupancy (``s4``), excess RTT (``s2``), or proximity to
    the reconfiguration-phase centre (``s7``). The RTT and reconfiguration
    thresholds default to ``1.0`` -- i.e. disabled -- so existing protocols
    keep their behaviour until they opt in.
    """

    stock_action: int = 2
    min_logit_advantage: float = 0.15
    confidence_temperature: float = 0.10
    max_inflight_state: float = 0.45
    max_queue_state: float = 0.20
    max_excess_rtt_state: float = 1.0
    max_reconfig_phase_proximity: float = 1.0
    high_gain_actions: tuple[int, ...] = (3, 4)
    # Low native gains are admissible only under genuine queue/inflight
    # pressure. In headroom a sub-1.0 gain only under-fills the pipe -- a pure
    # throughput loss with no compensating benefit -- so a policy should never
    # converge to it there. Empty tuple (default) disables this guard, keeping
    # legacy behaviour. It is native-state admissibility over the frozen action
    # set, not an added action.
    low_gain_actions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.min_logit_advantage < 0.0:
            raise ValueError("min_logit_advantage must be non-negative.")
        if self.confidence_temperature <= 0.0:
            raise ValueError("confidence_temperature must be positive.")
        for name in ("max_inflight_state", "max_queue_state",
                     "max_excess_rtt_state", "max_reconfig_phase_proximity"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; the canonical state is clipped there.")
        if self.stock_action in self.high_gain_actions or self.stock_action in self.low_gain_actions:
            raise ValueError("The stock action cannot be a guarded high- or low-gain action.")

    def _state_values(self, state: torch.Tensor) -> tuple[float, float, float, float]:
        flat = state.detach().reshape(-1)
        if flat.numel() < 7:
            raise ValueError("Native action selection requires the canonical seven-state input.")
        # (s2 excess RTT, s3 inflight/BDP, s4 queue, s7 reconfig-phase proximity)
        return (float(flat[1].item()), float(flat[2].item()),
                float(flat[3].item()), float(flat[6].item()))

    def admissible_actions(
        self, logits: torch.Tensor, state: torch.Tensor, allowed_actions: Sequence[int],
    ) -> tuple[int, ...]:
        """Return actions admitted by the hard safety part of the selector."""

        allowed = tuple(int(item) for item in allowed_actions)
        if not allowed:
            raise ValueError("A native action mask must retain at least one action.")
        if self.stock_action not in allowed:
            return allowed

        excess_rtt, inflight_state, queue_state, reconfig_proximity = self._state_values(state)
        congested = (
            inflight_state >= self.max_inflight_state
            or queue_state >= self.max_queue_state
            or excess_rtt >= self.max_excess_rtt_state
            or reconfig_proximity >= self.max_reconfig_phase_proximity
        )
        admitted = []
        for action in allowed:
            if action == self.stock_action:
                admitted.append(action)
                continue
            if congested and action in self.high_gain_actions:
                continue
            if not congested and action in self.low_gain_actions:
                continue
            admitted.append(action)
        return tuple(admitted) if admitted else (self.stock_action,)

    def hard_distribution(
        self, logits: torch.Tensor, state: torch.Tensor, allowed_actions: Sequence[int],
    ) -> torch.distributions.Categorical:
        """Policy distribution for learning: native action mask plus hard safety.

        This is the behaviour policy for A2C rollouts and updates. It avoids
        an on-policy chicken-and-egg loop in which a soft stock anchor must be
        overcome before the actor can receive credit for a non-stock action.
        """

        admitted = self.admissible_actions(logits, state, allowed_actions)
        masked = torch.full_like(logits, float("-inf"))
        masked[list(admitted)] = logits[list(admitted)]
        return torch.distributions.Categorical(logits=masked)

    def distribution(
        self, logits: torch.Tensor, state: torch.Tensor, allowed_actions: Sequence[int],
    ) -> torch.distributions.Categorical:
        """Return the legacy soft-fold distribution for deployment audits only.

        Callers must use :meth:`hard_distribution` for A2C sampling and
        optimisation. Keeping this method makes the old shield observable in
        reports without silently changing the training policy.
        """

        allowed = tuple(int(item) for item in allowed_actions)
        admitted = self.admissible_actions(logits, state, allowed)
        masked = torch.full_like(logits, float("-inf"))
        masked[list(allowed)] = logits[list(allowed)]
        raw_probabilities = torch.softmax(masked, dim=-1)

        effective = torch.zeros_like(raw_probabilities)
        effective[list(admitted)] = raw_probabilities[list(admitted)]
        if self.stock_action in allowed:
            stock_logit = logits[self.stock_action]
            for action in admitted:
                if action == self.stock_action:
                    continue
                confidence = torch.sigmoid(
                    (logits[action] - stock_logit - self.min_logit_advantage)
                    / self.confidence_temperature
                )
                effective[action] = raw_probabilities[action] * confidence
            effective[self.stock_action] = (
                effective[self.stock_action]
                + raw_probabilities[
                    [action for action in allowed if action not in admitted]
                ].sum()
                + sum(
                    (
                        raw_probabilities[action] - effective[action]
                        for action in admitted
                        if action != self.stock_action
                    ),
                    torch.zeros_like(raw_probabilities[self.stock_action]),
                )
            )
        total = effective.sum()
        if not bool(torch.isfinite(total)) or float(total.detach().item()) <= 0.0:
            raise ValueError("Native selector produced an invalid policy distribution.")
        return torch.distributions.Categorical(probs=effective / total)

    def deployment_action(
        self, logits: torch.Tensor, state: torch.Tensor, allowed_actions: Sequence[int],
    ) -> int:
        """Choose one safe existing BBR action for deterministic deployment.

        A non-stock candidate must beat stock by the declared logit margin.
        This is a deterministic confidence rule, not an added BBR action, and
        avoids using ``argmax`` on a folded probability distribution whose
        accumulated stock mass can reverse the actor's ranking.
        """

        admitted = self.admissible_actions(logits, state, allowed_actions)
        hard_logits = torch.full_like(logits, float("-inf"))
        hard_logits[list(admitted)] = logits[list(admitted)]
        candidate = int(torch.argmax(hard_logits).item())
        if self.stock_action not in admitted or candidate == self.stock_action:
            return candidate
        if logits[candidate] >= logits[self.stock_action] + self.min_logit_advantage:
            return candidate
        return self.stock_action
