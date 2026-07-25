"""On-policy rollout storage for A2C.

Holds one episode's (state, action, log_prob, reward, value) tuples until
qbbr.train.loop consumes them for an advantage estimate and gradient
update, then clears -- A2C is on-policy, so nothing here is replayed
across episodes.

Not yet implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutBuffer:
    states: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(self, state: Any, action: int, log_prob: float, reward: float, value: float) -> None:
        raise NotImplementedError("Rollout buffer not yet implemented.")

    def clear(self) -> None:
        raise NotImplementedError("Rollout buffer not yet implemented.")
