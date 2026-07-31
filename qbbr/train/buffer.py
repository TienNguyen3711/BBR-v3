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
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)
