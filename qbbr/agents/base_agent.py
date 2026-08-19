from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Sequence

import torch


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state: Any) -> tuple[int, float]:
        """Sample an action index and its log-probability from pi_theta(state)."""

    @abc.abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """One A2C gradient step; returns a dict of loss/metric scalars."""

    @abc.abstractmethod
    def param_count(self) -> int:
        """Total trainable parameter count, for the RQ4 matched-budget comparison."""

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist actor/critic/optimizer weights to path (torch.save)."""

    @abc.abstractmethod
    def load(self, path: str | Path) -> None:
        """Load weights saved by save() into this already-constructed agent.

        The agent must already be built with matching hyperparameters
        (n_layers, reupload, ...); this only restores weights, not
        architecture.
        """


def discounted_returns(rewards: Sequence[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return torch.tensor(returns, dtype=torch.float32)


def a2c_losses(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    normalize_advantage: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:

    advantage = returns - values
    if normalize_advantage:
        actor_advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    else:
        actor_advantage = advantage
    actor_loss = -(log_probs * actor_advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()
    return actor_loss, critic_loss
