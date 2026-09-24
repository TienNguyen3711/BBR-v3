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
        """Load weights saved by save() into this already-constructed agent."""


def discounted_returns(
    rewards: Sequence[float], gamma: float, bootstrap: float = 0.0
) -> torch.Tensor:
    """Discounted returns, optionally bootstrapped from a successor value."""
    returns = []
    g = float(bootstrap)
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return torch.tensor(returns, dtype=torch.float32)


class RunningReturnNormalizer:
    """Running mean/variance of discounted returns."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, returns: torch.Tensor) -> None:
        batch_mean = float(returns.mean())
        batch_var = float(returns.var(unbiased=False))
        batch_count = int(returns.numel())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.mean += delta * batch_count / total
        self.var = (m_a + m_b + delta * delta * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        return (returns - self.mean) / (self.var ** 0.5 + 1e-8)

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


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
