"""Shared actor-critic interface for both the quantum and classical cores.

Lets qbbr.train.loop treat QA2CAgent and MLPA2CAgent identically, so an
RQ4 quantum-vs-classical run differs only in which agent class is
instantiated -- both are built to the same parameter budget (~36-54 params).
"""
from __future__ import annotations

import abc
from typing import Any


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
