from __future__ import annotations

import abc
from typing import Any


class BaseEnv(abc.ABC):
    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Any:
        """Return the initial observation s_0 for this environment's declared MDP."""

    @abc.abstractmethod
    def step(self, action: int) -> tuple[Any, float, bool, dict]:
        """Apply a pacing_gain action and advance one decision interval T_dec.

        Returns (next_state, reward, done, info).
        """

    @property
    @abc.abstractmethod
    def action_space_size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def observation_dim(self) -> int:
        ...
