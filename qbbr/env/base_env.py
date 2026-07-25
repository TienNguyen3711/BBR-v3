"""Abstract Gym-style environment interface shared by FluidSimEnv and TestbedEnv.

Both offline training (the fluid-model simulator) and sim-to-real evaluation
(the real BBR-v3 testbed) implement this same interface, so qbbr.train.loop
and qbbr.agents never need to know which one they're talking to.
"""
from __future__ import annotations

import abc
from typing import Any


class BaseEnv(abc.ABC):
    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> Any:
        """Return the initial 6-dim state s_0."""

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
