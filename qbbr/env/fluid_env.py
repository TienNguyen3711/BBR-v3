"""FluidSimEnv: BaseEnv wrapping fluid_sim + features + reward for offline A2C training.

One episode corresponds to a 300s run (matching the base paper's measurement
protocol); step() advances by T_dec, the minRTT-100 decision interval.
Domain randomization around qbbr.env.calibration's per-location parameters
is the planned sim-to-real robustness mechanism.

Not yet implemented -- depends on qbbr.env.fluid_sim (dynamics), qbbr.risk.ptot
(risk process), and qbbr.reward.alpha_fair (reward).
"""
from __future__ import annotations

from qbbr.env.base_env import BaseEnv


class FluidSimEnv(BaseEnv):
    def __init__(
        self,
        location: str,
        direction: str,
        calibration: dict,
        risk_mode: str = "stub_constant",
    ) -> None:
        raise NotImplementedError("FluidSimEnv not yet implemented; see qbbr.env.fluid_sim.")

    def reset(self, seed: int | None = None):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    @property
    def action_space_size(self) -> int:
        raise NotImplementedError

    @property
    def observation_dim(self) -> int:
        return 6
