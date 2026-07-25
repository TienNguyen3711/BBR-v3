"""TestbedEnv: BaseEnv driving the real BBR-v3 kernel over a live Starlink link.

Exposes the same reset()/step() API as FluidSimEnv so qbbr.eval.scenario_a
and qbbr.eval.scenario_b can run either interchangeably. Reserved for
sim-to-real evaluation under the deployment guardrails described in
main.tex (ProbeBW_CRUISE-only clamp, retransmission fallback, latency
budget).

Not yet implemented -- depends on the qbbr.action.bbr_hook kernel hook.
"""
from __future__ import annotations

from qbbr.env.base_env import BaseEnv


class TestbedEnv(BaseEnv):
    def __init__(self, iface: str, decision_interval_s: float) -> None:
        raise NotImplementedError("TestbedEnv not yet implemented; see qbbr.action.bbr_hook.")

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
