"""Offline A2C training loop: rollout -> advantage -> gradient step.

Wraps a qbbr.env (FluidSimEnv for training, TestbedEnv for sim-to-real
evaluation) with a qbbr.agents.base_agent.BaseAgent (quantum QA2C or
classical MLP), collecting qbbr.train.buffer.RolloutBuffer episodes and
applying Adam updates (parameter-shift gradients on real quantum hardware,
analytic backprop on the PennyLane simulator during development -- see
main.tex Sec. "Gradients"). Target: 200-500 episodes per
configuration-location pair, per main.tex's training methodology.

Not yet implemented -- this is the Phase 3 deliverable, gated on
qbbr.env.fluid_env and qbbr.agents being implemented first.
"""
from __future__ import annotations

from typing import Any


def train(agent: Any, env: Any, n_episodes: int, config: dict[str, Any]) -> dict[str, Any]:
    """Run n_episodes of on-policy A2C training; returns summary training metrics."""
    raise NotImplementedError(
        "A2C training loop not yet implemented; see qbbr.train.buffer, qbbr.train.logging."
    )
