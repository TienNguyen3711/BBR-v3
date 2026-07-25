"""Classical MLP actor-critic baseline, parameter-matched to QA2C (~36-54 params).

Trained under identical state, action, reward, environment, and seeding to
qbbr.agents.quantum.qa2c.QA2CAgent, in order to isolate the quantum
contribution -- see main.tex Sec. "Classical baseline" (RQ4).

Not yet implemented.
"""
from __future__ import annotations

from typing import Any

from qbbr.agents.base_agent import BaseAgent


class MLPA2CAgent(BaseAgent):
    def __init__(self, param_budget: int = 45, n_actions: int = 5) -> None:
        raise NotImplementedError("Classical MLP baseline not yet implemented.")

    def act(self, state: Any) -> tuple[int, float]:
        raise NotImplementedError

    def update(self, batch: Any) -> dict[str, float]:
        raise NotImplementedError

    def param_count(self) -> int:
        raise NotImplementedError
