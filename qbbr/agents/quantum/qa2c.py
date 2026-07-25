"""QA2C: quantum actor pi_theta + quantum critic V_phi (Stage 4 core).

Actor: per-qubit <Z_i> -> 6x5 linear head -> softmax over the 5
pacing_gain levels. Critic: sum_i <Z_i> -> linear head -> scalar value.
Same ansatz, independent weights, following the qbbr.agents.base_agent
interface so qbbr.train.loop can swap this for
qbbr.agents.classical.mlp_a2c.MLPA2CAgent without changes (RQ4).

Not yet implemented -- depends on qbbr.agents.quantum.qnn.
"""
from __future__ import annotations

from typing import Any

from qbbr.agents.base_agent import BaseAgent


class QA2CAgent(BaseAgent):
    def __init__(self, n_qubits: int = 6, n_layers: int = 2, n_actions: int = 5) -> None:
        raise NotImplementedError("QA2C agent not yet implemented; see qbbr.agents.quantum.qnn.")

    def act(self, state: Any) -> tuple[int, float]:
        raise NotImplementedError

    def update(self, batch: Any) -> dict[str, float]:
        raise NotImplementedError

    def param_count(self) -> int:
        raise NotImplementedError
