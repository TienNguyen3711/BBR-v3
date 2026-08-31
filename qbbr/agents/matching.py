"""Exact full-agent parameter matching for RQ4 comparisons.

The archived RQ4 design targeted the quantum variational block only, leaving
the actor heads and critics unmatched.  New RQ4 runs use this module instead:
the classical actor and critic widths are searched jointly and the resulting
full trainable-parameter total must exactly equal the quantum total.
"""

from __future__ import annotations

from dataclasses import dataclass

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.agents.quantum.qa2c import QA2CAgent


@dataclass(frozen=True)
class FullParameterMatch:
    n_qubits: int
    n_layers: int
    action_dims: tuple[int, ...]
    actor_hidden: int
    critic_hidden: int
    quantum_params: int
    classical_params: int

    @property
    def difference(self) -> int:
        return self.classical_params - self.quantum_params

    @property
    def exact(self) -> bool:
        return self.difference == 0


def classical_full_param_count(
    n_qubits: int, action_dims: tuple[int, ...], actor_hidden: int, critic_hidden: int
) -> int:
    """Return the total for MLPA2CAgent's actor trunk/heads and critic."""
    if min(n_qubits, actor_hidden, critic_hidden) < 1 or not action_dims or min(action_dims) < 1:
        raise ValueError("network dimensions must be positive")
    actor_outputs = sum(action_dims)
    actor_params = (n_qubits + 1) * actor_hidden + (actor_hidden + 1) * actor_outputs
    critic_params = (n_qubits + 1) * critic_hidden + (critic_hidden + 1)
    return actor_params + critic_params


def full_parameter_match(
    n_qubits: int = 7,
    n_layers: int = 2,
    action_dims: tuple[int, ...] = (5,),
    reupload: bool = False,
    max_hidden: int = 128,
) -> FullParameterMatch:
    """Find the closest classical full-agent count to the quantum one.

    Ties prefer smaller widths.  Callers running RQ4 must require
    ``match.exact``; returning the nearest candidate makes unsupported state
    dimensions diagnosable rather than silently selecting an unfair baseline.
    """
    if max_hidden < 1:
        raise ValueError("max_hidden must be positive")
    quantum_params = QA2CAgent(
        n_qubits=n_qubits, n_layers=n_layers, action_dims=action_dims, reupload=reupload
    ).param_count()
    candidates = [
        (
            abs(classical_full_param_count(n_qubits, action_dims, actor_hidden, critic_hidden) - quantum_params),
            actor_hidden + critic_hidden,
            actor_hidden,
            critic_hidden,
        )
        for actor_hidden in range(1, max_hidden + 1)
        for critic_hidden in range(1, max_hidden + 1)
    ]
    _difference, _width, actor_hidden, critic_hidden = min(candidates)
    classical_params = classical_full_param_count(n_qubits, action_dims, actor_hidden, critic_hidden)
    return FullParameterMatch(
        n_qubits=n_qubits,
        n_layers=n_layers,
        action_dims=tuple(action_dims),
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        quantum_params=quantum_params,
        classical_params=classical_params,
    )


def build_full_parameter_matched_classical(
    n_qubits: int = 7,
    n_layers: int = 2,
    action_dims: tuple[int, ...] = (5,),
    reupload: bool = False,
    lr: float = 1e-3,
    gamma: float = 0.99,
    normalize_advantage: bool = True,
) -> tuple[MLPA2CAgent, FullParameterMatch]:
    """Build an exact full-agent-matched classical core for a new RQ4 run."""
    match = full_parameter_match(n_qubits, n_layers, action_dims, reupload)
    if not match.exact:
        raise ValueError(
            "no exact full-agent parameter match exists for "
            f"n_qubits={n_qubits}, n_layers={n_layers}, action_dims={action_dims}; "
            f"closest classical total is {match.classical_params}, quantum total is {match.quantum_params}. "
            "Do not run this configuration as RQ4 without revising the architectures."
        )
    agent = MLPA2CAgent(
        n_qubits=n_qubits,
        n_layers=n_layers,
        action_dims=action_dims,
        actor_hidden=match.actor_hidden,
        critic_hidden=match.critic_hidden,
        lr=lr,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
    )
    if agent.param_count() != match.quantum_params:
        raise RuntimeError("constructed classical agent does not satisfy its RQ4 parameter match")
    return agent, match
