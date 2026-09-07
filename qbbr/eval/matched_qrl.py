"""Construction and invariant checks for the recurrent-QDQN ablation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from qbbr.agents.classical.recurrent_prioritized_ddqn import (
    RecurrentPrioritizedClassicalDoubleDQN,
)
from qbbr.agents.quantum.recurrent_prioritized_ddqn import (
    RecurrentPrioritizedVariationalDoubleDQN,
)


@dataclass(frozen=True)
class MatchedQRLBenchmarkSpec:
    """All factors that must be shared by the classical and quantum arms."""

    observation_dim: int
    action_count: int
    history_window: int = 8
    latent_dim: int = 3
    variational_layers: int = 1
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    replay_capacity: int = 10_000
    priority_alpha: float = 0.6
    importance_beta: float = 0.4
    baseline_action: int = 2
    min_relative_q_advantage: float = 0.0
    min_absolute_q_advantage: float = 0.0
    empirical_certification_min_samples: int = 0
    min_empirical_reward_advantage: float = 0.0
    reward_contract: str = "supervisor-locked-throughput-only-v1"
    action_set_contract: str = "native-rl-bbr-v3-canonical-7state-fixed-actions"

    def __post_init__(self) -> None:
        if min(self.observation_dim, self.action_count, self.history_window, self.latent_dim) < 1:
            raise ValueError("observation_dim, action_count, history_window and latent_dim must be positive")
        # A 3x3 bias-free classical map exactly matches a one-layer QNN's
        # 3 * latent_dim rotation parameters only at latent_dim=3.
        if (self.latent_dim, self.variational_layers) != (3, 1):
            raise ValueError(
                "The exact matched benchmark requires latent_dim=3 and variational_layers=1."
            )
        if self.reward_contract != "supervisor-locked-throughput-only-v1":
            raise ValueError("The successor benchmark reward must remain throughput-only.")
        if not 0 <= self.baseline_action < self.action_count:
            raise ValueError("baseline_action must refer to an existing fixed BBR action.")
        if (
            self.min_relative_q_advantage < 0.0 or self.min_absolute_q_advantage < 0.0
            or self.empirical_certification_min_samples < 0 or self.min_empirical_reward_advantage < 0.0
        ):
            raise ValueError("Q-value advantage margins must be non-negative.")

    def as_dict(self) -> dict:
        return asdict(self)


def build_matched_agents(spec: MatchedQRLBenchmarkSpec, seed: int):
    """Build equal-parameter agents with an otherwise identical learning loop."""

    quantum = RecurrentPrioritizedVariationalDoubleDQN(
        observation_dim=spec.observation_dim,
        action_count=spec.action_count,
        quantum_dim=spec.latent_dim,
        n_layers=spec.variational_layers,
        learning_rate=spec.learning_rate,
        discount_factor=spec.discount_factor,
        replay_capacity=spec.replay_capacity,
        priority_alpha=spec.priority_alpha,
        baseline_action=spec.baseline_action,
        min_relative_q_advantage=spec.min_relative_q_advantage,
        min_absolute_q_advantage=spec.min_absolute_q_advantage,
        empirical_certification_min_samples=spec.empirical_certification_min_samples,
        min_empirical_reward_advantage=spec.min_empirical_reward_advantage,
        seed=seed,
    )
    classical = RecurrentPrioritizedClassicalDoubleDQN(
        observation_dim=spec.observation_dim,
        action_count=spec.action_count,
        latent_dim=spec.latent_dim,
        learning_rate=spec.learning_rate,
        discount_factor=spec.discount_factor,
        replay_capacity=spec.replay_capacity,
        priority_alpha=spec.priority_alpha,
        baseline_action=spec.baseline_action,
        min_relative_q_advantage=spec.min_relative_q_advantage,
        min_absolute_q_advantage=spec.min_absolute_q_advantage,
        empirical_certification_min_samples=spec.empirical_certification_min_samples,
        min_empirical_reward_advantage=spec.min_empirical_reward_advantage,
        seed=seed,
    )
    if quantum.param_count() != classical.param_count():
        raise RuntimeError(
            "Matched QRL comparison rejected: trainable-parameter counts differ "
            f"(quantum={quantum.param_count()}, classical={classical.param_count()})."
        )
    return quantum, classical
