"""Recurrent, prioritized, variational Double-DQN for a fixed BBR action set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from qbbr.agents.quantum.qnn import build_qnn
from qbbr.agents.quantum.variational_ddqn import (
    _baseline_anchored_masked_argmax,
    _empirically_certified_action,
)


@dataclass(frozen=True)
class RecurrentTransition:
    history: np.ndarray
    action: int
    reward: float
    next_history: np.ndarray
    done: bool
    next_action_mask: tuple[int, ...]


class PrioritizedReplayBuffer:

    def __init__(self, capacity: int = 10_000, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.items: list[RecurrentTransition] = []
        self.priorities: list[float] = []

    def add(self, item: RecurrentTransition) -> None:
        priority = max(self.priorities, default=1.0)
        if len(self.items) >= self.capacity:
            self.items.pop(0)
            self.priorities.pop(0)
        self.items.append(item)
        self.priorities.append(priority)

    def sample(
        self, batch_size: int, beta: float, rng: np.random.RandomState
    ) -> tuple[list[RecurrentTransition], np.ndarray, np.ndarray]:
        priorities = np.asarray(self.priorities, dtype=float) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = rng.choice(len(self.items), size=batch_size, replace=False, p=probabilities)
        weights = (len(self.items) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        return [self.items[int(index)] for index in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        for index, error in zip(indices, td_errors):
            self.priorities[int(index)] = max(abs(float(error)), 1e-6)

    def __len__(self) -> int:
        return len(self.items)


class RecurrentPrioritizedVariationalDoubleDQN:
    """GRU context encoder + QNN Double-DQN head, with frozen BBR actions.

    The GRU models temporal RTT/queue/reconfiguration patterns.  The QNN still
    produces Q-values only for the fixed choices supplied by the action mask.
    """

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        quantum_dim: int = 4,
        n_layers: int = 1,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        replay_capacity: int = 10_000,
        priority_alpha: float = 0.6,
        baseline_action: int | None = 2,
        min_relative_q_advantage: float = 0.0,
        min_absolute_q_advantage: float = 0.0,
        empirical_certification_min_samples: int = 0,
        min_empirical_reward_advantage: float = 0.0,
        seed: int = 0,
    ) -> None:
        if min(observation_dim, action_count, quantum_dim) < 1:
            raise ValueError("observation_dim, action_count, and quantum_dim must be positive")
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.discount_factor = discount_factor
        if baseline_action is not None and not 0 <= baseline_action < action_count:
            raise ValueError("baseline_action must be an existing fixed BBR action.")
        if min_relative_q_advantage < 0.0 or min_absolute_q_advantage < 0.0:
            raise ValueError("Q-value advantage margins must be non-negative.")
        if empirical_certification_min_samples < 0 or min_empirical_reward_advantage < 0.0:
            raise ValueError("Empirical certification settings must be non-negative.")
        self.baseline_action = baseline_action
        self.min_relative_q_advantage = min_relative_q_advantage
        self.min_absolute_q_advantage = min_absolute_q_advantage
        self.empirical_certification_min_samples = empirical_certification_min_samples
        self.min_empirical_reward_advantage = min_empirical_reward_advantage
        self.action_samples = np.zeros(action_count, dtype=np.int64)
        self.action_mean_rewards = np.zeros(action_count, dtype=np.float64)
        self.rng = np.random.RandomState(seed)
        self.online_gru = torch.nn.GRU(observation_dim, quantum_dim, batch_first=True)
        self.online_qnn = build_qnn(quantum_dim, n_layers)
        self.online_head = torch.nn.Linear(quantum_dim, action_count)
        self.target_gru = torch.nn.GRU(observation_dim, quantum_dim, batch_first=True)
        self.target_qnn = build_qnn(quantum_dim, n_layers)
        self.target_head = torch.nn.Linear(quantum_dim, action_count)
        self.target_gru.load_state_dict(self.online_gru.state_dict())
        self.target_qnn.load_state_dict(self.online_qnn.state_dict())
        self.target_head.load_state_dict(self.online_head.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.online_gru.parameters())
            + list(self.online_qnn.parameters())
            + list(self.online_head.parameters()),
            lr=learning_rate,
        )
        self.replay = PrioritizedReplayBuffer(replay_capacity, priority_alpha)

    @staticmethod
    def _history_tensor(history: np.ndarray) -> torch.Tensor:
        value = torch.as_tensor(np.asarray(history), dtype=torch.float32)
        if value.ndim != 2:
            raise ValueError("A recurrent QRL history must have shape [time, features].")
        return value.unsqueeze(0)

    def _values(self, history: np.ndarray, target: bool = False) -> torch.Tensor:
        gru = self.target_gru if target else self.online_gru
        qnn = self.target_qnn if target else self.online_qnn
        head = self.target_head if target else self.online_head
        _sequence, hidden = gru(self._history_tensor(history))
        return head(qnn(hidden[-1, 0]))

    def act(self, history: np.ndarray, allowed_actions: Sequence[int], epsilon: float = 0.0) -> int:
        if not allowed_actions:
            raise ValueError("No BBR action is available to recurrent Double-DQN.")
        if self.rng.rand() < epsilon:
            return int(self.rng.choice(allowed_actions))
        with torch.no_grad():
            selected = _baseline_anchored_masked_argmax(
                self._values(history), allowed_actions, self.baseline_action,
                self.min_relative_q_advantage, self.min_absolute_q_advantage,
            )
            return _empirically_certified_action(
                selected, allowed_actions, self.baseline_action, self.action_samples,
                self.action_mean_rewards, self.empirical_certification_min_samples,
                self.min_empirical_reward_advantage,
            )

    def observe(
        self,
        history: np.ndarray,
        action: int,
        reward: float,
        next_history: np.ndarray,
        done: bool,
        next_action_mask: Sequence[int],
    ) -> None:
        if not 0 <= action < self.action_count:
            raise ValueError("Observed action is outside the fixed BBR action set.")
        self.action_samples[action] += 1
        self.action_mean_rewards[action] += (
            float(reward) - self.action_mean_rewards[action]
        ) / self.action_samples[action]
        self.replay.add(
            RecurrentTransition(
                np.asarray(history, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_history, dtype=np.float32),
                bool(done),
                tuple(int(item) for item in next_action_mask),
            )
        )

    def optimize(self, batch_size: int = 16, importance_beta: float = 0.4) -> float | None:
        if len(self.replay) < batch_size:
            return None
        batch, indices, weights = self.replay.sample(batch_size, importance_beta, self.rng)
        losses, td_errors = [], []
        for item, weight in zip(batch, weights):
            predicted = self._values(item.history)[item.action]
            with torch.no_grad():
                if item.done:
                    target = torch.as_tensor(item.reward, dtype=torch.float32)
                else:
                    selected = _baseline_anchored_masked_argmax(
                        self._values(item.next_history), item.next_action_mask, self.baseline_action,
                        self.min_relative_q_advantage, self.min_absolute_q_advantage,
                    )
                    selected = _empirically_certified_action(
                        selected, item.next_action_mask, self.baseline_action, self.action_samples,
                        self.action_mean_rewards, self.empirical_certification_min_samples,
                        self.min_empirical_reward_advantage,
                    )
                    target = torch.as_tensor(item.reward, dtype=torch.float32) + self.discount_factor * self._values(item.next_history, target=True)[selected]
            td_error = target - predicted
            losses.append(torch.as_tensor(weight) * torch.nn.functional.smooth_l1_loss(predicted, target))
            td_errors.append(float(td_error.detach().abs().item()))
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay.update_priorities(indices, td_errors)
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_gru.load_state_dict(self.online_gru.state_dict())
        self.target_qnn.load_state_dict(self.online_qnn.state_dict())
        self.target_head.load_state_dict(self.online_head.state_dict())

    def param_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.online_gru, self.online_qnn, self.online_head)
            for parameter in module.parameters()
        )

    def training_state_dict(self) -> dict:
        """State needed to resume this local simulator training run exactly."""
        return {
            "online_gru": self.online_gru.state_dict(),
            "online_qnn": self.online_qnn.state_dict(),
            "online_head": self.online_head.state_dict(),
            "target_gru": self.target_gru.state_dict(),
            "target_qnn": self.target_qnn.state_dict(),
            "target_head": self.target_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_items": self.replay.items,
            "replay_priorities": self.replay.priorities,
            "rng_state": self.rng.get_state(),
            "action_samples": self.action_samples,
            "action_mean_rewards": self.action_mean_rewards,
            "observation_dim": self.observation_dim,
            "action_count": self.action_count,
        }

    def load_training_state_dict(self, state: dict) -> None:
        if (state["observation_dim"], state["action_count"]) != (
            self.observation_dim, self.action_count,
        ):
            raise ValueError("Checkpoint dimensions do not match the fixed successor contract.")
        for name in ("online_gru", "online_qnn", "online_head", "target_gru", "target_qnn", "target_head"):
            getattr(self, name).load_state_dict(state[name])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.items = list(state["replay_items"])
        self.replay.priorities = list(state["replay_priorities"])
        self.rng.set_state(state["rng_state"])
        self.action_samples = np.asarray(state["action_samples"], dtype=np.int64)
        self.action_mean_rewards = np.asarray(state["action_mean_rewards"], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        torch.save(self.training_state_dict(), path)
