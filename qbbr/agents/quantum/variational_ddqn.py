"""Masked variational Double-DQN over a fixed BBR action set."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from qbbr.agents.quantum.qnn import build_qnn


@dataclass(frozen=True)
class QTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: tuple[int, ...]


class QReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self._items: deque[QTransition] = deque(maxlen=capacity)

    def add(self, transition: QTransition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int, rng: np.random.RandomState) -> list[QTransition]:
        indices = rng.choice(len(self._items), size=batch_size, replace=False)
        return [self._items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._items)


def _masked_argmax(values: torch.Tensor, allowed: Sequence[int]) -> int:
    if not allowed:
        raise ValueError("A Double-DQN action mask must retain at least one BBR action.")
    masked = torch.full_like(values, float("-inf"))
    masked[list(allowed)] = values[list(allowed)]
    return int(torch.argmax(masked).item())


def _baseline_anchored_masked_argmax(
    values: torch.Tensor,
    allowed: Sequence[int],
    baseline_action: int | None,
    min_relative_advantage: float = 0.0,
    min_absolute_advantage: float = 0.0,
) -> int:

    selected = _masked_argmax(values, allowed)
    if baseline_action is None or baseline_action not in allowed or selected == baseline_action:
        return selected
    baseline_value = float(values[baseline_action].item())
    candidate_value = float(values[selected].item())
    required_advantage = max(
        float(min_absolute_advantage),
        abs(baseline_value) * float(min_relative_advantage),
    )
    if candidate_value < baseline_value + required_advantage:
        return int(baseline_action)
    return selected


def _empirically_certified_action(
    selected: int,
    allowed: Sequence[int],
    baseline_action: int | None,
    action_samples: Sequence[int],
    action_mean_rewards: Sequence[float],
    min_samples: int = 0,
    min_mean_reward_advantage: float = 0.0,
) -> int:
    """Require replay evidence before exploiting a non-stock fixed action."""
    if baseline_action is None or baseline_action not in allowed or selected == baseline_action:
        return selected
    if action_samples[selected] < min_samples or action_samples[baseline_action] < min_samples:
        return int(baseline_action)
    if action_mean_rewards[selected] < action_mean_rewards[baseline_action] + min_mean_reward_advantage:
        return int(baseline_action)
    return selected


class VariationalDoubleDQN:
    """QRL value learner which selects only a fixed, externally supplied mask.

    Double-DQN selects the next action with the online QNN and evaluates it
    with a target QNN.  Thus it changes the learning rule, not BBR's action
    alphabet.
    """

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        n_layers: int = 1,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        replay_capacity: int = 10_000,
        seed: int = 0,
    ) -> None:
        if observation_dim < 1 or action_count < 1:
            raise ValueError("observation_dim and action_count must be positive")
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.discount_factor = discount_factor
        self.rng = np.random.RandomState(seed)
        self.online_qnn = build_qnn(observation_dim, n_layers)
        self.online_head = torch.nn.Linear(observation_dim, action_count)
        self.target_qnn = build_qnn(observation_dim, n_layers)
        self.target_head = torch.nn.Linear(observation_dim, action_count)
        self.target_qnn.load_state_dict(self.online_qnn.state_dict())
        self.target_head.load_state_dict(self.online_head.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.online_qnn.parameters()) + list(self.online_head.parameters()),
            lr=learning_rate,
        )
        self.replay = QReplayBuffer(replay_capacity)

    def _online_values(self, state: np.ndarray) -> torch.Tensor:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        return self.online_head(self.online_qnn(state_t))

    def _target_values(self, state: np.ndarray) -> torch.Tensor:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        return self.target_head(self.target_qnn(state_t))

    def act(self, state: np.ndarray, allowed_actions: Sequence[int], epsilon: float = 0.0) -> int:
        if not allowed_actions:
            raise ValueError("No BBR action is available to Double-DQN.")
        if self.rng.rand() < epsilon:
            return int(self.rng.choice(allowed_actions))
        with torch.no_grad():
            return _masked_argmax(self._online_values(state), allowed_actions)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: Sequence[int],
    ) -> None:
        if not 0 <= action < self.action_count:
            raise ValueError("Observed action is outside the fixed BBR action set.")
        self.replay.add(
            QTransition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                tuple(int(item) for item in next_action_mask),
            )
        )

    def optimize(self, batch_size: int = 16) -> float | None:
        if len(self.replay) < batch_size:
            return None
        transitions = self.replay.sample(batch_size, self.rng)
        predicted, targets = [], []
        for item in transitions:
            q_values = self._online_values(item.state)
            predicted.append(q_values[item.action])
            with torch.no_grad():
                if item.done:
                    target = torch.as_tensor(item.reward, dtype=torch.float32)
                else:
                    selected = _masked_argmax(self._online_values(item.next_state), item.next_action_mask)
                    target = torch.as_tensor(item.reward, dtype=torch.float32) + self.discount_factor * self._target_values(item.next_state)[selected]
                targets.append(target)
        loss = torch.nn.functional.smooth_l1_loss(torch.stack(predicted), torch.stack(targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_qnn.load_state_dict(self.online_qnn.state_dict())
        self.target_head.load_state_dict(self.online_head.state_dict())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online_qnn": self.online_qnn.state_dict(),
                "online_head": self.online_head.state_dict(),
                "target_qnn": self.target_qnn.state_dict(),
                "target_head": self.target_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "observation_dim": self.observation_dim,
                "action_count": self.action_count,
            },
            path,
        )
