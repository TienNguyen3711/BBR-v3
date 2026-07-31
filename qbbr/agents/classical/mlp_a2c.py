from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from qbbr.agents.base_agent import BaseAgent, a2c_losses, discounted_returns


def _mlp_param_count(in_dim: int, hidden: int, out_dim: int) -> int:
    return (in_dim * hidden + hidden) + (hidden * out_dim + out_dim)


def _best_hidden_size(in_dim: int, out_dim: int, target_params: int, max_hidden: int = 64) -> int:
    candidates = range(1, max_hidden + 1)
    return min(candidates, key=lambda h: abs(_mlp_param_count(in_dim, h, out_dim) - target_params))


class MLPA2CAgent(BaseAgent):
    def __init__(
        self,
        n_qubits: int = 6,
        n_actions: int = 5,
        n_layers: int = 2,
        param_budget: int | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        normalize_advantage: bool = True,
    ) -> None:

        target = param_budget if param_budget is not None else 18 * n_layers
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        actor_hidden = _best_hidden_size(n_qubits, n_actions, target)
        critic_hidden = _best_hidden_size(n_qubits, 1, target)
        self.actor = nn.Sequential(
            nn.Linear(n_qubits, actor_hidden), nn.Tanh(), nn.Linear(actor_hidden, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_qubits, critic_hidden), nn.Tanh(), nn.Linear(critic_hidden, 1)
        )

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def act(self, state: Any) -> tuple[int, float]:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            logits = self.actor(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    def value(self, state: Any) -> float:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            return float(self.critic(state_t).squeeze(0).item())

    def update(self, batch: Any) -> dict[str, float]:
        returns = discounted_returns(batch.rewards, self.gamma)
        states_t = torch.as_tensor(np.stack([np.asarray(s) for s in batch.states]), dtype=torch.float32)
        actions_t = torch.as_tensor(batch.actions, dtype=torch.long)

        logits = self.actor(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_t = dist.log_prob(actions_t)
        values_t = self.critic(states_t).squeeze(-1)

        actor_loss, critic_loss = a2c_losses(log_probs_t, values_t, returns, self.normalize_advantage)
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "loss": float(loss.item()),
        }

    def param_count(self) -> int:
        return sum(p.numel() for module in (self.actor, self.critic) for p in module.parameters())
