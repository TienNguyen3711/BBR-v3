from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qbbr.agents.base_agent import BaseAgent, a2c_losses, discounted_returns
from qbbr.agents.quantum.qnn import build_qnn


class QA2CAgent(BaseAgent):
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        n_actions: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        reupload: bool = False,
        normalize_advantage: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        self.actor_qnn = build_qnn(n_qubits, n_layers, reupload=reupload)
        self.actor_head = torch.nn.Linear(n_qubits, n_actions)
        self.critic_qnn = build_qnn(n_qubits, n_layers, reupload=reupload)
        self.critic_head = torch.nn.Linear(1, 1)

        params = (
            list(self.actor_qnn.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_qnn.parameters())
            + list(self.critic_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _actor_logits(self, state_t: torch.Tensor) -> torch.Tensor:
        z = self.actor_qnn(state_t)
        return self.actor_head(z)

    def _critic_value(self, state_t: torch.Tensor) -> torch.Tensor:
        z_sum = self.critic_qnn(state_t).sum().unsqueeze(0)
        return self.critic_head(z_sum).squeeze(0)

    def act(self, state: Any) -> tuple[int, float]:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            logits = self._actor_logits(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    def value(self, state: Any) -> float:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            return float(self._critic_value(state_t).item())

    def update(self, batch: Any) -> dict[str, float]:
        returns = discounted_returns(batch.rewards, self.gamma)
        actions_t = torch.as_tensor(batch.actions, dtype=torch.long)

        log_probs, values = [], []
        for state, action in zip(batch.states, actions_t):
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
            logits = self._actor_logits(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(action))
            values.append(self._critic_value(state_t))
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)

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
        return sum(
            p.numel()
            for module in (self.actor_qnn, self.actor_head, self.critic_qnn, self.critic_head)
            for p in module.parameters()
        )
