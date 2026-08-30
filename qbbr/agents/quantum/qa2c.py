from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from qbbr.action.registry import decode_flat_action, encode_flat_action
from qbbr.agents.base_agent import BaseAgent, a2c_losses, discounted_returns
from qbbr.agents.quantum.qnn import build_qnn


class QA2CAgent(BaseAgent):

    def __init__(
        self,
        n_qubits: int = 7,
        n_layers: int = 2,
        action_dims: tuple[int, ...] = (5,),
        lr: float = 1e-3,
        gamma: float = 0.99,
        reupload: bool = False,
        normalize_advantage: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.action_dims = tuple(action_dims)
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        self.actor_qnn = build_qnn(n_qubits, n_layers, reupload=reupload)
        self.actor_heads = torch.nn.ModuleList([torch.nn.Linear(n_qubits, d) for d in self.action_dims])
        self.critic_qnn = build_qnn(n_qubits, n_layers, reupload=reupload)
        self.critic_head = torch.nn.Linear(1, 1)

        params = (
            list(self.actor_qnn.parameters())
            + list(self.actor_heads.parameters())
            + list(self.critic_qnn.parameters())
            + list(self.critic_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _actor_logits(self, state_t: torch.Tensor) -> list[torch.Tensor]:
        z = self.actor_qnn(state_t)
        return [head(z) for head in self.actor_heads]

    def _critic_value(self, state_t: torch.Tensor) -> torch.Tensor:
        z_sum = self.critic_qnn(state_t).sum().unsqueeze(0)
        return self.critic_head(z_sum).squeeze(0)

    def act(self, state: Any) -> tuple[int, float]:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            indices, log_prob_sum = [], 0.0
            for logits in self._actor_logits(state_t):
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                indices.append(int(a.item()))
                log_prob_sum += float(dist.log_prob(a).item())
        flat_action = encode_flat_action(indices, list(self.action_dims))
        return flat_action, log_prob_sum

    def value(self, state: Any) -> float:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            return float(self._critic_value(state_t).item())

    def update(self, batch: Any) -> dict[str, float]:
        returns = discounted_returns(batch.rewards, self.gamma)
        decoded = [decode_flat_action(a, list(self.action_dims)) for a in batch.actions]

        log_probs, values = [], []
        for state, head_indices in zip(batch.states, decoded):
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
            log_prob = torch.zeros(())
            for logits, idx in zip(self._actor_logits(state_t), head_indices):
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = log_prob + dist.log_prob(torch.as_tensor(idx, dtype=torch.long))
            log_probs.append(log_prob)
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
            for module in (self.actor_qnn, self.actor_heads, self.critic_qnn, self.critic_head)
            for p in module.parameters()
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor_qnn": self.actor_qnn.state_dict(),
                "actor_heads": self.actor_heads.state_dict(),
                "critic_qnn": self.critic_qnn.state_dict(),
                "critic_head": self.critic_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if "actor_head" in checkpoint and len(self.action_dims) != 1:
            # Legacy pre-ModuleList format (a single nn.Linear "actor_head", not yet
            # wrapped in actor_heads: ModuleList -- e.g. RQ4's frozen quantum
            # checkpoints, outputs/checkpoints/quantum/, predating that wrap) is only
            # valid for the single-head case it was ever trained under; reject before
            # touching any state so a bad load never leaves the agent half-updated.
            raise ValueError(
                f"legacy single-head checkpoint {path!r} cannot be loaded into a "
                f"{len(self.action_dims)}-head agent"
            )
        self.actor_qnn.load_state_dict(checkpoint["actor_qnn"])
        if "actor_head" in checkpoint:
            # The legacy Linear's flat weight/bias are exactly today's actor_heads[0].
            legacy_head = checkpoint["actor_head"]
            self.actor_heads.load_state_dict({"0.weight": legacy_head["weight"], "0.bias": legacy_head["bias"]})
        else:
            self.actor_heads.load_state_dict(checkpoint["actor_heads"])
        self.critic_qnn.load_state_dict(checkpoint["critic_qnn"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
