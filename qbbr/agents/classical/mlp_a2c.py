from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from qbbr.action.registry import decode_flat_action, encode_flat_action
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
        action_dims: tuple[int, ...] = (5,),
        n_layers: int = 2,
        param_budget: int | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        normalize_advantage: bool = True,
    ) -> None:
        self.action_dims = tuple(action_dims)
        target = param_budget if param_budget is not None else 18 * n_layers
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        actor_hidden = _best_hidden_size(n_qubits, sum(self.action_dims), target)
        critic_hidden = _best_hidden_size(n_qubits, 1, target)
        self.actor_trunk = nn.Sequential(nn.Linear(n_qubits, actor_hidden), nn.Tanh())
        self.actor_heads = nn.ModuleList([nn.Linear(actor_hidden, d) for d in self.action_dims])
        self.critic = nn.Sequential(
            nn.Linear(n_qubits, critic_hidden), nn.Tanh(), nn.Linear(critic_hidden, 1)
        )

        params = (
            list(self.actor_trunk.parameters())
            + list(self.actor_heads.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _actor_logits(self, state_t: torch.Tensor) -> list[torch.Tensor]:
        features = self.actor_trunk(state_t)
        return [head(features) for head in self.actor_heads]

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
            return float(self.critic(state_t).squeeze(0).item())

    def update(self, batch: Any) -> dict[str, float]:
        returns = discounted_returns(batch.rewards, self.gamma)
        states_t = torch.as_tensor(np.stack([np.asarray(s) for s in batch.states]), dtype=torch.float32)
        decoded = [decode_flat_action(a, list(self.action_dims)) for a in batch.actions]
        actions_per_head = list(zip(*decoded))  # transpose: n_heads tuples, each length T

        logits_list = self._actor_logits(states_t)
        log_probs_t = torch.zeros(len(batch.actions))
        for logits, head_actions in zip(logits_list, actions_per_head):
            dist = torch.distributions.Categorical(logits=logits)
            head_actions_t = torch.as_tensor(head_actions, dtype=torch.long)
            log_probs_t = log_probs_t + dist.log_prob(head_actions_t)

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
        return sum(
            p.numel()
            for module in (self.actor_trunk, self.actor_heads, self.critic)
            for p in module.parameters()
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor_trunk": self.actor_trunk.state_dict(),
                "actor_heads": self.actor_heads.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if "actor" in checkpoint:
            # Legacy pre-actor_trunk/actor_heads-split format (single nn.Sequential
            # "actor", e.g. RQ1a's frozen pacing_gain-only baseline checkpoints,
            # outputs/checkpoints/, predating this split). Only valid for the
            # single-head case the legacy format was ever trained under; the
            # Sequential's first Linear is exactly today's actor_trunk, its second
            # (final) Linear is exactly today's actor_heads[0].
            if len(self.action_dims) != 1:
                raise ValueError(
                    f"legacy single-head checkpoint {path!r} cannot be loaded into a "
                    f"{len(self.action_dims)}-head agent"
                )
            legacy_actor = checkpoint["actor"]
            self.actor_trunk.load_state_dict({"0.weight": legacy_actor["0.weight"], "0.bias": legacy_actor["0.bias"]})
            self.actor_heads.load_state_dict({"0.weight": legacy_actor["2.weight"], "0.bias": legacy_actor["2.bias"]})
            self.critic.load_state_dict(checkpoint["critic"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            return
        self.actor_trunk.load_state_dict(checkpoint["actor_trunk"])
        self.actor_heads.load_state_dict(checkpoint["actor_heads"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
