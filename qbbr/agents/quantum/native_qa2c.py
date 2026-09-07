"""Masked Q-A2C brain for a fixed, named BBR-v3 action inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from qbbr.agents.base_agent import BaseAgent, a2c_losses, discounted_returns
from qbbr.agents.quantum.qnn import build_qnn
from qbbr.control.native_action_selector import NativeActionSelector


def _masked_logits(logits: torch.Tensor, allowed_indices: Sequence[int]) -> torch.Tensor:
    if not allowed_indices:
        raise ValueError("A native action mask must allow at least stock BBR.")
    mask = torch.full_like(logits, float("-inf"))
    mask[list(allowed_indices)] = 0.0
    return logits + mask


class NativeQA2CAgent(BaseAgent):

    def __init__(
        self,
        observation_dim: int,
        native_action_count: int,
        n_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        reupload: bool = False,
        normalize_advantage: bool = True,
        selector: NativeActionSelector | None = None,
        entropy_coef: float = 0.0,
    ) -> None:
        if observation_dim < 1 or native_action_count < 1:
            raise ValueError("observation_dim and native_action_count must be positive")
        self.observation_dim = observation_dim
        self.native_action_count = native_action_count
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        if entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        self.selector = selector
        self.entropy_coef = entropy_coef
        self.actor_qnn = build_qnn(observation_dim, n_layers, reupload=reupload)
        self.actor_head = torch.nn.Linear(observation_dim, native_action_count)
        self.critic_qnn = build_qnn(observation_dim, n_layers, reupload=reupload)
        self.critic_head = torch.nn.Linear(1, 1)
        params = (
            list(self.actor_qnn.parameters())
            + list(self.actor_head.parameters())
            + list(self.critic_qnn.parameters())
            + list(self.critic_head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _actor_logits(self, state_t: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.actor_qnn(state_t))

    def _critic_value(self, state_t: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_qnn(state_t).sum().reshape(1)).squeeze(0)

    def _distribution(
        self, state_t: torch.Tensor, allowed_indices: Sequence[int],
    ) -> torch.distributions.Categorical:
        logits = self._actor_logits(state_t)
        if self.selector is not None:
            return self.selector.hard_distribution(logits, state_t, allowed_indices)
        return torch.distributions.Categorical(logits=_masked_logits(logits, allowed_indices))

    def _deployment_action(self, state_t: torch.Tensor, allowed_indices: Sequence[int]) -> int:
        logits = self._actor_logits(state_t)
        if self.selector is not None:
            return self.selector.deployment_action(logits, state_t, allowed_indices)
        return int(torch.argmax(_masked_logits(logits, allowed_indices)).item())

    def act(
        self, state: Any, allowed_indices: Sequence[int] | None = None, deterministic: bool = False,
        deployment: bool = False,
    ) -> tuple[int, float]:
        if allowed_indices is None:
            raise ValueError("NativeQA2CAgent requires an action mask from NativeActionSpace.")
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            if deterministic and deployment:
                action = torch.as_tensor(self._deployment_action(state_t, allowed_indices), dtype=torch.long)
                distribution = self._distribution(state_t, allowed_indices)
                return int(action.item()), float(distribution.log_prob(action).item())
            distribution = self._distribution(state_t, allowed_indices)
            action = torch.argmax(distribution.logits) if deterministic else distribution.sample()
            return int(action.item()), float(distribution.log_prob(action).item())

    def value(self, state: Any) -> float:
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        with torch.no_grad():
            return float(self._critic_value(state_t).item())

    def update(self, batch: Any) -> dict[str, float]:
        if not hasattr(batch, "action_masks"):
            raise ValueError("NativeQA2CAgent requires a rollout with per-step action_masks.")
        returns = discounted_returns(batch.rewards, self.gamma)
        log_probs, values, entropies, greedy_actions, stock_probabilities = [], [], [], [], []
        for state, action, allowed_indices in zip(batch.states, batch.actions, batch.action_masks):
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
            distribution = self._distribution(state_t, allowed_indices)
            log_probs.append(distribution.log_prob(torch.as_tensor(action, dtype=torch.long)))
            entropies.append(distribution.entropy())
            greedy_actions.append(int(torch.argmax(distribution.probs).item()))
            stock_probabilities.append(distribution.probs[2] if self.native_action_count > 2 else distribution.probs[0])
            values.append(self._critic_value(state_t))
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()
        actor_loss, critic_loss = a2c_losses(
            torch.stack(log_probs), values_t, returns, self.normalize_advantage
        )
        mean_entropy = torch.stack(entropies).mean()
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * mean_entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        phase = np.asarray([np.asarray(state, dtype=float)[6] if len(np.asarray(state).reshape(-1)) >= 7 else 0.0 for state in batch.states])
        near = phase >= 0.60
        advantage_np = advantages.detach().cpu().numpy()
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "policy_entropy": float(mean_entropy.item()),
            "value_std": float(values_t.detach().std(unbiased=False).item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "near_handover_advantage_mean": float(advantage_np[near].mean()) if np.any(near) else 0.0,
            "non_handover_advantage_mean": float(advantage_np[~near].mean()) if np.any(~near) else 0.0,
            "sampled_nonstock_fraction": float(np.mean(np.asarray(batch.actions) != 2)),
            "greedy_nonstock_fraction": float(np.mean(np.asarray(greedy_actions) != 2)),
            "mean_stock_probability": float(torch.stack(stock_probabilities).mean().item()),
            "loss": float(loss.item()),
        }

    def param_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.actor_qnn, self.actor_head, self.critic_qnn, self.critic_head)
            for parameter in module.parameters()
        )

    def training_state_dict(self) -> dict:
        return {
            "actor_qnn": self.actor_qnn.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic_qnn": self.critic_qnn.state_dict(),
            "critic_head": self.critic_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "observation_dim": self.observation_dim,
            "native_action_count": self.native_action_count,
            "entropy_coef": self.entropy_coef,
        }

    def load_training_state_dict(self, checkpoint: dict) -> None:
        if checkpoint["observation_dim"] != self.observation_dim or checkpoint["native_action_count"] != self.native_action_count:
            raise ValueError("Checkpoint dimensions do not match native QA2C contract.")
        if checkpoint.get("entropy_coef", self.entropy_coef) != self.entropy_coef:
            raise ValueError("Checkpoint entropy coefficient does not match native QA2C contract.")
        self.actor_qnn.load_state_dict(checkpoint["actor_qnn"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_qnn.load_state_dict(checkpoint["critic_qnn"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, path: str | Path) -> None:
        torch.save(self.training_state_dict(), path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("observation_dim") not in (None, self.observation_dim):
            raise ValueError("Checkpoint observation dimension does not match native contract.")
        if checkpoint.get("native_action_count") not in (None, self.native_action_count):
            raise ValueError("Checkpoint native action count does not match native contract.")
        self.actor_qnn.load_state_dict(checkpoint["actor_qnn"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_qnn.load_state_dict(checkpoint["critic_qnn"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
