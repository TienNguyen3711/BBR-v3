"""Classical masked A2C baseline matched to the native QRL--BBR contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from qbbr.agents.base_agent import BaseAgent, a2c_losses, discounted_returns
from qbbr.control.native_action_selector import NativeActionSelector


def _masked_logits(logits: torch.Tensor, allowed_indices: Sequence[int]) -> torch.Tensor:
    if not allowed_indices:
        raise ValueError("A native action mask must allow at least stock BBR.")
    mask = torch.full_like(logits, float("-inf"))
    mask[list(allowed_indices)] = 0.0
    return logits + mask


class NativeMLPA2CAgent(BaseAgent):
    """Non-quantum baseline with exactly the QRL brain's MDP/action mask."""

    def __init__(
        self,
        observation_dim: int,
        native_action_count: int,
        hidden_dim: int = 16,
        actor_hidden_dim: int | None = None,
        critic_hidden_dim: int | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        normalize_advantage: bool = True,
        selector: NativeActionSelector | None = None,
        entropy_coef: float = 0.0,
    ) -> None:
        actor_hidden_dim = hidden_dim if actor_hidden_dim is None else actor_hidden_dim
        critic_hidden_dim = hidden_dim if critic_hidden_dim is None else critic_hidden_dim
        if min(observation_dim, native_action_count, actor_hidden_dim, critic_hidden_dim) < 1:
            raise ValueError("observation_dim, native_action_count, and hidden_dim must be positive")
        self.observation_dim = observation_dim
        self.native_action_count = native_action_count
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        if entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        self.selector = selector
        self.entropy_coef = entropy_coef
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, actor_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(actor_hidden_dim, native_action_count),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, critic_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(critic_hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def _distribution(
        self, state_t: torch.Tensor, allowed_indices: Sequence[int],
    ) -> torch.distributions.Categorical:
        logits = self.actor(state_t)
        if self.selector is not None:
            return self.selector.hard_distribution(logits, state_t, allowed_indices)
        return torch.distributions.Categorical(logits=_masked_logits(logits, allowed_indices))

    def _deployment_action(self, state_t: torch.Tensor, allowed_indices: Sequence[int]) -> int:
        logits = self.actor(state_t)
        if self.selector is not None:
            return self.selector.deployment_action(logits, state_t, allowed_indices)
        return int(torch.argmax(_masked_logits(logits, allowed_indices)).item())

    def act(
        self, state: Any, allowed_indices: Sequence[int] | None = None, deterministic: bool = False,
        deployment: bool = False,
    ) -> tuple[int, float]:
        if allowed_indices is None:
            raise ValueError("NativeMLPA2CAgent requires an action mask from NativeActionSpace.")
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
            return float(self.critic(state_t).squeeze(0).item())

    def update(self, batch: Any, entropy_coef: float | None = None) -> dict[str, float]:
        if not hasattr(batch, "action_masks"):
            raise ValueError("NativeMLPA2CAgent requires a rollout with per-step action_masks.")
        ec = self.entropy_coef if entropy_coef is None else float(entropy_coef)
        returns = discounted_returns(batch.rewards, self.gamma)
        log_probs, values, entropies, greedy_actions, stock_probabilities = [], [], [], [], []
        for state, action, allowed_indices in zip(batch.states, batch.actions, batch.action_masks):
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32)
            distribution = self._distribution(state_t, allowed_indices)
            log_probs.append(distribution.log_prob(torch.as_tensor(action, dtype=torch.long)))
            entropies.append(distribution.entropy())
            greedy_actions.append(int(torch.argmax(distribution.probs).item()))
            stock_probabilities.append(distribution.probs[2] if self.native_action_count > 2 else distribution.probs[0])
            values.append(self.critic(state_t).squeeze(0))
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()
        actor_loss, critic_loss = a2c_losses(
            torch.stack(log_probs), values_t, returns, self.normalize_advantage
        )
        mean_entropy = torch.stack(entropies).mean()
        loss = actor_loss + 0.5 * critic_loss - ec * mean_entropy
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
            "entropy_coef_effective": ec,
            "loss": float(loss.item()),
        }

    def param_count(self) -> int:
        return sum(parameter.numel() for module in (self.actor, self.critic) for parameter in module.parameters())

    def training_state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(), "observation_dim": self.observation_dim,
            "native_action_count": self.native_action_count,
            "actor_hidden_dim": self.actor_hidden_dim, "critic_hidden_dim": self.critic_hidden_dim,
            "entropy_coef": self.entropy_coef,
        }

    def load_training_state_dict(self, checkpoint: dict) -> None:
        expected = (self.observation_dim, self.native_action_count, self.actor_hidden_dim, self.critic_hidden_dim)
        observed = (
            checkpoint["observation_dim"], checkpoint["native_action_count"],
            checkpoint["actor_hidden_dim"], checkpoint["critic_hidden_dim"],
        )
        if observed != expected:
            raise ValueError("Checkpoint dimensions do not match native classical A2C contract.")
        if checkpoint.get("entropy_coef", self.entropy_coef) != self.entropy_coef:
            raise ValueError("Checkpoint entropy coefficient does not match native classical A2C contract.")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, path: str | Path) -> None:
        torch.save(self.training_state_dict(), path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("observation_dim") not in (None, self.observation_dim):
            raise ValueError("Checkpoint observation dimension does not match native contract.")
        if checkpoint.get("native_action_count") not in (None, self.native_action_count):
            raise ValueError("Checkpoint native action count does not match native contract.")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
