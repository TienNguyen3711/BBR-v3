"""Exact Classical-A2C counterpart for the fixed-action NativeQA2C brain."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qbbr.agents.classical.native_a2c import NativeMLPA2CAgent
from qbbr.agents.quantum.native_qa2c import NativeQA2CAgent
from qbbr.control.native_action_selector import NativeActionSelector


@dataclass(frozen=True)
class NativeQA2CMatch:
    observation_dim: int
    action_count: int
    n_layers: int
    actor_hidden_dim: int
    critic_hidden_dim: int
    quantum_params: int
    classical_params: int


def _classical_count(observation_dim: int, action_count: int, actor_hidden: int, critic_hidden: int) -> int:
    actor = (observation_dim + 1) * actor_hidden + (actor_hidden + 1) * action_count
    critic = (observation_dim + 1) * critic_hidden + (critic_hidden + 1)
    return actor + critic


def _apply_stock_init_bias(final_linear, stock_action: int, bias: float) -> None:
    """Warm-start the actor toward stock BBR: a positive prior on the stock
    logit so a seed does not begin by favouring -- and then self-reinforcing --
    a sub-1.0 gain. Exploration and every other logit are untouched, and the
    bias term already exists so no parameter count changes."""
    if bias:
        with torch.no_grad():
            final_linear.bias[stock_action] += float(bias)


def build_matched_native_a2c_agents(
    observation_dim: int, action_count: int, n_layers: int = 2, lr: float = 1e-3,
    gamma: float = 0.99, reupload: bool = False, max_hidden: int = 128,
    selector: NativeActionSelector | None = None, entropy_coef: float = 0.0,
    stock_action: int = 2, stock_init_bias: float = 0.0,
) -> tuple[NativeQA2CAgent, NativeMLPA2CAgent, NativeQA2CMatch]:
    """Construct the primary QA2C and an exactly parameter-matched A2C arm."""
    quantum = NativeQA2CAgent(
        observation_dim=observation_dim, native_action_count=action_count,
        n_layers=n_layers, lr=lr, gamma=gamma, reupload=reupload,
        selector=selector, entropy_coef=entropy_coef,
    )
    _apply_stock_init_bias(quantum.actor_head, stock_action, stock_init_bias)
    target = quantum.param_count()
    candidates = [
        (actor, critic)
        for actor in range(1, max_hidden + 1)
        for critic in range(1, max_hidden + 1)
        if _classical_count(observation_dim, action_count, actor, critic) == target
    ]
    if not candidates:
        raise ValueError("No exact NativeQA2C/Classical-A2C parameter match exists.")
    actor_hidden, critic_hidden = min(candidates, key=lambda widths: sum(widths))
    classical = NativeMLPA2CAgent(
        observation_dim=observation_dim, native_action_count=action_count,
        actor_hidden_dim=actor_hidden, critic_hidden_dim=critic_hidden, lr=lr, gamma=gamma,
        selector=selector, entropy_coef=entropy_coef,
    )
    _apply_stock_init_bias(classical.actor[-1], stock_action, stock_init_bias)
    if classical.param_count() != target:
        raise RuntimeError("Native QA2C parameter match construction failed.")
    return quantum, classical, NativeQA2CMatch(
        observation_dim, action_count, n_layers, actor_hidden, critic_hidden, target, classical.param_count(),
    )
