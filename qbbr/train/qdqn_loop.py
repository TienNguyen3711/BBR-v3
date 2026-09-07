"""Generic training loop for a masked Double-DQN BBR action policy."""

from __future__ import annotations

from typing import Any

import numpy as np


def _available_actions(env: Any) -> tuple[int, ...]:
    if hasattr(env, "allowed_action_indices"):
        return tuple(env.allowed_action_indices())
    return tuple(range(env.action_space_size))


def train_variational_ddqn(
    agent: Any,
    env: Any,
    episodes: int,
    warmup_steps: int = 32,
    batch_size: int = 16,
    update_every: int = 4,
    target_sync_every: int = 50,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> dict[str, float | list[float]]:
    """Train while preserving the environment's fixed action availability."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    total_steps, losses, episode_rewards = 0, [], []
    for episode in range(episodes):
        state, done, total_reward = env.reset(), False, 0.0
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode / max(episodes - 1, 1)
        while not done:
            action_mask = _available_actions(env)
            action = agent.act(state, action_mask, epsilon=epsilon)
            next_state, reward, done, _info = env.step(action)
            next_mask = _available_actions(env) if not done else tuple(range(env.action_space_size))
            agent.observe(state, action, reward, next_state, done, next_mask)
            total_steps += 1
            if total_steps >= warmup_steps and total_steps % update_every == 0:
                loss = agent.optimize(batch_size)
                if loss is not None:
                    losses.append(loss)
            if total_steps % target_sync_every == 0:
                agent.sync_target()
            state, total_reward = next_state, total_reward + reward
        episode_rewards.append(total_reward)
    agent.sync_target()
    return {
        "episode_rewards": episode_rewards,
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "mean_td_loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": float(total_steps),
    }
