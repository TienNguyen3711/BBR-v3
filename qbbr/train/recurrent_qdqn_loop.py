"""Sequence-aware training loop for the recurrent prioritized QRL brain."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from qbbr.train.qdqn_loop import _available_actions


def train_recurrent_prioritized_ddqn(
    agent: Any,
    env: Any,
    episodes: int,
    history_window: int = 8,
    warmup_steps: int = 32,
    batch_size: int = 16,
    update_every: int = 4,
    target_sync_every: int = 50,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    reward_scale_mbps: float = 1.0,
    start_episode: int = 0,
    total_episodes: int | None = None,
    environment_seed_base: int | None = None,
) -> dict[str, float | list[float]]:
    if episodes < 1 or history_window < 1:
        raise ValueError("episodes and history_window must be positive")
    if not np.isfinite(reward_scale_mbps) or reward_scale_mbps <= 0.0:
        raise ValueError("reward_scale_mbps must be a positive finite constant.")
    if start_episode < 0:
        raise ValueError("start_episode must be non-negative.")
    total_episodes = total_episodes if total_episodes is not None else start_episode + episodes
    if total_episodes < start_episode + episodes:
        raise ValueError("total_episodes cannot end before the requested training chunk.")
    steps, losses, rewards = 0, [], []
    for local_episode in range(episodes):
        episode = start_episode + local_episode
        reset_seed = None if environment_seed_base is None else environment_seed_base + episode
        state, done, total_reward = env.reset(seed=reset_seed), False, 0.0
        history = deque([np.asarray(state, dtype=np.float32)], maxlen=history_window)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode / max(total_episodes - 1, 1)
        while not done:
            action_mask = _available_actions(env)
            history_array = np.stack(history)
            action = agent.act(history_array, action_mask, epsilon)
            next_state, reward, done, _info = env.step(action)
            next_history = deque(history, maxlen=history_window)
            next_history.append(np.asarray(next_state, dtype=np.float32))
            next_mask = _available_actions(env) if not done else tuple(range(env.action_space_size))
            # Positive constant scaling changes numerical conditioning only:
            # maximizing this signal remains exactly maximizing throughput.
            agent.observe(
                history_array, action, reward / reward_scale_mbps,
                np.stack(next_history), done, next_mask,
            )
            history = next_history
            steps += 1
            if steps >= warmup_steps and steps % update_every == 0:
                loss = agent.optimize(batch_size)
                if loss is not None:
                    losses.append(loss)
            if steps % target_sync_every == 0:
                agent.sync_target()
            total_reward += reward
        rewards.append(total_reward)
    agent.sync_target()
    return {
        "episode_rewards": rewards,
        "mean_episode_reward": float(np.mean(rewards)),
        "mean_td_loss": float(np.mean(losses)) if losses else float("nan"),
        "steps": float(steps),
        "reward_scale_mbps": float(reward_scale_mbps),
        "start_episode": float(start_episode),
        "completed_episode": float(start_episode + episodes),
    }
