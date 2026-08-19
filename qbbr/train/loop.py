from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from qbbr.train.buffer import RolloutBuffer
from qbbr.train.logging import log_episode


def train(
    agent: Any,
    env: Any,
    n_episodes: int,
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    on_episode: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:

    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    episode_rewards: list[float] = []
    mean_rewards: list[float] = []
    for episode in range(n_episodes):
        state = env.reset()
        buffer = RolloutBuffer()
        episode_reward = 0.0
        done = False
        while not done:
            action, log_prob = agent.act(state)
            value = agent.value(state)
            next_state, reward, done, _info = env.step(action)
            buffer.add(state, action, log_prob, reward, value)
            state = next_state
            episode_reward += reward

        update_metrics = agent.update(buffer)
        summary = {
            "episode_reward": episode_reward,
            "episode_length": len(buffer),
            "mean_reward": episode_reward / max(len(buffer), 1),
            **update_metrics,
        }
        episode_rewards.append(episode_reward)
        mean_rewards.append(summary["mean_reward"])

        if run_dir is not None:
            log_episode(run_dir, episode, summary)
        if on_episode is not None:
            on_episode(episode, summary)

    if run_dir is not None:
        agent.save(Path(run_dir) / "checkpoint.pt")

    return {
        "n_episodes": n_episodes,
        "episode_rewards": episode_rewards,
        "final_mean_reward": mean_rewards[-1] if mean_rewards else float("nan"),
    }
