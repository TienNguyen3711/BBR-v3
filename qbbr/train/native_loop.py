"""Training loop that records the native-action availability mask each step."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from qbbr.agents.base_agent import discounted_returns
from qbbr.train.logging import log_episode

_STOCK_ACTION = 2
_LOW_GAIN = (0, 1)
_HIGH_GAIN = (3, 4)
_NEAR_HANDOVER_S7 = 0.6     # s7 reconfiguration-phase proximity
_PRESSURE_S3 = 0.45         # s3 inflight/BDP
_PRESSURE_S4 = 0.20         # s4 queue


def _phase_stratified_diagnostics(agent, rollout, gamma: float) -> dict[str, float]:
    """Per-episode action x context breakdown: is the policy conditioning its
    high gains on handover proximity and its low gains on queue pressure, or
    just mixing them at a constant rate? Uses a Monte-Carlo advantage
    (discounted return minus the critic baseline); cheap, no extra gradient."""
    if not rollout.states:
        return {}
    returns = discounted_returns(rollout.rewards, gamma).tolist()
    near_high = far_total = near_total = far_high = 0
    pressure_low = pressure_total = headroom_low = headroom_total = 0
    adv = {"high_near": [], "high_far": [], "low_pressure": [], "low_headroom": []}
    for state, action, ret in zip(rollout.states, rollout.actions, returns):
        s = np.asarray(state, dtype=float)
        near = s[6] >= _NEAR_HANDOVER_S7
        pressure = s[2] >= _PRESSURE_S3 or s[3] >= _PRESSURE_S4
        a = int(action)
        advantage = float(ret) - agent.value(state)
        if near:
            near_total += 1
            if a in _HIGH_GAIN:
                near_high += 1
                adv["high_near"].append(advantage)
        else:
            far_total += 1
            if a in _HIGH_GAIN:
                far_high += 1
                adv["high_far"].append(advantage)
        if pressure:
            pressure_total += 1
            if a in _LOW_GAIN:
                pressure_low += 1
                adv["low_pressure"].append(advantage)
        else:
            headroom_total += 1
            if a in _LOW_GAIN:
                headroom_low += 1
                adv["low_headroom"].append(advantage)
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "high_gain_share_near_handover": near_high / max(near_total, 1),
        "high_gain_share_far": far_high / max(far_total, 1),
        "low_gain_share_under_pressure": pressure_low / max(pressure_total, 1),
        "low_gain_share_in_headroom": headroom_low / max(headroom_total, 1),
        "adv_high_gain_near": mean(adv["high_near"]),
        "adv_high_gain_far": mean(adv["high_far"]),
        "adv_low_gain_under_pressure": mean(adv["low_pressure"]),
        "adv_low_gain_in_headroom": mean(adv["low_headroom"]),
    }


@dataclass
class NativeRolloutBuffer:
    states: list[Any] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    action_masks: list[tuple[int, ...]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, state: Any, action: int, action_mask: tuple[int, ...], reward: float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.rewards.append(reward)

    def __len__(self) -> int:
        return len(self.states)


def train_native_qrl(
    agent: Any,
    env: Any,
    n_episodes: int,
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    on_episode: Callable[[int, dict[str, float]], None] | None = None,
    start_episode: int = 0,
    total_episodes: int | None = None,
    environment_seed_base: int | None = None,
    reward_scale_mbps: float = 1.0,
) -> dict[str, Any]:
    """Train only on actions actually available to BBR at each decision time."""

    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Optional entropy schedule: start high, decay linearly to the agent's
    # own entropy_coef floor over `entropy_decay_episodes` (global episode
    # index). Keeps exploration alive through the fragile early phase where
    # some seeds otherwise collapse to constant stock. Off when unset.
    entropy_start = config.get("entropy_start")
    entropy_decay_episodes = float(config.get("entropy_decay_episodes") or (total_episodes or 1))
    if reward_scale_mbps <= 0.0:
        raise ValueError("reward_scale_mbps must be positive.")
    total_episodes = total_episodes if total_episodes is not None else start_episode + n_episodes
    if total_episodes < start_episode + n_episodes:
        raise ValueError("total_episodes cannot end before this training chunk.")
    rewards: list[float] = []
    episode_diagnostics: list[dict[str, float]] = []
    for local_episode in range(n_episodes):
        episode = start_episode + local_episode
        reset_seed = None if environment_seed_base is None else environment_seed_base + episode
        state = env.reset(seed=reset_seed)
        rollout = NativeRolloutBuffer()
        done, episode_reward = False, 0.0
        fallback_count = 0
        stock_only_mask_count = 0
        while not done:
            action_mask = env.allowed_action_indices()
            stock_only_mask_count += int(len(action_mask) == 1)
            action, _log_probability = agent.act(state, action_mask)
            next_state, reward, done, info = env.step(action)
            # Positive scaling is numerical conditioning only; the objective
            # remains delivered throughput alone.
            rollout.add(state, action, action_mask, reward / reward_scale_mbps)
            # The live native environment reports an explicit safety fallback.
            # FluidSimEnv represents the same BBR restriction through its
            # action mask, so this field is intentionally optional there.
            fallback_count += int(info.get("used_stock_fallback", False))
            state = next_state
            episode_reward += reward
        strata = _phase_stratified_diagnostics(agent, rollout, agent.gamma)
        scheduled_entropy = None
        if entropy_start is not None:
            frac = max(0.0, 1.0 - episode / max(entropy_decay_episodes, 1e-9))
            scheduled_entropy = agent.entropy_coef + (float(entropy_start) - agent.entropy_coef) * frac
        metrics = agent.update(rollout, entropy_coef=scheduled_entropy)
        summary = {
            "episode_reward": episode_reward,
            "episode_length": len(rollout),
            "mean_reward": episode_reward / max(len(rollout), 1),
            "stock_fallback_fraction": fallback_count / max(len(rollout), 1),
            "stock_only_action_mask_fraction": stock_only_mask_count / max(len(rollout), 1),
            **strata,
            **metrics,
        }
        rewards.append(episode_reward)
        episode_diagnostics.append({key: float(value) for key, value in summary.items()})
        if run_dir is not None:
            log_episode(run_dir, episode, summary)
        if on_episode is not None:
            on_episode(episode, summary)
    if run_dir is not None:
        agent.save(Path(run_dir) / "checkpoint.pt")
    return {
        "n_episodes": n_episodes,
        "episode_rewards": rewards,
        "episode_diagnostics": episode_diagnostics,
        "final_mean_reward": rewards[-1] / max(len(rollout), 1) if rewards else float("nan"),
        "completed_episode": start_episode + n_episodes,
    }
