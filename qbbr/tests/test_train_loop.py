from __future__ import annotations

import numpy as np
import torch

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.train.loop import train


def test_train_runs_n_episodes_and_returns_summary(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.3)
    agent = MLPA2CAgent()

    result = train(agent, env, n_episodes=3, config={"seed": 0})

    assert result["n_episodes"] == 3
    assert len(result["episode_rewards"]) == 3
    assert np.isfinite(result["episode_rewards"]).all()
    assert np.isfinite(result["final_mean_reward"])


def test_train_seed_makes_two_runs_reproducible(sample_calibration):
    # train()'s seeding only covers the rollout/update phase, not agent
    # construction (the agent already exists by the time train() sees it) --
    # so full reproducibility requires seeding *before* building the agent
    # too, matching how qbbr/scripts/train.py orders it.
    def run():
        torch.manual_seed(42)
        np.random.seed(42)
        env = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.3)
        agent = MLPA2CAgent()
        return train(agent, env, n_episodes=2, config={"seed": 42})["episode_rewards"]

    assert run() == run()


def test_on_episode_callback_is_invoked_once_per_episode(sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.3)
    agent = MLPA2CAgent()

    seen = []
    train(agent, env, n_episodes=3, config={}, on_episode=lambda ep, summary: seen.append(ep))
    assert seen == [0, 1, 2]


def test_run_dir_logging_writes_one_row_per_episode(tmp_path, sample_calibration):
    env = FluidSimEnv("Sydney", "downlink", sample_calibration, episode_s=0.3)
    agent = MLPA2CAgent()

    train(agent, env, n_episodes=3, config={}, run_dir=tmp_path)
    lines = (tmp_path / "episodes.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 3  # header + 3 episodes
