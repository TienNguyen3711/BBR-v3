from __future__ import annotations

import copy

import numpy as np
import torch

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent, _best_hidden_size, _mlp_param_count
from qbbr.train.buffer import RolloutBuffer


def _random_rollout(agent, n_steps=12):
    buf = RolloutBuffer()
    for _ in range(n_steps):
        s = np.random.rand(6)
        a, log_prob = agent.act(s)
        v = agent.value(s)
        buf.add(s, a, log_prob, reward=float(np.random.randn()), value=v)
    return buf


def test_best_hidden_size_minimizes_distance_to_target():
    h = _best_hidden_size(in_dim=6, out_dim=5, target_params=36)
    achieved = _mlp_param_count(6, h, 5)
    for candidate in range(max(1, h - 2), h + 3):
        assert abs(achieved - 36) <= abs(_mlp_param_count(6, candidate, 5) - 36)


def test_param_count_is_within_reach_of_default_budget():
    # default budget is 18*n_layers (matching QA2C's own per-circuit count)
    agent = MLPA2CAgent(n_layers=2)
    assert agent.param_count() > 0
    # each sub-network's own params shouldn't wildly overshoot the 36-target
    assert sum(p.numel() for p in agent.actor.parameters()) < 100
    assert sum(p.numel() for p in agent.critic.parameters()) < 100


def test_act_returns_valid_action_and_log_prob():
    agent = MLPA2CAgent()
    for _ in range(10):
        action, log_prob = agent.act(np.random.rand(6))
        assert 0 <= action < 5
        assert np.isfinite(log_prob)


def test_update_returns_finite_losses_and_changes_weights():
    torch.manual_seed(0)
    agent = MLPA2CAgent()
    before = copy.deepcopy(next(agent.actor.parameters()).detach())
    buf = _random_rollout(agent, n_steps=20)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["loss"])
    after = next(agent.actor.parameters()).detach()
    assert not torch.allclose(before, after)


def test_normalize_advantage_defaults_true_and_is_configurable():
    assert MLPA2CAgent().normalize_advantage is True
    assert MLPA2CAgent(normalize_advantage=False).normalize_advantage is False


def test_update_works_with_normalize_advantage_disabled():
    agent = MLPA2CAgent(normalize_advantage=False)
    buf = _random_rollout(agent)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["loss"])
