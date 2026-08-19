from __future__ import annotations

import copy

import numpy as np
import pytest
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
    actor_params = sum(p.numel() for p in agent.actor_trunk.parameters())
    actor_params += sum(p.numel() for p in agent.actor_heads.parameters())
    assert actor_params < 100
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
    before = copy.deepcopy(next(agent.actor_trunk.parameters()).detach())
    buf = _random_rollout(agent, n_steps=20)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["loss"])
    after = next(agent.actor_trunk.parameters()).detach()
    assert not torch.allclose(before, after)


def test_normalize_advantage_defaults_true_and_is_configurable():
    assert MLPA2CAgent().normalize_advantage is True
    assert MLPA2CAgent(normalize_advantage=False).normalize_advantage is False


def test_update_works_with_normalize_advantage_disabled():
    agent = MLPA2CAgent(normalize_advantage=False)
    buf = _random_rollout(agent)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["loss"])


def test_save_then_load_restores_exact_behavior(tmp_path):
    torch.manual_seed(0)
    agent = MLPA2CAgent()
    state = np.random.rand(6)
    value_before = agent.value(state)
    path = tmp_path / "checkpoint.pt"
    agent.save(path)

    torch.manual_seed(123)  # different init, so a no-op load would be caught
    fresh_agent = MLPA2CAgent()
    assert fresh_agent.value(state) != value_before

    fresh_agent.load(path)
    assert fresh_agent.value(state) == value_before


def test_load_legacy_single_head_checkpoint_restores_exact_behavior(tmp_path):
    torch.manual_seed(0)
    agent = MLPA2CAgent(action_dims=(5,))
    state = np.random.rand(6)
    state_t = torch.as_tensor(state, dtype=torch.float32)
    logits_before = agent._actor_logits(state_t)[0].detach().clone()
    value_before = agent.value(state)

    # Simulate a pre-actor_trunk/actor_heads-split checkpoint (a single "actor"
    # Sequential, e.g. RQ1a's frozen baseline checkpoints under outputs/checkpoints/):
    # its two Linear layers are exactly today's actor_trunk and actor_heads[0].
    legacy_checkpoint = {
        "actor": {
            "0.weight": agent.actor_trunk[0].weight.detach().clone(),
            "0.bias": agent.actor_trunk[0].bias.detach().clone(),
            "2.weight": agent.actor_heads[0].weight.detach().clone(),
            "2.bias": agent.actor_heads[0].bias.detach().clone(),
        },
        "critic": agent.critic.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy_checkpoint, path)

    torch.manual_seed(123)  # different init, so a no-op load would be caught
    fresh_agent = MLPA2CAgent(action_dims=(5,))
    assert fresh_agent.value(state) != value_before

    fresh_agent.load(path)
    assert fresh_agent.value(state) == value_before
    assert torch.allclose(fresh_agent._actor_logits(state_t)[0].detach(), logits_before)


def test_load_legacy_single_head_checkpoint_rejects_multihead_agent(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"actor": {}, "critic": {}, "optimizer": {}}, path)
    agent = MLPA2CAgent(action_dims=(5, 5))
    with pytest.raises(ValueError):
        agent.load(path)


def test_multihead_action_dims_produces_flat_action_in_range():
    agent = MLPA2CAgent(action_dims=(5, 5, 5))
    assert len(agent.actor_heads) == 3
    for _ in range(30):
        action, log_prob = agent.act(np.random.rand(6))
        assert 0 <= action < 125
        assert np.isfinite(log_prob)


def test_multihead_update_returns_finite_losses_and_changes_weights():
    torch.manual_seed(0)
    agent = MLPA2CAgent(action_dims=(5, 5, 5))
    before = copy.deepcopy(next(agent.actor_trunk.parameters()).detach())
    buf = _random_rollout(agent, n_steps=20)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["loss"])
    after = next(agent.actor_trunk.parameters()).detach()
    assert not torch.allclose(before, after)


def test_single_head_action_dims_matches_flat_action_semantics():
    # action_dims=(5,) (the default) must behave exactly like the pre-
    # multihead agent: flat action == the single head's sampled index.
    agent = MLPA2CAgent(action_dims=(5,))
    for _ in range(10):
        action, _log_prob = agent.act(np.random.rand(6))
        assert 0 <= action < 5
