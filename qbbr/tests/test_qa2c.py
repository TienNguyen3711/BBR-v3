from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from qbbr.agents.quantum.qa2c import QA2CAgent
from qbbr.train.buffer import RolloutBuffer


def _random_rollout(agent, n_steps=12):
    buf = RolloutBuffer()
    for _ in range(n_steps):
        s = np.random.rand(6)
        a, log_prob = agent.act(s)
        v = agent.value(s)
        buf.add(s, a, log_prob, reward=float(np.random.randn()), value=v)
    return buf


def test_param_count_matches_main_tex_formula():
    agent = QA2CAgent(n_layers=2)
    # 2x(18*L quantum circuit) + 6x5+5 actor head + 1x1+1 critic head
    assert agent.param_count() == 2 * 36 + 35 + 2

    agent3 = QA2CAgent(n_layers=3)
    assert agent3.param_count() == 2 * 54 + 35 + 2


def test_act_returns_valid_action_and_log_prob():
    agent = QA2CAgent()
    for _ in range(10):
        action, log_prob = agent.act(np.random.rand(6))
        assert 0 <= action < agent.action_dims[0]
        assert np.isfinite(log_prob)


def test_value_returns_finite_scalar():
    agent = QA2CAgent()
    v = agent.value(np.random.rand(6))
    assert isinstance(v, float)
    assert np.isfinite(v)


def test_update_returns_finite_losses():
    agent = QA2CAgent()
    buf = _random_rollout(agent)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["loss"])


def test_update_actually_changes_the_weights():
    torch.manual_seed(0)
    np.random.seed(0)
    agent = QA2CAgent()
    before = copy.deepcopy(agent.actor_qnn.weights.detach())
    buf = _random_rollout(agent, n_steps=20)
    agent.update(buf)
    after = agent.actor_qnn.weights.detach()
    assert not torch.allclose(before, after)


def test_actor_and_critic_circuits_have_independent_weights():
    agent = QA2CAgent()
    assert not torch.allclose(agent.actor_qnn.weights, agent.critic_qnn.weights)


def test_normalize_advantage_defaults_true_and_is_configurable():
    assert QA2CAgent().normalize_advantage is True
    assert QA2CAgent(normalize_advantage=False).normalize_advantage is False


def test_update_works_with_normalize_advantage_disabled():
    agent = QA2CAgent(normalize_advantage=False)
    buf = _random_rollout(agent)
    metrics = agent.update(buf)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["critic_loss"])


def test_save_then_load_restores_exact_behavior(tmp_path):
    torch.manual_seed(0)
    agent = QA2CAgent()
    state = np.random.rand(6)
    value_before = agent.value(state)
    path = tmp_path / "checkpoint.pt"
    agent.save(path)

    torch.manual_seed(123)  # different init, so a no-op load would be caught
    fresh_agent = QA2CAgent()
    assert fresh_agent.value(state) != value_before

    fresh_agent.load(path)
    assert fresh_agent.value(state) == value_before


def test_load_legacy_single_head_checkpoint_restores_exact_behavior(tmp_path):
    torch.manual_seed(0)
    agent = QA2CAgent(action_dims=(5,))
    state = np.random.rand(6)
    state_t = torch.as_tensor(state, dtype=torch.float32)
    logits_before = agent._actor_logits(state_t)[0].detach().clone()
    value_before = agent.value(state)

    # Simulate a pre-ModuleList-wrap checkpoint (a single "actor_head" nn.Linear,
    # e.g. RQ4's frozen checkpoints under outputs/checkpoints/quantum/): its flat
    # weight/bias are exactly today's actor_heads[0].
    legacy_checkpoint = {
        "actor_qnn": agent.actor_qnn.state_dict(),
        "actor_head": {
            "weight": agent.actor_heads[0].weight.detach().clone(),
            "bias": agent.actor_heads[0].bias.detach().clone(),
        },
        "critic_qnn": agent.critic_qnn.state_dict(),
        "critic_head": agent.critic_head.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy_checkpoint, path)

    torch.manual_seed(123)  # different init, so a no-op load would be caught
    fresh_agent = QA2CAgent(action_dims=(5,))
    assert fresh_agent.value(state) != value_before

    fresh_agent.load(path)
    assert fresh_agent.value(state) == value_before
    assert torch.allclose(fresh_agent._actor_logits(state_t)[0].detach(), logits_before)


def test_load_legacy_single_head_checkpoint_rejects_multihead_agent(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"actor_qnn": {}, "actor_head": {}, "critic_qnn": {}, "critic_head": {}}, path)
    agent = QA2CAgent(action_dims=(5, 5))
    with pytest.raises(ValueError):
        agent.load(path)


def test_multihead_action_dims_produces_flat_action_in_range():
    agent = QA2CAgent(action_dims=(5, 5, 5))
    assert len(agent.actor_heads) == 3
    for _ in range(30):
        action, log_prob = agent.act(np.random.rand(6))
        assert 0 <= action < 125
        assert np.isfinite(log_prob)


def test_multihead_update_actually_changes_the_weights():
    torch.manual_seed(0)
    np.random.seed(0)
    agent = QA2CAgent(action_dims=(5, 5, 5))
    before = copy.deepcopy(agent.actor_qnn.weights.detach())
    buf = _random_rollout(agent, n_steps=20)
    agent.update(buf)
    after = agent.actor_qnn.weights.detach()
    assert not torch.allclose(before, after)


def test_multihead_param_count_adds_two_more_heads():
    agent5 = QA2CAgent(n_layers=2, action_dims=(5,))
    agent555 = QA2CAgent(n_layers=2, action_dims=(5, 5, 5))
    # same trunk/critic; actor_heads grows from one Linear(6,5) to three
    extra_heads_params = 2 * (6 * 5 + 5)  # two more Linear(6,5) heads
    assert agent555.param_count() == agent5.param_count() + extra_heads_params
