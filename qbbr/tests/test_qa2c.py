from __future__ import annotations

import copy

import numpy as np
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
        assert 0 <= action < agent.n_actions
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
