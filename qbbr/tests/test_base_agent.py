from __future__ import annotations

import torch

from qbbr.agents.base_agent import a2c_losses, discounted_returns


def test_discounted_returns_matches_manual_calc():
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.5
    returns = discounted_returns(rewards, gamma)
    expected = [
        1.0 + 0.5 * (2.0 + 0.5 * 3.0),
        2.0 + 0.5 * 3.0,
        3.0,
    ]
    assert torch.allclose(returns, torch.tensor(expected), atol=1e-6)


def test_discounted_returns_gamma_zero_is_immediate_reward():
    returns = discounted_returns([1.0, 2.0, 3.0], gamma=0.0)
    assert torch.allclose(returns, torch.tensor([1.0, 2.0, 3.0]))


def test_a2c_losses_zero_advantage_gives_zero_losses():
    log_probs = torch.tensor([-0.1, -0.2, -0.3])
    values = torch.tensor([1.0, 2.0, 3.0])
    returns = values.clone()
    actor_loss, critic_loss = a2c_losses(log_probs, values, returns)
    assert torch.isclose(actor_loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(critic_loss, torch.tensor(0.0), atol=1e-6)


def test_a2c_advantage_does_not_backprop_into_critic_via_actor_loss():
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    log_probs = torch.tensor([-0.1, -0.2], requires_grad=True)
    returns = torch.tensor([2.0, 1.0])
    actor_loss, _ = a2c_losses(log_probs, values, returns)
    actor_loss.backward()
    assert values.grad is None or torch.allclose(values.grad, torch.zeros_like(values))


def test_normalize_advantage_is_the_new_default():
    log_probs = torch.tensor([-0.1, -0.2, -0.3])
    values = torch.tensor([0.0, 0.0, 0.0])
    returns = torch.tensor([1.0, 100.0, 300.0])  # large, uneven scale, like a real rollout
    default_actor_loss, default_critic_loss = a2c_losses(log_probs, values, returns)
    explicit_actor_loss, explicit_critic_loss = a2c_losses(log_probs, values, returns, normalize_advantage=True)
    assert torch.isclose(default_actor_loss, explicit_actor_loss)
    assert torch.isclose(default_critic_loss, explicit_critic_loss)


def test_normalize_advantage_true_rescales_actor_gradient_signal_to_unit_variance():
    log_probs = torch.tensor([-0.1, -0.2, -0.3, -0.4])
    values = torch.tensor([0.0, 0.0, 0.0, 0.0])
    returns = torch.tensor([1.0, 100.0, 300.0, 500.0])  # huge, uneven raw advantage scale
    raw_actor_loss, _ = a2c_losses(log_probs, values, returns, normalize_advantage=False)
    normalized_actor_loss, _ = a2c_losses(log_probs, values, returns, normalize_advantage=True)
    # the raw-scale version is dominated by the huge advantage magnitude;
    # the normalized version stays on the same order as log_probs itself.
    assert abs(raw_actor_loss.item()) > 50.0
    assert abs(normalized_actor_loss.item()) < 1.0


def test_normalize_advantage_false_preserves_the_original_raw_formula():
    log_probs = torch.tensor([-0.1, -0.2, -0.3])
    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([2.0, 2.0, 2.0])
    advantage = returns - values
    expected_actor_loss = -(log_probs * advantage).mean()
    expected_critic_loss = advantage.pow(2).mean()

    actor_loss, critic_loss = a2c_losses(log_probs, values, returns, normalize_advantage=False)
    assert torch.isclose(actor_loss, expected_actor_loss)
    assert torch.isclose(critic_loss, expected_critic_loss)


def test_normalize_advantage_handles_zero_variance_without_blowing_up():
    # every sample has the identical advantage -- std=0, so normalization
    # must not divide by exactly zero.
    log_probs = torch.tensor([-0.1, -0.2, -0.3])
    values = torch.tensor([5.0, 5.0, 5.0])
    returns = torch.tensor([5.0, 5.0, 5.0])
    actor_loss, critic_loss = a2c_losses(log_probs, values, returns, normalize_advantage=True)
    assert torch.isfinite(actor_loss)
    assert torch.isclose(actor_loss, torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(critic_loss, torch.tensor(0.0), atol=1e-6)


def test_normalize_advantage_still_blocks_backprop_into_critic():
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    log_probs = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)
    returns = torch.tensor([5.0, 1.0, 9.0])
    actor_loss, _ = a2c_losses(log_probs, values, returns, normalize_advantage=True)
    actor_loss.backward()
    assert values.grad is None or torch.allclose(values.grad, torch.zeros_like(values))
