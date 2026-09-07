from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.agents.classical.native_a2c import NativeMLPA2CAgent
from qbbr.agents.quantum.native_qa2c import NativeQA2CAgent
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.train.native_loop import NativeRolloutBuffer


def test_native_qrl_brain_requires_and_obeys_action_mask() -> None:
    agent = NativeQA2CAgent(observation_dim=2, native_action_count=3, n_layers=1)
    state = np.array([0.1, 0.2], dtype=np.float32)
    action, _log_probability = agent.act(state, allowed_indices=(0,))
    assert action == 0
    rollout = NativeRolloutBuffer()
    rollout.add(state, 0, (0,), 1.0)
    rollout.add(state, 0, (0,), 0.5)
    metrics = agent.update(rollout)
    assert np.isfinite(metrics["loss"])


def test_classical_native_baseline_obeys_the_same_mask() -> None:
    agent = NativeMLPA2CAgent(observation_dim=2, native_action_count=3, hidden_dim=4)
    action, _log_probability = agent.act(np.array([0.1, 0.2], dtype=np.float32), allowed_indices=(0,))
    assert action == 0


def test_primary_native_qa2c_has_an_exact_classical_a2c_match() -> None:
    quantum, classical, match = build_matched_native_a2c_agents(
        observation_dim=7, action_count=5, n_layers=2
    )
    assert quantum.param_count() == classical.param_count() == match.quantum_params
    assert (match.actor_hidden_dim, match.critic_hidden_dim) == (3, 9)


def test_stock_init_bias_warm_starts_the_stock_logit_without_changing_param_count() -> None:
    torch.manual_seed(0)
    plain_q, plain_c, plain_match = build_matched_native_a2c_agents(
        observation_dim=7, action_count=5, n_layers=2, stock_action=2, stock_init_bias=0.0
    )
    torch.manual_seed(0)
    biased_q, biased_c, biased_match = build_matched_native_a2c_agents(
        observation_dim=7, action_count=5, n_layers=2, stock_action=2, stock_init_bias=1.5
    )
    # The warm start is a shift of the stock bias term only -- identical
    # structure, so the exact parameter match is untouched.
    assert biased_q.param_count() == plain_q.param_count() == biased_match.quantum_params
    assert biased_c.param_count() == plain_c.param_count() == biased_match.classical_params
    q_shift = (biased_q.actor_head.bias.detach() - plain_q.actor_head.bias.detach())
    c_shift = (biased_c.actor[-1].bias.detach() - plain_c.actor[-1].bias.detach())
    assert np.isclose(q_shift[2].item(), 1.5) and np.allclose(q_shift[[0, 1, 3, 4]].numpy(), 0.0)
    assert np.isclose(c_shift[2].item(), 1.5) and np.allclose(c_shift[[0, 1, 3, 4]].numpy(), 0.0)


def test_primary_and_ablation_protocols_have_unambiguous_roles() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    primary = yaml.safe_load((config_dir / "tier1_native_qa2c_successor_protocol.yaml").read_text())
    full_primary = yaml.safe_load((config_dir / "full_native_qa2c_successor_protocol.yaml").read_text())
    qdqn = yaml.safe_load((config_dir / "full_recurrent_qdqn_ablation_protocol.yaml").read_text())

    assert primary["control"]["role"] == "primary_successor_quantum_brain"
    assert primary["control"]["primary_quantum_agent"] == "NativeQA2CAgent"
    assert primary["control"]["matched_classical_agent"] == "NativeMLPA2CAgent"
    assert primary["control"]["fixed_action_count"] == 5
    assert full_primary["control"]["role"] == "primary_successor_quantum_brain"
    assert full_primary["training"]["locations"] and len(full_primary["training"]["training_seeds"]) == 10
    assert qdqn["algorithm_role"] == "ablation_only_not_primary_successor"
