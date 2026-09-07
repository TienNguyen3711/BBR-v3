import numpy as np
import pytest
import torch

from qbbr.eval.matched_qrl import MatchedQRLBenchmarkSpec, build_matched_agents
from qbbr.agents.quantum.variational_ddqn import (
    _baseline_anchored_masked_argmax,
    _empirically_certified_action,
)
from qbbr.scripts.run_matched_qrl_benchmark import _assess_records


def test_matched_recurrent_agents_have_equal_parameters_and_respect_mask() -> None:
    spec = MatchedQRLBenchmarkSpec(observation_dim=2, action_count=5)
    quantum, classical = build_matched_agents(spec, seed=0)
    assert quantum.param_count() == classical.param_count()
    history = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    assert quantum.act(history, (2,), epsilon=0.0) == 2
    assert classical.act(history, (2,), epsilon=0.0) == 2


def test_matched_benchmark_rejects_a_non_exact_core_shape() -> None:
    with pytest.raises(ValueError, match="latent_dim=3"):
        MatchedQRLBenchmarkSpec(observation_dim=2, action_count=5, latent_dim=4)


def test_selection_assessment_is_post_training_and_throughput_first() -> None:
    def record(seed: int, delta: float, action_zero_share: float) -> dict:
        return {
            "location": "London",
            "seed": seed,
            "core": "quantum",
            "evaluation": {
                "throughput_delta_vs_stock_pct": delta,
                "retransmits_per_s_mean": 1.0,
                "action_shares": {
                    "0": action_zero_share, "1": 0.0, "2": 1.0 - action_zero_share,
                    "3": 0.0, "4": 0.0,
                },
            },
        }

    assessment = _assess_records(
        [record(0, 1.0, 0.0), record(1, 0.0, 0.0), record(2, 2.0, 0.0)],
        {
            "min_median_throughput_delta_vs_stock_pct": 0.0,
            "max_low_gain_action_share": 0.10,
            "max_mean_action_js_divergence": 0.20,
        },
    )[0]
    assert assessment["qualified_for_longer_training"]
    assert assessment["criteria_pass"]["throughput_vs_stock"]


def test_baseline_anchored_selector_keeps_stock_without_q_evidence() -> None:
    values = torch.tensor([1.03, 0.5, 1.0, 0.9, 0.8])
    assert _baseline_anchored_masked_argmax(
        values, (0, 1, 2, 3, 4), baseline_action=2,
        min_relative_advantage=0.05, min_absolute_advantage=0.10,
    ) == 2
    values[3] = 1.20
    assert _baseline_anchored_masked_argmax(
        values, (0, 1, 2, 3, 4), baseline_action=2,
        min_relative_advantage=0.05, min_absolute_advantage=0.10,
    ) == 3


def test_empirical_certification_does_not_mask_the_native_action() -> None:
    # Action 0 remains in the action alphabet; it is simply not exploited
    # until throughput-only replay data certifies it against stock action 2.
    assert _empirically_certified_action(
        0, (0, 1, 2, 3, 4), baseline_action=2,
        action_samples=(20, 0, 100, 0, 0),
        action_mean_rewards=(0.9, 0.0, 1.0, 0.0, 0.0),
        min_samples=16,
    ) == 2
    assert _empirically_certified_action(
        0, (0, 1, 2, 3, 4), baseline_action=2,
        action_samples=(20, 0, 100, 0, 0),
        action_mean_rewards=(1.1, 0.0, 1.0, 0.0, 0.0),
        min_samples=16,
    ) == 0
