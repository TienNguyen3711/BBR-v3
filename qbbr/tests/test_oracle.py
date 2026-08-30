from __future__ import annotations

import pytest

from qbbr.action.registry import load_action_space
from qbbr.eval.oracle import action_sensitivity, stock_action_index


def test_stock_action_is_neutral_pacing_gain():
    config = load_action_space("qbbr/configs/action_pacing_gain.yaml")
    assert stock_action_index(config, 5) == 2


def test_stock_action_is_the_multihead_no_op():
    config = load_action_space("qbbr/configs/action_multihead.yaml")
    assert stock_action_index(config, 125) == 50


def test_one_step_oracle_reports_all_actions(sample_calibration):
    config = load_action_space("qbbr/configs/action_pacing_gain.yaml")
    result = action_sensitivity(
        "Sydney", "downlink", sample_calibration, config, seeds=[7], episode_s=0.2, risk_mode="stub_constant"
    )
    assert result["stock_trajectory"]["decisions"] > 0
    assert len(result["actions"]) == 5
    assert result["actions"][2]["delta_retransmits_vs_stock"] == pytest.approx(0.0)
    assert result["actions"][2]["mean_delivery_retention"] == pytest.approx(1.0)
    assert 0.0 <= result["one_step_oracle_bound"]["delivery_retention_vs_stock"]
