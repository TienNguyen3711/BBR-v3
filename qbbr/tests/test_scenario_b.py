from __future__ import annotations

import math

import numpy as np
import pytest

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.eval.scenario_b import run_scenario_b


def test_run_scenario_b_returns_expected_shape(sample_calibration):
    agent = MLPA2CAgent()
    result = run_scenario_b(
        agent, "Sydney", "downlink", sample_calibration,
        n_episodes=1, episode_s=1.0, competing_ccas=("cubic", "vegas"),
    )
    assert result["flow_names"] == ["qbbr", "cubic", "vegas"]
    assert len(result["x_achieved_bps"]) == 3
    assert all(np.isfinite(x) for x in result["x_achieved_bps"])


def test_rho_alpha_covers_full_sweep_and_stays_in_expected_ranges(sample_calibration):
    agent = MLPA2CAgent()
    result = run_scenario_b(
        agent, "Sydney", "downlink", sample_calibration,
        n_episodes=1, episode_s=1.0, competing_ccas=("cubic",),
    )
    assert set(result["rho_alpha"].keys()) == {"0.0", "1.0", "2.0", "inf"}
    # alpha=0 (throughput-max) is always exactly 1.0, alpha=inf always in (0,1].
    assert result["rho_alpha"]["0.0"] == pytest.approx(1.0)
    assert 0.0 < result["rho_alpha"]["inf"] <= 1.0 + 1e-9


def test_min_throughput_matches_the_smallest_flow(sample_calibration):
    agent = MLPA2CAgent()
    result = run_scenario_b(
        agent, "Sydney", "downlink", sample_calibration,
        n_episodes=1, episode_s=1.0, competing_ccas=("cubic", "vegas", "hybla"),
    )
    assert result["min_throughput_bps"] == min(result["x_achieved_bps"])


def test_custom_alpha_sweep_is_respected(sample_calibration):
    agent = MLPA2CAgent()
    result = run_scenario_b(
        agent, "Sydney", "downlink", sample_calibration,
        n_episodes=1, episode_s=1.0, competing_ccas=("cubic",), alpha_sweep=(1.0, math.inf),
    )
    assert set(result["rho_alpha"].keys()) == {"1.0", "inf"}
