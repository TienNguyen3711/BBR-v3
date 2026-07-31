from __future__ import annotations

import numpy as np
import pytest

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.eval.metrics import real_cca_distribution_stats
from qbbr.eval.scenario_a import run_scenario_a, simulated_agent_stats

_EXPECTED_KEYS = {
    "throughput_mbps_median", "throughput_mbps_iqr",
    "rtt_ms_median", "rtt_ms_iqr",
    "retransmits_per_s_median", "retransmits_per_s_iqr",
}


def test_real_cca_distribution_stats_matches_expected_shape(dataset_root):
    stats = real_cca_distribution_stats(dataset_root, "Sydney", "downlink", cca="cubic")
    assert set(stats.keys()) == _EXPECTED_KEYS
    assert stats["throughput_mbps_median"] > 0
    assert stats["rtt_ms_median"] > 0


def test_real_cca_distribution_stats_raises_for_unknown_combo(dataset_root):
    with pytest.raises(ValueError):
        real_cca_distribution_stats(dataset_root, "Sydney", "downlink", cca="not_a_real_cca")


def test_simulated_agent_stats_matches_expected_shape(sample_calibration):
    agent = MLPA2CAgent()
    stats = simulated_agent_stats(agent, "Sydney", "downlink", sample_calibration, n_episodes=1, episode_s=1.0)
    assert set(stats.keys()) == _EXPECTED_KEYS
    for v in stats.values():
        assert np.isfinite(v)


def test_run_scenario_a_returns_qbbr_plus_every_comparison_cca(dataset_root, sample_calibration):
    agent = MLPA2CAgent()
    results = run_scenario_a(
        agent, "Sydney", "downlink", sample_calibration, dataset_root,
        n_episodes=1, episode_s=1.0, comparison_ccas=("bbr", "cubic"),
    )
    assert set(results.keys()) == {"qbbr", "bbr", "cubic"}
    for cca, stats in results.items():
        assert set(stats.keys()) == _EXPECTED_KEYS, cca
