from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qbbr.agents.base_agent import BaseAgent
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.metrics import real_cca_distribution_stats

DEFAULT_COMPARISON_CCAS = ("bbr", "cubic", "vegas", "hybla")


def _summarize(bps: pd.Series, rtt: pd.Series, rtx_per_s: pd.Series) -> dict[str, float]:
    return {
        "throughput_mbps_median": float(bps.median() / 1e6),
        "throughput_mbps_iqr": float(bps.quantile(0.75) / 1e6 - bps.quantile(0.25) / 1e6),
        "rtt_ms_median": float(rtt.median()),
        "rtt_ms_iqr": float(rtt.quantile(0.75) - rtt.quantile(0.25)),
        "retransmits_per_s_median": float(rtx_per_s.median()),
        "retransmits_per_s_iqr": float(rtx_per_s.quantile(0.75) - rtx_per_s.quantile(0.25)),
    }


def simulated_agent_stats(
    agent: BaseAgent,
    location: str,
    direction: str,
    calibration: dict[str, Any],
    n_episodes: int,
    episode_s: float,
    risk_mode: str = "stub_constant",
) -> dict[str, float]:

    bps_all, rtt_all, rtx_per_s_all = [], [], []
    for _ep in range(n_episodes):
        env = FluidSimEnv(location, direction, calibration, episode_s=episode_s, risk_mode=risk_mode)
        state = env.reset()
        done = False
        while not done:
            action, _log_prob = agent.act(state)
            state, _reward, done, info = env.step(action)
            bps_all.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            rtt_all.append(info["rtt_ms"])
            rtx_per_s_all.append(info["retransmits"] / info["t_dec_s"])

    return _summarize(pd.Series(bps_all), pd.Series(rtt_all), pd.Series(rtx_per_s_all))


def run_scenario_a(
    agent: BaseAgent,
    location: str,
    direction: str,
    calibration: dict[str, Any],
    dataset_root: str | Path,
    n_episodes: int = 10,
    episode_s: float = 300.0,
    risk_mode: str = "stub_constant",
    comparison_ccas: tuple[str, ...] = DEFAULT_COMPARISON_CCAS,
) -> dict[str, dict[str, float]]:

    results = {
        "qbbr": simulated_agent_stats(agent, location, direction, calibration, n_episodes, episode_s, risk_mode)
    }
    for cca in comparison_ccas:
        results[cca] = real_cca_distribution_stats(dataset_root, location, direction, cca)
    return results
