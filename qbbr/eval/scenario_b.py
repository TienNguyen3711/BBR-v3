from __future__ import annotations

import math
from typing import Any

import numpy as np

from qbbr.agents.base_agent import BaseAgent
from qbbr.env.multi_flow_env import MultiFlowFluidEnv
from qbbr.eval.metrics import alpha_fair_efficiency_ratio, min_per_flow_throughput

DEFAULT_ALPHA_SWEEP = (0.0, 1.0, 2.0, math.inf)
DEFAULT_COMPETING_CCAS = ("cubic", "vegas", "hybla")


def run_scenario_b(
    agent: BaseAgent,
    location: str,
    direction: str,
    calibration: dict[str, Any],
    n_episodes: int = 10,
    episode_s: float = 300.0,
    risk_mode: str = "stub_constant",
    alpha_sweep: tuple[float, ...] = DEFAULT_ALPHA_SWEEP,
    competing_ccas: tuple[str, ...] = DEFAULT_COMPETING_CCAS,
    action_config: Any = None,
) -> dict[str, Any]:
    flow_names = ("qbbr", *competing_ccas)
    per_episode_mean_bps = {name: [] for name in flow_names}

    for _ep in range(n_episodes):
        env = MultiFlowFluidEnv(
            location, direction, calibration, action_config=action_config, risk_mode=risk_mode,
            episode_s=episode_s, competing_ccas=competing_ccas,
        )
        state = env.reset()
        done = False
        step_bps = {name: [] for name in flow_names}
        while not done:
            action, _log_prob = agent.act(state)
            state, _reward, done, info = env.step(action)
            for name, bps in info["flow_throughput_bps"].items():
                step_bps[name].append(bps)
        for name in flow_names:
            per_episode_mean_bps[name].append(float(np.mean(step_bps[name])))

    x_achieved_bps = [float(np.mean(per_episode_mean_bps[name])) for name in flow_names]

    rho_alpha = {}
    for alpha in alpha_sweep:
        label = "inf" if math.isinf(alpha) else str(alpha)
        rho_alpha[label] = alpha_fair_efficiency_ratio(x_achieved_bps, alpha)

    return {
        "flow_names": list(flow_names),
        "x_achieved_bps": x_achieved_bps,
        "per_episode_mean_bps": per_episode_mean_bps,
        "rho_alpha": rho_alpha,
        "min_throughput_bps": min_per_flow_throughput(x_achieved_bps),
    }
