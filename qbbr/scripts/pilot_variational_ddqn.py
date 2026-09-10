"""Run a clearly labelled simulator-only variational Double-DQN ablation.

This script is a technical screening experiment.  Its output is not field
evidence and must not be used for an RL--BBR performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qbbr.agents.quantum.variational_ddqn import VariationalDoubleDQN
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.train.qdqn_loop import train_variational_ddqn


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _summarize_policy(agent, calibration, location: str, direction: str, episodes: int, duration_s: float):
    bps, retransmits, rtts = [], [], []
    action_counts: dict[str, int] = {}
    for seed in range(episodes):
        env = FluidSimEnv(location, direction, calibration, risk_mode="stub_constant", episode_s=duration_s, reward_mode="throughput_only")
        state, done = env.reset(seed=seed), False
        while not done:
            action = agent.act(state, env.allowed_action_indices(), epsilon=0.0)
            action_counts[str(action)] = action_counts.get(str(action), 0) + 1
            state, _reward, done, info = env.step(action)
            bps.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            retransmits.append(info["retransmits"] / info["t_dec_s"])
            rtts.append(info["rtt_ms"])
    return {
        "throughput_mbps_mean": float(np.mean(bps) / 1e6),
        "retransmits_per_s_mean": float(np.mean(retransmits)),
        "rtt_ms_mean": float(np.mean(rtts)),
        "action_counts": action_counts,
    }


def _summarize_stock(calibration, location: str, direction: str, episodes: int, duration_s: float):
    bps, retransmits, rtts = [], [], []
    for seed in range(episodes):
        env = FluidSimEnv(location, direction, calibration, risk_mode="stub_constant", episode_s=duration_s, reward_mode="throughput_only")
        _state, done = env.reset(seed=seed), False
        while not done:
            _state, _reward, done, info = env.step(2)  # action_pacing_gain.yaml: 1.0 stock choice
            bps.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            retransmits.append(info["retransmits"] / info["t_dec_s"])
            rtts.append(info["rtt_ms"])
    return {
        "throughput_mbps_mean": float(np.mean(bps) / 1e6),
        "retransmits_per_s_mean": float(np.mean(retransmits)),
        "rtt_ms_mean": float(np.mean(rtts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", default="Sydney")
    parser.add_argument("--direction", choices=["uplink", "downlink"], default="downlink")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "pilot_variational_ddqn.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    calibration = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
    env = FluidSimEnv(args.location, args.direction, calibration, risk_mode="stub_constant", episode_s=args.duration_s, reward_mode="throughput_only")
    agent = VariationalDoubleDQN(
        observation_dim=env.observation_dim,
        action_count=env.action_space_size,
        n_layers=1,
        seed=args.seed,
    )
    training = train_variational_ddqn(agent, env, args.episodes)
    learned = _summarize_policy(agent, calibration, args.location, args.direction, args.evaluation_episodes, args.duration_s)
    stock = _summarize_stock(calibration, args.location, args.direction, args.evaluation_episodes, args.duration_s)
    result = {
        "algorithm_role": "ablation_only_not_primary_successor",
        "evidence_tier": "simulator_proxy_only",
        "warning": "Not evidence of field throughput improvement or BBR-v3 kernel deployment.",
        "reward_contract": "supervisor-locked-throughput-only-v1",
        "location": args.location,
        "direction": args.direction,
        "training": training,
        "variational_double_dqn": learned,
        "stock_pacing_gain_100": stock,
        "throughput_delta_pct": 100.0 * (learned["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
