"""Screen the fixed-action recurrent-QDQN QRL ablation in the legacy simulator.

This is explicitly a simulator proxy, not a BBR-v3 kernel or Starlink field
result.  It keeps the five pacing gains, seven-feature observation, BBR-state/
reconfiguration action mask and EMA actuation represented by ``FluidSimEnv``.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from qbbr.agents.quantum.recurrent_prioritized_ddqn import (
    RecurrentPrioritizedVariationalDoubleDQN,
)
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.train.recurrent_qdqn_loop import train_recurrent_prioritized_ddqn


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
_STOCK_GAIN_ACTION = 2  # action_pacing_gain.yaml's fixed 1.0 choice.


def _make_env(calibration, location: str, direction: str, duration_s: float) -> FluidSimEnv:
    return FluidSimEnv(
        location,
        direction,
        calibration,
        risk_mode="stub_constant",
        episode_s=duration_s,
        reward_mode="throughput_only",
    )


def _summary(agent, calibration, location: str, direction: str, episodes: int, duration_s: float, history_window: int):
    throughputs, retransmits, rtts = [], [], []
    action_counts: dict[str, int] = {}
    for seed in range(episodes):
        env = _make_env(calibration, location, direction, duration_s)
        state, done = env.reset(seed=seed), False
        history = deque([np.asarray(state, dtype=np.float32)], maxlen=history_window)
        while not done:
            action = agent.act(np.stack(history), env.allowed_action_indices(), epsilon=0.0)
            action_counts[str(action)] = action_counts.get(str(action), 0) + 1
            state, _reward, done, info = env.step(action)
            history.append(np.asarray(state, dtype=np.float32))
            throughputs.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            retransmits.append(info["retransmits"] / info["t_dec_s"])
            rtts.append(info["rtt_ms"])
    return {
        "throughput_mbps_mean": float(np.mean(throughputs) / 1e6),
        "retransmits_per_s_mean": float(np.mean(retransmits)),
        "rtt_ms_mean": float(np.mean(rtts)),
        "action_counts": action_counts,
    }


def _stock_summary(calibration, location: str, direction: str, episodes: int, duration_s: float):
    throughputs, retransmits, rtts = [], [], []
    for seed in range(episodes):
        env = _make_env(calibration, location, direction, duration_s)
        _state, done = env.reset(seed=seed), False
        while not done:
            _state, _reward, done, info = env.step(_STOCK_GAIN_ACTION)
            throughputs.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            retransmits.append(info["retransmits"] / info["t_dec_s"])
            rtts.append(info["rtt_ms"])
    return {
        "throughput_mbps_mean": float(np.mean(throughputs) / 1e6),
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
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "pilot_recurrent_prioritized_ddqn.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    calibration = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
    env = _make_env(calibration, args.location, args.direction, args.duration_s)
    agent = RecurrentPrioritizedVariationalDoubleDQN(
        observation_dim=env.observation_dim,
        action_count=env.action_space_size,
        n_layers=1,
        seed=args.seed,
    )
    training = train_recurrent_prioritized_ddqn(
        agent, env, args.episodes, history_window=args.history_window
    )
    learned = _summary(agent, calibration, args.location, args.direction, args.evaluation_episodes, args.duration_s, args.history_window)
    stock = _stock_summary(calibration, args.location, args.direction, args.evaluation_episodes, args.duration_s)
    result = {
        "algorithm_role": "ablation_only_not_primary_successor",
        "evidence_tier": "simulator_proxy_only",
        "warning": "Not evidence of field throughput improvement or BBR-v3 kernel deployment.",
        "reward_contract": "supervisor-locked-throughput-only-v1",
        "state_and_actuation": "legacy FluidSimEnv: seven features, fixed five gains, STARTUP/reconfiguration mask, EMA smoothing",
        "location": args.location,
        "direction": args.direction,
        "history_window": args.history_window,
        "training": training,
        "recurrent_prioritized_variational_double_dqn": learned,
        "stock_pacing_gain_100": stock,
        "throughput_delta_pct": 100.0 * (learned["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
