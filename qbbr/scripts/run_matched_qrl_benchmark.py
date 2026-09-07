"""Preflight or execute a recurrent-QDQN classical-vs-quantum ablation.

It does not replace the primary QA2C/A2C successor. By default this command is
a no-write preflight. Training requires both ``--execute`` and
``--allow-simulator-proxy`` so a simulator screen cannot be mistaken for a
field BBR-v3 experiment.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.matched_qrl import MatchedQRLBenchmarkSpec, build_matched_agents
from qbbr.train.recurrent_qdqn_loop import train_recurrent_prioritized_ddqn


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "matched_recurrent_qrl_benchmark.yaml"
STOCK_ACTION = 2  # action_pacing_gain.yaml's existing 1.0 choice


def _env(
    calibration, location: str, direction: str, duration_s: float,
    dynamics_overrides: dict[str, float] | None = None,
) -> FluidSimEnv:
    return FluidSimEnv(
        location, direction, calibration, risk_mode="stub_constant", episode_s=duration_s,
        reward_mode="throughput_only", dynamics_overrides=dynamics_overrides,
    )


def _evaluate(
    agent, calibration, location: str, direction: str, duration_s: float, episodes: int,
    history_window: int, dynamics_overrides: dict[str, float] | None = None,
):
    metrics = {"throughput_mbps": [], "retransmits_per_s": [], "rtt_ms": []}
    action_counts: dict[str, int] = {}
    for seed in range(episodes):
        env = _env(calibration, location, direction, duration_s, dynamics_overrides)
        state, done = env.reset(seed=seed), False
        history = deque([np.asarray(state, dtype=np.float32)], maxlen=history_window)
        while not done:
            action = agent.act(np.stack(history), env.allowed_action_indices(), epsilon=0.0)
            action_counts[str(action)] = action_counts.get(str(action), 0) + 1
            state, _reward, done, info = env.step(action)
            history.append(np.asarray(state, dtype=np.float32))
            metrics["throughput_mbps"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            metrics["retransmits_per_s"].append(info["retransmits"] / info["t_dec_s"])
            metrics["rtt_ms"].append(info["rtt_ms"])
    total_actions = sum(action_counts.values())
    action_shares = {
        str(action): action_counts.get(str(action), 0) / total_actions
        for action in range(5)
    }
    return (
        {f"{name}_mean": float(np.mean(values)) for name, values in metrics.items()}
        | {"action_counts": action_counts, "action_shares": action_shares}
    )


def _evaluate_stock(
    calibration, location: str, direction: str, duration_s: float, episodes: int,
    dynamics_overrides: dict[str, float] | None = None,
):
    metrics = {"throughput_mbps": [], "retransmits_per_s": [], "rtt_ms": []}
    for seed in range(episodes):
        env = _env(calibration, location, direction, duration_s, dynamics_overrides)
        _state, done = env.reset(seed=seed), False
        while not done:
            _state, _reward, done, info = env.step(STOCK_ACTION)
            metrics["throughput_mbps"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            metrics["retransmits_per_s"].append(info["retransmits"] / info["t_dec_s"])
            metrics["rtt_ms"].append(info["rtt_ms"])
    total_actions = len(metrics["throughput_mbps"])
    return (
        {f"{name}_mean": float(np.mean(values)) for name, values in metrics.items()}
        | {
            "action_counts": {str(STOCK_ACTION): total_actions},
            "action_shares": {
                str(action): float(action == STOCK_ACTION) for action in range(5)
            },
        }
    )


def _mean_js_divergence(action_shares: list[list[float]]) -> float:
    """Return mean Jensen-Shannon divergence across seed policies in nats."""
    distributions = np.asarray(action_shares, dtype=float)
    mean_distribution = np.mean(distributions, axis=0)
    eps = np.finfo(float).tiny
    divergences = []
    for distribution in distributions:
        midpoint = 0.5 * (distribution + mean_distribution)
        kl_to_midpoint = np.sum(
            np.where(distribution > 0.0, distribution * np.log((distribution + eps) / (midpoint + eps)), 0.0)
        )
        kl_mean_to_midpoint = np.sum(
            np.where(mean_distribution > 0.0, mean_distribution * np.log((mean_distribution + eps) / (midpoint + eps)), 0.0)
        )
        divergences.append(0.5 * (kl_to_midpoint + kl_mean_to_midpoint))
    return float(np.mean(divergences))


def _assess_records(records: list[dict], criteria: dict) -> list[dict]:
    """Summarise throughput-first selection criteria without changing reward."""
    assessments = []
    for location in sorted({record["location"] for record in records}):
        for core in ("quantum", "classical"):
            group = [record for record in records if record["location"] == location and record["core"] == core]
            if not group:
                continue
            deltas = [record["evaluation"]["throughput_delta_vs_stock_pct"] for record in group]
            retransmit_deltas = [
                record["evaluation"]["retransmits_per_s_mean"] for record in group
            ]
            shares = [
                [record["evaluation"]["action_shares"][str(action)] for action in range(5)]
                for record in group
            ]
            mean_shares = np.mean(np.asarray(shares), axis=0)
            low_gain_share = float(mean_shares[0])
            median_delta = float(np.median(deltas))
            mean_jsd = _mean_js_divergence(shares)
            throughput_pass = median_delta >= criteria["min_median_throughput_delta_vs_stock_pct"]
            low_gain_pass = low_gain_share <= criteria["max_low_gain_action_share"]
            stability_pass = mean_jsd <= criteria["max_mean_action_js_divergence"]
            assessments.append(
                {
                    "location": location,
                    "core": core,
                    "seed_count": len(group),
                    "median_throughput_delta_vs_stock_pct": median_delta,
                    "per_seed_throughput_delta_vs_stock_pct": deltas,
                    "mean_retransmits_per_s": float(np.mean(retransmit_deltas)),
                    "mean_action_shares": {str(action): float(mean_shares[action]) for action in range(5)},
                    "low_gain_action_075_share": low_gain_share,
                    "mean_action_js_divergence": mean_jsd,
                    "criteria_pass": {
                        "throughput_vs_stock": throughput_pass,
                        "avoid_systematic_low_gain": low_gain_pass,
                        "stable_action_distribution": stability_pass,
                    },
                    "qualified_for_longer_training": bool(throughput_pass and low_gain_pass and stability_pass),
                }
            )
    return assessments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--locations", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-simulator-proxy", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "matched_qrl_benchmark.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("algorithm_role") != "ablation_only_not_primary_successor":
        raise ValueError("This runner is reserved for an explicitly labelled recurrent-QDQN ablation.")
    run = config["run"]
    locations = args.locations or run["locations"]
    seeds = args.seeds or run["seeds"]
    episodes = args.episodes or run["episodes"]
    duration_s = args.duration_s or run["duration_s"]
    evaluation_episodes = args.evaluation_episodes or run["evaluation_episodes"]
    calibration = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
    dynamics_overrides = config.get("dynamics_overrides")
    selection_criteria = config["selection_criteria"]
    probe_env = _env(calibration, locations[0], run["direction"], duration_s, dynamics_overrides)
    spec = MatchedQRLBenchmarkSpec(observation_dim=probe_env.observation_dim, action_count=probe_env.action_space_size, **config["agent"])
    quantum, classical = build_matched_agents(spec, seed=seeds[0])
    plan = {
        "mode": "simulator_proxy_training" if args.execute else "preflight_only",
        "algorithm_role": config["algorithm_role"],
        "evidence_tier": "simulator_proxy_only",
        "locations": locations,
        "direction": run["direction"],
        "seeds": seeds,
        "episodes": episodes,
        "duration_s": duration_s,
        "evaluation_episodes": evaluation_episodes,
        "jobs": len(locations) * len(seeds) * 2,
        "shared_contract": spec.as_dict(),
        "parameter_counts": {"quantum": quantum.param_count(), "classical": classical.param_count()},
        "dynamics_overrides": dynamics_overrides or {},
        "dynamics_evidence": config.get("dynamics_evidence", "legacy instantaneous-capacity path"),
        "selection_criteria": selection_criteria,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not args.allow_simulator_proxy:
        raise SystemExit("Refusing to train: pass --allow-simulator-proxy to acknowledge this is not field evidence.")

    records = []
    stock_evaluations = []
    for location in locations:
        stock_evaluations.append(
            {
                "location": location,
                "evaluation": _evaluate_stock(
                    calibration, location, run["direction"], duration_s,
                    evaluation_episodes, dynamics_overrides,
                ),
            }
        )
        for seed in seeds:
            # Build both arms from a declared seed before either train loop
            # consumes randomness. Their architectures differ, but their
            # initialisation/replay RNG provenance is recorded identically.
            torch.manual_seed(seed)
            np.random.seed(seed)
            quantum_agent, classical_agent = build_matched_agents(spec, seed)
            for core, agent in (("quantum", quantum_agent), ("classical", classical_agent)):
                torch.manual_seed(seed)
                np.random.seed(seed)
                env = _env(calibration, location, run["direction"], duration_s, dynamics_overrides)
                training = train_recurrent_prioritized_ddqn(
                    agent, env, episodes, history_window=spec.history_window,
                    batch_size=run["batch_size"], warmup_steps=run["warmup_steps"],
                    update_every=run["update_every"], target_sync_every=run["target_sync_every"],
                    epsilon_start=run["epsilon_start"], epsilon_end=run["epsilon_end"],
                    reward_scale_mbps=run["reward_scale_mbps"],
                )
                evaluation = _evaluate(
                    agent, calibration, location, run["direction"], duration_s,
                    evaluation_episodes, spec.history_window, dynamics_overrides,
                )
                stock = stock_evaluations[-1]["evaluation"]
                evaluation["throughput_delta_vs_stock_pct"] = 100.0 * (
                    evaluation["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0
                )
                records.append(
                    {
                        "location": location,
                        "seed": seed,
                        "core": core,
                        "param_count": agent.param_count(),
                        "training": training,
                        "empirical_action_evidence": {
                            "samples": {str(index): int(value) for index, value in enumerate(agent.action_samples)},
                            "mean_scaled_throughput_reward": {
                                str(index): float(value) for index, value in enumerate(agent.action_mean_rewards)
                            },
                        },
                        "evaluation": evaluation,
                    }
                )
    plan["records"] = records
    plan["stock_evaluations"] = stock_evaluations
    plan["selection_assessment"] = _assess_records(records, selection_criteria)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
