"""Run the frozen full recurrent-QDQN ablation protocol, with checkpoint/resume.

This is not the primary QA2C/A2C successor path. The default is a read-only
preflight. Training needs both explicit flags so a long simulator workload
cannot be mistaken for a field BBR-v3 experiment.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.matched_qrl import MatchedQRLBenchmarkSpec, build_matched_agents
from qbbr.eval.successor_protocol import assess_full_successor_records
from qbbr.train.recurrent_qdqn_loop import train_recurrent_prioritized_ddqn


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "full_recurrent_qdqn_ablation_protocol.yaml"
STOCK_ACTION = 2


def _environment(calibration, location: str, direction: str, config: dict) -> FluidSimEnv:
    simulator = config["simulator"]
    training = config["training"]
    return FluidSimEnv(
        location, direction, calibration, risk_mode=simulator["risk_mode"],
        episode_s=training["duration_s"], reward_mode="throughput_only",
        dynamics_overrides=simulator.get("dynamics_overrides"),
    )


def _evaluate(agent, calibration, location: str, direction: str, config: dict) -> dict:
    metrics = {"throughput_mbps": [], "retransmits_per_s": [], "rtt_ms": []}
    counts: dict[str, int] = {}
    history_window = config["agent"]["history_window"]
    for seed in config["evaluation"]["holdout_seeds"]:
        env = _environment(calibration, location, direction, config)
        state, done = env.reset(seed=seed), False
        history = deque([np.asarray(state, dtype=np.float32)], maxlen=history_window)
        while not done:
            action = agent.act(np.stack(history), env.allowed_action_indices(), epsilon=0.0)
            counts[str(action)] = counts.get(str(action), 0) + 1
            state, _reward, done, info = env.step(action)
            history.append(np.asarray(state, dtype=np.float32))
            metrics["throughput_mbps"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            metrics["retransmits_per_s"].append(info["retransmits"] / info["t_dec_s"])
            metrics["rtt_ms"].append(info["rtt_ms"])
    total = sum(counts.values())
    return (
        {f"{name}_mean": float(np.mean(values)) for name, values in metrics.items()}
        | {"action_counts": counts, "action_shares": {str(i): counts.get(str(i), 0) / total for i in range(5)}}
    )


def _evaluate_stock(calibration, location: str, direction: str, config: dict) -> dict:
    metrics = {"throughput_mbps": [], "retransmits_per_s": [], "rtt_ms": []}
    count = 0
    for seed in config["evaluation"]["holdout_seeds"]:
        env = _environment(calibration, location, direction, config)
        _state, done = env.reset(seed=seed), False
        while not done:
            _state, _reward, done, info = env.step(STOCK_ACTION)
            count += 1
            metrics["throughput_mbps"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            metrics["retransmits_per_s"].append(info["retransmits"] / info["t_dec_s"])
            metrics["rtt_ms"].append(info["rtt_ms"])
    return (
        {f"{name}_mean": float(np.mean(values)) for name, values in metrics.items()}
        | {"action_counts": {str(STOCK_ACTION): count}, "action_shares": {str(i): float(i == STOCK_ACTION) for i in range(5)}}
    )


def _checkpoint_path(root: Path, core: str, location: str, direction: str, seed: int) -> Path:
    return root / core / location / direction / f"seed{seed}.pt"


def _save_checkpoint(path: Path, protocol_id: str, shared_contract: dict, completed: int, rewards: list[float], agent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_id": protocol_id,
            "shared_contract": shared_contract,
            "completed_episodes": completed,
            "episode_rewards": rewards,
            "agent_state": agent.training_state_dict(),
        },
        path,
    )


def _restore_checkpoint(path: Path, protocol_id: str, shared_contract: dict, agent) -> tuple[int, list[float]]:
    # Checkpoints are local artifacts written by _save_checkpoint and include
    # replay transitions, so PyTorch 2.6's weights-only default cannot restore
    # them. Never point this runner at an untrusted checkpoint path.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with PyTorch versions before this option
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["protocol_id"] != protocol_id or checkpoint["shared_contract"] != shared_contract:
        raise ValueError(f"Checkpoint {path} does not belong to this frozen QDQN ablation protocol.")
    agent.load_training_state_dict(checkpoint["agent_state"])
    return int(checkpoint["completed_episodes"]), list(checkpoint["episode_rewards"])


def _train_arm(
    agent, calibration, location: str, direction: str, seed: int, core: str,
    config: dict, spec: MatchedQRLBenchmarkSpec, checkpoint_root: Path,
) -> tuple[dict, dict]:
    training_config = config["training"]
    path = _checkpoint_path(checkpoint_root, core, location, direction, seed)
    target_episodes = training_config["episodes"]
    completed, rewards = 0, []
    if training_config["resume"] and path.exists():
        completed, rewards = _restore_checkpoint(path, config["protocol_id"], spec.as_dict(), agent)
    if completed > target_episodes:
        raise ValueError(f"Checkpoint {path} has more episodes than this protocol permits.")
    env = _environment(calibration, location, direction, config)
    chunk_losses = []
    while completed < target_episodes:
        chunk = min(training_config["checkpoint_every_episodes"], target_episodes - completed)
        result = train_recurrent_prioritized_ddqn(
            agent, env, chunk, history_window=spec.history_window,
            batch_size=training_config["batch_size"], warmup_steps=training_config["warmup_steps"],
            update_every=training_config["update_every"], target_sync_every=training_config["target_sync_every"],
            epsilon_start=training_config["epsilon_start"], epsilon_end=training_config["epsilon_end"],
            reward_scale_mbps=training_config["reward_scale_mbps"], start_episode=completed,
            total_episodes=target_episodes, environment_seed_base=seed * 1_000_000,
        )
        rewards.extend(result["episode_rewards"])
        completed = int(result["completed_episode"])
        if np.isfinite(result["mean_td_loss"]):
            chunk_losses.append(result["mean_td_loss"])
        _save_checkpoint(path, config["protocol_id"], spec.as_dict(), completed, rewards, agent)
    training = {
        "episode_rewards": rewards,
        "completed_episodes": completed,
        "mean_episode_reward": float(np.mean(rewards)),
        "mean_chunk_td_loss": float(np.mean(chunk_losses)) if chunk_losses else float("nan"),
        "checkpoint": str(path),
    }
    evidence = {
        "samples": {str(i): int(value) for i, value in enumerate(agent.action_samples)},
        "mean_scaled_throughput_reward": {str(i): float(value) for i, value in enumerate(agent.action_mean_rewards)},
    }
    return training, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--locations", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", choices=("downlink", "uplink"), default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "outputs" / "full_recurrent_qdqn_ablation_checkpoints")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "full_recurrent_qdqn_ablation_report.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-simulator-proxy", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("algorithm_role") != "ablation_only_not_primary_successor":
        raise ValueError("This runner is reserved for an explicitly labelled recurrent-QDQN ablation.")
    if config["control"]["reward_contract"] != "supervisor-locked-throughput-only-v1":
        raise ValueError("QDQN ablation runner refuses a non-throughput reward contract.")
    if config["control"]["action_set_contract"] != "native-rl-bbr-v3-canonical-7state-fixed-actions":
        raise ValueError("QDQN ablation runner refuses an expanded or non-canonical action contract.")
    training = config["training"]
    if args.locations:
        training["locations"] = args.locations
    if args.directions:
        training["directions"] = args.directions
    if args.training_seeds:
        training["training_seeds"] = args.training_seeds
    if args.episodes:
        training["episodes"] = args.episodes
    if args.duration_s:
        training["duration_s"] = args.duration_s
    if args.holdout_seeds:
        config["evaluation"]["holdout_seeds"] = args.holdout_seeds
    calibration = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
    probe = _environment(calibration, training["locations"][0], training["directions"][0], config)
    spec = MatchedQRLBenchmarkSpec(
        observation_dim=probe.observation_dim, action_count=probe.action_space_size, **config["agent"],
    )
    plan = {
        "protocol_id": config["protocol_id"], "mode": "simulator_proxy_training" if args.execute else "preflight_only",
        "algorithm_role": config["algorithm_role"],
        "evidence_tier": config["evidence_tier"], "evidence_statement": config["evidence_statement"],
        "locations": training["locations"], "directions": training["directions"],
        "training_seeds": training["training_seeds"], "holdout_seeds": config["evaluation"]["holdout_seeds"],
        "episodes": training["episodes"], "duration_s": training["duration_s"],
        "jobs": len(training["locations"]) * len(training["directions"]) * len(training["training_seeds"]) * 2,
        "shared_contract": spec.as_dict(), "selection_criteria": config["selection_criteria"],
        "bootstrap": config["bootstrap"], "simulator": config["simulator"],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not args.allow_simulator_proxy:
        raise SystemExit("Refusing to train: acknowledge simulator-only evidence with --allow-simulator-proxy.")
    records, stock_evaluations = [], []
    total_jobs = plan["jobs"]
    completed_jobs = 0
    started_at = time.time()
    partial_out = args.out.with_suffix(".partial.json")
    for location in training["locations"]:
        for direction in training["directions"]:
            stock = _evaluate_stock(calibration, location, direction, config)
            stock_evaluations.append({"location": location, "direction": direction, "evaluation": stock})
            for seed in training["training_seeds"]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                quantum, classical = build_matched_agents(spec, seed)
                for core, agent in (("quantum", quantum), ("classical", classical)):
                    print(
                        f"[{completed_jobs + 1}/{total_jobs}] start {core} {location} {direction} seed={seed}",
                        flush=True,
                    )
                    training_result, evidence = _train_arm(
                        agent, calibration, location, direction, seed, core, config, spec, args.checkpoint_root,
                    )
                    evaluation = _evaluate(agent, calibration, location, direction, config)
                    evaluation["throughput_delta_vs_stock_pct"] = 100.0 * (
                        evaluation["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0
                    )
                    evaluation["retransmits_delta_vs_stock_per_s"] = (
                        evaluation["retransmits_per_s_mean"] - stock["retransmits_per_s_mean"]
                    )
                    records.append({
                        "location": location, "direction": direction, "seed": seed, "core": core,
                        "param_count": agent.param_count(), "training": training_result,
                        "empirical_action_evidence": evidence, "evaluation": evaluation,
                    })
                    completed_jobs += 1
                    partial = dict(plan)
                    partial["records"] = records
                    partial["stock_evaluations"] = stock_evaluations
                    partial["progress"] = {
                        "completed_jobs": completed_jobs,
                        "total_jobs": total_jobs,
                        "elapsed_s": time.time() - started_at,
                        "resume_manifest": str(partial_out),
                    }
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    partial_out.write_text(json.dumps(partial, indent=2))
                    print(
                        f"[{completed_jobs}/{total_jobs}] complete {core} {location} {direction} seed={seed} "
                        f"throughput_delta={evaluation['throughput_delta_vs_stock_pct']:.3f}%",
                        flush=True,
                    )
    plan["records"] = records
    plan["stock_evaluations"] = stock_evaluations
    plan["selection_assessment"] = assess_full_successor_records(records, config["selection_criteria"], config["bootstrap"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
