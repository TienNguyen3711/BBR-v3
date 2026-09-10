"""Primary successor runner: throughput-only NativeQA2C vs matched Classical A2C.

QDQN is deliberately absent here: it is maintained as an ablation, not the
primary successor controller.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.control.contracts import mdp_contract_from_mapping
from qbbr.control.native_actions import validate_fixed_action_set
from qbbr.control.native_action_selector import NativeActionSelector
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.eval.successor_protocol import assess_full_successor_records
from qbbr.train.native_loop import train_native_qrl


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "tier1_native_qa2c_successor_protocol.yaml"
STOCK_ACTION = 2


def _env(calibration, location: str, direction: str, config: dict) -> FluidSimEnv:
    return FluidSimEnv(
        location, direction, calibration, risk_mode=config["simulator"]["risk_mode"],
        episode_s=config["training"]["duration_s"], reward_mode="throughput_only",
        dynamics_overrides=config["simulator"].get("dynamics_overrides"),
        probe_bw_phase_gate=bool(config["simulator"].get("probe_bw_phase_gate", False)),
    )


def _selector(config: dict) -> NativeActionSelector:
    """Build the shared QA2C/A2C selector from the declared agent contract."""

    values = config["agent"].get("selection")
    if not values:
        raise ValueError("Primary selector-v2 protocol requires a declared native action selector.")
    return NativeActionSelector(
        stock_action=int(values["stock_action"]),
        min_logit_advantage=float(values["min_logit_advantage"]),
        confidence_temperature=float(values["confidence_temperature"]),
        max_inflight_state=float(values["max_inflight_state"]),
        max_queue_state=float(values["max_queue_state"]),
        max_excess_rtt_state=float(values.get("max_excess_rtt_state", 1.0)),
        max_reconfig_phase_proximity=float(values.get("max_reconfig_phase_proximity", 1.0)),
        high_gain_actions=tuple(int(item) for item in values["high_gain_actions"]),
        low_gain_actions=tuple(int(item) for item in values.get("low_gain_actions", ())),
        queue_budget_max_inflight=float(values.get("queue_budget_max_inflight", 1.0)),
        queue_budget_max_queue=float(values.get("queue_budget_max_queue", 1.0)),
    )


def _validate_primary_contract(config: dict, probe: FluidSimEnv) -> dict:
    """Reject drift between the canonical MDP declaration and simulator path."""

    control = config["control"]
    if control.get("role") != "primary_successor_quantum_brain":
        raise ValueError("Primary runner requires an explicitly labelled QA2C successor role.")
    contract_path = PACKAGE_ROOT / "configs" / control["canonical_state_contract"]
    contract = mdp_contract_from_mapping(yaml.safe_load(contract_path.read_text()))
    if contract.contract_id != control["action_set_contract"]:
        raise ValueError("Primary protocol and canonical MDP contract identify different action/state sets.")
    validate_fixed_action_set(contract.action_space, contract_path.parent / "action_pacing_gain.yaml")
    if len(contract.observation_space) != probe.observation_dim:
        raise ValueError(
            "Primary QA2C state mismatch: canonical contract and FluidSimEnv must both expose seven features."
        )
    if len(contract.action_space) != probe.action_space_size:
        raise ValueError("Primary QA2C action mismatch: canonical contract and FluidSimEnv must both expose five actions.")
    return {
        "contract_id": contract.contract_id,
        "observation_names": [item.name for item in contract.observation_space],
        "observation_dim": len(contract.observation_space),
        "action_ids": [item.action_id for item in contract.action_space],
        "action_count": len(contract.action_space),
    }


def _evaluate(agent, calibration, location: str, direction: str, config: dict, deployment: bool) -> dict:
    metrics = {"throughput_mbps": [], "retransmits_per_s": [], "rtt_ms": []}
    durations = []
    delivered = retransmitted = 0.0
    counts: dict[str, int] = {}
    near_high = near_total = far_high = far_total = 0  # high gain (3/4) by s7 handover proximity
    for seed in config["evaluation"]["holdout_seeds"]:
        env, done = _env(calibration, location, direction, config), False
        state = env.reset(seed=seed)
        while not done:
            action, _ = agent.act(
                state, env.allowed_action_indices(), deterministic=True, deployment=deployment,
            )
            counts[str(action)] = counts.get(str(action), 0) + 1
            near = float(np.asarray(state, dtype=float)[6]) >= 0.6
            is_high = action in (3, 4)
            near_total += int(near); near_high += int(near and is_high)
            far_total += int(not near); far_high += int((not near) and is_high)
            state, _reward, done, info = env.step(action)
            metrics["throughput_mbps"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            metrics["retransmits_per_s"].append(info["retransmits"] / info["t_dec_s"])
            metrics["rtt_ms"].append(info["rtt_ms"])
            durations.append(info["t_dec_s"])
            delivered += info["delivered_bytes"]
            retransmitted += info["retransmitted_bytes"]
    total = sum(counts.values())
    return ({f"{key}_mean": float(np.average(value, weights=durations)) for key, value in metrics.items()}
            | {"rtt_p90_ms": float(np.percentile(metrics["rtt_ms"], 90)),
               "rtt_p95_ms": float(np.percentile(metrics["rtt_ms"], 95)),
               "retransmission_ratio": retransmitted / max(delivered + retransmitted, 1.0)}
            | {"action_counts": counts, "action_shares": {str(i): counts.get(str(i), 0) / total for i in range(5)},
               "high_gain_share_near_handover": near_high / max(near_total, 1),
               "high_gain_share_far": far_high / max(far_total, 1),
               "policy_view": "deployed_hard_masked_logit_margin" if deployment else "learned_hard_masked_argmax"})


def _stock(calibration, location: str, direction: str, config: dict) -> dict:
    metrics = {"throughput_mbps": [], "retransmits_per_s": [], "rtt_ms": []}
    durations = []
    delivered = retransmitted = 0.0
    count = 0
    for seed in config["evaluation"]["holdout_seeds"]:
        env, done = _env(calibration, location, direction, config), False
        _state = env.reset(seed=seed)
        while not done:
            _state, _reward, done, info = env.step(STOCK_ACTION)
            count += 1
            metrics["throughput_mbps"].append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            metrics["retransmits_per_s"].append(info["retransmits"] / info["t_dec_s"])
            metrics["rtt_ms"].append(info["rtt_ms"])
            durations.append(info["t_dec_s"])
            delivered += info["delivered_bytes"]
            retransmitted += info["retransmitted_bytes"]
    return ({f"{key}_mean": float(np.average(value, weights=durations)) for key, value in metrics.items()}
            | {"rtt_p90_ms": float(np.percentile(metrics["rtt_ms"], 90)),
               "rtt_p95_ms": float(np.percentile(metrics["rtt_ms"], 95)),
               "retransmission_ratio": retransmitted / max(delivered + retransmitted, 1.0)}
            | {"action_counts": {str(STOCK_ACTION): count}, "action_shares": {str(i): float(i == STOCK_ACTION) for i in range(5)}})


def _checkpoint(
    path: Path, protocol_id: str, contract: dict, completed: int, rewards: list[float],
    episode_diagnostics: list[dict[str, float]], agent,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol_id": protocol_id, "contract": contract, "completed_episodes": completed,
        "episode_rewards": rewards, "episode_diagnostics": episode_diagnostics,
        "agent_state": agent.training_state_dict(),
        "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
    }, path)


def _restore(path: Path, protocol_id: str, contract: dict, agent) -> tuple[int, list[float], list[dict[str, float]]]:
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    if saved["protocol_id"] != protocol_id or saved["contract"] != contract:
        raise ValueError(f"Checkpoint {path} belongs to a different QA2C successor contract.")
    agent.load_training_state_dict(saved["agent_state"])
    torch.set_rng_state(saved["torch_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    return (
        int(saved["completed_episodes"]), list(saved["episode_rewards"]),
        list(saved.get("episode_diagnostics", [])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-simulator-proxy", action="store_true")
    parser.add_argument("--locations", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", choices=("downlink", "uplink"), default=None)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="override per-location constants JSON (default: data/calibrated/per_location_constants.json)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    protocol_stem = "full_native_qa2c" if config["protocol_id"].startswith("full-") else "tier1_native_qa2c"
    if config["protocol_id"].endswith("selector-v2"):
        protocol_stem += "_selector_v2"
    if args.checkpoint_root is None:
        args.checkpoint_root = PROJECT_ROOT / "outputs" / f"{protocol_stem}_checkpoints"
    if args.out is None:
        args.out = PROJECT_ROOT / "outputs" / f"{protocol_stem}_report.json"
    control, training, agent_config = config["control"], config["training"], config["agent"]
    if control["reward_contract"] != "supervisor-locked-throughput-only-v1" or control["fixed_action_count"] != 5:
        raise ValueError("Primary QA2C runner requires the frozen throughput-only, five-action contract.")
    if args.locations:
        training["locations"] = args.locations
    if args.directions:
        training["directions"] = args.directions
    if args.training_seeds:
        training["training_seeds"] = args.training_seeds
    if args.episodes is not None:
        training["episodes"] = args.episodes
    if args.duration_s is not None:
        training["duration_s"] = args.duration_s
    if args.holdout_seeds:
        config["evaluation"]["holdout_seeds"] = args.holdout_seeds
    calibration = load_calibration(args.calibration or PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
    probe = _env(calibration, training["locations"][0], training["directions"][0], config)
    canonical_mdp = _validate_primary_contract(config, probe)
    quantum, classical, match = build_matched_native_a2c_agents(
        probe.observation_dim, probe.action_space_size, n_layers=agent_config["n_layers"],
        lr=agent_config["learning_rate"], gamma=agent_config["discount_factor"],
        reupload=agent_config["reupload"], max_hidden=agent_config["max_classical_hidden"],
        selector=_selector(config), entropy_coef=float(agent_config.get("entropy_coef", 0.0)),
        stock_action=int(control["stock_action"]),
        stock_init_bias=float(agent_config.get("stock_init_bias", 0.0)),
    )
    contract = {"control": control, "agent": agent_config, "canonical_mdp": canonical_mdp, "simulator": config["simulator"], "parameter_match": match.__dict__}
    plan = {
        "protocol_id": config["protocol_id"], "mode": "simulator_proxy_training" if args.execute else "preflight_only",
        "algorithm_role": control["role"],
        "primary_quantum_agent": control["primary_quantum_agent"], "matched_classical_agent": control["matched_classical_agent"],
        "qdqn_role": config["notes"]["qdqn_role"], "locations": training["locations"],
        "directions": training["directions"], "training_seeds": training["training_seeds"],
        "holdout_seeds": config["evaluation"]["holdout_seeds"], "episodes": training["episodes"],
        "duration_s": training["duration_s"], "jobs": len(training["locations"]) * len(training["directions"]) * len(training["training_seeds"]) * 2,
        "contract": contract,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not args.allow_simulator_proxy:
        raise SystemExit("Pass --allow-simulator-proxy to run this non-field protocol.")
    records, stock_evaluations = [], []
    total_jobs = plan["jobs"]
    completed_jobs = 0
    started_at = time.time()
    partial_out = args.out.with_suffix(".partial.json")
    for location in training["locations"]:
        for direction in training["directions"]:
            stock = _stock(calibration, location, direction, config)
            stock_evaluations.append({"location": location, "direction": direction, "evaluation": stock})
            for seed in training["training_seeds"]:
                torch.manual_seed(seed); np.random.seed(seed)
                quantum, classical, match = build_matched_native_a2c_agents(
                    probe.observation_dim, probe.action_space_size, n_layers=agent_config["n_layers"],
                    lr=agent_config["learning_rate"], gamma=agent_config["discount_factor"],
                    reupload=agent_config["reupload"], max_hidden=agent_config["max_classical_hidden"],
                    selector=_selector(config), entropy_coef=float(agent_config.get("entropy_coef", 0.0)),
                    stock_action=int(control["stock_action"]),
                    stock_init_bias=float(agent_config.get("stock_init_bias", 0.0)),
                )
                for core, model in (("quantum", quantum), ("classical", classical)):
                    print(
                        f"[{completed_jobs + 1}/{total_jobs}] start {core} {location} {direction} seed={seed}",
                        flush=True,
                    )
                    path = args.checkpoint_root / core / location / direction / f"seed{seed}.pt"
                    completed, rewards, episode_diagnostics = (0, [], [])
                    if training["resume"] and path.exists():
                        completed, rewards, episode_diagnostics = _restore(path, config["protocol_id"], contract, model)
                    loop_config = {
                        k: agent_config[k] for k in ("entropy_start", "entropy_decay_episodes")
                        if k in agent_config
                    }
                    while completed < training["episodes"]:
                        chunk = min(training["checkpoint_every_episodes"], training["episodes"] - completed)
                        result = train_native_qrl(
                            model, _env(calibration, location, direction, config), chunk, loop_config, start_episode=completed,
                            total_episodes=training["episodes"], environment_seed_base=seed * 1_000_000,
                            reward_scale_mbps=training["reward_scale_mbps"],
                        )
                        rewards.extend(result["episode_rewards"])
                        episode_diagnostics.extend(result["episode_diagnostics"])
                        completed = int(result["completed_episode"])
                        _checkpoint(
                            path, config["protocol_id"], contract, completed, rewards,
                            episode_diagnostics, model,
                        )
                    evaluation = _evaluate(model, calibration, location, direction, config, deployment=True)
                    learned_evaluation = _evaluate(model, calibration, location, direction, config, deployment=False)
                    evaluation["throughput_delta_vs_stock_pct"] = 100.0 * (evaluation["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0)
                    evaluation["retransmits_delta_vs_stock_per_s"] = evaluation["retransmits_per_s_mean"] - stock["retransmits_per_s_mean"]
                    evaluation["rtt_p90_delta_vs_stock_ms"] = evaluation["rtt_p90_ms"] - stock["rtt_p90_ms"]
                    evaluation["rtt_p95_delta_vs_stock_ms"] = evaluation["rtt_p95_ms"] - stock["rtt_p95_ms"]
                    evaluation["retransmission_ratio_delta_vs_stock"] = evaluation["retransmission_ratio"] - stock["retransmission_ratio"]
                    records.append({"location": location, "direction": direction, "seed": seed, "core": core,
                                    "param_count": model.param_count(),
                                    "training": {"completed_episodes": completed, "episode_rewards": rewards,
                                                 "episode_diagnostics": episode_diagnostics, "checkpoint": str(path)},
                                    "evaluation": evaluation, "learned_policy_evaluation": learned_evaluation})
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
