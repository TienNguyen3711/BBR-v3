from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "action_sensitivity_diagnostic.yaml"
STOCK_ACTION = 2  # action_pacing_gain.yaml: 1.0


def _run_forced_action(
    calibration, location: str, direction: str, action: int, seed: int, duration_s: float,
    dynamics_overrides: dict[str, float] | None = None,
) -> dict:
    env = FluidSimEnv(
        location, direction, calibration, risk_mode="stub_constant", episode_s=duration_s,
        reward_mode="throughput_only", dynamics_overrides=dynamics_overrides,
    )
    _state, done = env.reset(seed=seed), False
    throughput, retransmits, rtts, queues, effective_gains = [], [], [], [], []
    opportunities = decisions = 0
    while not done:
        allowed = env.allowed_action_indices()
        selected = action if action in allowed else STOCK_ACTION
        if len(allowed) > 1:
            opportunities += 1
        _state, _reward, done, info = env.step(selected)
        decisions += 1
        throughput.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
        retransmits.append(info["retransmits"] / info["t_dec_s"])
        rtts.append(info["rtt_ms"])
        queues.append(max(info["i_dwn"], 0.0))
        effective_gains.append(info["pacing_gain"])
    return {
        "location": location,
        "direction": direction,
        "forced_action": action,
        "seed": seed,
        "throughput_mbps_mean": float(np.mean(throughput)),
        "retransmits_per_s_mean": float(np.mean(retransmits)),
        "rtt_ms_mean": float(np.mean(rtts)),
        "drawdown_indicator_mean": float(np.mean(queues)),
        "effective_pacing_gain_mean": float(np.mean(effective_gains)),
        "decisions": decisions,
        "action_opportunities": opportunities,
        "action_opportunity_fraction": opportunities / decisions if decisions else 0.0,
    }


def _summarize(records: list[dict], min_throughput_span_pct: float, min_improvement_pct: float) -> list[dict]:
    scenarios = sorted({(row["location"], row["direction"]) for row in records})
    rows = []
    for location, direction in scenarios:
        scenario = [row for row in records if (row["location"], row["direction"]) == (location, direction)]
        by_action = {
            action: [row for row in scenario if row["forced_action"] == action]
            for action in sorted({row["forced_action"] for row in scenario})
        }
        action_means = {
            action: {
                metric: float(np.mean([row[metric] for row in values]))
                for metric in ("throughput_mbps_mean", "retransmits_per_s_mean", "rtt_ms_mean", "drawdown_indicator_mean", "effective_pacing_gain_mean", "action_opportunity_fraction")
            }
            for action, values in by_action.items()
        }
        stock = action_means[STOCK_ACTION]["throughput_mbps_mean"]
        throughputs = [value["throughput_mbps_mean"] for value in action_means.values()]
        span_pct = 100.0 * (max(throughputs) - min(throughputs)) / stock if stock else 0.0
        best_action = max(action_means, key=lambda action: action_means[action]["throughput_mbps_mean"])
        best_delta_pct = 100.0 * (action_means[best_action]["throughput_mbps_mean"] / stock - 1.0) if stock else 0.0
        rows.append(
            {
                "location": location,
                "direction": direction,
                "stock_throughput_mbps": stock,
                "throughput_span_pct_across_fixed_actions": span_pct,
                "best_action": best_action,
                "best_throughput_delta_vs_stock_pct": best_delta_pct,
                "max_action_opportunity_fraction": max(value["action_opportunity_fraction"] for value in action_means.values()),
                "throughput_sensitive": span_pct >= min_throughput_span_pct,
                "throughput_improvable": best_delta_pct >= min_improvement_pct,
                "per_action": {str(action): value for action, value in action_means.items()},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--locations", nargs="+", default=None)
    parser.add_argument("--directions", nargs="+", choices=["uplink", "downlink"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "action_sensitivity_diagnostic.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run = config["run"]
    locations = args.locations or run["locations"]
    directions = args.directions or run["directions"]
    seeds = args.seeds or run["seeds"]
    duration_s = args.duration_s or run["duration_s"]
    actions = run["fixed_action_indices"]
    dynamics_overrides = config.get("dynamics_overrides")
    calibration = load_calibration(PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json")
    records = [
        _run_forced_action(
            calibration, location, direction, action, seed, duration_s, dynamics_overrides
        )
        for location in locations
        for direction in directions
        for action in actions
        for seed in seeds
    ]
    summary = _summarize(
        records,
        config["go_no_go"]["minimum_throughput_span_pct"],
        config["go_no_go"]["minimum_improvement_vs_stock_pct"],
    )
    sensitive = [row for row in summary if row["throughput_sensitive"]]
    improvable = [row for row in summary if row["throughput_improvable"]]
    result = {
        "evidence_tier": "simulator_diagnostic_only",
        "warning": "Forced-action sensitivity, not an RL or field-performance result.",
        "reward_contract": "supervisor-locked-throughput-only-v1",
        "action_config": "action_pacing_gain.yaml",
        "dynamics_overrides": dynamics_overrides or {},
        "dynamics_evidence": config.get("dynamics_evidence", "legacy instantaneous-capacity path"),
        "run": {"locations": locations, "directions": directions, "seeds": seeds, "duration_s": duration_s, "fixed_action_indices": actions},
        "go_no_go": {
            "minimum_throughput_span_pct": config["go_no_go"]["minimum_throughput_span_pct"],
            "minimum_improvement_vs_stock_pct": config["go_no_go"]["minimum_improvement_vs_stock_pct"],
            "sensitive_scenarios": len(sensitive),
            "improvable_scenarios": len(improvable),
            "total_scenarios": len(summary),
            "training_ready": bool(improvable),
            "reason": "at least one scenario has a fixed action that improves throughput over stock" if improvable else "no tested scenario has a fixed action that improves throughput over stock by the required margin",
        },
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["go_no_go"], indent=2))
    for row in summary:
        print(f"{row['location']:10s} {row['direction']:8s} span={row['throughput_span_pct_across_fixed_actions']:.4f}% best={row['best_action']} delta={row['best_throughput_delta_vs_stock_pct']:.4f}% opportunities={row['max_action_opportunity_fraction']:.3f} ready={row['throughput_improvable']}")


if __name__ == "__main__":
    main()
