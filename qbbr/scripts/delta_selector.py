"""Choose the difference reward's delta PER CELL, against the three gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from qbbr.agents.base_agent import discounted_returns
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.reward.difference import EPSILON, EPSILON_RTX

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "tier1_v10_difference.yaml"
DEFAULT_CALIBRATION = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"


def collect_components(env: FluidSimEnv, seeds, rng):
    """Per-decision reward COMPONENTS, which are independent of delta and beta."""
    actions, log_thr, log_rtt, rtx_delta = [], [], [], []
    rtt_agent, rtt_stock, episode_bounds = [], [], []
    for seed in seeds:
        env.reset(seed=seed)
        baseline = env._difference_baselines[seed]
        index, done, start = 0, False, len(actions)
        while not done:
            allowed = list(env.allowed_action_indices())
            action = int(allowed[rng.randint(len(allowed))])
            _state, _reward, done, info = env.step(action)
            i = min(index, len(baseline["bits_per_second"]) - 1)
            thr_a = info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6
            thr_s = baseline["bits_per_second"][i] / 1e6
            actions.append(action)
            log_thr.append(math.log((thr_a + EPSILON) / (thr_s + EPSILON)))
            log_rtt.append(math.log((info["rtt_ms"] + EPSILON) / (baseline["rtt_ms"][i] + EPSILON)))
            rtx_delta.append((info["retransmits"] - baseline["retransmits"][i])
                             / (baseline["retransmits"][i] + EPSILON_RTX))
            rtt_agent.append(info["rtt_ms"])
            rtt_stock.append(baseline["rtt_ms"][i])
            index += 1
        episode_bounds.append((start, len(actions)))
    return {k: np.asarray(v) for k, v in dict(
        actions=actions, log_thr=log_thr, log_rtt=log_rtt, rtx_delta=rtx_delta,
        rtt_agent=rtt_agent, rtt_stock=rtt_stock).items()} | {"episodes": episode_bounds}


def score_delta(data, states, delta: float, beta: float, gamma: float):
    rewards = data["log_thr"] - delta * data["log_rtt"] - beta * data["rtx_delta"]
    returns = np.concatenate([
        np.asarray(discounted_returns(rewards[a:b].tolist(), gamma, 0.0), dtype=float)
        for a, b in data["episodes"]
    ])
    design = np.hstack([states, np.ones((states.shape[0], 1))])
    coefficients, *_ = np.linalg.lstsq(design, returns, rcond=None)
    advantage = returns - design @ coefficients

    stats, best_sigma = {}, 0.0
    for action in sorted(set(data["actions"].tolist())):
        sample = advantage[data["actions"] == action]
        stats[action] = (float(sample.mean()),
                         float(sample.std(ddof=1) / np.sqrt(len(sample))))
    keys = sorted(stats)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            se = math.sqrt(stats[a][1] ** 2 + stats[b][1] ** 2)
            if se > 0:
                best_sigma = max(best_sigma, abs(stats[a][0] - stats[b][0]) / se)

    argmax = max(stats, key=lambda a: stats[a][0])
    chosen = data["actions"] == argmax
    return {
        "delta": delta,
        "argmax_action": int(argmax),
        "separation_sigma": best_sigma,
        # Physical consequences of the preferred action, measured not assumed.
        "throughput_delta_pct": float((math.exp(data["log_thr"][chosen].mean()) - 1.0) * 100.0),
        "rtt_p90_delta_ms": float(np.percentile(data["rtt_agent"][chosen], 90)
                                  - np.percentile(data["rtt_stock"][chosen], 90)),
        "rtx_delta_rel": float(data["rtx_delta"][chosen].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--location", default="London")
    parser.add_argument("--direction", default="downlink")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--deltas", nargs="+", type=float,
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    simulator = config["simulator"]
    criteria = config["selection_criteria"]
    gamma = float(config["agent"]["discount_factor"])
    env = FluidSimEnv(
        args.location, args.direction, load_calibration(args.calibration),
        risk_mode=simulator["risk_mode"], episode_s=float(config["training"]["duration_s"]),
        reward_mode="difference", reward_kwargs={"delta": 1.0, "beta": args.beta},
        dynamics_overrides=simulator.get("dynamics_overrides"),
        probe_bw_phase_gate=bool(simulator.get("probe_bw_phase_gate", False)),
    )
    rng = np.random.RandomState(args.seed)
    seeds = [args.seed * 1000 + i for i in range(args.episodes)]

    # States are collected once under a fixed reward; they do not depend on
    # delta, and neither do the components, so one rollout set serves the sweep.
    from qbbr.scripts.snr_probe import collect
    states, _actions, _returns = collect(env, seeds, gamma, np.random.RandomState(args.seed))
    data = collect_components(env, seeds, np.random.RandomState(args.seed))

    rtt_budget = float(criteria["max_rtt_p90_delta_vs_stock_ms"])
    rtx_budget = float(criteria["max_retransmits_delta_vs_stock_per_s"])
    print(f"{args.location} {args.direction}   gates: RTT p90 <= {rtt_budget} ms, throughput > 0, sep >= 2 sigma")
    print(f"{'delta':>6}{'argmax':>8}{'sigma':>8}{'thr %':>9}{'RTT p90 ms':>12}{'rtx rel':>9}  verdict")
    rows, admissible = [], []
    for delta in args.deltas:
        row = score_delta(data, states, delta, args.beta, gamma)
        ok = (row["separation_sigma"] >= 2.0 and row["throughput_delta_pct"] > 0.0
              and row["rtt_p90_delta_ms"] <= rtt_budget)
        row["admissible"] = bool(ok)
        rows.append(row)
        if ok:
            admissible.append(row)
        reason = ""
        if not ok:
            if row["separation_sigma"] < 2.0:
                reason = "no separation"
            elif row["throughput_delta_pct"] <= 0.0:
                reason = "loses throughput"
            else:
                reason = f"RTT +{row['rtt_p90_delta_ms']:.1f} ms over budget"
        print(f"{delta:>6.1f}{row['argmax_action']:>8}{row['separation_sigma']:>8.2f}"
              f"{row['throughput_delta_pct']:>9.2f}{row['rtt_p90_delta_ms']:>12.2f}"
              f"{row['rtx_delta_rel']:>9.3f}  {'OK' if ok else reason}")

    best = max(admissible, key=lambda r: r["throughput_delta_pct"]) if admissible else None
    print()
    if best:
        print(f"RECOMMENDED delta = {best['delta']}  "
              f"(action a{best['argmax_action']}, +{best['throughput_delta_pct']:.2f}% throughput, "
              f"RTT p90 {best['rtt_p90_delta_ms']:+.2f} ms)")
    else:
        print("NO admissible delta: every weighting either loses throughput, busts the RTT "
              "budget, or cannot separate actions. This cell needs a change other than delta.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "location": args.location, "direction": args.direction, "beta": args.beta,
            "rtt_budget_ms": rtt_budget, "rtx_budget_per_s": rtx_budget,
            "sweep": rows, "recommended": best,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
