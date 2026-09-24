"""SNR gate: does the learning signal distinguish actions at all?"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from qbbr.agents.base_agent import discounted_returns
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "tier1_v9_fixedcritic.yaml"
DEFAULT_CALIBRATION = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"

GATE_SEPARATION_SIGMA = 2.0
GATE_MIN_R2 = 0.05


def collect(env: FluidSimEnv, seeds: list[int], gamma: float, rng: np.random.RandomState):
    states, actions, rewards_by_episode = [], [], []
    for seed in seeds:
        state = env.reset(seed=seed)
        episode_rewards: list[float] = []
        done = False
        while not done:
            allowed = list(env.allowed_action_indices())
            action = int(allowed[rng.randint(len(allowed))])
            states.append(np.asarray(state, dtype=float))
            actions.append(action)
            state, reward, done, _info = env.step(action)
            episode_rewards.append(float(reward))
        rewards_by_episode.append(episode_rewards)
    returns = np.concatenate([
        np.asarray(discounted_returns(r, gamma, 0.0), dtype=float) for r in rewards_by_episode
    ])
    return np.asarray(states), np.asarray(actions), returns


def linear_r2(features: np.ndarray, target: np.ndarray) -> float:
    """R^2 of the best least-squares linear fit -- a ceiling for any linear head."""
    design = np.hstack([features, np.ones((features.shape[0], 1))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - float(np.sum(residual**2)) / total if total > 0 else 0.0


STATE_NAMES = ["s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue",
               "s5_handover_eta", "s6_p_tot", "s7_reconfig_phase"]


def conditional_argmax(states, actions, advantage, n_strata: int = 3):
    """Does the best action change with the state, or is it a constant?"""
    rows = []
    for index, name in enumerate(STATE_NAMES[: states.shape[1]]):
        column = states[:, index]
        edges = np.quantile(column, np.linspace(0, 1, n_strata + 1)[1:-1])
        stratum = np.digitize(column, edges)
        winners = []
        for level in range(n_strata):
            in_stratum = stratum == level
            if in_stratum.sum() < 20:
                winners.append(None)
                continue
            means = {
                action: advantage[in_stratum & (actions == action)].mean()
                for action in sorted(set(actions.tolist()))
                if (in_stratum & (actions == action)).sum() >= 5
            }
            winners.append(max(means, key=means.get) if means else None)
        rows.append((name, winners))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Training protocol whose simulator block defines the environment.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--location", default="London")
    parser.add_argument("--direction", default="downlink")
    parser.add_argument("--reward-mode", default="difference",
                        choices=["legacy_alpha_fair", "throughput_only", "difference"])
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--episode-s", type=float, default=None,
                        help="Defaults to the config's training duration_s.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Defaults to the config's discount_factor.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta", type=float, default=None,
                        help="Defaults to the config's reward_kwargs delta.")
    parser.add_argument("--beta", type=float, default=None,
                        help="Defaults to the config's reward_kwargs beta.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    simulator = config["simulator"]
    episode_s = args.episode_s if args.episode_s is not None else float(config["training"]["duration_s"])
    gamma = args.gamma if args.gamma is not None else float(config["agent"]["discount_factor"])

    declared = simulator.get("reward_kwargs") or {}
    if args.reward_mode == "difference":
        # Flags override the protocol; otherwise the probe measures exactly the
        # weighting the protocol would train on.
        reward_kwargs = {
            "delta": args.delta if args.delta is not None else float(declared.get("delta", 1.0)),
            "beta": args.beta if args.beta is not None else float(declared.get("beta", 0.5)),
        }
    else:
        reward_kwargs = declared or None
    env = FluidSimEnv(
        args.location, args.direction, load_calibration(args.calibration),
        risk_mode=simulator["risk_mode"], episode_s=episode_s,
        reward_mode=args.reward_mode, reward_kwargs=reward_kwargs,
        dynamics_overrides=simulator.get("dynamics_overrides"),
        probe_bw_phase_gate=bool(simulator.get("probe_bw_phase_gate", False)),
    )
    rng = np.random.RandomState(args.seed)
    seeds = [args.seed * 1000 + i for i in range(args.episodes)]
    states, actions, returns = collect(env, seeds, gamma, rng)

    r2_state = linear_r2(states, returns)
    design = np.hstack([states, np.ones((states.shape[0], 1))])
    coefficients, *_ = np.linalg.lstsq(design, returns, rcond=None)
    advantage = returns - design @ coefficients
    corr_adv_return = float(np.corrcoef(advantage, returns)[0, 1])

    print(f"reward mode      {args.reward_mode}")
    print(f"cell             {args.location} {args.direction}")
    print(f"samples          {len(returns)}  ({args.episodes} episodes, gamma {gamma})")
    print(f"return           mean {returns.mean():+.4f}  std {returns.std():.4f}")
    print()
    print(f"R2(return|state)         {r2_state:+.4f}   (gate: > {GATE_MIN_R2})")
    print(f"corr(advantage, return)  {corr_adv_return:+.4f}   (1.0 = inert critic)")
    print()
    print("mean advantage by action")
    stats = {}
    for action in sorted(set(actions.tolist())):
        sample = advantage[actions == action]
        stderr = float(sample.std(ddof=1) / np.sqrt(len(sample))) if len(sample) > 1 else float("inf")
        stats[action] = (float(sample.mean()), stderr, len(sample))
        print(f"  a{action}: {sample.mean():+.5f}  +/- {stderr:.5f} (se)   n={len(sample)}")

    print()
    print("pairwise separation (|difference| / se of difference)")
    best_sigma, best_pair = 0.0, None
    for a, b in itertools.combinations(sorted(stats), 2):
        mean_a, se_a, _ = stats[a]
        mean_b, se_b, _ = stats[b]
        se_difference = float(np.sqrt(se_a**2 + se_b**2))
        sigma = abs(mean_a - mean_b) / se_difference if se_difference > 0 else 0.0
        if sigma > best_sigma:
            best_sigma, best_pair = sigma, (a, b)
        print(f"  a{a} vs a{b}: {mean_a - mean_b:+.5f}  = {sigma:.2f} sigma")

    print()
    print("best action per state stratum (low / mid / high)")
    conditional = conditional_argmax(states, actions, advantage)
    varying = []
    for name, winners in conditional:
        labels = "  ".join("--" if w is None else f"a{w}" for w in winners)
        present = [w for w in winners if w is not None]
        changes = len(set(present)) > 1
        if changes:
            varying.append(name)
        print(f"  {name:<20} {labels}   {'<- varies' if changes else ''}")
    pooled_argmax = max(stats, key=lambda a: stats[a][0])
    print(f"  pooled best action: a{pooled_argmax}")
    if not varying:
        print("  WARNING: the same action wins everywhere -- the optimum is a CONSTANT gain,")
        print("           so a state-conditioned policy has nothing to learn here.")

    passed = best_sigma >= GATE_SEPARATION_SIGMA and r2_state > GATE_MIN_R2
    print()
    print(f"best separation  {best_sigma:.2f} sigma" + (f"  (a{best_pair[0]} vs a{best_pair[1]})" if best_pair else ""))
    print(f"GATE: {'PASS -- training can distinguish actions' if passed else 'FAIL -- do not train'}")
    if not passed:
        if r2_state <= GATE_MIN_R2:
            print(f"  reason: state explains only {r2_state:.1%} of return variance")
        if best_sigma < GATE_SEPARATION_SIGMA:
            print(f"  reason: no action pair separates by {GATE_SEPARATION_SIGMA} sigma "
                  f"-- the gradient cannot prefer any action")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "location": args.location, "direction": args.direction,
            "reward_mode": args.reward_mode, "episodes": args.episodes,
            # Provenance: without these a separation figure cannot be traced back
            # to the weighting and calibration that produced it.
            "config": str(args.config), "calibration": str(args.calibration),
            "reward_kwargs": reward_kwargs, "seed": args.seed, "episode_s": episode_s,
            "best_pair": list(best_pair) if best_pair else None,
            "samples": int(len(returns)), "gamma": gamma,
            "r2_state": r2_state, "corr_advantage_return": corr_adv_return,
            "action_stats": {str(k): {"mean": v[0], "stderr": v[1], "n": v[2]} for k, v in stats.items()},
            "best_separation_sigma": best_sigma, "passed": bool(passed),
            "pooled_best_action": int(pooled_argmax),
            "state_dependent_features": varying,
            "conditional_argmax": {name: [None if w is None else int(w) for w in winners] for name, winners in conditional},
        }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
