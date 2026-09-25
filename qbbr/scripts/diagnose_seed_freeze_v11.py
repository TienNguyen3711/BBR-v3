"""Why do some seeds never leave stock, on some cells but not others?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.run_native_qa2c_successor import _env, _selector
from qbbr.train.native_loop import train_native_qrl

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "tier1_v11_6city.yaml"
DEFAULT_CALIBRATION = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
STOCK_ACTION = 2


def _actor_parameters(agent):
    module = getattr(agent, "actor", None)
    if isinstance(module, torch.nn.Module):
        return list(module.parameters())
    # NativeQA2CAgent: quantum weights plus the classical actor head.
    params = []
    for name in ("actor_qnn", "actor_head"):
        item = getattr(agent, name, None)
        if isinstance(item, torch.nn.Module):
            params.extend(item.parameters())
        elif isinstance(item, torch.Tensor):
            params.append(item)
    return params


def _flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params]) if params else torch.zeros(1)


def _probe_states(env, seed: int, count: int = 64):
    """A FIXED batch of states, so the logit margin is comparable across episodes."""
    states, state = [], env.reset(seed=seed)
    for _ in range(count):
        allowed = env.allowed_action_indices()
        states.append(np.asarray(state, dtype=float))
        state, _r, done, _i = env.step(STOCK_ACTION if STOCK_ACTION in allowed else allowed[0])
        if done:
            state = env.reset(seed=seed)
    return states


def _stock_logit_margin(agent, probe_states) -> float:
    """logit(stock) - max logit(non-stock). Positive = stock still preferred."""
    margins = []
    with torch.no_grad():
        for state in probe_states:
            state_t = torch.as_tensor(state, dtype=torch.float32)
            if hasattr(agent, "_actor_logits"):
                logits = agent._actor_logits(state_t)          # NativeQA2CAgent
            else:
                logits = agent.actor(state_t)                  # NativeMLPA2CAgent
            logits = logits.reshape(-1)
            stock = float(logits[STOCK_ACTION])
            other = float(torch.max(torch.cat([logits[:STOCK_ACTION], logits[STOCK_ACTION + 1:]])))
            margins.append(stock - other)
    return float(np.mean(margins))


def run_case(config, calibration, location, direction, core, seed, episodes,
             stock_init_bias=None, actor_lr=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = _env(calibration, location, direction, config)
    selector = _selector(config)
    agent_cfg = config["agent"]
    quantum, classical, _match = build_matched_native_a2c_agents(
        observation_dim=env.observation_dim, action_count=env.action_space_size,
        n_layers=agent_cfg["n_layers"],
        lr=agent_cfg["learning_rate"] if actor_lr is None else float(actor_lr),
        gamma=agent_cfg["discount_factor"], reupload=agent_cfg["reupload"],
        max_hidden=agent_cfg["max_classical_hidden"], selector=selector,
        entropy_coef=agent_cfg.get("entropy_coef", 0.0), stock_action=STOCK_ACTION,
        stock_init_bias=(agent_cfg.get("stock_init_bias", 0.0)
                         if stock_init_bias is None else float(stock_init_bias)),
        normalize_returns=agent_cfg.get("normalize_returns", False),
        critic_lr=agent_cfg.get("critic_lr"),
    )
    agent = quantum if core == "quantum" else classical
    probe_states = _probe_states(_env(calibration, location, direction, config), seed)
    params = _actor_parameters(agent)
    previous = _flat(params).clone()
    rows = []

    def on_episode(episode: int, summary: dict) -> None:
        nonlocal previous
        current = _flat(_actor_parameters(agent))
        step = float(torch.norm(current - previous).item())
        previous = current.clone()
        rows.append({
            "episode": episode,
            "actor_step_l2": step,
            "stock_logit_margin": _stock_logit_margin(agent, probe_states),
            "greedy_nonstock_frac": float(summary.get("greedy_nonstock_fraction", float("nan"))),
            "sampled_nonstock_frac": float(summary.get("sampled_nonstock_fraction", float("nan"))),
            "stock_only_mask_frac": float(summary.get("stock_only_action_mask_fraction", float("nan"))),
            "policy_entropy": float(summary.get("policy_entropy", float("nan"))),
            "advantage_std": float(summary.get("advantage_std", float("nan"))),
            "mean_reward": float(summary.get("mean_reward", float("nan"))),
        })

    # Mirror the runner's loop_config exactly, so this diagnoses the run that
    # actually happened rather than a differently-configured one.
    loop_config = {k: agent_cfg[k] for k in ("entropy_start", "entropy_decay_episodes", "n_step_update")
                   if k in agent_cfg}
    train_native_qrl(
        agent, env, episodes, loop_config, on_episode=on_episode,
        total_episodes=episodes, environment_seed_base=seed * 1_000_000,
        reward_scale_mbps=config["training"]["reward_scale_mbps"],
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--cases", nargs="+",
                        default=["quantum:London:downlink:1", "quantum:Sydney:downlink:1",
                                 "quantum:London:downlink:0", "classical:London:downlink:4"],
                        help="core:location:direction:seed")
    parser.add_argument("--stock-init-bias", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    calibration = load_calibration(args.calibration)
    out = {}
    for case in args.cases:
        core, location, direction, seed = case.split(":")
        rows = run_case(config, calibration, location, direction, core, int(seed), args.episodes,
                        stock_init_bias=args.stock_init_bias, actor_lr=args.actor_lr)
        out[case] = rows
        first, last = rows[0], rows[-1]
        moved = sum(r["actor_step_l2"] for r in rows)
        crossed = any(r["stock_logit_margin"] < 0 for r in rows)
        print(f"\n=== {case} ===")
        print(f"  actor movement (sum L2 over {len(rows)} ep) : {moved:.4f}")
        print(f"  stock logit margin  first {first['stock_logit_margin']:+.4f} "
              f"-> last {last['stock_logit_margin']:+.4f}   crossed zero: {crossed}")
        print(f"  greedy non-stock fraction  first {first['greedy_nonstock_frac']:.3f} "
              f"-> last {last['greedy_nonstock_frac']:.3f}")
        print(f"  sampled non-stock fraction (mean) : "
              f"{np.mean([r['sampled_nonstock_frac'] for r in rows]):.3f}")
        print(f"  stock-only mask fraction (mean)   : "
              f"{np.mean([r['stock_only_mask_frac'] for r in rows]):.3f}")
        print(f"  advantage std (mean)              : "
              f"{np.mean([r['advantage_std'] for r in rows]):.4f}")
        print(f"  policy entropy  first {first['policy_entropy']:.3f} -> last {last['policy_entropy']:.3f}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
