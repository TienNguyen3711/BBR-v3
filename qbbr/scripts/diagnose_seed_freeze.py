"""Fork (1): why do training seeds 1 and 3 never leave stock?

Loads the v7b Tier-1 checkpoints and, per seed, reports:
  * raw actor logits (mean over an eval episode) -- has the actor moved off the
    stock warm-start at all, or is it still argmax=stock by the +bias?
  * greedy (argmax) non-stock fraction
  * STOCHASTIC eval throughput delta vs stock -- does the sampled policy win
    even where the deterministic/deployed one shows 0.00%?
Does not train. Simulator-proxy only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, yaml

from qbbr.env.calibration import load_calibration
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.run_native_qa2c_successor import _env, _selector, _restore, _stock, STOCK_ACTION

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent


def _agents(cfg, probe):
    a = cfg["agent"]
    return build_matched_native_a2c_agents(
        probe.observation_dim, probe.action_space_size, n_layers=a["n_layers"],
        lr=a["learning_rate"], gamma=a["discount_factor"], reupload=a["reupload"],
        max_hidden=a["max_classical_hidden"], selector=_selector(cfg),
        entropy_coef=float(a.get("entropy_coef", 0.0)),
        stock_action=int(cfg["control"]["stock_action"]),
        stock_init_bias=float(a.get("stock_init_bias", 0.0)),
    )


def _episode(agent, env, seeds, mode):
    """mode: 'greedy' (argmax), 'stoch' (sample)."""
    thr, logit_rows, greedy_nonstock, n = [], [], 0, 0
    for s in seeds:
        st, done = env.reset(seed=s), False
        while not done:
            allowed = env.allowed_action_indices()
            state_t = torch.as_tensor(np.asarray(st), dtype=torch.float32)
            with torch.no_grad():
                raw = agent._actor_logits(state_t).numpy()
            logit_rows.append(raw)
            g, _ = agent.act(st, allowed, deterministic=True, deployment=False)
            greedy_nonstock += int(g != STOCK_ACTION)
            n += 1
            a = g if mode == "greedy" else agent.act(st, allowed, deterministic=False)[0]
            st, _r, done, info = env.step(a)
            thr.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
    return float(np.mean(thr)), np.mean(logit_rows, axis=0), greedy_nonstock / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=PKG / "configs" / "tier1_native_qa2c_successor_protocol.yaml")
    ap.add_argument("--ckpt-root", type=Path, default=ROOT / "outputs" / "tier1_v7b_par")
    ap.add_argument("--location", default="Sydney")
    ap.add_argument("--direction", default="downlink")
    ap.add_argument("--core", default="quantum")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "diagnose_seed_freeze.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    calib = load_calibration(PKG / "data" / "calibrated" / "per_location_constants.json")
    probe = _env(calib, args.location, args.direction, cfg)
    stock = _stock(calib, args.location, args.direction, cfg)["throughput_mbps_mean"]

    contract = None
    rows = []
    print(f"{args.core} {args.location} {args.direction}  stock throughput = {stock:.2f} Mbps\n")
    print(f"{'seed':5}{'greedy_nonstock%':>18}{'d_greedy%':>11}{'d_stoch%':>11}   actor_logits (mean over eval)")
    for seed in args.seeds:
        q, c, match = _agents(cfg, probe)
        model = q if args.core == "quantum" else c
        if contract is None:
            contract = {"control": cfg["control"], "agent": cfg["agent"],
                        "canonical_mdp": None, "simulator": cfg["simulator"], "parameter_match": match.__dict__}
        # match the runner's contract dict shape enough for _restore's equality check to pass;
        # fall back to loading the raw state if the strict check trips.
        path = args.ckpt_root / f"ckpt_{args.location}_{args.direction}_s{seed}" / args.core / args.location / args.direction / f"seed{seed}.pt"
        if not path.exists():  # London used a/b split dirs
            for tag in ("a", "b"):
                p2 = args.ckpt_root / f"ckpt_{args.location}_{args.direction}_{tag}" / args.core / args.location / args.direction / f"seed{seed}.pt"
                if p2.exists():
                    path = p2; break
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_training_state_dict(saved["agent_state"])
        env = _env(calib, args.location, args.direction, cfg)
        d_g, logits, gns = _episode(model, env, args.eval_seeds, "greedy")
        env = _env(calib, args.location, args.direction, cfg)
        d_s, _, _ = _episode(model, env, args.eval_seeds, "stoch")
        dg = 100.0 * (d_g / stock - 1.0)
        ds = 100.0 * (d_s / stock - 1.0)
        print(f"{seed:<5}{gns*100:17.1f}%{dg:10.2f}%{ds:10.2f}%   [" +
              " ".join(f"{x:+.3f}" for x in logits) + "]")
        rows.append({"seed": seed, "greedy_nonstock_fraction": gns,
                     "delta_greedy_pct": dg, "delta_stochastic_pct": ds,
                     "mean_actor_logits": [float(x) for x in logits],
                     "completed_episodes": int(saved.get("completed_episodes", -1))})
    args.out.write_text(json.dumps({"evidence_tier": "simulator_proxy_only",
        "location": args.location, "direction": args.direction, "core": args.core,
        "stock_throughput_mbps": stock, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
