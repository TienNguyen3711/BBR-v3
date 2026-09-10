"""Task 2 -- train the native QRL policy ON real replayed capacity forcing.

`trace_replay.py` showed that a policy trained against the SYNTHETIC periodic
handover model loses almost all of its advantage on real capacity traces
(+3..6% -> +0.1..1.2%). This script tests the obvious follow-up: if the policy
is *trained* on real, irregular capacity forcing, does it find an advantage
that survives on held-out real traces?

Split: --train-runs (default 1..7) drive training episodes, --eval-runs
(default 8,9,10) are held out for evaluation. The replay CCA matches the
calibration provenance of per_location_constants_v7c.json (downlink <- bbr,
uplink <- bbr2).

Caveat kept in the report: the calibration constants themselves were fit on
all runs, so the ENVIRONMENT is mildly in-sample. That bias applies identically
to stock and to the agent, so it largely cancels in the agent-vs-stock delta.

Simulator-proxy only. No Starlink field claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.run_native_qa2c_successor import _selector, STOCK_ACTION
from qbbr.scripts.trace_replay import trace_forcing, align_phase_offset, _run
from qbbr.train.native_loop import train_native_qrl

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent


def build_pool(catalog, location, direction, cca, runs, capacity_proxy) -> list[dict]:
    sub = catalog[(catalog.category == "sequential") & (catalog.location == location)
                  & (catalog.direction == direction) & (catalog.cca == cca)
                  & (catalog.run.isin(runs))]
    pool = []
    for record in iter_file_records(sub):
        f = trace_forcing(load_trace(record).intervals, capacity_proxy)
        if f["duration_s"] < 30.0:
            continue
        pool.append({
            "run": int(record.run),
            "times_s": f["times_s"], "capacity_bytes_s": f["capacity_bytes_s"],
            "phase_offset_s": align_phase_offset(f["times_s"], f["capacity_bytes_s"]),
            "duration_s": f["duration_s"], "realised_mbps_mean": f["realised_mbps_mean"],
        })
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=PKG / "configs" / "tier1_native_qa2c_successor_protocol.yaml")
    ap.add_argument("--calibration", type=Path, default=PKG / "data" / "calibrated" / "per_location_constants_v7c.json")
    ap.add_argument("--dataset-root", type=Path, default=PKG / "data" / "raw")
    ap.add_argument("--locations", nargs="+", default=["London", "Sydney"])
    ap.add_argument("--directions", nargs="+", default=["downlink", "uplink"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--cores", nargs="+", default=["quantum", "classical"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--train-runs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--eval-runs", nargs="+", type=int, default=[8, 9, 10])
    ap.add_argument("--capacity-proxy", default="envq90_w5")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "train_on_traces_v7final.json")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--allow-simulator-proxy", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    calib = load_calibration(args.calibration)
    catalog = build_catalog(args.dataset_root)
    ov = cfg["simulator"]["dynamics_overrides"]
    gate = bool(cfg["simulator"].get("probe_bw_phase_gate", False))
    a = cfg["agent"]

    plan = {"locations": args.locations, "directions": args.directions, "seeds": args.seeds,
            "cores": args.cores, "episodes": args.episodes,
            "train_runs": args.train_runs, "eval_runs": args.eval_runs,
            "jobs": len(args.locations) * len(args.directions) * len(args.seeds) * len(args.cores)}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return
    if not args.allow_simulator_proxy:
        raise SystemExit("Pass --allow-simulator-proxy to run this non-field protocol.")

    rows = []
    for location in args.locations:
        for direction in args.directions:
            cca = "bbr" if direction == "downlink" else "bbr2"
            train_pool = build_pool(catalog, location, direction, cca, args.train_runs, args.capacity_proxy)
            eval_pool = build_pool(catalog, location, direction, cca, args.eval_runs, args.capacity_proxy)
            if not train_pool or not eval_pool:
                print(f"!! skipping {location} {direction}: pools {len(train_pool)}/{len(eval_pool)}")
                continue
            print(f"\n=== {location} {direction} (cca={cca}) train={len(train_pool)} traces "
                  f"eval={len(eval_pool)} traces ===", flush=True)

            def make_env(pool):
                return FluidSimEnv(
                    location, direction, calib, episode_s=300.0, reward_mode="throughput_only",
                    risk_mode=cfg["simulator"]["risk_mode"], dynamics_overrides=ov,
                    probe_bw_phase_gate=gate, capacity_trace_pool=pool)

            # stock baseline on each held-out trace
            stock_per_trace = []
            for entry in eval_pool:
                env = FluidSimEnv(
                    location, direction, calib, episode_s=entry["duration_s"],
                    reward_mode="throughput_only", risk_mode=cfg["simulator"]["risk_mode"],
                    dynamics_overrides=ov, probe_bw_phase_gate=gate,
                    capacity_trace=(entry["times_s"], entry["capacity_bytes_s"]),
                    phase_offset_s=entry["phase_offset_s"])
                s = _run(env, None, lambda st, al: STOCK_ACTION)
                s["run"] = entry["run"]; s["realised_mbps_mean"] = entry["realised_mbps_mean"]
                stock_per_trace.append(s)
            fid = float(np.median([s["throughput_mbps_mean"] / max(s["realised_mbps_mean"], 1e-9)
                                   for s in stock_per_trace]))
            print(f"  replay fidelity (stock vs real, median) = {fid:.2f}x", flush=True)

            probe_env = make_env(train_pool)
            for seed in args.seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                quantum, classical, _m = build_matched_native_a2c_agents(
                    probe_env.observation_dim, probe_env.action_space_size,
                    n_layers=a["n_layers"], lr=a["learning_rate"], gamma=a["discount_factor"],
                    reupload=a["reupload"], max_hidden=a["max_classical_hidden"],
                    selector=_selector(cfg), entropy_coef=float(a.get("entropy_coef", 0.0)),
                    stock_action=int(cfg["control"]["stock_action"]),
                    stock_init_bias=float(a.get("stock_init_bias", 0.0)))
                for core in args.cores:
                    model = quantum if core == "quantum" else classical
                    train_native_qrl(
                        model, make_env(train_pool), args.episodes, {}, start_episode=0,
                        total_episodes=args.episodes, environment_seed_base=seed * 1_000_000,
                        reward_scale_mbps=cfg["training"]["reward_scale_mbps"])
                    deltas = []
                    for entry, stock in zip(eval_pool, stock_per_trace):
                        env = FluidSimEnv(
                            location, direction, calib, episode_s=entry["duration_s"],
                            reward_mode="throughput_only", risk_mode=cfg["simulator"]["risk_mode"],
                            dynamics_overrides=ov, probe_bw_phase_gate=gate,
                            capacity_trace=(entry["times_s"], entry["capacity_bytes_s"]),
                            phase_offset_s=entry["phase_offset_s"])
                        ag = _run(env, model,
                                  lambda st, al: model.act(st, al, deterministic=True, deployment=True)[0])
                        deltas.append({
                            "run": entry["run"],
                            "throughput_delta_vs_stock_pct":
                                100.0 * (ag["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0),
                            "rtt_p90_delta_vs_stock_ms": ag["rtt_p90_ms"] - stock["rtt_p90_ms"],
                            "retransmits_delta_vs_stock_per_s":
                                ag["retransmits_per_s_mean"] - stock["retransmits_per_s_mean"],
                            "action_shares": ag["action_shares"],
                        })
                    med_thr = float(np.median([d["throughput_delta_vs_stock_pct"] for d in deltas]))
                    med_rtt = float(np.median([d["rtt_p90_delta_vs_stock_ms"] for d in deltas]))
                    print(f"    {core:9} seed{seed}: heldout dThru {med_thr:+6.2f}%  "
                          f"dRTTp90 {med_rtt:+6.2f} ms", flush=True)
                    rows.append({"location": location, "direction": direction, "core": core,
                                 "seed": seed, "cca": cca, "replay_fidelity_median": fid,
                                 "heldout_median_throughput_delta_pct": med_thr,
                                 "heldout_median_rtt_p90_delta_ms": med_rtt,
                                 "per_trace": deltas})

    report = {"evidence_tier": "simulator_proxy_only",
              "method": "policy TRAINED on replayed real capacity forcing; evaluated on held-out real traces",
              "caveat": "Calibration constants were fit on all runs, so the environment is mildly "
                        "in-sample; that bias applies equally to stock and agent and largely cancels "
                        "in the delta. Transport response is still the fluid proxy.",
              "protocol_id": cfg["protocol_id"], "plan": plan,
              "capacity_proxy": args.capacity_proxy, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
