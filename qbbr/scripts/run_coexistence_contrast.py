"""Does a policy trained on an idle path still help on a contended one?"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch

from qbbr.env.native_multi_flow_env import AGENT_FLOW, NativeMultiFlowEnv
from qbbr.eval.state_ablation import PolicyObservationEnv
from qbbr.eval.successor_protocol import bootstrap_median_ci
from qbbr.scripts.run_native_qa2c_successor import _selector
from qbbr.study import runner
from qbbr.study.protocol import atomic_json, load_protocol

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "outputs/rq_study/native-bbr-rq-screen-v2-sixcity"
CONFIG = ROOT / "qbbr/configs/rq_study_screen.yaml"
OVERLAY = ROOT / "qbbr/configs/native_coexistence.yaml"


def make_env(study, calibration, overlay, location, direction, duration):
    simulator = study["base"]["simulator"]
    overrides = dict(simulator["dynamics_overrides"], **overlay["dynamics_overrides"])
    env = NativeMultiFlowEnv(
        location, direction, calibration, risk_mode=simulator["risk_mode"],
        episode_s=duration, reward_mode=simulator["reward_mode"],
        reward_kwargs=simulator.get("reward_kwargs"), dynamics_overrides=overrides,
        probe_bw_phase_gate=simulator.get("probe_bw_phase_gate", False),
        competing_ccas=tuple(overlay["competing_ccas"]))
    return PolicyObservationEnv(env, _selector(study["base"]), "full")


def rollout(agent, env, seed):
    state = env.reset(seed=seed)
    rows, flows, done = [], defaultdict(float), False
    while not done:
        action = agent.act(state, env.allowed_action_indices(), deterministic=True)[0]
        state, _, done, info = env.step(action)
        interval = float(info["t_dec_s"])
        rows.append((interval, float(info["delivered_bytes"]), float(info["retransmits"]),
                     float(info["rtt_ms"])))
        for name, bps in info["flow_throughput_bps"].items():
            flows[name] += bps * interval
    duration = sum(r[0] for r in rows)
    weights = np.asarray([r[0] for r in rows])
    rtts = np.asarray([r[3] for r in rows])
    order = np.argsort(rtts)
    p90 = float(rtts[order][np.searchsorted(np.cumsum(weights[order]), .9 * duration)])
    total = sum(flows.values())
    return {
        "throughput_mbps": sum(r[1] for r in rows) * 8 / duration / 1e6,
        "retransmits_per_s": sum(r[2] for r in rows) / duration,
        "rtt_p90_ms": p90,
        "agent_share_pct": 100.0 * flows[AGENT_FLOW] / total if total > 0 else 0.0,
        "flow_mbps": {k: v / duration / 1e6 for k, v in flows.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "qbbr/data/calibrated/per_location_constants_v14.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/coexistence_contrast.json")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--cores", nargs="+", default=["qa2c", "a2c"])
    args = parser.parse_args()

    study = load_protocol(CONFIG)
    overlay = json.loads(json.dumps(__import__("yaml").safe_load(OVERLAY.read_text())))
    calibration = json.loads(args.calibration.read_text())
    direction = study["directions"][0]

    rows, stock_cache = [], {}
    for location in study["locations"]:
        for holdout in study["holdout_seeds"]:
            key = (location, holdout)
            if key not in stock_cache:
                env = make_env(study, calibration, overlay, location, direction, args.duration_s)
                stock_cache[key] = rollout(runner.Stock(), env, holdout)
            stock = stock_cache[key]
            for core in args.cores:
                for seed in study["training_seeds"]:
                    directory = SCREEN / f"synthetic__{location}__{direction}__{core}__full__{seed}"
                    checkpoint = directory / "checkpoint.pt"
                    if not checkpoint.exists():
                        raise SystemExit(f"Missing trained policy: {checkpoint}")
                    agent = runner.build_agent(study, core, seed)
                    agent.load_training_state_dict(
                        torch.load(checkpoint, map_location="cpu", weights_only=False)["agent"])
                    env = make_env(study, calibration, overlay, location, direction, args.duration_s)
                    policy = rollout(agent, env, holdout)
                    rows.append({
                        "location": location, "core": core, "seed": seed, "holdout_seed": holdout,
                        "stock": stock, "policy": policy,
                        "throughput_delta_pct": 100 * (policy["throughput_mbps"] / stock["throughput_mbps"] - 1),
                        "share_delta_pp": policy["agent_share_pct"] - stock["agent_share_pct"],
                        "rtt_p90_delta_ms": policy["rtt_p90_ms"] - stock["rtt_p90_ms"],
                    })
            print(f"{location} holdout {holdout}: stock share {stock['agent_share_pct']:.1f}%", flush=True)

    summary = {}
    for core in args.cores:
        cells = defaultdict(list)
        for row in rows:
            if row["core"] == core:
                cells[(row["location"], row["seed"])].append(row)
        deltas = {k: median(r["throughput_delta_pct"] for r in v) for k, v in cells.items()}
        shares = {k: median(r["share_delta_pp"] for r in v) for k, v in cells.items()}
        rtts = {k: median(r["rtt_p90_delta_ms"] for r in v) for k, v in cells.items()}
        summary[core] = {
            "cells": len(deltas),
            "throughput_delta_pct": {
                "median": round(median(deltas.values()), 4),
                "positive": f"{sum(1 for v in deltas.values() if v > 0)}/{len(deltas)}",
                "ci95": [round(x, 4) for x in
                         (bootstrap_median_ci(list(deltas.values()), seed=20260921)[k]
                          for k in ("lower", "upper"))]},
            "share_delta_pp_median": round(median(shares.values()), 4),
            "rtt_p90_delta_ms_median": round(median(rtts.values()), 4),
        }
    zero_rtt = sum(1 for row in rows if row["rtt_p90_delta_ms"] == 0.0)
    report = {
        "design": "zero-shot: policies trained single-flow, evaluated under contention",
        "claim_scope": "NOT REPORTABLE -- see validity_warning",
        "validity_warning": (
            f"RTT p90 is identical between arms in {zero_rtt} of {len(rows)} rows while "
            "throughput differs several-fold. Displacing a flow costs the displacer "
            "nothing in this model (the aggressive gain wins throughput AND delay), so "
            "these deltas measure the allocation rule, not the controller. See "
            "qbbr/configs/native_coexistence.yaml."),
        "competing_ccas": overlay["competing_ccas"],
        "stock_agent_share_pct_median": round(
            median(v["agent_share_pct"] for v in stock_cache.values()), 3),
        "summary": summary, "rows": rows,
    }
    atomic_json(args.out, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
