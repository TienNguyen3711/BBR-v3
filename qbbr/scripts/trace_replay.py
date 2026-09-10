"""Stage 2.5 -- trace-driven replay: run stock and the trained QA2C policy
through capacity forcing taken from REAL measured Starlink BBR runs.

The exogenous disturbance (bottleneck capacity over time, including its real
handover dips) comes from `qbbr/data/raw/`; the queue / loss / ProbeBW
response is still this repository's fluid proxy. That upgrades "synthetic
conditions" to "real conditions"; it does NOT upgrade "modelled transport" to
"real transport" -- only a kernel-in-the-loop testbed closes that gap.

Reports, per (location, direction):
  * fidelity  -- stock sim on the replayed forcing vs the trace's own realised
                 throughput. If these disagree the replay is not trustworthy
                 and the agent delta below means nothing.
  * agent     -- trained QA2C policy vs stock, on the SAME replayed forcing.
                 The policy was trained only on synthetic forcing, so every
                 replay trace is out-of-sample for it.

Simulator-proxy only. No Starlink field claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.run_native_qa2c_successor import _selector, STOCK_ACTION

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent
_RECONFIG_MEAN_PHASE_S = 10.5   # matches fluid_env._HANDOVER_PHASE_PROFILE
_RECONFIG_CYCLE_S = 15.0


def trace_forcing(intervals, capacity_proxy: str = "envq90_w5") -> dict:
    """(times, capacity bytes/s) plus the trace's own realised statistics.

    capacity_proxy: how available capacity is recovered from BBR's *achieved*
    rate. "achieved" takes the delivered rate as-is, which under-states the
    path (BBR under-fills during its own bandwidth-estimate recovery, so the
    replayed stock run then under-delivers ~25%). "envq90_wN" takes a rolling
    N-second centred 90th percentile -- what the path sustained in BBR's best
    moments of that window. The window is picked so the replayed STOCK run
    reproduces the trace's realised throughput (fidelity ~1.0); the agent
    delta is then measured against that validated baseline.
    """
    iv = intervals[(intervals["omitted"] != True) & intervals["bits_per_second"].notna()]  # noqa: E712
    t = iv["t_start"].to_numpy(dtype=float)
    bps = iv["bits_per_second"].to_numpy(dtype=float)
    order = np.argsort(t)
    t, bps = t[order], bps[order]
    series = pd.Series(bps / 8.0)
    if capacity_proxy == "achieved":
        capacity = series.to_numpy()
    elif capacity_proxy.startswith("envq90_w"):
        window = int(capacity_proxy.split("_w")[1])
        capacity = series.rolling(window, center=True, min_periods=1).quantile(0.9).to_numpy()
    elif capacity_proxy.startswith("envmax_w"):
        window = int(capacity_proxy.split("_w")[1])
        capacity = series.rolling(window, center=True, min_periods=1).max().to_numpy()
    else:
        raise ValueError(f"unknown capacity_proxy {capacity_proxy!r}")
    return {
        "times_s": t,
        "capacity_bytes_s": capacity,
        "capacity_proxy": capacity_proxy,
        "realised_mbps_mean": float(np.mean(bps) / 1e6),
        "realised_mbps_median": float(np.median(bps) / 1e6),
        "realised_rtt_p90_ms": float(np.nanpercentile(iv["rtt_ms"].to_numpy(dtype=float), 90)),
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
    }


def align_phase_offset(times: np.ndarray, capacity: np.ndarray) -> float:
    """Pin the simulator's reconfiguration clock to the trace's own dips.

    Dips = samples in the lowest decile of capacity. Their mean position on
    the 15 s cycle is mapped onto the profile's mean_phase_s, so the modelled
    freeze window and retransmit phase-lock line up with the real handovers.
    """
    if times.size < 4:
        return 0.0
    threshold = np.quantile(capacity, 0.10)
    dips = times[capacity <= threshold]
    if dips.size == 0:
        return 0.0
    angles = 2 * np.pi * (dips % _RECONFIG_CYCLE_S) / _RECONFIG_CYCLE_S
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    dip_phase = (mean_angle % (2 * np.pi)) / (2 * np.pi) * _RECONFIG_CYCLE_S
    return float((_RECONFIG_MEAN_PHASE_S - dip_phase) % _RECONFIG_CYCLE_S)


def _run(env, agent, action_fn) -> dict:
    state, done = env.reset(seed=0), False
    thr, rtt, retx, dur = [], [], [], []
    counts = {}
    while not done:
        allowed = env.allowed_action_indices()
        action = action_fn(state, allowed)
        counts[str(action)] = counts.get(str(action), 0) + 1
        state, _r, done, info = env.step(action)
        thr.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
        rtt.append(info["rtt_ms"])
        retx.append(info["retransmits"] / info["t_dec_s"])
        dur.append(info["t_dec_s"])
    return {
        "throughput_mbps_mean": float(np.average(thr, weights=dur)),
        "rtt_p90_ms": float(np.percentile(rtt, 90)),
        "rtt_median_ms": float(np.median(rtt)),
        "rtt_std_ms": float(np.std(rtt)),
        "retransmits_per_s_mean": float(np.average(retx, weights=dur)),
        "action_shares": {str(i): counts.get(str(i), 0) / max(sum(counts.values()), 1) for i in range(5)},
    }


def _load_agent(cfg, probe, ckpt_root: Path, location: str, direction: str, seed: int, core: str):
    a = cfg["agent"]
    quantum, classical, _m = build_matched_native_a2c_agents(
        probe.observation_dim, probe.action_space_size, n_layers=a["n_layers"],
        lr=a["learning_rate"], gamma=a["discount_factor"], reupload=a["reupload"],
        max_hidden=a["max_classical_hidden"], selector=_selector(cfg),
        entropy_coef=float(a.get("entropy_coef", 0.0)),
        stock_action=int(cfg["control"]["stock_action"]),
        stock_init_bias=float(a.get("stock_init_bias", 0.0)),
    )
    model = quantum if core == "quantum" else classical
    candidates = [ckpt_root / f"ckpt_{location}_{direction}_s{seed}",
                  ckpt_root / f"ckpt_{location}_{direction}"]
    candidates += [ckpt_root / f"ckpt_{location}_{direction}_{tag}" for tag in ("a", "b")]
    for base in candidates:
        path = base / core / location / direction / f"seed{seed}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model.load_training_state_dict(saved["agent_state"])
            return model
    raise FileNotFoundError(f"no {core} checkpoint for {location}/{direction}/seed{seed} under {ckpt_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=PKG / "configs" / "tier1_native_qa2c_successor_protocol.yaml")
    ap.add_argument("--calibration", type=Path, default=PKG / "data" / "calibrated" / "per_location_constants_v7c.json")
    ap.add_argument("--ckpt-root", type=Path, default=ROOT / "outputs" / "tier1_v7final_par")
    ap.add_argument("--dataset-root", type=Path, default=PKG / "data" / "raw")
    ap.add_argument("--locations", nargs="+", default=["London", "Sydney"])
    ap.add_argument("--directions", nargs="+", default=["downlink", "uplink"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--core", default="quantum")
    ap.add_argument("--replay-runs", nargs="+", type=int, default=[8, 9, 10],
                    help="run numbers used as the replay set")
    ap.add_argument("--cca", default="auto",
                    help="replay CCA; 'auto' matches the calibration provenance of "
                         "per_location_constants_v7c.json (downlink<-bbr, uplink<-bbr2)")
    ap.add_argument("--capacity-proxy", default="envq90_w5",
                    help="achieved | envq90_wN | envmax_wN (default envq90_w5: stock-sim fidelity ~1.0)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "trace_replay_v7final.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    calib = load_calibration(args.calibration)
    catalog = build_catalog(args.dataset_root)
    ov = cfg["simulator"]["dynamics_overrides"]
    gate = bool(cfg["simulator"].get("probe_bw_phase_gate", False))

    rows = []
    for location in args.locations:
        for direction in args.directions:
            # Replay the CCA the environment was calibrated from, or the replay
            # is internally inconsistent (v7c: downlink<-bbr, uplink<-bbr2).
            cca = ("bbr" if direction == "downlink" else "bbr2") if args.cca == "auto" else args.cca
            sub = catalog[(catalog.category == "sequential") & (catalog.location == location)
                          & (catalog.direction == direction) & (catalog.cca == cca)
                          & (catalog.run.isin(args.replay_runs))]
            if sub.empty:
                print(f"!! no replay traces for {location} {direction}")
                continue
            print(f"\n=== {location} {direction}  ({len(sub)} replay traces, cca={cca}, {args.core}) ===")
            for record in iter_file_records(sub):
                f = trace_forcing(load_trace(record).intervals, args.capacity_proxy)
                if f["duration_s"] < 30.0:
                    continue
                offset = align_phase_offset(f["times_s"], f["capacity_bytes_s"])

                def make_env():
                    return FluidSimEnv(
                        location, direction, calib, episode_s=f["duration_s"],
                        reward_mode="throughput_only", risk_mode=cfg["simulator"]["risk_mode"],
                        dynamics_overrides=ov, probe_bw_phase_gate=gate,
                        capacity_trace=(f["times_s"], f["capacity_bytes_s"]),
                        phase_offset_s=offset,
                    )

                probe_env = make_env()
                stock = _run(make_env(), None, lambda s, allowed: STOCK_ACTION)
                fidelity = stock["throughput_mbps_mean"] / max(f["realised_mbps_mean"], 1e-9)
                print(f"  run{record.run:<3} real {f['realised_mbps_mean']:7.1f} Mbps | "
                      f"stock-sim {stock['throughput_mbps_mean']:7.1f} Mbps  (fidelity {fidelity:.2f}x)")
                for seed in args.seeds:
                    model = _load_agent(cfg, probe_env, args.ckpt_root, location, direction, seed, args.core)
                    agent = _run(make_env(), model,
                                 lambda s, allowed: model.act(s, allowed, deterministic=True, deployment=True)[0])
                    d_thr = 100.0 * (agent["throughput_mbps_mean"] / stock["throughput_mbps_mean"] - 1.0)
                    d_rtt = agent["rtt_p90_ms"] - stock["rtt_p90_ms"]
                    d_retx = agent["retransmits_per_s_mean"] - stock["retransmits_per_s_mean"]
                    print(f"      seed{seed}: dThru {d_thr:+6.2f}%  dRTTp90 {d_rtt:+6.2f} ms  dRetx {d_retx:+7.4f}/s")
                    rows.append({
                        "location": location, "direction": direction, "core": args.core, "cca": cca,
                        "run": int(record.run), "seed": seed,
                        "trace_realised_mbps_mean": f["realised_mbps_mean"],
                        "trace_realised_rtt_p90_ms": f["realised_rtt_p90_ms"],
                        "stock_sim": stock, "agent_sim": agent,
                        "replay_fidelity_stock_over_real": fidelity,
                        "throughput_delta_vs_stock_pct": d_thr,
                        "rtt_p90_delta_vs_stock_ms": d_rtt,
                        "retransmits_delta_vs_stock_per_s": d_retx,
                        "phase_offset_s": offset,
                    })

    summary = {}
    for key in {(r["location"], r["direction"]) for r in rows}:
        grp = [r for r in rows if (r["location"], r["direction"]) == key]
        fid = [r["replay_fidelity_stock_over_real"] for r in grp]
        summary[f"{key[0]}_{key[1]}"] = {
            "replay_fidelity_median": float(np.median(fid)),
            "per_seed_median_throughput_delta_pct": {
                str(s): float(np.median([r["throughput_delta_vs_stock_pct"] for r in grp if r["seed"] == s]))
                for s in args.seeds
            },
            "per_seed_median_rtt_p90_delta_ms": {
                str(s): float(np.median([r["rtt_p90_delta_vs_stock_ms"] for r in grp if r["seed"] == s]))
                for s in args.seeds
            },
        }

    report = {
        "evidence_tier": "simulator_proxy_only",
        "method": "trace-driven replay: real measured capacity forcing, modelled transport response",
        "caveat": "The policy never saw a trace during training (synthetic forcing only), so replay is "
                  "out-of-sample for the policy. The calibration constants were fit on all bbr sequential "
                  "runs including these, so the ENVIRONMENT is in-sample; a fully clean split would refit "
                  "calibration on the complement of --replay-runs.",
        "protocol_id": cfg["protocol_id"], "calibration": str(args.calibration),
        "capacity_proxy": args.capacity_proxy, "cca": args.cca,
        "checkpoint_root": str(args.ckpt_root), "replay_runs": args.replay_runs,
        "core": args.core, "summary": summary, "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    for cell, s in sorted(summary.items()):
        print(f"\n{cell}  replay fidelity {s['replay_fidelity_median']:.2f}x")
        print("   median dThru%% by seed: " + str({k: round(v, 2) for k, v in s["per_seed_median_throughput_delta_pct"].items()}))
        print("   median dRTTp90 ms by seed: " + str({k: round(v, 2) for k, v in s["per_seed_median_rtt_p90_delta_ms"].items()}))


if __name__ == "__main__":
    main()
