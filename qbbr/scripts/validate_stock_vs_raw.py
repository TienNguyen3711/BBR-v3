"""Fork-4 Stage-0 check: does the v7b stock fluid sim reproduce the
distribution of the *real* sequential BBR iperf3 runs in qbbr/data/raw/?

This validates the ENVIRONMENT at the stock operating point -- not the agent.
It does not train, does not touch checkpoints, and makes no Starlink claim.
The raw runs are stock-CCA-on-Starlink; a match means the proxy is credible
for stock, a mismatch says which constant to recalibrate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qbbr.data.catalog import build_catalog, iter_file_records
from qbbr.data.loader import load_trace
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv

ROOT = Path(__file__).resolve().parents[1]
STOCK_ACTION = 2


def _stats(thr_mbps, rtt_ms, retr_per_s) -> dict:
    def q(a, p):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, p)) if a.size else float("nan")
    return {
        "throughput_mbps_median": q(thr_mbps, 50), "throughput_mbps_p10": q(thr_mbps, 10),
        "throughput_mbps_p90": q(thr_mbps, 90), "throughput_mbps_mean": float(np.nanmean(thr_mbps)),
        "rtt_ms_median": q(rtt_ms, 50), "rtt_ms_p90": q(rtt_ms, 90), "rtt_ms_p95": q(rtt_ms, 95),
        "retransmits_per_s_median": q(retr_per_s, 50), "retransmits_per_s_mean": float(np.nanmean(retr_per_s)),
    }


def real_stats(dataset_root: Path, location: str, direction: str, ccas: tuple[str, ...]) -> dict | None:
    cat = build_catalog(dataset_root)
    cat = cat[(cat.category == "sequential") & (cat.location == location)
              & (cat.direction == direction) & (cat.cca.isin(ccas))]
    if cat.empty:
        return None
    thr, rtt, retr = [], [], []
    n_runs = 0
    for rec in iter_file_records(cat):
        iv = load_trace(rec).intervals
        iv = iv[(iv["omitted"] != True) & iv["bits_per_second"].notna()]  # noqa: E712
        if iv.empty:
            continue
        n_runs += 1
        thr.extend((iv["bits_per_second"] / 1e6).tolist())
        rtt.extend(iv["rtt_ms"].dropna().tolist())
        secs = iv["seconds"].replace(0, np.nan)
        retr.extend((iv["retransmits"] / secs).dropna().tolist())
    out = _stats(thr, rtt, retr)
    out["n_runs"] = n_runs
    out["ccas"] = list(ccas)
    return out


def sim_stock_stats(calibration, cfg, location: str, direction: str, seeds: list[int], duration_s: float) -> dict:
    ov = cfg["simulator"]["dynamics_overrides"]
    thr, rtt, retr = [], [], []
    for seed in seeds:
        env = FluidSimEnv(location, direction, calibration, episode_s=duration_s,
                          reward_mode="throughput_only", risk_mode=cfg["simulator"]["risk_mode"],
                          dynamics_overrides=ov, probe_bw_phase_gate=True)
        env.reset(seed=seed)
        done = False
        while not done:
            _s, _r, done, info = env.step(STOCK_ACTION)
            thr.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"] / 1e6)
            rtt.append(info["rtt_ms"])
            retr.append(info["retransmits"] / info["t_dec_s"])
    out = _stats(thr, rtt, retr)
    out["n_seeds"] = len(seeds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "tier1_native_qa2c_successor_protocol.yaml")
    ap.add_argument("--ccas", nargs="+", default=["bbr", "bbr2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=ROOT.parent / "outputs" / "validate_stock_vs_raw_v7b.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    calibration = load_calibration(ROOT / "data" / "calibrated" / "per_location_constants.json")
    locations = cfg["training"]["locations"]

    rows = []
    for location in locations:
        for direction in ("downlink", "uplink"):
            real = real_stats(args.dataset_root, location, direction, tuple(args.ccas))
            sim = sim_stock_stats(calibration, cfg, location, direction, args.seeds, args.duration_s)
            flags = []
            if real:
                rt_err = abs(sim["throughput_mbps_median"] - real["throughput_mbps_median"]) / max(real["throughput_mbps_median"], 1e-9)
                rtt_err = sim["rtt_ms_p90"] - real["rtt_ms_p90"]
                retr_ratio = (sim["retransmits_per_s_mean"] + 1e-9) / (real["retransmits_per_s_mean"] + 1e-9)
                if rt_err > 0.25:
                    flags.append(f"throughput median off {rt_err*100:.0f}%")
                if abs(rtt_err) > 30:
                    flags.append(f"RTT p90 off {rtt_err:+.0f} ms")
                if not (1/3 <= retr_ratio <= 3):
                    flags.append(f"retransmit rate {retr_ratio:.2f}x real")
            rows.append({"location": location, "direction": direction, "real": real, "sim_stock": sim, "flags": flags})
            r = real or {}
            print(f"\n{location} {direction}  (real n={r.get('n_runs','-')} runs {args.ccas}, sim n={sim['n_seeds']} seeds)")
            print(f"  throughput Mbps median   real {r.get('throughput_mbps_median',float('nan')):8.1f}   sim {sim['throughput_mbps_median']:8.1f}")
            print(f"  RTT ms p90               real {r.get('rtt_ms_p90',float('nan')):8.1f}   sim {sim['rtt_ms_p90']:8.1f}")
            print(f"  retransmits/s mean       real {r.get('retransmits_per_s_mean',float('nan')):8.2f}   sim {sim['retransmits_per_s_mean']:8.2f}")
            print(f"  flags: {flags or 'OK'}")

    report = {
        "evidence_tier": "simulator_proxy_only",
        "purpose": "v7b stock fluid sim vs real sequential BBR iperf3 runs; environment validation only",
        "protocol_id": cfg["protocol_id"], "dataset_root": str(args.dataset_root),
        "ccas": args.ccas, "sim_seeds": args.seeds, "duration_s": args.duration_s,
        "rows": rows,
        "n_flagged": sum(1 for x in rows if x["flags"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}   ({report['n_flagged']}/{len(rows)} cells flagged)")


if __name__ == "__main__":
    main()
