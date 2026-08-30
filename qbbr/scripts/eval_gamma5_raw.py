"""Evaluates the gamma=5.0 checkpoints (see train_gamma5_parallel.py) vs.
stock BBR-v3, same protocol as eval_rq1_raw_for_boxplot.py's pacing_only
arm (downlink, closed_form risk, 10 seeds/10 episodes, raw per-seed arrays
saved). The published (gamma=0.0) numbers are already in
outputs/rq1_raw_for_boxplot.json's "London|pacing_only" etc. keys -- this
script only re-derives the gamma=5.0 side, then the two are compared
directly (same real BBR-v3 baseline either way, so no need to reload it).

Output -> outputs/rq1_gamma5_raw.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_gamma5" / "pacing_only" / "classical"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
N_EPISODES = 10
EPISODE_S = 300.0
RISK_MODE = "closed_form"
DIRECTION = "downlink"
OUT_PATH = PROJECT_ROOT / "outputs" / "rq1_gamma5_raw.json"


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.action.registry import load_action_space
    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_a import simulated_agent_stats

    calibration = load_calibration(CALIBRATION_PATH)
    action_config = load_action_space(ACTION_CONFIG_PATH)
    agent = MLPA2CAgent(n_layers=2)
    agent.load(job["checkpoint"])
    stats = simulated_agent_stats(
        agent, job["location"], DIRECTION, calibration,
        n_episodes=N_EPISODES, episode_s=EPISODE_S, risk_mode=RISK_MODE, action_config=action_config,
    )
    return {"location": job["location"], "seed": job["seed"], **stats}


def main() -> None:
    import statistics
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from scipy import stats as scipy_stats

    from qbbr.eval.metrics import real_cca_per_run_medians

    jobs = [
        {"location": loc, "seed": seed, "checkpoint": str(CHECKPOINT_ROOT / loc / f"seed{seed}.pt")}
        for loc in LOCATIONS for seed in range(N_SEEDS)
    ]
    print(f"{len(jobs)} qbbr eval job(s), gamma=5.0, direction={DIRECTION}")

    t0 = time.time()
    qbbr_raw: dict[str, dict[str, list[float]]] = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = r["location"]
            qbbr_raw.setdefault(key, {"throughput_mbps": [], "rtt_ms": [], "retransmits_per_s": []})
            qbbr_raw[key]["throughput_mbps"].append(r["throughput_mbps_median"])
            qbbr_raw[key]["rtt_ms"].append(r["rtt_ms_median"])
            qbbr_raw[key]["retransmits_per_s"].append(r["retransmits_per_s_median"])
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} seed={job['seed']}  "
                  f"tput={r['throughput_mbps_median']:7.2f}Mbps  rtt={r['rtt_ms_median']:7.1f}ms  "
                  f"rtx/s={r['retransmits_per_s_median']:6.2f}")

    # Reuse the already-computed real BBR-v3 per-run arrays saved by
    # eval_rq1_raw_for_boxplot.py -- same dataset, same category/direction,
    # no need to recompute.
    published = json.load(open(PROJECT_ROOT / "outputs" / "rq1_raw_for_boxplot.json"))
    published_pacing_only = published["qbbr"]

    print(f"\n{'=' * 100}\ngamma=5.0 vs. gamma=0.0 (published) vs. stock BBR-v3\n{'=' * 100}")
    report = []
    for location in LOCATIONS:
        g5 = qbbr_raw[location]
        g0 = published_pacing_only[f"{location}|pacing_only"]
        bbr = real_cca_per_run_medians(PACKAGE_ROOT / "data" / "raw", location, DIRECTION, "bbr")

        g5_rtx_med = statistics.median(g5["retransmits_per_s"])
        g0_rtx_med = statistics.median(g0["retransmits_per_s"])
        bbr_rtx_med = statistics.median(bbr["retransmits_per_s"])
        g5_tput_med = statistics.median(g5["throughput_mbps"])
        bbr_tput_med = statistics.median(bbr["throughput_mbps"])

        rtx_reduction_g5 = 1.0 - g5_rtx_med / bbr_rtx_med if bbr_rtx_med > 0 else float("nan")
        tput_retention_g5 = g5_tput_med / bbr_tput_med if bbr_tput_med > 0 else float("nan")
        _u, p_g5_vs_bbr = scipy_stats.mannwhitneyu(g5["retransmits_per_s"], bbr["retransmits_per_s"], alternative="two-sided")
        _u, p_g5_vs_g0 = scipy_stats.mannwhitneyu(g5["retransmits_per_s"], g0["retransmits_per_s"], alternative="two-sided")

        row = {
            "location": location,
            "gamma5_retransmits_per_s_median": g5_rtx_med, "gamma0_retransmits_per_s_median": g0_rtx_med,
            "bbr_retransmits_per_s_median": bbr_rtx_med,
            "retransmit_reduction_gamma5_vs_bbr": rtx_reduction_g5,
            "throughput_retention_gamma5_vs_bbr": tput_retention_g5,
            "p_gamma5_vs_bbr": float(p_g5_vs_bbr), "p_gamma5_vs_gamma0": float(p_g5_vs_g0),
        }
        report.append(row)
        print(f"{location:10s}  gamma5_rtx={g5_rtx_med:6.2f}  gamma0_rtx={g0_rtx_med:6.2f}  bbr_rtx={bbr_rtx_med:6.2f}  "
              f"rtx_reduction(g5 vs bbr)={rtx_reduction_g5:+7.1%}  p(g5 vs bbr)={p_g5_vs_bbr:.4f}  "
              f"p(g5 vs g0)={p_g5_vs_g0:.4f}")

    out = {
        "meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES, "risk_mode": RISK_MODE, "direction": DIRECTION,
                 "gamma": 5.0, "total_elapsed_s": time.time() - t0},
        "comparison": report,
        "qbbr_gamma5": qbbr_raw,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
