"""RQ1-style eval (qbbr vs. stock BBR-v3 + cubic/vegas/hybla as bonus
context) for the newly-trained UPLINK checkpoints (outputs/checkpoints_uplink
/pacing_only/classical -- see train_uplink_parallel.py). Unlike the old
downlink checkpoints, these were trained under the CURRENT (post-merge,
7-feature) state -- no n_qubits=6 / state-truncation workaround needed;
verified directly (MLPA2CAgent default input dim == FluidSimEnv's current
state dim == 7) before this script was written.

Saves raw per-seed (qbbr) / per-trace-file (real CCAs) arrays, same as
eval_rq1_raw_for_boxplot.py, so a boxplot directly comparable to the
downlink one can be built from this file too.

Output -> outputs/rq1_uplink_raw.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_uplink" / "pacing_only" / "classical"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
COMPARISON_CCAS = ["bbr", "cubic", "vegas", "hybla"]
N_SEEDS = 10
N_EPISODES = 10
EPISODE_S = 300.0
RISK_MODE = "closed_form"
DIRECTION = "uplink"
OUT_PATH = PROJECT_ROOT / "outputs" / "rq1_uplink_raw.json"


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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from qbbr.eval.metrics import real_cca_per_run_medians

    jobs = [
        {"location": loc, "seed": seed, "checkpoint": str(CHECKPOINT_ROOT / loc / f"seed{seed}.pt")}
        for loc in LOCATIONS for seed in range(N_SEEDS)
    ]
    print(f"{len(jobs)} qbbr eval job(s), direction={DIRECTION}")

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

    print("\nloading real per-run CCA baselines (bbr/cubic/vegas/hybla), uplink...")
    real_raw: dict[str, dict[str, list[float]]] = {}
    for location in LOCATIONS:
        for cca in COMPARISON_CCAS:
            real_raw[f"{location}|{cca}"] = real_cca_per_run_medians(DATASET_ROOT, location, DIRECTION, cca)

    import statistics

    from scipy import stats as scipy_stats

    print(f"\n{'=' * 100}\nRQ1-uplink verdict (vs. stock BBR-v3, uplink)\n{'=' * 100}")
    report = []
    for location in LOCATIONS:
        qbbr = qbbr_raw[location]
        bbr = real_raw[f"{location}|bbr"]
        qbbr_tput_med = statistics.median(qbbr["throughput_mbps"])
        qbbr_rtx_med = statistics.median(qbbr["retransmits_per_s"])
        bbr_tput_med = statistics.median(bbr["throughput_mbps"])
        bbr_rtx_med = statistics.median(bbr["retransmits_per_s"])
        tput_retention = qbbr_tput_med / bbr_tput_med if bbr_tput_med > 0 else float("nan")
        rtx_reduction = 1.0 - qbbr_rtx_med / bbr_rtx_med if bbr_rtx_med > 0 else float("nan")
        _u, p_rtx = scipy_stats.mannwhitneyu(qbbr["retransmits_per_s"], bbr["retransmits_per_s"], alternative="two-sided")
        _u, p_tput = scipy_stats.mannwhitneyu(qbbr["throughput_mbps"], bbr["throughput_mbps"], alternative="two-sided")
        verdict = rtx_reduction >= 0.20 and tput_retention >= 0.95 and p_rtx < 0.05
        row = {
            "location": location, "qbbr_throughput_mbps_median": qbbr_tput_med, "bbr_throughput_mbps_median": bbr_tput_med,
            "qbbr_retransmits_per_s_median": qbbr_rtx_med, "bbr_retransmits_per_s_median": bbr_rtx_med,
            "throughput_retention": tput_retention, "retransmit_reduction": rtx_reduction,
            "p_retransmits": float(p_rtx), "p_throughput": float(p_tput), "rq1_pass": bool(verdict),
        }
        report.append(row)
        flag = "PASS" if verdict else "FAIL"
        print(f"{location:10s}  tput_retention={tput_retention:6.1%}  rtx_reduction={rtx_reduction:+7.1%}  "
              f"p_rtx={p_rtx:.4f}  p_tput={p_tput:.4f}  [{flag}]")

    out = {
        "meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES, "risk_mode": RISK_MODE, "direction": DIRECTION,
                 "total_elapsed_s": time.time() - t0},
        "rq1_uplink": report,
        "qbbr": qbbr_raw,
        "real_cca": real_raw,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
