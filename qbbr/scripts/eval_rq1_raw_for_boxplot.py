"""Re-runs the exact RQ1 headline comparison (qbbr vs. stock BBR-v3, +
cubic/vegas/hybla as bonus context) against the FINAL checkpoints
(outputs/checkpoints_final/{pacing_only,multihead}/classical), but --
unlike eval_rq1_parallel.py, whose report only ever persisted the collapsed
per-location MEDIAN -- this script keeps the full per-seed array (qbbr) and
per-trace-file array (real CCAs, via real_cca_per_run_medians) so a genuine
boxplot/CDF can be drawn afterward instead of only a bar-of-medians. No
training; same checkpoints, same action configs, same risk_mode as the
numbers already published in main.tex's RQ1a/RQ1b tables -- this only adds
persistence of the raw arrays eval_rq1_parallel.py computes internally and
then discards.
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
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
COMPARISON_CCAS = ["bbr", "cubic", "vegas", "hybla"]
N_SEEDS = 10
N_EPISODES = 10
EPISODE_S = 300.0
RISK_MODE = "closed_form"
DIRECTION = "downlink"

ARMS = {
    "pacing_only": {
        "checkpoint_root": PROJECT_ROOT / "outputs" / "checkpoints_final" / "pacing_only" / "classical",
        "action_config_path": PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml",
    },
    "extended": {
        "checkpoint_root": PROJECT_ROOT / "outputs" / "checkpoints_final" / "multihead" / "classical",
        "action_config_path": PACKAGE_ROOT / "configs" / "action_multihead.yaml",
    },
}
OUT_PATH = PROJECT_ROOT / "outputs" / "rq1_raw_for_boxplot.json"


class _StateTruncatingAgent:
    """These checkpoints (outputs/checkpoints_final/*) predate s7_reconfig_phase
    (the state vector has since grown from 6 to 7 features under FluidSimEnv);
    n_qubits=6 on construction plus truncating the env's state to its first 6
    entries here keeps this a like-for-like replay of what they actually saw
    during training, not a mismatched forward pass (same pattern already used
    in eval_rq2_extended_action_space.py and analyze_ecn_behavior_shift.py)."""

    def __init__(self, agent):
        self._agent = agent

    def act(self, state):
        return self._agent.act(state[:6])


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_a import simulated_agent_stats

    calibration = load_calibration(CALIBRATION_PATH)
    base_agent = MLPA2CAgent(n_layers=2, n_qubits=6, action_dims=job["action_dims"])
    base_agent.load(job["checkpoint"])
    agent = _StateTruncatingAgent(base_agent)
    stats = simulated_agent_stats(
        agent, job["location"], DIRECTION, calibration,
        n_episodes=N_EPISODES, episode_s=EPISODE_S, risk_mode=RISK_MODE, action_config=job["action_config"],
    )
    return {"location": job["location"], "arm": job["arm"], "seed": job["seed"], **stats}


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from qbbr.action.registry import dimension_sizes, is_multihead, load_action_space
    from qbbr.eval.metrics import real_cca_per_run_medians

    jobs = []
    action_configs = {}
    for arm, cfg in ARMS.items():
        action_config = load_action_space(cfg["action_config_path"])
        action_configs[arm] = action_config
        action_dims = tuple(dimension_sizes(action_config)) if is_multihead(action_config) else (
            len(action_config["levels"]),
        )
        for location in LOCATIONS:
            for seed in range(N_SEEDS):
                checkpoint = cfg["checkpoint_root"] / location / f"seed{seed}.pt"
                jobs.append({
                    "location": location, "arm": arm, "seed": seed,
                    "checkpoint": str(checkpoint), "action_config": action_config, "action_dims": action_dims,
                })

    print(f"{len(jobs)} qbbr eval job(s): {len(LOCATIONS)} location(s) x {len(ARMS)} arm(s) x {N_SEEDS} seed(s)")
    t0 = time.time()
    qbbr_raw: dict[tuple, dict[str, list[float]]] = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["arm"])
            qbbr_raw.setdefault(key, {"throughput_mbps": [], "rtt_ms": [], "retransmits_per_s": []})
            qbbr_raw[key]["throughput_mbps"].append(r["throughput_mbps_median"])
            qbbr_raw[key]["rtt_ms"].append(r["rtt_ms_median"])
            qbbr_raw[key]["retransmits_per_s"].append(r["retransmits_per_s_median"])
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} {job['arm']:12s} seed={job['seed']}  "
                  f"tput={r['throughput_mbps_median']:7.1f}Mbps  rtt={r['rtt_ms_median']:7.1f}ms  "
                  f"rtx/s={r['retransmits_per_s_median']:6.2f}")

    print("\nloading real per-run CCA baselines (bbr/cubic/vegas/hybla)...")
    real_raw: dict[tuple, dict[str, list[float]]] = {}
    for location in LOCATIONS:
        for cca in COMPARISON_CCAS:
            real_raw[(location, cca)] = real_cca_per_run_medians(DATASET_ROOT, location, DIRECTION, cca)

    out = {
        "meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES, "risk_mode": RISK_MODE, "direction": DIRECTION,
                 "total_elapsed_s": time.time() - t0},
        "qbbr": {f"{loc}|{arm}": vals for (loc, arm), vals in qbbr_raw.items()},
        "real_cca": {f"{loc}|{cca}": vals for (loc, cca), vals in real_raw.items()},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
