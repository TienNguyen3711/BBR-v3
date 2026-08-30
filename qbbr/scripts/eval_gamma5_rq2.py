"""Side-effect check (Step 1 continued): does activating the reward's
ECN-level term (gamma=5.0 -- see train_gamma5_parallel.py, which found a
large retransmit-rate improvement over the published gamma=0.0 reward)
come at a coexistence-fairness cost under RQ2's Scenario B (competing
cubic/vegas/hybla flows)? Same alpha sweep and stock-agent baseline as
eval_rq2_parallel.py, just pointed at the gamma=5.0 checkpoints, so the
result is directly comparable to outputs/rq2_report.json's published
(gamma=0.0) numbers at the same alpha values.

Output -> outputs/rq2_gamma5_report.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_gamma5" / "pacing_only" / "classical"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
N_EPISODES = 10
EPISODE_S = 300.0
DIRECTION = "downlink"
RISK_MODE = "closed_form"
ALPHA_LABELS = ["0.0", "0.25", "0.5", "0.75", "1.0", "inf"]
ALPHA_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0, float("inf"))
OUT_PATH = PROJECT_ROOT / "outputs" / "rq2_gamma5_report.json"


class _StockAgent:
    def act(self, state: Any) -> tuple[int, float]:
        return 2, 0.0


class _StateTruncatingAgent:
    """This checkpoint was trained on FluidSimEnv (single-flow, 7-feature
    state: s1..s7). MultiFlowFluidEnv (Scenario B) now additionally emits
    s8_fairness_ratio (8 features) -- added after this checkpoint's
    training setup was fixed, so state[:7] keeps this a like-for-like
    replay of what the checkpoint actually saw, not a mismatched forward
    pass (same pattern as eval_rq2_extended_action_space.py's wrapper,
    truncating to 7 instead of 6 since this checkpoint generation is
    post-s7_reconfig_phase)."""

    def __init__(self, agent):
        self._agent = agent

    def act(self, state):
        return self._agent.act(state[:7])


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_b import run_scenario_b

    calibration = load_calibration(CALIBRATION_PATH)
    if job["arm"] == "qbbr":
        base_agent = MLPA2CAgent(n_layers=2, action_dims=(5,))
        base_agent.load(job["checkpoint"])
        agent = _StateTruncatingAgent(base_agent)
    else:
        agent = _StockAgent()

    result = run_scenario_b(
        agent, job["location"], DIRECTION, calibration,
        n_episodes=N_EPISODES, episode_s=EPISODE_S, risk_mode=RISK_MODE, alpha_sweep=ALPHA_SWEEP,
    )
    return {"location": job["location"], "arm": job["arm"], "seed": job["seed"], **result}


def build_jobs() -> list[dict[str, Any]]:
    jobs = []
    for location in LOCATIONS:
        for arm in ("qbbr", "stock"):
            for seed in range(N_SEEDS):
                checkpoint = CHECKPOINT_ROOT / location / f"seed{seed}.pt"
                jobs.append({"location": location, "arm": arm, "seed": seed, "checkpoint": str(checkpoint)})
    return jobs


def mann_whitney(a: list[float], b: list[float]) -> float:
    from scipy import stats as scipy_stats

    if len(set(a + b)) == 1:
        return 1.0
    _u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def main() -> None:
    jobs = build_jobs()
    print(f"{len(jobs)} eval job(s): {len(LOCATIONS)} location(s) x 2 arm(s) x {N_SEEDS} seed(s), gamma=5.0")

    t0 = time.time()
    rho_results: dict[tuple[str, str], dict[str, list[float]]] = {}
    min_tput_results: dict[tuple[str, str], list[float]] = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["arm"])
            rho_results.setdefault(key, {label: [] for label in ALPHA_LABELS})
            for label in ALPHA_LABELS:
                rho_results[key][label].append(r["rho_alpha"][label])
            min_tput_results.setdefault(key, []).append(r["min_throughput_bps"] / 1e6)
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} {job['arm']:6s} seed={job['seed']}  "
                  f"rho_1={r['rho_alpha']['1.0']:.4f}  min_tput={r['min_throughput_bps']/1e6:6.2f}Mbps")

    report = []
    print(f"\n{'=' * 100}\nRQ2 under gamma=5.0: qbbr vs. stock, classical core\n{'=' * 100}")
    for location in LOCATIONS:
        row: dict[str, Any] = {"location": location}
        print(f"\n{location}:")
        for label in ALPHA_LABELS:
            qbbr_vals = rho_results[(location, "qbbr")][label]
            stock_vals = rho_results[(location, "stock")][label]
            qbbr_med = statistics.median(qbbr_vals)
            stock_med = statistics.median(stock_vals)
            p = mann_whitney(qbbr_vals, stock_vals)
            row[f"rho_{label}_qbbr_median"] = qbbr_med
            row[f"rho_{label}_stock_median"] = stock_med
            row[f"rho_{label}_p"] = p
            flag = "SIGNIFICANT" if p < 0.05 else "n.s."
            print(f"  alpha={label:4s}  qbbr_rho={qbbr_med:.4f}  stock_rho={stock_med:.4f}  "
                  f"delta={qbbr_med - stock_med:+.4f}  p={p:.4f}  [{flag}]")
        qbbr_min = statistics.median(min_tput_results[(location, "qbbr")])
        stock_min = statistics.median(min_tput_results[(location, "stock")])
        row["min_throughput_mbps_qbbr_median"] = qbbr_min
        row["min_throughput_mbps_stock_median"] = stock_min
        print(f"  min_throughput: qbbr={qbbr_min:6.2f}Mbps  stock={stock_min:6.2f}Mbps")
        report.append(row)

    OUT_PATH.write_text(json.dumps({"meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES, "gamma": 5.0}, "rq2_gamma5": report}, indent=2))
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
