from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
N_EPISODES = 10
ALPHA_LABELS = ["0.0", "1.0", "inf"]  # alpha=2.0 EXCLUDED: alpha_fair_efficiency_ratio is


class _StockAgent:
    def act(self, state: Any) -> tuple[int, float]:
        return 2, 0.0


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_b import run_scenario_b

    calibration = load_calibration(job["calibration_path"])
    if job["arm"] == "qbbr":
        agent = MLPA2CAgent(n_layers=job["n_layers"], action_dims=(5,))
        agent.load(job["checkpoint"])
    else:
        agent = _StockAgent()

    result = run_scenario_b(
        agent, job["location"], job["direction"], calibration,
        n_episodes=job["n_episodes"], episode_s=job["episode_s"], risk_mode=job["risk_mode"],
        alpha_sweep=(0.0, 1.0, float("inf")),  # alpha=2.0 excluded -- see ALPHA_LABELS' comment
    )
    return {"location": job["location"], "arm": job["arm"], "seed": job["seed"], **result}


def build_jobs(
    direction: str, calibration_path: Path, n_layers: int, n_episodes: int, episode_s: float,
    risk_mode: str, checkpoint_root: Path,
) -> list[dict[str, Any]]:
    jobs = []
    for location in LOCATIONS:
        for arm in ("qbbr", "stock"):
            for seed in range(N_SEEDS):
                checkpoint = Path(checkpoint_root) / "classical" / location / f"seed{seed}.pt"
                jobs.append({
                    "location": location, "arm": arm, "seed": seed, "direction": direction,
                    "checkpoint": str(checkpoint), "calibration_path": calibration_path,
                    "n_layers": n_layers, "n_episodes": n_episodes, "episode_s": episode_s,
                    "risk_mode": risk_mode,
                })
    return jobs


def mann_whitney(a: list[float], b: list[float]) -> float:
    from scipy import stats as scipy_stats

    if len(set(a + b)) == 1:  # identical constants on both sides (e.g. rho_alpha always 1.0)
        return 1.0
    _u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", default="downlink", choices=["downlink", "uplink"])
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--episode-s", type=float, default=300.0)
    parser.add_argument("--risk-mode", default="closed_form")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "rq2_report.json")
    args = parser.parse_args()

    jobs = build_jobs(
        args.direction, args.calibration_path, args.n_layers, args.n_episodes, args.episode_s,
        args.risk_mode, args.checkpoint_root,
    )
    print(f"{len(jobs)} eval job(s): {len(LOCATIONS)} location(s) x 2 arm(s) x {N_SEEDS} seed(s), "
          f"{args.n_episodes} episodes each, max_workers={args.max_workers}")

    t0 = time.time()
    # results[(location, arm)][alpha_label] = list of rho_alpha values (one per seed)
    rho_results: dict[tuple[str, str], dict[str, list[float]]] = {}
    min_tput_results: dict[tuple[str, str], list[float]] = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["arm"])
            rho_results.setdefault(key, {label: [] for label in ALPHA_LABELS})
            for label in ALPHA_LABELS:
                rho_results[key][label].append(r["rho_alpha"][label])
            min_tput_results.setdefault(key, [])
            min_tput_results[key].append(r["min_throughput_bps"] / 1e6)
            elapsed = time.time() - t0
            rho1 = r["rho_alpha"]["1.0"]
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} {job['arm']:6s} seed={job['seed']}  "
                  f"rho_1={rho1:.4f}  min_tput={r['min_throughput_bps']/1e6:6.1f}Mbps")

    report = []
    print(f"\n{'=' * 100}\nRQ2: qbbr vs. stock BBR-v3 (as the competing-flow anchor), classical core, "
          f"Mann-Whitney p<0.05\n{'=' * 100}")
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
            row[f"rho_{label}_qbbr_better"] = qbbr_med > stock_med and p < 0.05
            flag = "SIGNIFICANT" if p < 0.05 else "n.s."
            print(f"  alpha={label:4s}  qbbr_rho={qbbr_med:.4f}  stock_rho={stock_med:.4f}  "
                  f"delta={qbbr_med - stock_med:+.4f}  p={p:.4f}  [{flag}]")
        qbbr_min = statistics.median(min_tput_results[(location, "qbbr")])
        stock_min = statistics.median(min_tput_results[(location, "stock")])
        row["min_throughput_mbps_qbbr_median"] = qbbr_min
        row["min_throughput_mbps_stock_median"] = stock_min
        print(f"  min_throughput: qbbr={qbbr_min:6.1f}Mbps  stock={stock_min:6.1f}Mbps")
        report.append(row)

    out_payload = {
        "meta": {
            "direction": args.direction, "n_seeds": N_SEEDS, "n_episodes": args.n_episodes,
            "alpha_sweep": ALPHA_LABELS, "total_elapsed_s": time.time() - t0,
        },
        "rq2": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_payload, indent=2, sort_keys=True))
    try:
        shown_path = args.out.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = args.out
    print(f"\ndone in {time.time() - t0:.1f}s -> {shown_path}")


if __name__ == "__main__":
    main()
