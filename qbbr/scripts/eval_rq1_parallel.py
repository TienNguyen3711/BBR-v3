from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
CORES = ["classical", "quantum"]
COMPARISON_CCAS = ["bbr", "cubic", "vegas", "hybla"]
N_SEEDS = 10
N_EPISODES = 10  # matches configs/eval_scenarioA.yaml's runs_per_config
RETRANSMIT_REDUCTION_THRESHOLD = 0.20
THROUGHPUT_RETENTION_THRESHOLD = 0.95
ALPHA = 0.05
RETRANSMIT_MEANINGFUL_FLOOR_PER_S = 0.5  # below this, a CCA's retransmit rate reflects near-idle


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.agents.quantum.qa2c import QA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_a import simulated_agent_stats

    calibration = load_calibration(job["calibration_path"])
    action_dims = job["action_dims"]
    if job["core"] == "quantum":
        agent = QA2CAgent(n_layers=job["n_layers"], action_dims=action_dims)
    else:
        agent = MLPA2CAgent(n_layers=job["n_layers"], action_dims=action_dims)
    agent.load(job["checkpoint"])

    stats = simulated_agent_stats(
        agent, job["location"], job["direction"], calibration,
        n_episodes=job["n_episodes"], episode_s=job["episode_s"], risk_mode=job["risk_mode"],
        action_config=job["action_config"],
    )
    return {"location": job["location"], "core": job["core"], "seed": job["seed"], **stats}


def build_jobs(
    direction: str, checkpoint_root: Path, calibration_path: Path,
    n_layers: int, n_episodes: float, episode_s: float, risk_mode: str,
    action_config: dict[str, Any], action_dims: tuple[int, ...],
    cores: list[str] = CORES,
) -> list[dict[str, Any]]:
    jobs = []
    for location in LOCATIONS:
        for core in cores:
            for seed in range(N_SEEDS):
                checkpoint = Path(checkpoint_root) / core / location / f"seed{seed}.pt"
                jobs.append({
                    "location": location, "core": core, "seed": seed, "direction": direction,
                    "checkpoint": str(checkpoint), "calibration_path": calibration_path,
                    "n_layers": n_layers, "n_episodes": n_episodes, "episode_s": episode_s, "risk_mode": risk_mode,
                    "action_config": action_config, "action_dims": action_dims,
                })
    return jobs


def mann_whitney(a: list[float], b: list[float]) -> float:
    from scipy import stats as scipy_stats

    _u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", default="downlink", choices=["downlink", "uplink"])
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--episode-s", type=float, default=300.0)
    parser.add_argument("--risk-mode", default="closed_form")
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG_PATH)
    parser.add_argument("--cores", type=lambda s: s.split(","), default=CORES,
                         help="e.g. classical or classical,quantum")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "rq1_report.json")
    args = parser.parse_args()

    from qbbr.action.registry import dimension_sizes, is_multihead, load_action_space

    action_config = load_action_space(args.action_config)
    action_dims = tuple(dimension_sizes(action_config)) if is_multihead(action_config) else (
        len(action_config["levels"]),
    )

    jobs = build_jobs(
        args.direction, args.checkpoint_root, args.calibration_path,
        args.n_layers, args.n_episodes, args.episode_s, args.risk_mode,
        action_config, action_dims, args.cores,
    )
    print(f"{len(jobs)} eval job(s): {len(LOCATIONS)} location(s) x {len(args.cores)} core(s) x {N_SEEDS} seed(s), "
          f"{args.n_episodes} episodes each, max_workers={args.max_workers}")

    t0 = time.time()
    qbbr_results: dict[tuple[str, str], dict[str, list[float]]] = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["core"])
            qbbr_results.setdefault(key, {"throughput_mbps": [], "rtt_ms": [], "retransmits_per_s": []})
            qbbr_results[key]["throughput_mbps"].append(r["throughput_mbps_median"])
            qbbr_results[key]["rtt_ms"].append(r["rtt_ms_median"])
            qbbr_results[key]["retransmits_per_s"].append(r["retransmits_per_s_median"])
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} {job['core']:9s} seed={job['seed']}  "
                  f"tput={r['throughput_mbps_median']:7.1f}Mbps  rtt={r['rtt_ms_median']:7.1f}ms  "
                  f"rtx/s={r['retransmits_per_s_median']:6.2f}")

    print("\nloading real per-run baselines...")
    from qbbr.eval.metrics import real_cca_per_run_medians

    real_by_loc_cca: dict[tuple[str, str], dict[str, list[float]]] = {}
    for location in LOCATIONS:
        for cca in COMPARISON_CCAS:
            real_by_loc_cca[(location, cca)] = real_cca_per_run_medians(
                args.dataset_root, location, args.direction, cca
            )

    report = []
    print(f"\n{'=' * 100}\nRQ1 verdict (vs. stock BBR-v3): retransmit reduction >= {RETRANSMIT_REDUCTION_THRESHOLD:.0%}, "
          f"throughput retention >= {THROUGHPUT_RETENTION_THRESHOLD:.0%}, retransmit Mann-Whitney p < {ALPHA}\n{'=' * 100}")
    for location in LOCATIONS:
        for core in args.cores:
            qbbr = qbbr_results[(location, core)]
            bbr = real_by_loc_cca[(location, "bbr")]

            import statistics
            qbbr_tput_med = statistics.median(qbbr["throughput_mbps"])
            qbbr_rtx_med = statistics.median(qbbr["retransmits_per_s"])
            bbr_tput_med = statistics.median(bbr["throughput_mbps"])
            bbr_rtx_med = statistics.median(bbr["retransmits_per_s"])

            tput_retention = qbbr_tput_med / bbr_tput_med if bbr_tput_med > 0 else float("nan")
            rtx_reduction = 1.0 - qbbr_rtx_med / bbr_rtx_med if bbr_rtx_med > 0 else float("nan")
            p_rtx = mann_whitney(qbbr["retransmits_per_s"], bbr["retransmits_per_s"])
            p_tput = mann_whitney(qbbr["throughput_mbps"], bbr["throughput_mbps"])

            verdict = (
                rtx_reduction >= RETRANSMIT_REDUCTION_THRESHOLD
                and tput_retention >= THROUGHPUT_RETENTION_THRESHOLD
                and p_rtx < ALPHA
            )
            row = {
                "location": location, "core": core,
                "qbbr_throughput_mbps_median": qbbr_tput_med, "bbr_throughput_mbps_median": bbr_tput_med,
                "qbbr_retransmits_per_s_median": qbbr_rtx_med, "bbr_retransmits_per_s_median": bbr_rtx_med,
                "throughput_retention": tput_retention, "retransmit_reduction": rtx_reduction,
                "p_retransmits": p_rtx, "p_throughput": p_tput, "rq1_pass": verdict,
            }
            report.append(row)
            flag = "PASS" if verdict else "FAIL"
            print(f"{location:10s} {core:9s}  tput_retention={tput_retention:6.1%}  "
                  f"rtx_reduction={rtx_reduction:+7.1%}  p_rtx={p_rtx:.4f}  p_tput={p_tput:.4f}  [{flag}]")

    print(f"\n{'=' * 100}\nFull comparison vs. all CCAs (descriptive + Mann-Whitney; not a pass/fail gate -- "
          f"RQ1's pre-registered criterion is vs. stock BBR-v3 only, above)\n{'=' * 100}")
    comparison_report = []
    for location in LOCATIONS:
        for core in args.cores:
            qbbr = qbbr_results[(location, core)]
            qbbr_tput_med = statistics.median(qbbr["throughput_mbps"])
            qbbr_rtx_med = statistics.median(qbbr["retransmits_per_s"])
            for cca in COMPARISON_CCAS:
                real = real_by_loc_cca[(location, cca)]
                real_tput_med = statistics.median(real["throughput_mbps"])
                real_rtx_med = statistics.median(real["retransmits_per_s"])
                tput_retention = qbbr_tput_med / real_tput_med if real_tput_med > 0 else float("nan")
                # A near-idle competitor (real_rtx_med close to 0, typical of Cubic/Vegas/Hybla on
                # these high-RTT links -- they trivially avoid loss by barely sending anything) makes
                # a retransmit RATIO meaningless: it's an artifact of the denominator, not a genuine
                # comparison. RETRANSMIT_MEANINGFUL_FLOOR_PER_S gates this off rather than reporting
                # a NaN or a three-digit swing that looks broken.
                rtx_comparison_meaningful = real_rtx_med >= RETRANSMIT_MEANINGFUL_FLOOR_PER_S
                rtx_reduction = (
                    1.0 - qbbr_rtx_med / real_rtx_med if rtx_comparison_meaningful else None
                )
                p_tput = mann_whitney(qbbr["throughput_mbps"], real["throughput_mbps"])
                p_rtx = mann_whitney(qbbr["retransmits_per_s"], real["retransmits_per_s"]) if rtx_comparison_meaningful else None
                comparison_report.append({
                    "location": location, "core": core, "vs_cca": cca,
                    "qbbr_throughput_mbps_median": qbbr_tput_med, "cca_throughput_mbps_median": real_tput_med,
                    "qbbr_retransmits_per_s_median": qbbr_rtx_med, "cca_retransmits_per_s_median": real_rtx_med,
                    "throughput_retention": tput_retention, "retransmit_reduction": rtx_reduction,
                    "p_throughput": p_tput, "p_retransmits": p_rtx,
                    "retransmit_comparison_meaningful": rtx_comparison_meaningful,
                })
                rtx_str = f"{rtx_reduction:+7.1%}" if rtx_comparison_meaningful else "  n/a  "
                p_rtx_str = f"{p_rtx:.4f}" if rtx_comparison_meaningful else " n/a  "
                print(f"{location:10s} {core:9s} vs {cca:6s}  tput_retention={tput_retention:7.1%}  "
                      f"rtx_reduction={rtx_str}  p_tput={p_tput:.4f}  p_rtx={p_rtx_str}"
                      + ("" if rtx_comparison_meaningful else "  (rtx comparison n/a: near-idle baseline)"))

    out_payload = {
        "meta": {
            "direction": args.direction, "n_seeds": N_SEEDS, "n_episodes": args.n_episodes,
            "thresholds": {
                "retransmit_reduction": RETRANSMIT_REDUCTION_THRESHOLD,
                "throughput_retention": THROUGHPUT_RETENTION_THRESHOLD, "alpha": ALPHA,
            },
            "total_elapsed_s": time.time() - t0,
        },
        "rq1": report,
        "comparison_vs_all_ccas": comparison_report,
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
