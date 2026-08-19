"""RQ3 (risk-feature contribution): evaluates the risk-on (closed_form) vs.
risk-off (stub_constant) checkpoint pairs -- same action space, same
episode/reward dynamics, same seeds 0-9, the only difference is whether
s5/s6 carry real signal (closed_form) or a constant placeholder
(stub_constant) during BOTH training and evaluation.

Statistical design: mirrors eval_rq1_parallel.py exactly -- qbbr's 10
per-seed medians (risk-on) vs. qbbr's 10 per-seed medians (risk-off),
Mann-Whitney U, per location. This is the pre-registered design in
main.tex Sec. "RQ3: Risk-Feature Contribution" ("seeds matched pairwise to
the risk-on runs" refers to using the SAME seed range 0-9 as a controlled
comparison, not a paired statistical test -- the pre-registered analysis
plan itself names Mann-Whitney, not Wilcoxon signed-rank, and is honored
as originally written rather than substituted post-hoc).

Pre-registered RQ3 decision rule (main.tex Sec. V-C): strong correlation
retains s5/s6 as a core contribution; weak correlation demotes them to
ablation-only status, which is itself a reportable finding on proxy
validity, not a failure to hide.
"""
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
DEFAULT_ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
N_EPISODES = 10
ALPHA = 0.05


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_a import simulated_agent_stats

    calibration = load_calibration(job["calibration_path"])
    agent = MLPA2CAgent(n_layers=job["n_layers"], action_dims=job["action_dims"])
    agent.load(job["checkpoint"])

    stats = simulated_agent_stats(
        agent, job["location"], job["direction"], calibration,
        n_episodes=job["n_episodes"], episode_s=job["episode_s"], risk_mode=job["risk_mode"],
        action_config=job["action_config"],
    )
    return {"location": job["location"], "arm": job["arm"], "seed": job["seed"], **stats}


def build_jobs(
    direction: str, calibration_path: Path, n_layers: int, n_episodes: int, episode_s: float,
    action_config: dict[str, Any], action_dims: tuple[int, ...],
    risk_on_checkpoint_root: Path, risk_off_checkpoint_root: Path,
) -> list[dict[str, Any]]:
    jobs = []
    arms = [
        ("risk_on", risk_on_checkpoint_root, "closed_form"),
        ("risk_off", risk_off_checkpoint_root, "stub_constant"),
    ]
    for location in LOCATIONS:
        for arm, checkpoint_root, risk_mode in arms:
            for seed in range(N_SEEDS):
                checkpoint = Path(checkpoint_root) / "classical" / location / f"seed{seed}.pt"
                jobs.append({
                    "location": location, "arm": arm, "seed": seed, "direction": direction,
                    "checkpoint": str(checkpoint), "calibration_path": calibration_path,
                    "n_layers": n_layers, "n_episodes": n_episodes, "episode_s": episode_s,
                    "risk_mode": risk_mode, "action_config": action_config, "action_dims": action_dims,
                })
    return jobs


def mann_whitney(a: list[float], b: list[float]) -> float:
    from scipy import stats as scipy_stats

    _u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", default="downlink", choices=["downlink", "uplink"])
    parser.add_argument("--risk-on-checkpoint-root", type=Path, required=True)
    parser.add_argument("--risk-off-checkpoint-root", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--episode-s", type=float, default=300.0)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG_PATH)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "rq3_report.json")
    args = parser.parse_args()

    from qbbr.action.registry import dimension_sizes, is_multihead, load_action_space

    action_config = load_action_space(args.action_config)
    action_dims = tuple(dimension_sizes(action_config)) if is_multihead(action_config) else (
        len(action_config["levels"]),
    )

    jobs = build_jobs(
        args.direction, args.calibration_path, args.n_layers, args.n_episodes, args.episode_s,
        action_config, action_dims, args.risk_on_checkpoint_root, args.risk_off_checkpoint_root,
    )
    print(f"{len(jobs)} eval job(s): {len(LOCATIONS)} location(s) x 2 arm(s) x {N_SEEDS} seed(s), "
          f"{args.n_episodes} episodes each, max_workers={args.max_workers}")

    t0 = time.time()
    results: dict[tuple[str, str], dict[str, list[float]]] = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["arm"])
            results.setdefault(key, {"throughput_mbps": [], "rtt_ms": [], "retransmits_per_s": []})
            results[key]["throughput_mbps"].append(r["throughput_mbps_median"])
            results[key]["rtt_ms"].append(r["rtt_ms_median"])
            results[key]["retransmits_per_s"].append(r["retransmits_per_s_median"])
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} {job['arm']:9s} seed={job['seed']}  "
                  f"tput={r['throughput_mbps_median']:7.1f}Mbps  rtt={r['rtt_ms_median']:7.1f}ms  "
                  f"rtx/s={r['retransmits_per_s_median']:6.2f}")

    report = []
    print(f"\n{'=' * 100}\nRQ3: risk-on (closed_form) vs. risk-off (stub_constant), classical core, "
          f"Mann-Whitney p<{ALPHA}\n{'=' * 100}")
    for location in LOCATIONS:
        on = results[(location, "risk_on")]
        off = results[(location, "risk_off")]

        on_tput_med = statistics.median(on["throughput_mbps"])
        off_tput_med = statistics.median(off["throughput_mbps"])
        on_rtx_med = statistics.median(on["retransmits_per_s"])
        off_rtx_med = statistics.median(off["retransmits_per_s"])

        tput_delta = on_tput_med / off_tput_med - 1.0 if off_tput_med > 0 else float("nan")
        rtx_delta = on_rtx_med / off_rtx_med - 1.0 if off_rtx_med > 0 else float("nan")
        p_tput = mann_whitney(on["throughput_mbps"], off["throughput_mbps"])
        p_rtx = mann_whitney(on["retransmits_per_s"], off["retransmits_per_s"])

        row = {
            "location": location,
            "risk_on_throughput_mbps_median": on_tput_med, "risk_off_throughput_mbps_median": off_tput_med,
            "risk_on_retransmits_per_s_median": on_rtx_med, "risk_off_retransmits_per_s_median": off_rtx_med,
            "throughput_delta_on_vs_off": tput_delta, "retransmit_delta_on_vs_off": rtx_delta,
            "p_throughput": p_tput, "p_retransmits": p_rtx,
            "significant": (p_tput < ALPHA) or (p_rtx < ALPHA),
        }
        report.append(row)
        flag = "SIGNIFICANT DIFFERENCE" if row["significant"] else "no significant difference"
        print(f"{location:10s}  tput: on={on_tput_med:7.1f} off={off_tput_med:7.1f} ({tput_delta:+6.1%}) p={p_tput:.4f}  "
              f"rtx: on={on_rtx_med:6.2f} off={off_rtx_med:6.2f} ({rtx_delta:+6.1%}) p={p_rtx:.4f}  [{flag}]")

    out_payload = {
        "meta": {
            "direction": args.direction, "n_seeds": N_SEEDS, "n_episodes": args.n_episodes,
            "alpha": ALPHA, "total_elapsed_s": time.time() - t0,
        },
        "rq3": report,
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
