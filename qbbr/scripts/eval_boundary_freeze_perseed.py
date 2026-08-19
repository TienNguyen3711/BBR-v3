"""Per-seed boundary-freeze effect-size table: baseline (pre-registered,
pacing-only, no freeze/ECN/STARTUP/s5-fix -- outputs/checkpoints, ORIGINAL
pre-multihead-refactor format) vs. freeze (exploratory extension: s5-fix +
boundary-freeze + EMA -- outputs/checkpoints_final/pacing_only, CURRENT
multi-head-compatible format), each seed's retransmit reduction vs. the
index-matched real BBR trace file, plus each seed's within-seed IQR.

The baseline checkpoints predate today's actor_trunk/actor_heads refactor
(single nn.Sequential("actor") with keys "0.*"/"2.*", not split into a
trunk + per-dimension heads). This script does NOT touch the shipped
MLPA2CAgent.load() -- it carries a local, read-only remap
(_load_legacy_single_head_agent) that is exact and lossless for the
single-head case: actor.0.* is the trunk's only linear layer (identical
before and after the refactor), and actor.2.* is the trunk output layer,
which is exactly what actor_heads[0] is when there is only one head.
Confirmed byte-shape-identical against the current _best_hidden_size(...)
computation before use (hidden=3 actor / 4 critic for n_qubits=6, actor
out_dim=5, budget=36).

Per-seed pairing here (agent seed i vs. real-trace file i) is an index
convention for building an interpretable per-row diagnostic table, not a
formal paired statistical test -- the pre-registered RQ1/RQ1b comparisons
already run the correct unpaired n-vs-n Mann-Whitney (eval_rq1_parallel.py).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
FREEZE_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_final" / "pacing_only"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
N_EPISODES = 10
EPISODE_S = 300.0
DIRECTION = "downlink"
OUT_PATH = PROJECT_ROOT / "outputs" / "boundary_freeze_perseed.json"


def _load_legacy_single_head_agent(checkpoint_path: Path, n_layers: int = 2):
    import torch

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent

    agent = MLPA2CAgent(n_layers=n_layers, action_dims=(5,))
    legacy = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent.actor_trunk.load_state_dict({"0.weight": legacy["actor"]["0.weight"], "0.bias": legacy["actor"]["0.bias"]})
    agent.actor_heads.load_state_dict({"0.weight": legacy["actor"]["2.weight"], "0.bias": legacy["actor"]["2.bias"]})
    agent.critic.load_state_dict(legacy["critic"])
    return agent


def _eval_one_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)

    from qbbr.action.registry import load_action_space
    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.eval.scenario_a import simulated_agent_stats

    calibration = load_calibration(CALIBRATION_PATH)
    action_config = load_action_space(ACTION_CONFIG_PATH)

    if job["arm"] == "baseline":
        agent = _load_legacy_single_head_agent(Path(job["checkpoint"]))
    else:
        agent = MLPA2CAgent(n_layers=2, action_dims=(5,))
        agent.load(job["checkpoint"])

    stats = simulated_agent_stats(
        agent, job["location"], DIRECTION, calibration,
        n_episodes=N_EPISODES, episode_s=EPISODE_S, risk_mode="closed_form",
        action_config=action_config,
    )
    return {"location": job["location"], "arm": job["arm"], "seed": job["seed"], **stats}


def build_jobs() -> list[dict[str, Any]]:
    jobs = []
    for location in LOCATIONS:
        for arm, root in (("baseline", BASELINE_ROOT), ("freeze", FREEZE_ROOT)):
            for seed in range(N_SEEDS):
                checkpoint = root / "classical" / location / f"seed{seed}.pt"
                jobs.append({"location": location, "arm": arm, "seed": seed, "checkpoint": str(checkpoint)})
    return jobs


def main() -> None:
    from qbbr.eval.metrics import real_cca_per_run_medians

    jobs = build_jobs()
    print(f"{len(jobs)} eval job(s): {len(LOCATIONS)} location(s) x 2 arm(s) x {N_SEEDS} seed(s)")

    t0 = time.time()
    results: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_eval_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            key = (r["location"], r["arm"])
            results.setdefault(key, {})[r["seed"]] = r
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} {job['arm']:8s} seed={job['seed']}  "
                  f"rtx/s={r['retransmits_per_s_median']:6.2f}  iqr={r['retransmits_per_s_iqr']:6.2f}")

    real_by_location = {
        location: real_cca_per_run_medians(DATASET_ROOT, location, DIRECTION, "bbr")["retransmits_per_s"]
        for location in LOCATIONS
    }

    rows = []
    print(f"\n{'=' * 110}\nBoundary-freeze per-seed table (baseline=pre-registered pacing-only, "
          f"freeze=exploratory s5-fix+freeze+EMA extension)\n{'=' * 110}")
    print(f"{'location':10s} {'seed':4s} {'tag':12s} {'rtx/s':>7s} {'iqr':>6s} {'real_bbr':>9s} {'reduction':>10s}")
    for location in LOCATIONS:
        real_vals = real_by_location[location]
        for arm, tag in (("baseline", "pre-reg"), ("freeze", "exploratory")):
            for seed in range(N_SEEDS):
                r = results[(location, arm)][seed]
                real_rtx = real_vals[seed]
                reduction = 1.0 - r["retransmits_per_s_median"] / real_rtx if real_rtx > 0 else float("nan")
                row = {
                    "location": location, "seed": seed, "arm": arm, "tag": tag,
                    "retransmits_per_s_median": r["retransmits_per_s_median"],
                    "retransmits_per_s_iqr": r["retransmits_per_s_iqr"],
                    "real_bbr_retransmits_per_s": real_rtx,
                    "retransmit_reduction_vs_real_bbr": reduction,
                }
                rows.append(row)
                print(f"{location:10s} {seed:<4d} {tag:12s} {r['retransmits_per_s_median']:7.2f} "
                      f"{r['retransmits_per_s_iqr']:6.2f} {real_rtx:9.2f} {reduction:+9.1%}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"meta": {"n_seeds": N_SEEDS, "n_episodes": N_EPISODES}, "rows": rows}, indent=2))
    print(f"\ndone in {time.time() - t0:.1f}s -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
