"""RQ4-under-gamma5 (Step 2 of the gamma=5.0 follow-up): trains the quantum
core under the same gamma=5.0 reward as train_gamma5_parallel.py's
classical checkpoints, at RQ4's exact published grid point (alpha=1, L=2,
no data re-uploading, risk features on, 200 episodes, pacing_gain-only) --
see main.tex Sec. sec:rq4 -- so the quantum-vs-classical comparison
(Table tab:qvc's design) can be re-run under gamma=5.0 and checked against
the gamma=0.0 published null (quantum and classical statistically
indistinguishable at every location).

Output -> outputs/checkpoints_gamma5/pacing_only/quantum/{location}/seed{N}.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PACKAGE_ROOT / "configs" / "pilot_gamma5.yaml"
CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 3  # pilot scale (matches this project's own precedent -- main.tex sec:rq4's
# "earlier 3-seed pilot"), chosen after measuring quantum training at ~42s/episode
# (~20x classical): a full 10-seed run would take ~11-23h depending on scheduling,
# vs. ~3-7h for this pilot. Not a substitute for a full run if the pilot result
# looks decision-relevant -- see main.tex's own caveat about 3-seed's p-value floor
# (p=2/C(6,3)=0.10, cannot reach significance at any effect size).
DIRECTION = "downlink"
CORE = "quantum"
RISK_MODE = "closed_form"
REUPLOAD = False  # matches RQ4's published grid point: L=2, no data re-uploading
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_gamma5" / "pacing_only" / CORE


def _train_one(job: dict) -> dict:
    import os

    # MUST happen before numpy/pennylane/torch are imported in this worker
    # process (spawn start method -> fresh interpreter per worker, so this
    # is still early enough): torch.set_num_threads(1) alone does not
    # constrain PennyLane's underlying numpy/BLAS thread pool, which was
    # silently oversubscribing across all 6 concurrent workers -- the
    # likely cause of an 11h stall producing zero completed checkpoints
    # (each worker fighting the others for the same physical cores via
    # independent BLAS thread pools, instead of one thread each).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    import numpy as np
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.quantum.qa2c import QA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.env.fluid_env import FluidSimEnv
    from qbbr.train.loop import train

    config = dict(job["base_config"])
    config["seed"] = job["seed"]
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])

    calibration = load_calibration(CALIBRATION_PATH)
    env = FluidSimEnv(
        job["location"], DIRECTION, calibration, risk_mode=RISK_MODE,
        episode_s=config.get("episode", {}).get("duration_s", 300.0),
        reward_kwargs=config.get("reward", {}),
    )
    agent = QA2CAgent(
        n_layers=config.get("n_layers", 2), lr=config.get("learning_rate", 1e-3),
        gamma=config.get("gamma", 0.99), reupload=REUPLOAD,
    )
    n_episodes = config.get("episode", {}).get("target_episodes", [200])[0]

    t0 = time.time()
    result = train(agent, env, n_episodes, config, run_dir=None)
    elapsed = time.time() - t0

    out_dir = CHECKPOINT_ROOT / job["location"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{job['seed']}.pt"
    agent.save(str(out_path))

    return {
        "location": job["location"], "seed": job["seed"], "elapsed_s": elapsed,
        "final_mean_reward": result["final_mean_reward"], "n_episodes": n_episodes,
        "checkpoint": str(out_path),
    }


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    base_config = yaml.safe_load(CONFIG_PATH.read_text())
    jobs = [
        {"location": loc, "seed": seed, "base_config": base_config}
        for loc in LOCATIONS for seed in range(N_SEEDS)
    ]
    print(f"{len(jobs)} training job(s): {len(LOCATIONS)} location(s) x {N_SEEDS} seed(s), "
          f"direction={DIRECTION}, core={CORE}, reupload={REUPLOAD}, risk_mode={RISK_MODE}, "
          f"gamma={base_config['reward']['gamma']}")

    t0 = time.time()
    # Fewer workers than the classical runs (max_workers=10): this job runs
    # concurrently alongside the risk-off classical retrain (also 10
    # workers) on an 18-core machine, and per-step quantum circuit
    # simulation is heavier than the classical MLP forward pass.
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_train_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} seed={job['seed']}  "
                  f"run_time={r['elapsed_s']:6.1f}s  final_mean_reward={r['final_mean_reward']:.3f}  "
                  f"-> {Path(r['checkpoint']).relative_to(PROJECT_ROOT)}")

    print(f"\ndone in {time.time() - t0:.1f}s. checkpoints -> "
          f"{CHECKPOINT_ROOT.relative_to(PROJECT_ROOT)}/<location>/seed<N>.pt")


if __name__ == "__main__":
    main()
