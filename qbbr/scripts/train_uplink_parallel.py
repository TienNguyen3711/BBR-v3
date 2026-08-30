"""Trains qbbr (classical core, pacing_gain-only action space, closed_form
risk mode) from scratch under UPLINK dynamics -- every existing checkpoint
in this project (checkpoints_final, checkpoints_multihead, ...) was trained
under downlink only, even though per_location_constants.json already holds
a fully-calibrated, physically distinct uplink parameter set (much lower
B_max, much higher baseline retransmit rate -- see qbbr/env/calibration.py)
and the real trace dataset already has uplink coverage for every CCA. This
is the first script to actually exercise that half of the calibration.

Mirrors base.yaml (seed 0, gamma 0.99, lr 1e-3, n_layers 2, 200 episodes --
config's target_episodes[0]) and eval_rq1_parallel.py's action space
default (action_pacing_gain.yaml) exactly, just under direction="uplink"
instead of "downlink", so the resulting checkpoints are the direct uplink
counterpart of outputs/checkpoints_final/pacing_only/classical.

No `action_config` is passed to FluidSimEnv -- its own default already is
the single-head pacing_gain space (confirmed: mirrors what train.py does
for a plain classical/pacing-only run).

Output -> outputs/checkpoints_uplink/pacing_only/classical/{location}/seed{N}.pt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

BASE_CONFIG_PATH = PACKAGE_ROOT / "configs" / "base.yaml"
CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
DIRECTION = "uplink"
CORE = "classical"
RISK_MODE = "closed_form"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_uplink" / "pacing_only" / CORE


def _train_one(job: dict) -> dict:
    import torch
    import numpy as np

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
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
    agent = MLPA2CAgent(
        n_layers=config.get("n_layers", 2), lr=config.get("learning_rate", 1e-3), gamma=config.get("gamma", 0.99),
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

    base_config = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    jobs = [
        {"location": loc, "seed": seed, "base_config": base_config}
        for loc in LOCATIONS for seed in range(N_SEEDS)
    ]
    print(f"{len(jobs)} training job(s): {len(LOCATIONS)} location(s) x {N_SEEDS} seed(s), "
          f"direction={DIRECTION}, core={CORE}, risk_mode={RISK_MODE}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=10) as pool:
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
