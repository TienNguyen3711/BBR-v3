"""RQ3-under-gamma5 (Step 2 of the gamma=5.0 follow-up): trains the
risk_off counterpart (risk_mode=stub_constant -- s5/s6 frozen at their
constant placeholder values, agent effectively blind to risk) under the
same gamma=5.0 reward as train_gamma5_parallel.py's risk_on
(risk_mode=closed_form) checkpoints, so RQ3's risk-on-vs-off comparison
(eval_rq3_parallel.py's design) can be re-run under gamma=5.0 and compared
against the published (gamma=0.0) RQ3 null.

Output -> outputs/checkpoints_gamma5_riskoff/pacing_only/classical/{location}/seed{N}.pt
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
N_SEEDS = 10
DIRECTION = "downlink"
CORE = "classical"
RISK_MODE = "stub_constant"  # the only difference vs. train_gamma5_parallel.py
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints_gamma5_riskoff" / "pacing_only" / CORE


def _train_one(job: dict) -> dict:
    import numpy as np
    import torch

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

    base_config = yaml.safe_load(CONFIG_PATH.read_text())
    jobs = [
        {"location": loc, "seed": seed, "base_config": base_config}
        for loc in LOCATIONS for seed in range(N_SEEDS)
    ]
    print(f"{len(jobs)} training job(s): {len(LOCATIONS)} location(s) x {N_SEEDS} seed(s), "
          f"direction={DIRECTION}, core={CORE}, risk_mode={RISK_MODE}, gamma={base_config['reward']['gamma']}")

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
