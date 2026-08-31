"""Train the per-location classical checkpoints an RQ3 risk-feature ablation
needs, in the ``<out-root>/classical/<location>/seed<N>.pt`` layout that
``qbbr.scripts.eval_rq3_parallel`` consumes.

RQ3 compares a policy trained *with* the closed-form risk features
(``s5``/``s6``) against one trained *without* them (``stub_constant``).
Both arms use the pacing_gain-only action space and the published reward,
so the only difference between an arm's checkpoints is ``--risk-mode``:

    # risk-on arm, new dynamic risk model (atmospheric + ISL + handover hazard)
    python -m qbbr.scripts.train_rq3_checkpoints --risk-mode closed_form_dynamic \
        --locations London Mumbai Sydney SaoPaulo --seeds 0 1 2 3 4 \
        --out-root outputs/checkpoints_rq3_dynamic/risk_on

    # matched risk-off arm
    python -m qbbr.scripts.train_rq3_checkpoints --risk-mode stub_constant \
        --locations London Mumbai Sydney SaoPaulo --seeds 0 1 2 3 4 \
        --out-root outputs/checkpoints_rq3_dynamic/risk_off

This is the same training protocol as ``train_gamma5_parallel.py`` (classical
core, pacing_gain-only, downlink); it exists so the RQ3 checkpoint-building
step is a committed, parameterised entry point rather than an ad-hoc script.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "base.yaml"
DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ALL_LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
RISK_MODES = ["stub_constant", "empirical_proxy", "closed_form", "closed_form_dynamic"]


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

    calibration = load_calibration(job["calibration_path"])
    env = FluidSimEnv(
        job["location"], job["direction"], calibration, risk_mode=job["risk_mode"],
        episode_s=config.get("episode", {}).get("duration_s", 300.0),
        reward_kwargs=config.get("reward", {}),
    )
    agent = MLPA2CAgent(
        n_layers=config.get("n_layers", 2),
        lr=config.get("learning_rate", 1e-3),
        gamma=config.get("gamma", 0.99),
    )

    t0 = time.time()
    result = train(agent, env, job["n_episodes"], config, run_dir=None)
    elapsed = time.time() - t0

    out_dir = Path(job["out_root"]) / "classical" / job["location"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{job['seed']}.pt"
    agent.save(str(out_path))

    return {
        "location": job["location"], "seed": job["seed"], "elapsed_s": elapsed,
        "final_mean_reward": result["final_mean_reward"], "n_episodes": job["n_episodes"],
        "checkpoint": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--risk-mode", required=True, choices=RISK_MODES,
                        help="risk feature the checkpoints are trained with; the RQ3 arm this builds")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="checkpoints are written to <out-root>/classical/<location>/seed<N>.pt")
    parser.add_argument("--locations", nargs="+", default=ALL_LOCATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--direction", default="downlink", choices=["downlink", "uplink"])
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="default: config's episode.target_episodes[0] (200)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    base_config = yaml.safe_load(args.config.read_text())
    n_episodes = args.n_episodes or base_config.get("episode", {}).get("target_episodes", [200])[0]

    jobs = [
        {
            "location": loc, "seed": seed, "base_config": base_config,
            "risk_mode": args.risk_mode, "direction": args.direction,
            "out_root": str(args.out_root), "n_episodes": n_episodes,
            "calibration_path": args.calibration_path,
        }
        for loc in args.locations for seed in args.seeds
    ]
    print(f"{len(jobs)} training job(s): {len(args.locations)} location(s) x {len(args.seeds)} seed(s), "
          f"direction={args.direction}, core=classical, risk_mode={args.risk_mode}, "
          f"{n_episodes} episodes each -> {args.out_root}/classical/<location>/seed<N>.pt")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_train_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            r = future.result()
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s  {job['location']:10s} seed={job['seed']}  "
                  f"run_time={r['elapsed_s']:6.1f}s  final_mean_reward={r['final_mean_reward']:.3f}  "
                  f"-> {Path(r['checkpoint']).relative_to(PROJECT_ROOT)}")

    print(f"\ndone in {time.time() - t0:.1f}s. checkpoints -> "
          f"{args.out_root}/classical/<location>/seed<N>.pt")


if __name__ == "__main__":
    main()
