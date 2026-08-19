from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DEFAULT_ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
DEFAULT_LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "ablation"


def _train_one_job(job: dict[str, Any]) -> dict[str, Any]:
    """Runs in its own process (ProcessPoolExecutor, spawn start method on
    macOS): all imports are local so the worker doesn't need the parent's
    already-imported state, and each process gets its own single-threaded
    torch (see the comment below) to avoid oversubscribing the machine's
    cores when many workers run at once.
    """
    import numpy as np
    import torch

    torch.set_num_threads(1)

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.agents.quantum.qa2c import QA2CAgent
    from qbbr.env.calibration import load_calibration
    from qbbr.env.fluid_env import FluidSimEnv
    from qbbr.train.loop import train

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])

    calibration = load_calibration(job["calibration_path"])
    env = FluidSimEnv(
        job["location"], job["direction"], calibration, action_config=job["action_config"],
        risk_mode=job["risk_mode"], episode_s=job["episode_s"], reward_kwargs=job["reward_kwargs"],
    )
    action_dims = job["action_dims"]
    if job["core"] == "quantum":
        agent = QA2CAgent(
            n_layers=job["n_layers"], action_dims=action_dims, lr=job["lr"], gamma=job["gamma"],
            reupload=job["reupload"],
        )
    else:
        agent = MLPA2CAgent(n_layers=job["n_layers"], action_dims=action_dims, lr=job["lr"], gamma=job["gamma"])

    t0 = time.time()
    result = train(agent, env, job["n_episodes"], config={"seed": job["seed"]})
    elapsed = time.time() - t0

    checkpoint_path = None
    if job["checkpoint_root"] is not None:
        checkpoint_dir = Path(job["checkpoint_root"]) / job["location"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"seed{job['seed']}.pt"
        agent.save(checkpoint_path)

    return {
        "location": job["location"],
        "seed": job["seed"],
        "core": job["core"],
        "final_mean_reward": result["final_mean_reward"],
        "elapsed_s": elapsed,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


def build_jobs(
    locations: list[str], direction: str, core: str, config: dict[str, Any],
    n_episodes: int, n_runs: int, base_seed: int,
    calibration_path: Path, checkpoint_root: Path | None,
    action_config: dict[str, Any], action_dims: tuple[int, ...],
) -> list[dict[str, Any]]:
    jobs = []
    for location in locations:
        for run_idx in range(n_runs):
            jobs.append({
                "location": location, "direction": direction, "core": core,
                "seed": base_seed + run_idx,
                "n_episodes": n_episodes, "episode_s": config.get("episode", {}).get("duration_s", 300.0),
                "risk_mode": config.get("risk_mode", "closed_form"),
                "reward_kwargs": config.get("reward", {}),
                "n_layers": config.get("n_layers", 2), "lr": config.get("learning_rate", 1e-3),
                "gamma": config.get("gamma", 0.99), "reupload": config.get("reupload", False),
                "calibration_path": calibration_path,
                "checkpoint_root": checkpoint_root,
                "action_config": action_config, "action_dims": action_dims,
            })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a qbbr/configs/*.yaml file")
    parser.add_argument("--locations", type=lambda s: s.split(","), default=DEFAULT_LOCATIONS)
    parser.add_argument("--direction", required=True, choices=["downlink", "uplink"])
    parser.add_argument("--core", choices=["quantum", "classical"], required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=None, help="default: config's episode.target_episodes[0]")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--risk-mode", choices=["stub_constant", "empirical_proxy", "closed_form"], default=None)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                         help="if set, save each (location, seed) agent under <root>/<location>/seed<N>.pt")
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/ablation/parallel_<timestamp>.json")
    parser.add_argument("--max-workers", type=int, default=12,
                         help="concurrent training processes; leave headroom below the core count "
                              "(this machine has 18) for the OS and any other work")
    args = parser.parse_args()

    from qbbr.action.registry import dimension_sizes, is_multihead, load_action_space

    config = yaml.safe_load(Path(args.config).read_text())
    if args.risk_mode is not None:
        config["risk_mode"] = args.risk_mode
    n_episodes = args.n_episodes or config.get("episode", {}).get("target_episodes", [200])[0]
    action_config = load_action_space(args.action_config)
    action_dims = tuple(dimension_sizes(action_config)) if is_multihead(action_config) else (
        len(action_config["levels"]),
    )

    jobs = build_jobs(
        args.locations, args.direction, args.core, config, n_episodes, args.n_runs, args.base_seed,
        args.calibration_path, args.checkpoint_root, action_config, action_dims,
    )
    print(f"{len(jobs)} job(s): {len(args.locations)} location(s) x {args.n_runs} seed(s), "
          f"{n_episodes} episodes each, core={args.core}, max_workers={args.max_workers}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_train_one_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            result = future.result()
            results.append(result)
            elapsed = time.time() - t0
            print(f"[{i}/{len(jobs)}] {elapsed:7.1f}s elapsed  {job['location']:10s} seed={job['seed']:2d}  "
                  f"job_time={result['elapsed_s']:6.1f}s  final_mean_reward={result['final_mean_reward']:.4f}")

    out_path = args.out or (DEFAULT_OUT_DIR / f"parallel_{args.core}_{args.direction}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_path.write_text(json.dumps({
        "meta": {
            "locations": args.locations, "direction": args.direction, "core": args.core,
            "n_runs": args.n_runs, "n_episodes": n_episodes, "max_workers": args.max_workers,
            "total_elapsed_s": time.time() - t0,
        },
        "results": results,
    }, indent=2, sort_keys=True))
    print(f"\ndone in {time.time() - t0:.1f}s (vs. {sum(r['elapsed_s'] for r in results):.1f}s sequential) -> {out_path}")


if __name__ == "__main__":
    main()
