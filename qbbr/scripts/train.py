from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.agents.quantum.qa2c import QA2CAgent
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.train.logging import start_run
from qbbr.train.loop import train

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "runs"


def build_agent(core: str, config: dict, reupload: bool):
    n_layers = config.get("n_layers", 2)
    gamma = config.get("gamma", 0.99)
    lr = config.get("learning_rate", 1e-3)
    if core == "quantum":
        return QA2CAgent(n_layers=n_layers, lr=lr, gamma=gamma, reupload=reupload)
    return MLPA2CAgent(n_layers=n_layers, lr=lr, gamma=gamma)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a qbbr/configs/*.yaml file")
    parser.add_argument("--location", required=True)
    parser.add_argument("--direction", required=True, choices=["downlink", "uplink"])
    parser.add_argument("--core", choices=["quantum", "classical"], default="quantum")
    parser.add_argument("--n-episodes", type=int, default=None, help="default: config's episode.target_episodes[0]")
    parser.add_argument(
        "--risk-mode",
        choices=["stub_constant", "empirical_proxy", "closed_form", "closed_form_dynamic"],
        default="stub_constant",
    )
    parser.add_argument("--reupload", action="store_true", help="data re-uploading (quantum core only)")
    parser.add_argument(
        "--ablate-s7", action="store_true",
        help="freeze s7_reconfig_phase at its neutral midpoint (0.5) instead of the real wall-clock phase, "
             "for the Point-2 causal-isolation ablation study",
    )
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    n_episodes = args.n_episodes or config.get("episode", {}).get("target_episodes", [200])[0]

    # Seeded here, before the agent's weights are initialized: train()'s own
    # seeding only covers the rollout/update phase, which is too late to
    # make a run fully reproducible (agent construction already happened).
    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    calibration = load_calibration(args.calibration_path)
    env = FluidSimEnv(
        args.location,
        args.direction,
        calibration,
        risk_mode=args.risk_mode,
        episode_s=config.get("episode", {}).get("duration_s", 300.0),
        reward_kwargs=config.get("reward", {}),
        ablate_s7=args.ablate_s7,
    )
    agent = build_agent(args.core, config, args.reupload)

    run_config = {
        **config, "location": args.location, "direction": args.direction, "core": args.core,
        "ablate_s7": args.ablate_s7,
    }
    run_dir = start_run(run_config, args.out_dir)
    print(f"run -> {run_dir}")
    print(f"{args.core} agent, param_count={agent.param_count()}, {n_episodes} episodes")

    t0 = time.time()

    def on_episode(episode: int, summary: dict) -> None:
        print(
            f"  episode {episode + 1}/{n_episodes}  "
            f"reward={summary['episode_reward']:.2f}  "
            f"len={summary['episode_length']}  "
            f"loss={summary['loss']:.3f}"
        )

    result = train(agent, env, n_episodes, config, run_dir=run_dir, on_episode=on_episode)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s. final_mean_reward={result['final_mean_reward']:.3f}")
    print(f"episode log -> {run_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
