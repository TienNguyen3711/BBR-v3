from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.action.registry import load_action_space
from qbbr.env.calibration import load_calibration
from qbbr.eval.ablation import ablation_grid, run_ablation, save_ablation_results

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "ablation"
DEFAULT_ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"


def _parse_bool_list(raw: str) -> list[bool]:
    return [{"true": True, "false": False}[v.strip().lower()] for v in raw.split(",")]


def _parse_float_list(raw: str) -> list[float]:
    return [float(v.strip()) for v in raw.split(",")]


def _parse_int_list(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",")]


def _parse_str_list(raw: str) -> list[str]:
    return [v.strip() for v in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a qbbr/configs/*.yaml file")
    parser.add_argument("--location", required=True)
    parser.add_argument("--direction", required=True, choices=["downlink", "uplink"])
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=None, help="default: config's episode.target_episodes[0]")
    parser.add_argument("--episode-s", type=float, default=None, help="override config's episode.duration_s")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--alpha", type=_parse_float_list, default=None, help="e.g. 1 or 0.5,1,2")
    parser.add_argument("--n-layers", type=_parse_int_list, default=None, help="e.g. 2 or 2,3")
    parser.add_argument("--reupload", type=_parse_bool_list, default=None, help="e.g. false or false,true")
    parser.add_argument("--risk-features", type=_parse_bool_list, default=None, help="e.g. false or false,true")
    parser.add_argument("--core", type=_parse_str_list, default=None, help="e.g. quantum or quantum,classical")
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/ablation/<timestamp>.json")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=None,
        help="if set, save each grid point's trained agents under <root>/<point>/seed<N>.pt "
             "for later Scenario A/B evaluation",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.episode_s is not None:
        config.setdefault("episode", {})["duration_s"] = args.episode_s
    if args.alpha is not None:
        config["ablation"]["alpha"] = args.alpha
    if args.n_layers is not None:
        config["ablation"]["n_layers"] = args.n_layers
    if args.reupload is not None:
        config["ablation"]["data_reuploading"] = args.reupload
    if args.risk_features is not None:
        config["ablation"]["risk_features"] = args.risk_features
    if args.core is not None:
        config["ablation"]["core"] = args.core

    n_episodes = args.n_episodes or config.get("episode", {}).get("target_episodes", [200])[0]
    grid = list(ablation_grid(config))
    print(f"grid: {len(grid)} point(s), {args.n_runs} run(s) each, {n_episodes} episode(s) each")
    for point in grid:
        print(f"  {point}")

    calibration = load_calibration(args.calibration_path)
    action_config = load_action_space(args.action_config)

    t0 = time.time()

    def on_point(i: int, total: int, result: dict) -> None:
        elapsed = time.time() - t0
        s = result["summary"]
        print(
            f"[{i + 1}/{total}] {elapsed:6.1f}s elapsed  {result['config']}  "
            f"median={s['median']:.3f} iqr={s['iqr']:.3f}"
        )

    results = run_ablation(
        config, args.location, args.direction, calibration,
        n_episodes=n_episodes, n_runs=args.n_runs, base_seed=args.base_seed, on_point=on_point,
        checkpoint_root=args.checkpoint_root, action_config=action_config,
    )

    out_path = args.out or (DEFAULT_OUT_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}.json")
    meta = {
        "location": args.location, "direction": args.direction,
        "n_runs": args.n_runs, "n_episodes": n_episodes,
        "episode_s": config.get("episode", {}).get("duration_s"),
        "elapsed_s": time.time() - t0,
    }
    save_ablation_results(results, out_path, meta=meta)
    print(f"\ndone in {meta['elapsed_s']:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
