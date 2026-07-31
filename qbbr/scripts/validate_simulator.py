from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.metrics import real_cca_distribution_stats

_STOCK_ACTION_INDEX = 2  # configs/action_pacing_gain.yaml: levels[2] == 1.0
DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"


def real_stock_bbr_stats(dataset_root: Path, location: str, direction: str) -> dict[str, float]:
    return real_cca_distribution_stats(dataset_root, location, direction, cca="bbr")


def simulated_stock_bbr_stats(
    calibration: dict, location: str, direction: str, n_episodes: int, episode_s: float
) -> dict[str, float]:
    bps_all, rtt_all, rtx_per_s_all = [], [], []
    for _ep in range(n_episodes):
        env = FluidSimEnv(location, direction, calibration, episode_s=episode_s)
        env.reset()
        done = False
        while not done:
            _s, _r, done, info = env.step(_STOCK_ACTION_INDEX)
            bps_all.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            rtt_all.append(info["rtt_ms"])
            rtx_per_s_all.append(info["retransmits"] / info["t_dec_s"])
    bps = pd.Series(bps_all)
    rtt = pd.Series(rtt_all)
    rtx = pd.Series(rtx_per_s_all)
    return {
        "throughput_mbps_median": float(bps.median() / 1e6),
        "throughput_mbps_iqr": float(bps.quantile(0.75) / 1e6 - bps.quantile(0.25) / 1e6),
        "rtt_ms_median": float(rtt.median()),
        "rtt_ms_iqr": float(rtt.quantile(0.75) - rtt.quantile(0.25)),
        "retransmits_per_s_median": float(rtx.median()),
        "retransmits_per_s_iqr": float(rtx.quantile(0.75) - rtx.quantile(0.25)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--episodes", type=int, default=3, help="simulated episodes per location/direction")
    parser.add_argument("--episode-s", type=float, default=300.0)
    args = parser.parse_args()

    calibration = load_calibration(args.calibration_path)

    rows = []
    for location, directions in calibration.items():
        for direction in directions:
            real = real_stock_bbr_stats(args.dataset_root, location, direction)
            sim = simulated_stock_bbr_stats(calibration, location, direction, args.episodes, args.episode_s)
            row = {"location": location, "direction": direction}
            for key in real:
                row[f"real_{key}"] = real[key]
                row[f"sim_{key}"] = sim[key]
            rows.append(row)
            print(f"{location:10s} {direction:9s}  "
                  f"throughput Mbps: real={real['throughput_mbps_median']:8.1f}  sim={sim['throughput_mbps_median']:8.1f}  |  "
                  f"RTT ms: real={real['rtt_ms_median']:7.1f}  sim={sim['rtt_ms_median']:7.1f}  |  "
                  f"rtx/s: real={real['retransmits_per_s_median']:6.1f}  sim={sim['retransmits_per_s_median']:6.1f}")

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "outputs" / "simulator_validation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path.relative_to(PROJECT_ROOT)}")

    print("\nNOTE: this is main.tex's required pre-training validation report, not a pass/fail gate.")
    print("Large gaps here mean the fluid-model parameters/dynamics need further calibration")
    print("before any training run on this simulator should be trusted.")


if __name__ == "__main__":
    main()
