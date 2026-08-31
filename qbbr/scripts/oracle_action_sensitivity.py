from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.action.registry import load_action_space
from qbbr.env.calibration import load_calibration
from qbbr.eval.oracle import action_sensitivity

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DEFAULT_ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", nargs="+", default=None, help="default: all calibrated locations")
    parser.add_argument("--direction", choices=["downlink", "uplink"], default="downlink")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--episode-s", type=float, default=300.0)
    parser.add_argument("--risk-mode", choices=["stub_constant", "empirical_proxy", "closed_form", "closed_form_dynamic"], default="closed_form")
    parser.add_argument("--throughput-retention", type=float, default=0.95)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG_PATH)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "oracle_action_sensitivity.json")
    args = parser.parse_args()

    calibration = load_calibration(args.calibration_path)
    locations = args.locations or sorted(location for location, directions in calibration.items() if args.direction in directions)
    action_config = load_action_space(args.action_config)
    results = {}
    for location in locations:
        if location not in calibration or args.direction not in calibration[location]:
            raise ValueError(f"no calibration for {location}/{args.direction}")
        result = action_sensitivity(location, args.direction, calibration, action_config,
                                    seeds=args.seeds, episode_s=args.episode_s, risk_mode=args.risk_mode,
                                    throughput_retention=args.throughput_retention)
        results[location] = result
        bound = result["one_step_oracle_bound"]
        print(f"{location:12s} local oracle rtx reduction={bound['retransmit_reduction_vs_stock']:.1%} "
              f"delivery retention={bound['delivery_retention_vs_stock']:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
