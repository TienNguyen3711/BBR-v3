from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

from qbbr.action.registry import dimension_sizes, is_multihead, load_action_space
from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
from qbbr.agents.quantum.qa2c import QA2CAgent
from qbbr.env.calibration import load_calibration
from qbbr.eval.scenario_a import run_scenario_a
from qbbr.eval.scenario_b import run_scenario_b

DEFAULT_CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
DEFAULT_DATASET_ROOT = PACKAGE_ROOT / "data" / "raw"
DEFAULT_ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"


def _action_dims(action_config: dict) -> tuple[int, ...]:
    if is_multihead(action_config):
        return tuple(dimension_sizes(action_config))
    return (len(action_config["levels"]),)


def build_agent(
    core: str, n_layers: int, reupload: bool, action_dims: tuple[int, ...], n_qubits: int
) -> QA2CAgent | MLPA2CAgent:
    if core == "quantum":
        return QA2CAgent(n_qubits=n_qubits, n_layers=n_layers, action_dims=action_dims, reupload=reupload)
    return MLPA2CAgent(n_qubits=n_qubits, n_layers=n_layers, action_dims=action_dims)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["a", "b"], required=True)
    parser.add_argument("--config", required=True, help="path to a qbbr/configs/eval_scenario*.yaml file")
    parser.add_argument("--checkpoint", required=True, type=Path, help="agent checkpoint from BaseAgent.save")
    parser.add_argument("--core", choices=["quantum", "classical"], required=True)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--reupload", action="store_true", help="data re-uploading (quantum core only)")
    parser.add_argument(
        "--ablate-s7", action="store_true",
        help="must match the --ablate-s7 setting the checkpoint was trained with (Point-2 ablation study)",
    )
    parser.add_argument("--location", required=True)
    parser.add_argument("--direction", required=True, choices=["downlink", "uplink"])
    parser.add_argument("--risk-mode", choices=["stub_constant", "empirical_proxy", "closed_form"],
                         default="closed_form")
    parser.add_argument("--episode-s", type=float, default=300.0)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                         help="Scenario A only: real traces for the stock-CCA comparison columns")
    parser.add_argument("--out", type=Path, default=None,
                         help="default: outputs/eval/scenario_<a|b>_<location>_<direction>.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    n_episodes = config.get("runs_per_config", 10)
    calibration = load_calibration(args.calibration_path)
    action_config = load_action_space(args.action_config)
    action_dims = _action_dims(action_config)
    # FluidSimEnv (scenario a) exposes 7 state features; MultiFlowFluidEnv
    # (scenario b) adds s8_fairness_ratio on top, so a scenario-b-trained
    # checkpoint needs one more input than a scenario-a one.
    n_qubits = 8 if args.scenario == "b" else 7

    agent = build_agent(args.core, args.n_layers, args.reupload, action_dims, n_qubits)
    agent.load(args.checkpoint)
    print(f"loaded {args.core} agent (n_layers={args.n_layers}, reupload={args.reupload}, "
          f"action_dims={action_dims}, n_qubits={n_qubits}) from {args.checkpoint}")

    if args.scenario == "a":
        comparison_ccas = tuple(c for c in config.get("ccas", []) if c != "qbbr")
        result = run_scenario_a(
            agent, args.location, args.direction, calibration, args.dataset_root,
            n_episodes=n_episodes, episode_s=args.episode_s, risk_mode=args.risk_mode,
            comparison_ccas=comparison_ccas, action_config=action_config, ablate_s7=args.ablate_s7,
        )
        for name, stats in result.items():
            print(f"  {name:8s} throughput={stats['throughput_mbps_median']:8.1f}Mbps  "
                  f"rtt={stats['rtt_ms_median']:7.1f}ms  rtx/s={stats['retransmits_per_s_median']:6.1f}")
    else:
        alpha_sweep = tuple(float("inf") if a in (".inf", "inf") else float(a)
                             for a in config.get("alpha_sweep", [0, 1, 2, ".inf"]))
        result = run_scenario_b(
            agent, args.location, args.direction, calibration,
            n_episodes=n_episodes, episode_s=args.episode_s, risk_mode=args.risk_mode,
            alpha_sweep=alpha_sweep,
        )
        for name, bps in zip(result["flow_names"], result["x_achieved_bps"]):
            print(f"  {name:8s} throughput={bps / 1e6:8.1f}Mbps")
        print(f"  rho_alpha={result['rho_alpha']}")
        print(f"  min_throughput_bps={result['min_throughput_bps']:.1f}")

    out_path = args.out or (
        PROJECT_ROOT / "outputs" / "eval" / f"scenario_{args.scenario}_{args.location}_{args.direction}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    try:
        shown_path = out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = out_path
    print(f"\nsaved -> {shown_path}")


if __name__ == "__main__":
    main()
