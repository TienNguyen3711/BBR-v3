"""Mechanism diagnosis (Step 2, mechanism half): how does activating the
reward's ECN-level term (gamma=5.0) actually reshape the trained policy,
compared to the published gamma=0.0 checkpoints? Rolls out both checkpoint
generations under the SAME single-head pacing_gain action space and
records the discrete pacing_gain INDEX chosen at every decision (index
0..4 -> levels [0.75, 0.9, 1.0, 1.1, 1.25], ascending), plus the resulting
RTT/retransmit profile -- same style as analyze_ecn_behavior_shift.py, but
comparing gamma=0.0 vs. gamma=5.0 checkpoints (both trained on the CURRENT,
post-merge single-head action space -- no state-truncation workaround
needed, unlike the older exploit-era checkpoints that script covers).

Output -> outputs/gamma5_mechanism_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_pacing_gain.yaml"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 5  # qualitative behavioral analysis, not the full statistical protocol
N_EPISODES = 5
EPISODE_S = 300.0
RISK_MODE = "closed_form"
DIRECTION = "downlink"
ARMS = {
    "gamma0": PROJECT_ROOT / "outputs" / "checkpoints_final" / "pacing_only" / "classical",
    "gamma5": PROJECT_ROOT / "outputs" / "checkpoints_gamma5" / "pacing_only" / "classical",
}
OUT_PATH = PROJECT_ROOT / "outputs" / "gamma5_mechanism_report.json"


def _rollout_one(checkpoint: Path, location: str, calibration, action_config, legacy_6dim: bool) -> dict[str, Any]:
    import numpy as np

    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.fluid_env import FluidSimEnv

    # checkpoints_final/pacing_only (gamma0, published) predates s7_reconfig_phase
    # (state grew 6->7); gamma5 (trained this session) is current-gen and needs
    # no truncation. Same pattern as eval_rq1_raw_for_boxplot.py.
    n_qubits = 6 if legacy_6dim else 7
    agent = MLPA2CAgent(n_layers=2, n_qubits=n_qubits, action_dims=(5,))
    agent.load(str(checkpoint))

    index_counts: Counter = Counter()
    level_sum = 0.0
    rtt_all, rtx_per_s_all = [], []
    levels = action_config["levels"]
    for _ep in range(N_EPISODES):
        env = FluidSimEnv(
            location, DIRECTION, calibration, action_config=action_config,
            episode_s=EPISODE_S, risk_mode=RISK_MODE,
        )
        state = env.reset()
        if legacy_6dim:
            state = state[:6]
        done = False
        while not done:
            action, _log_prob = agent.act(state)
            index_counts[action] += 1
            level_sum += levels[action]
            state, _reward, done, info = env.step(action)
            if legacy_6dim:
                state = state[:6]
            rtt_all.append(info["rtt_ms"])
            rtx_per_s_all.append(info["retransmits"] / info["t_dec_s"])

    n_steps = sum(index_counts.values())
    return {
        "index_fractions": {str(idx): count / n_steps for idx, count in index_counts.items()},
        "mean_index": sum(idx * count for idx, count in index_counts.items()) / n_steps,
        "mean_pacing_gain_level": level_sum / n_steps,
        "rtt_ms_median": float(np.median(rtt_all)),
        "retransmits_per_s_median": float(np.median(rtx_per_s_all)),
        "n_steps": n_steps,
    }


def main() -> None:
    from qbbr.action.registry import load_action_space
    from qbbr.env.calibration import load_calibration

    calibration = load_calibration(CALIBRATION_PATH)
    action_config = load_action_space(ACTION_CONFIG_PATH)
    levels = action_config["levels"]
    print(f"pacing_gain levels: {levels} (index 0..{len(levels)-1})")

    report: dict[str, Any] = {}
    for arm, root in ARMS.items():
        legacy_6dim = arm == "gamma0"
        print(f"\n=== arm: {arm} (legacy_6dim={legacy_6dim}) ===")
        per_location = {}
        for location in LOCATIONS:
            agg_counts: Counter = Counter()
            rtts, rtxs = [], []
            n = 0
            for seed in range(N_SEEDS):
                checkpoint = root / location / f"seed{seed}.pt"
                if not checkpoint.exists():
                    continue
                r = _rollout_one(checkpoint, location, calibration, action_config, legacy_6dim)
                for idx_str, frac in r["index_fractions"].items():
                    agg_counts[idx_str] += frac
                rtts.append(r["rtt_ms_median"])
                rtxs.append(r["retransmits_per_s_median"])
                n += 1
            total = sum(agg_counts.values())
            mean_index = sum(int(idx_str) * frac for idx_str, frac in agg_counts.items()) / total
            mean_level = sum(levels[int(idx_str)] * frac for idx_str, frac in agg_counts.items()) / total
            per_location[location] = {
                "n_seeds": n,
                "mean_index": mean_index,
                "mean_pacing_gain_level": mean_level,
                "index_fractions": {k: v / n for k, v in agg_counts.items()},
                "rtt_ms_median": sum(rtts) / n,
                "retransmits_per_s_median": sum(rtxs) / n,
            }
            print(f"  {location:10s}  mean_index={mean_index:.2f}  mean_pacing_gain={mean_level:.3f}  "
                  f"rtt={sum(rtts)/n:6.1f}ms  rtx/s={sum(rtxs)/n:6.2f}")
        report[arm] = per_location

    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nsaved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
