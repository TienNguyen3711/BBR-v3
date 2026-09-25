"""
Deep-dive analysis: how did closing the inflight_hi/lo reward exploit actually reshape the trained
policy's behavior?
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../qbbr
PROJECT_ROOT = PACKAGE_ROOT.parent  # .../Codebase
sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants.json"
ACTION_CONFIG_PATH = PACKAGE_ROOT / "configs" / "action_multihead.yaml"
LOCATIONS = ["London", "Mumbai", "Ohio", "SaoPaulo", "Sydney", "Tokyo"]
N_SEEDS = 10
N_EPISODES = 5
EPISODE_S = 300.0
ACTION_DIMS = (5, 5, 5)  # pacing_gain, inflight_hi_mult, inflight_lo_mult
HEAD_NAMES = ["pacing_gain", "inflight_hi_mult", "inflight_lo_mult"]
ARMS = {
    "exploit": PROJECT_ROOT / "outputs" / "checkpoints_multihead_symmetric_exploit",
    "fixed": PROJECT_ROOT / "outputs" / "checkpoints_multihead",
}
OUT_PATH = PROJECT_ROOT / "outputs" / "ecn_behavior_shift_report.json"


def _reconstruct_symmetric_exploit_config(current_config: dict) -> dict:
    """
    Deep-copy the current (fixed) action config and swap in the old symmetric inflight_hi/lo
    ranges the "exploit" checkpoints were actually trained under (see module docstring -- read
    back from action_multihead .yaml's own comment, not guessed). pacing_gain levels, hooks, and
    clamp_state are untouched since only inflight_hi/lo's range changed between the two
    checkpoint generations.
    """
    import copy

    cfg = copy.deepcopy(current_config)
    cfg["dimensions"]["inflight_hi_mult"]["levels"] = [1.5, 1.75, 2.0, 2.25, 2.5]
    cfg["dimensions"]["inflight_lo_mult"]["levels"] = [0.75, 0.875, 1.0, 1.125, 1.25]
    return cfg


def _rollout_one(checkpoint: Path, location: str, calibration, action_config) -> dict[str, Any]:
    import numpy as np

    from qbbr.action.registry import decode_flat_action
    from qbbr.agents.classical.mlp_a2c import MLPA2CAgent
    from qbbr.env.fluid_env import FluidSimEnv

    agent = MLPA2CAgent(n_layers=2, n_qubits=6, action_dims=ACTION_DIMS)
    agent.load(str(checkpoint))

    head_index_counts = [Counter() for _ in HEAD_NAMES]
    bps_all, rtt_all, rtx_per_s_all = [], [], []
    for _ep in range(N_EPISODES):
        env = FluidSimEnv(
            location, "downlink", calibration, action_config=action_config,
            episode_s=EPISODE_S, risk_mode="closed_form",
        )
        state = env.reset()[:6]
        done = False
        while not done:
            action, _log_prob = agent.act(state)
            indices = decode_flat_action(action, list(ACTION_DIMS))
            for head_i, idx in enumerate(indices):
                head_index_counts[head_i][idx] += 1
            state, _reward, done, info = env.step(action)
            state = state[:6]
            bps_all.append(info["delivered_bytes"] * 8.0 / info["t_dec_s"])
            rtt_all.append(info["rtt_ms"])
            rtx_per_s_all.append(info["retransmits"] / info["t_dec_s"])

    n_steps = sum(head_index_counts[0].values())
    return {
        "head_index_fractions": [
            {str(idx): count / n_steps for idx, count in counter.items()} for counter in head_index_counts
        ],
        "head_mean_index": [
            sum(idx * count for idx, count in counter.items()) / n_steps for counter in head_index_counts
        ],
        "throughput_mbps_median": float(np.median(bps_all)) / 1e6,
        "rtt_ms_median": float(np.median(rtt_all)),
        "retransmits_per_s_median": float(np.median(rtx_per_s_all)),
        "n_steps": n_steps,
    }


def main() -> None:
    from qbbr.action.registry import load_action_space
    from qbbr.env.calibration import load_calibration

    calibration = load_calibration(CALIBRATION_PATH)
    fixed_config = load_action_space(ACTION_CONFIG_PATH)
    exploit_config = _reconstruct_symmetric_exploit_config(fixed_config)
    arm_configs = {"exploit": exploit_config, "fixed": fixed_config}
    # each arm's own default level, in physical units, for the
    # distance-from-default comparison below (index 2 / index 0 respectively)
    arm_defaults = {
        "exploit": {"inflight_hi_mult": 2.0, "inflight_lo_mult": 1.0},
        "fixed": {"inflight_hi_mult": 2.0, "inflight_lo_mult": 1.0},
    }

    report: dict[str, Any] = {}
    for arm, root in ARMS.items():
        action_config = arm_configs[arm]
        print(f"\n=== arm: {arm} ({root.name}) -- own action space, "
              f"inflight_hi levels={action_config['dimensions']['inflight_hi_mult']['levels']} ===")
        per_location = {}
        for location in LOCATIONS:
            agg_counts = [Counter() for _ in HEAD_NAMES]
            tputs, rtts, rtxs = [], [], []
            for seed in range(min(N_SEEDS, 5)):
                checkpoint = root / "classical" / location / f"seed{seed}.pt"
                if not checkpoint.exists():
                    continue
                r = _rollout_one(checkpoint, location, calibration, action_config)
                for head_i in range(len(HEAD_NAMES)):
                    for idx_str, frac in r["head_index_fractions"][head_i].items():
                        agg_counts[head_i][idx_str] += frac
                tputs.append(r["throughput_mbps_median"])
                rtts.append(r["rtt_ms_median"])
                rtxs.append(r["retransmits_per_s_median"])
            n = len(tputs)
            if n == 0:
                print(f"  {location:10s}  (no checkpoints found under {root})")
                continue
            mean_index = []
            mean_level = []
            for head_i, head_name in enumerate(HEAD_NAMES):
                total = sum(agg_counts[head_i].values())
                mi = sum(int(idx_str) * frac for idx_str, frac in agg_counts[head_i].items()) / total
                mean_index.append(mi)
                levels = action_config["dimensions"][head_name]["levels"]
                ml = sum(levels[int(idx_str)] * frac for idx_str, frac in agg_counts[head_i].items()) / total
                mean_level.append(ml)
            level_vs_default = {}
            for head_name, default_level in arm_defaults[arm].items():
                head_i = HEAD_NAMES.index(head_name)
                level_vs_default[head_name] = mean_level[head_i] - default_level
            per_location[location] = {
                "n_seeds": n,
                "mean_index": dict(zip(HEAD_NAMES, mean_index)),
                "mean_level": dict(zip(HEAD_NAMES, mean_level)),
                "mean_level_minus_default": level_vs_default,
                "throughput_mbps_median": sum(tputs) / n,
                "rtt_ms_median": sum(rtts) / n,
                "retransmits_per_s_median": sum(rtxs) / n,
            }
            print(f"  {location:10s}  mean_level: inflight_hi={mean_level[1]:.3f} "
                  f"({level_vs_default['inflight_hi_mult']:+.3f} vs default)  "
                  f"inflight_lo={mean_level[2]:.3f} ({level_vs_default['inflight_lo_mult']:+.3f} vs default)  "
                  f"| tput={sum(tputs)/n:6.1f}Mbps  rtt={sum(rtts)/n:6.1f}ms  rtx/s={sum(rtxs)/n:5.2f}")
        report[arm] = per_location

    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nsaved -> {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print("\nEach arm was replayed under its OWN native action space (v2 fix -- see module "
          "docstring): 'exploit' under the reconstructed symmetric [1.5..2.5]/[0.75..1.25] "
          "range it was actually trained on, 'fixed' under the current expansion-only range. "
          "mean_level_minus_default is directly comparable across arms in physical units "
          "(multiples of B_bar_DP): negative = shrunk below the shared 2.0/1.0 default, "
          "positive = expanded above it. 'fixed' cannot go negative by construction (its range "
          "is floored at the default); if 'exploit' comes out reliably negative where 'fixed' "
          "sits at/above 0, that is the exploit's shrink-toward-lower-RTT signature actually "
          "showing up, not an artifact of a shared action space.")


if __name__ == "__main__":
    main()
