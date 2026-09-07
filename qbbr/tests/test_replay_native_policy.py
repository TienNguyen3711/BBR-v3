import json
from pathlib import Path

from qbbr.control.native_action_selector import NativeActionSelector
from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.native_qa2c_matched import build_matched_native_a2c_agents
from qbbr.scripts.replay_native_policy import replay

_CAL = (
    Path(__file__).resolve().parent.parent / "data" / "calibrated" / "per_location_constants.json"
)


def _tiny_replay(deterministic: bool) -> dict:
    selector = NativeActionSelector()
    quantum, _classical, _match = build_matched_native_a2c_agents(
        observation_dim=7, action_count=5, n_layers=2, selector=None, entropy_coef=0.005,
    )
    env = FluidSimEnv(
        "London", "downlink", load_calibration(_CAL), risk_mode="stub_constant",
        episode_s=5.0, reward_mode="throughput_only",
    )
    return replay(quantum, selector, env, seed=1000, deterministic=deterministic)


def test_replay_reports_decision_cadence_and_action_shares():
    result = _tiny_replay(deterministic=True)
    assert result["decisions"] > 0
    assert result["probe_bw_phase_restriction"] is False
    assert abs(sum(result["action_shares"].values()) - 1.0) < 1e-6
    assert result["effective_decision_hz"] > 0.0
    # every logged row carries the canonical seven-state and a native action
    row = result["rows"][0]
    assert set(row["state"]) == {
        "s1_bhat", "s2_rtt_ratio", "s3_inflight_bdp", "s4_queue",
        "s5_handover_eta", "s6_p_tot", "s7_reconfig_phase",
    }
    assert 0 <= row["action"] <= 4


def test_replay_stratifies_high_gain_by_congestion_state():
    result = _tiny_replay(deterministic=False)
    strat = result["stratification"]
    assert set(strat) == {"s2_rtt_ratio", "s3_inflight_bdp", "s4_queue", "s7_reconfig_phase"}
    for entry in strat.values():
        assert "high_gain_share_by_bucket" in entry
    # result must be JSON-serialisable for the --out artifact
    json.dumps(result)
