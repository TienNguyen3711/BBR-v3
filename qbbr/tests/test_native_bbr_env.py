from pathlib import Path

import numpy as np
import yaml

from qbbr.control.contracts import mdp_contract_from_mapping
from qbbr.env.native_bbr_env import NativeBBRControlEnv


class FakeNativeAdapter:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.advanced_s = 0.0
        self._samples = [
            {
                "s1_bhat": 1.0,
                "s2_rtt_ratio": 2.0,
                "s3_inflight_bdp": 1.0,
                "s4_queue": 0.1,
                "s5_handover_eta": 0.0,
                "s6_p_tot": 0.01,
                "s7_reconfig_phase": 0.0,
                "delivery_rate_bps": 10_000_000.0,
                "rtt_s": 0.04,
                "min_rtt_s": 0.02,
                "inflight_bytes": 10_000,
                "cwnd_bytes": 12_000,
                "bbr_state": "PROBE_BW",
                "retransmission_rate": 0.0,
            },
            {
                "s1_bhat": 1.2,
                "s2_rtt_ratio": 1.5,
                "s3_inflight_bdp": 1.1,
                "s4_queue": 0.2,
                "s5_handover_eta": 0.0,
                "s6_p_tot": 0.01,
                "s7_reconfig_phase": 0.0,
                "delivery_rate_bps": 12_000_000.0,
                "rtt_s": 0.03,
                "min_rtt_s": 0.02,
                "inflight_bytes": 11_000,
                "cwnd_bytes": 12_000,
                "bbr_state": "PROBE_BW",
                "retransmission_rate": 0.01,
            },
        ]
        self._index = 0

    def snapshot(self):
        return self._samples[self._index]

    def apply_native_action(self, action) -> None:
        self.actions.append(action.action_id)

    def advance(self, decision_interval_s: float) -> None:
        self.advanced_s += decision_interval_s
        self._index = min(self._index + 1, len(self._samples) - 1)


def test_native_environment_applies_only_safe_contract_action() -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "native_rl_bbr_contract.yaml"
    with config.open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    adapter = FakeNativeAdapter()
    env = NativeBBRControlEnv(adapter, contract, decision_interval_s=1.0, episode_s=1.0)
    state = env.reset()
    assert state.shape == (7,)
    assert env.allowed_action_indices() == (2,)
    next_state, reward, done, info = env.step(2)
    assert next_state.shape == (7,)
    assert np.isfinite(reward)
    assert done is True
    assert adapter.actions == ["pacing_gain_100"]
    assert info["applied_fixed_parameters"] == {"pacing_gain": 1.0}
