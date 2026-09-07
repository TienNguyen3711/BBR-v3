from pathlib import Path

import numpy as np
import pytest
import yaml

from qbbr.control.contracts import ContractError, mdp_contract_from_mapping
from qbbr.control.native_actions import action_specs_from_mapping, validate_fixed_action_set
from qbbr.control.safety import enforce_action
from qbbr.control.space import NativeActionSpace, NativeObservationNormalizer, encode_observation


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_active_contract_loads_and_preserves_throughput_only_reward() -> None:
    with (CONFIG_DIR / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    assert contract.reward.throughput_utility == "delivered_throughput_only"
    assert contract.reward.rtt_penalty is False
    assert contract.reward.retransmission_penalty is False


def test_implicit_new_action_grid_is_rejected_from_native_inventory() -> None:
    with pytest.raises(ContractError, match="implicit action-set"):
        action_specs_from_mapping(
            {
                "actions": [
                    {
                        "action_id": "bad_gain",
                        "native_semantic": "invent a gain",
                        "allowed_bbr_states": ["PROBE_BW"],
                        "levels": [0.75, 1.0],
                    }
                ]
            }
        )


def test_fixed_action_contract_matches_existing_five_gain_choices() -> None:
    with (CONFIG_DIR / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    validate_fixed_action_set(contract.action_space, CONFIG_DIR / "action_pacing_gain.yaml")
    assert [action.parameters["pacing_gain"] for action in contract.action_space] == [
        0.75,
        0.9,
        1.0,
        1.1,
        1.25,
    ]


def test_fixed_baseline_falls_closed_before_kernel_audit() -> None:
    with (CONFIG_DIR / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    observed = {
        "s1_bhat": 1.0,
        "s2_rtt_ratio": 1.5,
        "s3_inflight_bdp": 1.0,
        "s4_queue": 0.1,
        "s5_handover_eta": 0.0,
        "s6_p_tot": 0.01,
        "s7_reconfig_phase": 0.0,
        "delivery_rate_bps": 1.0,
        "rtt_s": 0.03,
        "min_rtt_s": 0.02,
        "inflight_bytes": 10,
        "cwnd_bytes": 10,
        "bbr_state": "PROBE_BW",
        "retransmission_rate": 0.0,
    }
    decision = enforce_action(contract, "pacing_gain_100", "PROBE_BW", observed)
    assert decision.used_stock_fallback is True
    assert decision.applied_action == "pacing_gain_100"


def test_disabled_kernel_action_fails_closed() -> None:
    with (CONFIG_DIR / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    observed = {
        "s1_bhat": 1.0,
        "s2_rtt_ratio": 1.5,
        "s3_inflight_bdp": 1.0,
        "s4_queue": 0.1,
        "s5_handover_eta": 0.0,
        "s6_p_tot": 0.01,
        "s7_reconfig_phase": 0.0,
        "delivery_rate_bps": 1.0,
        "rtt_s": 0.03,
        "min_rtt_s": 0.02,
        "inflight_bytes": 10,
        "cwnd_bytes": 10,
        "bbr_state": "PROBE_BW",
        "retransmission_rate": 0.0,
    }
    decision = enforce_action(contract, "pacing_gain_075", "PROBE_BW", observed)
    assert decision.used_stock_fallback is True
    assert decision.reason == "action not kernel-enabled"


def test_native_rl_space_masks_unavailable_semantics() -> None:
    with (CONFIG_DIR / "native_rl_bbr_contract.yaml").open(encoding="utf-8") as handle:
        contract = mdp_contract_from_mapping(yaml.safe_load(handle))
    observed = {
        "s1_bhat": 1.0,
        "s2_rtt_ratio": 1.5,
        "s3_inflight_bdp": 1.0,
        "s4_queue": 0.1,
        "s5_handover_eta": 0.0,
        "s6_p_tot": 0.01,
        "s7_reconfig_phase": 0.0,
        "delivery_rate_bps": 1.0,
        "rtt_s": 0.03,
        "min_rtt_s": 0.02,
        "inflight_bytes": 10,
        "cwnd_bytes": 10,
        "bbr_state": "PROBE_BW",
        "retransmission_rate": 0.0,
    }
    space = NativeActionSpace(contract)
    allowed = space.allowed_indices("PROBE_BW", observed)
    assert allowed == (space.index_for("pacing_gain_100"),)
    masked = space.masked_logits([0.0, 10.0, 10.0, 10.0, 10.0], allowed)
    assert np.isfinite(masked[2])
    assert np.isneginf(np.delete(masked, 2)).all()
    encoded = encode_observation(contract, observed)
    assert encoded.shape == (7,)
    normalizer = NativeObservationNormalizer.fit(contract, [observed])
    normalized = normalizer.transform(contract, observed)
    assert normalized.shape == (7,)
    assert np.all(np.abs(normalized) <= np.pi)
