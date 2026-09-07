import torch

from qbbr.control.native_action_selector import NativeActionSelector


def test_selector_folds_unconfident_action_mass_into_existing_stock_gain():
    selector = NativeActionSelector(min_logit_advantage=0.20)
    logits = torch.tensor([0.0, 0.0, 0.1, 0.15, 0.05])
    state = torch.zeros(7)
    distribution = selector.distribution(logits, state, (0, 1, 2, 3, 4))
    assert distribution.probs[2].item() > 0.5
    assert torch.count_nonzero(distribution.probs).item() == 5


def test_selector_blocks_high_native_gains_under_queue_or_inflight_pressure():
    selector = NativeActionSelector(min_logit_advantage=0.0)
    logits = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    state = torch.tensor([0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0])
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (0, 1, 2)


def test_selector_retains_confident_native_high_gain_when_path_has_headroom():
    selector = NativeActionSelector(min_logit_advantage=0.10)
    logits = torch.tensor([0.0, 0.0, 0.0, 0.2, 2.0])
    state = torch.zeros(7)
    distribution = selector.distribution(logits, state, (0, 1, 2, 3, 4))
    assert distribution.probs[4] > distribution.probs[2]


def test_hard_training_distribution_does_not_fold_nonstock_mass_into_stock():
    selector = NativeActionSelector(min_logit_advantage=0.0)
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.7])
    state = torch.zeros(7)
    learning = selector.hard_distribution(logits, state, (0, 1, 2, 3, 4))
    legacy_soft = selector.distribution(logits, state, (0, 1, 2, 3, 4))
    assert torch.argmax(learning.probs).item() == 4
    assert torch.argmax(legacy_soft.probs).item() == 2


def test_deterministic_deployment_uses_safe_raw_logit_ranking_not_folded_argmax():
    selector = NativeActionSelector(min_logit_advantage=0.0)
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.7])
    state = torch.zeros(7)
    assert selector.deployment_action(logits, state, (0, 1, 2, 3, 4)) == 4


def test_rtt_and_reconfig_gates_are_disabled_by_default():
    selector = NativeActionSelector(min_logit_advantage=0.0)
    logits = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    # High s2 (excess RTT) and high s7 (near reconfiguration phase) but the
    # default thresholds are 1.0, so the high gains stay admissible.
    state = torch.tensor([0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.99])
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (0, 1, 2, 3, 4)


def test_excess_rtt_gate_blocks_high_native_gains_when_opted_in():
    selector = NativeActionSelector(min_logit_advantage=0.0, max_excess_rtt_state=0.35)
    logits = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    state = torch.tensor([0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (0, 1, 2)


def test_reconfig_proximity_gate_blocks_high_native_gains_when_opted_in():
    selector = NativeActionSelector(min_logit_advantage=0.0, max_reconfig_phase_proximity=0.85)
    logits = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90])
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (0, 1, 2)


def test_selector_rejects_out_of_range_threshold():
    import pytest

    with pytest.raises(ValueError):
        NativeActionSelector(max_excess_rtt_state=1.5)


def test_low_gain_guard_blocks_sub_stock_gains_when_the_path_has_headroom():
    selector = NativeActionSelector(min_logit_advantage=0.0, low_gain_actions=(0, 1))
    logits = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])
    state = torch.zeros(7)  # no inflight/BDP or queue pressure
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (2, 3, 4)


def test_low_gain_guard_admits_sub_stock_gains_under_queue_pressure():
    selector = NativeActionSelector(min_logit_advantage=0.0, low_gain_actions=(0, 1))
    logits = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])
    state = torch.tensor([0.0, 0.0, 0.0, 0.50, 0.0, 0.0, 0.0])  # s4 queue above 0.20
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (0, 1, 2)


def test_low_gain_guard_is_disabled_by_default_and_preserves_legacy_admissibility():
    selector = NativeActionSelector(min_logit_advantage=0.0)
    logits = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])
    state = torch.zeros(7)
    assert selector.admissible_actions(logits, state, (0, 1, 2, 3, 4)) == (0, 1, 2, 3, 4)


def test_stock_action_cannot_itself_be_a_guarded_low_gain():
    import pytest

    with pytest.raises(ValueError):
        NativeActionSelector(low_gain_actions=(2,))
