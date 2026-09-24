from dataclasses import replace

import pytest

from qbbr.env.fluid_sim import FluidParams, FluidState, step_fluid_state
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.eval.constrained_selection import summarize_candidate, select_candidate
from qbbr.scripts.audit_sydney_semantics import simple_action, paired_rows


PARAMS = FluidParams(x_btl_bps=1e6, rtt_rtp_s=0.1,
    consistent_transport=True, probe_bw_cycle=True, native_cruise_override=True)


def test_native_mode_rejects_incompatible_dynamics():
    with pytest.raises(ValueError):
        FluidParams(x_btl_bps=1e6, rtt_rtp_s=.1, native_cruise_override=True)


def test_stock_probes_autonomously_and_returns_to_cruise():
    state = FluidState()
    seen = set()
    for _ in range(1500):
        state, _, _ = step_fluid_state(state, 1.0, .01, PARAMS, 0, capacity_bps_override=1e6)
        seen.add(state.probe_phase)
    assert seen == {0, 1, 2, 3}


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_learner_cannot_override_native_probe_phases(phase):
    state = FluidState(probe_phase=phase)
    low = step_fluid_state(state, .75, .01, PARAMS, 0, capacity_bps_override=1e6)
    high = step_fluid_state(state, 1.25, .01, PARAMS, 0, capacity_bps_override=1e6)
    assert low == high


@pytest.mark.parametrize("state", [FluidState(startup_done=False), FluidState(in_probe_rtt=True)])
def test_startup_and_probe_rtt_do_not_apply_learner_gain(state):
    params = replace(PARAMS, probe_rtt_interval_s=5)
    assert step_fluid_state(state, .75, .01, params, 0, capacity_bps_override=1e6) == \
           step_fluid_state(state, 1.25, .01, params, 0, capacity_bps_override=1e6)


def test_cruise_gain_changes_delivery_without_triggering_probe():
    state = FluidState(bbr_bw_est_bps=5e5)
    params = replace(PARAMS, bandwidth_estimate_recovery_s=3)
    low = step_fluid_state(state, .75, .01, params, 0, capacity_bps_override=1e6)
    high = step_fluid_state(state, 1.25, .01, params, 0, capacity_bps_override=1e6)
    assert low[0].probe_phase == high[0].probe_phase == 0
    assert high[1] > low[1]


def make_env():
    return FluidSimEnv("test", "downlink",
        {"test": {"downlink": {"B_max_mbps": 8., "RTT_min_ms": 30., "RTT_max_ms": 90.}}},
        reward_mode="throughput_only", risk_mode="stub_constant", episode_s=1,
        phase_offset_s=0., dynamics_overrides={"consistent_transport": True,
            "probe_bw_cycle": True, "native_cruise_override": True})


def test_environment_exact_gains_and_explicit_phase_mask():
    env = make_env()
    env.reset(0)
    env._state = replace(env._state, startup_done=True, i_crs=1.0)
    assert env.allowed_action_indices() == (0, 1, 2, 3, 4)
    _, _, _, info = env.step(4)
    assert info["selected_pacing_gain"] == 1.25  # no EMA-created 1.125
    env._state = replace(env._state, probe_phase=3, i_crs=1.0)
    assert env.allowed_action_indices() == (2,)
    _, _, _, info = env.step(4)  # even an unmasked caller must fall back
    assert info["selected_pacing_gain"] == 1.0


def test_legacy_stock_still_has_no_autonomous_cycle():
    params = replace(PARAMS, native_cruise_override=False)
    state = FluidState()
    for _ in range(1500):
        state, _, _ = step_fluid_state(state, 1, .01, params, 0, capacity_bps_override=1e6)
        assert state.probe_phase == 0


def row(run=6, throughput=1., rtt=0., rtx=0.):
    return dict(run=run, seed=-1, throughput_delta_pct=throughput,
                rtt_p90_delta_ms=rtt, retransmits_delta_per_s=rtx)


def test_constraints_reject_tradeoff_and_check_every_trace():
    summary = summarize_candidate([row(rtt=1), row(7, rtt=-10)], {(6, -1), (7, -1)})
    assert summary["medians"]["rtt_p90_delta_ms"] < 0
    assert not summary["eligible"]
    assert select_candidate({"unsafe": summary}) == "stock"
    safe = summarize_candidate([row(rtt=-1, rtx=-.1)], {(6, -1)})
    assert safe["strict_three_metric_improvement"]
    assert select_candidate({"unsafe": summary, "safe": safe}) == "safe"


def test_constraints_reject_missing_duplicate_and_nonfinite_pairs():
    expected = {(6, -1), (7, -1)}
    for rows in ([row()], [row(), row()], [row(), row(7, rtt=float("nan"))]):
        assert not summarize_candidate(rows, expected)["eligible"]


def test_selection_does_not_cherry_pick_the_best_training_seed():
    rows = [{**row(throughput=gain), "seed": seed}
            for seed, gain in enumerate([5., 0., 0., 0., 0.])]
    summary = summarize_candidate(rows, {(6, seed) for seed in range(5)})
    assert summary["medians"]["throughput_delta_pct"] == 0.
    assert select_candidate({"one_lucky_seed": summary}) == "stock"


def test_simple_baselines_obey_mask_and_balanced_policy_drains():
    assert simple_action("highest_permitted", (0, 1, 2)) == 2
    assert simple_action("gain110", (2, 3, 4)) == 3
    assert simple_action("balanced125", (0, 1, 2)) == 1
    assert simple_action("balanced110", (2,)) == 2


def test_pairing_uses_each_semantics_own_stock():
    stock = dict(policy="stock", run=6, throughput_mbps=100, rtt_p90_ms=40, retransmits_per_s=5)
    rows = [{**stock, "semantics": "old"}, {**stock, "semantics": "new", "throughput_mbps": 110},
            {**stock, "policy": "agent", "semantics": "new", "throughput_mbps": 110}]
    assert paired_rows(rows)[-1]["throughput_delta_pct"] == 0.
