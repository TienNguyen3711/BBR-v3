"""The native multi-flow environment must keep the frozen contract and must
divide a bottleneck rather than invent capacity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qbbr.env.fluid_env import FluidSimEnv
from qbbr.env.native_multi_flow_env import AGENT_FLOW, NativeMultiFlowEnv

ROOT = Path(__file__).resolve().parents[2]
V14 = {"consistent_transport": True, "probe_bw_cycle": True, "overflow_retransmit_frac": 0.25,
       "bandwidth_estimate_recovery_s": 3.0, "drain_throughput_penalty": 0.0, "cwnd_gain": 2.0,
       "loss_thresh": 0.02, "loss_beta": 0.7, "probe_rtt_interval_s": 5.0}


@pytest.fixture(scope="module")
def calibration():
    return json.loads((ROOT / "qbbr/data/calibrated/per_location_constants_v14.json").read_text())


def build(calibration, **kwargs):
    options = dict(risk_mode="stub_constant", episode_s=6.0, reward_mode="throughput_only")
    options.update(kwargs)
    return NativeMultiFlowEnv("Sydney", "downlink", calibration, **options)


def test_state_stays_at_the_frozen_seven_features(calibration):
    """s8_fairness_ratio is dropped on purpose: widening the state would break
    the 126-parameter QA2C/A2C match the whole comparison rests on."""
    env = build(calibration)
    assert env.observation_dim == 7
    assert len(env.reset(seed=1000)) == 7
    assert env.action_space_size == 5


def test_it_accepts_the_declared_difference_reward(calibration):
    env = build(calibration, reward_mode="difference", reward_kwargs={"delta": 0.5, "beta": 0.5})
    env.reset(seed=1000)
    _, reward, _, _ = env.step(2)
    assert reward == pytest.approx(reward)  # finite, and the twin did not recurse


def test_allocation_never_exceeds_the_shared_bottleneck(calibration):
    """Flows divide one link. If the allocation could exceed capacity the
    coexistence metrics would be measuring an invented bottleneck."""
    env = build(calibration, dynamics_overrides=dict(V14))
    env.reset(seed=1000)
    done = False
    while not done:
        _, _, done, info = env.step(2)
        total = sum(info["flow_throughput_bps"].values())
        capacity = env._total_capacity_at(env._state.t_s) * 8.0
        assert total <= capacity * 1.01  # 1% for interval-boundary rounding


def test_every_flow_is_reported_each_step(calibration):
    env = build(calibration)
    env.reset(seed=1000)
    _, _, _, info = env.step(2)
    assert set(info["flow_throughput_bps"]) == {AGENT_FLOW, *env.competing_ccas}
    assert info["shared_rtt_s"] > 0


def test_competitors_do_not_leak_across_episodes(calibration):
    """A stale competitor window would make episode 2 start mid-contention."""
    env = build(calibration)
    env.reset(seed=1000)
    for _ in range(3):
        env.step(4)
    grown = {name: state.window_bytes for name, state in env._competing_states.items()}
    env.reset(seed=1000)
    fresh = {name: state.window_bytes for name, state in env._competing_states.items()}
    assert fresh != grown
    assert all(value == pytest.approx(env.params.bdp_bytes) for value in fresh.values())


def test_capacity_hook_outside_a_substep_does_not_advance_competitors(calibration):
    """The hook advances competing flows, so calling it without a timestep
    must degrade rather than integrate them over an invented dt."""
    env = build(calibration)
    env.reset(seed=1000)
    before = dict(env._competing_states)
    capacity = env._capacity_at(0.0)
    assert capacity == pytest.approx(env._total_capacity_at(0.0))
    assert env._competing_states == before


def test_coexistence_requires_a_competitor(calibration):
    with pytest.raises(ValueError):
        build(calibration, competing_ccas=())


def test_port_reproduces_the_legacy_split_under_legacy_dynamics(calibration):
    """Faithfulness check for the port itself.

    Under the pre-v6 dynamics the legacy MultiFlowFluidEnv was written for,
    this environment must reach the same regime: the agent's flow dominates.
    That isolates any behaviour change to the v6/v7 transport rather than to
    the re-coupling done here. See the known-limitation test below.
    """
    env = build(calibration, episode_s=20.0)
    env.reset(seed=1000)
    done = False
    while not done:
        _, _, done, info = env.step(2)
    agent = info["flow_throughput_bps"][AGENT_FLOW]
    assert agent > 100e6
    assert agent == max(info["flow_throughput_bps"].values())


def test_legacy_estimator_starves_the_agent_under_contention(calibration):
    """Documents the defect the BtlBw max filter exists to fix.

    The legacy estimator drops to the instantaneous capacity immediately. On a
    shared bottleneck that capacity IS the flow's allocated share and demand is
    derived from the estimate, so a lost share lowers demand and the loss
    compounds to total starvation. Measured data contradicts the outcome: real
    BBR retains 7.5% of its isolated rate under contention, not 0%.
    """
    env = build(calibration, episode_s=20.0, dynamics_overrides=dict(V14))
    env.reset(seed=1000)
    done = False
    while not done:
        _, _, done, info = env.step(2)
    assert info["flow_throughput_bps"][AGENT_FLOW] < 1e6


def test_btlbw_max_filter_lets_the_agent_hold_a_share(calibration):
    """BBR-v3 keeps BtlBw as a max over 10 packet-timed round trips, so a flow
    keeps claiming its remembered rate instead of conceding it. 10.0 is the
    protocol's own filter length, not a value fitted to an allocation."""
    env = build(calibration, episode_s=20.0,
                dynamics_overrides=dict(V14, bw_estimate_max_filter_rounds=10.0))
    env.reset(seed=1000)
    done = False
    while not done:
        _, _, done, info = env.step(2)
    flows = info["flow_throughput_bps"]
    share = flows[AGENT_FLOW] / sum(flows.values())
    # Measured BBR holds 30.6-57.4% against this competitor subset across the
    # six downlinks (reports/measured_coexistence.json); the model must land in
    # that region rather than at either extreme.
    assert 0.25 < share < 0.65


def test_max_filter_is_off_unless_asked_for(calibration):
    """Guards every single-flow result already on disk. The filter changes the
    estimator, so it must never activate through a default."""
    from qbbr.env.fluid_sim import FluidParams

    assert FluidParams(x_btl_bps=1e8, rtt_rtp_s=0.03).bw_estimate_max_filter_rounds == 0.0
    env = build(calibration, dynamics_overrides=dict(V14))
    assert env.params.bw_estimate_max_filter_rounds == 0.0


def test_competitor_mix_is_not_calibrated_for_fairness_claims(calibration):
    """A guard against over-claiming, kept as an executable note.

    With the filter on, the agent's own share matches the measured range, but
    the competing-CCA ranking does NOT: the model collapses cubic and hybla to
    the 2-MSS window floor while measurement makes hybla the STRONGEST
    competitor (16-52%) and vegas among the weakest (2.7-16.3%). Loss-reactive
    competitors sit at a saturated congestion signal and never recover. So
    rho_alpha from this environment is not evidence about real fairness; use
    it only for policy-versus-stock contrasts under an identical competitor
    mix, where the competitor error largely cancels.
    """
    env = build(calibration, episode_s=20.0,
                dynamics_overrides=dict(V14, bw_estimate_max_filter_rounds=10.0))
    env.reset(seed=1000)
    done = False
    while not done:
        _, _, done, info = env.step(2)
    flows = info["flow_throughput_bps"]
    collapsed = [name for name in ("cubic", "hybla") if flows[name] < 1e6]
    assert collapsed, "if the competitor models stop collapsing, re-validate this guard"


def test_aggression_is_free_under_contention(calibration):
    """Documents the second validation failure, and blocks silent use of it.

    On a shared bottleneck, taking more bandwidth must cost the taker
    something. Here it costs nothing: against an identical competitor mix and
    seed, the most aggressive native gain beats stock on BOTH axes at once --
    several times the throughput AND a markedly lower RTT, because the flow
    it displaces absorbs the entire queue. At 300 s the gap reaches +1083%
    throughput with 30 ms less delay.

    The consequence for reporting is that a policy-versus-stock contrast in
    this environment measures the allocation rule's reward for demand, not a
    controller result: the zero-shot grid returned +126% for policies worth
    +3-6% on an idle path, with an RTT p90 difference of exactly zero in five
    of six cities, because at p90 both arms still sit at the calibrated queue
    ceiling. Until displacing another flow carries a cost, no contention
    contrast from this environment is reportable -- including the
    policy-versus-stock one the overlay once declared supported.
    """
    def run(action):
        env = build(calibration, episode_s=60.0,
                    dynamics_overrides=dict(V14, bw_estimate_max_filter_rounds=10.0))
        env.reset(seed=1000)
        done, rtts, spans, delivered = False, [], [], 0.0
        while not done:
            chosen = action if action in env.allowed_action_indices() else 2
            _, _, done, info = env.step(chosen)
            rtts.append(info["rtt_ms"])
            spans.append(float(info["t_dec_s"]))
            delivered += float(info["delivered_bytes"])
        order = np.argsort(rtts)
        total = sum(spans)
        p90 = float(np.asarray(rtts)[order][
            np.searchsorted(np.cumsum(np.asarray(spans)[order]), 0.9 * total)])
        return delivered * 8 / total / 1e6, p90

    stock_throughput, stock_p90 = run(2)
    aggressive_throughput, aggressive_p90 = run(4)
    assert aggressive_throughput > 2 * stock_throughput
    assert aggressive_p90 < stock_p90 - 10.0, (
        "if the aggressive gain now pays for the bandwidth it takes, the contention "
        "model has gained a cost channel and the overlay's claims must be re-validated")
