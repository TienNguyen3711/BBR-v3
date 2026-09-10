from dataclasses import replace
import numpy as np
import pytest
from qbbr.env.fluid_sim import FluidParams, FluidState, step_fluid_state

P = FluidParams(x_btl_bps=1e6, rtt_rtp_s=.1, cwnd_gain=2, consistent_transport=True)


def test_reduced_cap_drains_existing_backlog():
    state=FluidState(v_bytes=P.bdp_bytes,inflight_hi_scale=.25)
    nxt, delivered, _=step_fluid_state(state,1.25,.01,P,0,capacity_bps_override=1e6)
    assert nxt.v_bytes < state.v_bytes
    assert delivered <= .5 * 1e6 * .01


def test_startup_plateau_exits_before_timeout_with_cap():
    state=FluidState(startup_done=False)
    for _ in range(200):
        state,_,_=step_fluid_state(state,1,.01,P,0,capacity_bps_override=1e6)
    assert state.startup_done


def test_loss_response_waits_for_round_and_is_timestep_stable():
    params=replace(P,loss_thresh=.02,base_retransmit_rate_pps=100,loss_beta=.7)
    scales=[]
    for dt in (.01,.005):
        state=FluidState()
        for _ in range(round(.1/dt)):
            state,_,_=step_fluid_state(state,1,dt,params,0,capacity_bps_override=1e6)
        scales.append(state.inflight_hi_scale)
    assert scales == pytest.approx([.7,.7])


def test_retransmission_consumes_wire_capacity():
    params=replace(P,base_retransmit_rate_pps=100)
    state=FluidState(v_bytes=P.bdp_bytes)
    _,goodput,retrans=step_fluid_state(state,1,.01,params,0,capacity_bps_override=1e6)
    assert goodput + retrans * 1500 <= 1e6*.01 + 1e-9
    assert goodput < 1e6*.01


def test_queue_telemetry_uses_backlog_and_current_service():
    from qbbr.env.fluid_env import FluidSimEnv
    env=object.__new__(FluidSimEnv)
    env.params=P
    state=FluidState(v_bytes=10000,pipe_bytes=30000,service_rate_bytes_s=500000)
    row=env._telemetry_row(0,state,1000,0,.1)
    assert row['rtt_ms'] == pytest.approx(120)
    assert row['q_packets'] == pytest.approx(10000/1500)
    assert row['v_over_bdp'] == pytest.approx(.4)


def test_safety_metric_gate_rejects_throughput_winner_and_missing_metrics():
    from qbbr.eval.successor_protocol import assess_full_successor_records
    criteria=dict(min_median_throughput_delta_vs_stock_pct=0,min_bootstrap_ci_lower_pct=0,
        min_positive_training_seed_fraction=.7,max_low_gain_action_share=.1,
        max_mean_action_js_divergence=.2,max_rtt_p95_delta_vs_stock_ms=0,
        max_retransmission_ratio_delta_vs_stock=0)
    evaluation=dict(throughput_delta_vs_stock_pct=5,retransmits_delta_vs_stock_per_s=0,
        action_shares={str(i):float(i==2) for i in range(5)},rtt_p95_delta_vs_stock_ms=1)
    rows=[dict(location='London',direction='downlink',core='test',evaluation=evaluation) for _ in range(2)]
    result=assess_full_successor_records(rows,criteria,dict(resamples=10,confidence=.95,seed=1))[0]
    assert not result['qualified_simulator_proxy_result']
    assert not result['criteria_pass']['rtt_p95_delta_vs_stock_ms']
    assert not result['criteria_pass']['retransmission_ratio_delta_vs_stock']


def test_full_and_tier1_share_transport_and_selector():
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]/'configs'
    tier=yaml.safe_load((root/'tier1_native_qa2c_successor_protocol.yaml').read_text())
    full=yaml.safe_load((root/'full_native_qa2c_successor_protocol.yaml').read_text())
    for key in ('agent','simulator','selection_criteria'):
        assert tier[key] == full[key]


def test_fractional_episode_terminates_without_zero_duration_steps():
    from qbbr.env.fluid_env import FluidSimEnv
    calibration={'test':{'downlink':{'B_max_mbps':8.,'B_median_mbps':8.,'RTT_min_ms':100.,'RTT_max_ms':200.}}}
    env=FluidSimEnv('test','downlink',calibration,episode_s=.6,risk_mode='stub_constant')
    env.reset(1)
    for _ in range(4):
        _,_,done,info=env.step(2)
        assert info['t_dec_s'] > 1e-9
        if done:
            break
    assert done


# --- v7 ProbeBW cycle + queue-pressure loss -------------------------------

CYCLE = replace(P, probe_bw_cycle=True)  # bdp = 1e5 bytes, probe interval 3 s, 1 round up/down


def _run(params, gain, steps, dt=.01, v0=0.0):
    state = FluidState(v_bytes=v0, startup_done=True, i_crs=1.0)
    peak_v = state.v_bytes
    for _ in range(steps):
        state, _d, _r = step_fluid_state(state, gain, dt, params, 0.0, capacity_bps_override=1e6)
        peak_v = max(peak_v, state.v_bytes)
    return state, peak_v


def test_probe_bw_cycle_defaults_off():
    assert FluidParams(x_btl_bps=1e6, rtt_rtp_s=.05).probe_bw_cycle is False
    assert FluidParams(x_btl_bps=1e6, rtt_rtp_s=.05).overflow_retransmit_frac == 0.0
    assert (FluidState().probe_phase, FluidState().probe_phase_elapsed_s) == (0, 0.0)


def test_probe_bw_cycle_bounds_standing_queue_vs_sustained_multiplier():
    bdp = P.bdp_bytes
    _s_no, peak_no = _run(replace(CYCLE, probe_bw_cycle=False), 1.25, 1200)  # 12 s sustained 1.25
    _s_cy, peak_cy = _run(CYCLE, 1.25, 1200)
    assert peak_no >= 0.9 * bdp            # legacy: standing backlog near the full cwnd cap
    assert peak_cy <= 0.4 * bdp            # v7: only a transient ProbeBW_UP pulse
    assert peak_cy < 0.5 * peak_no


def test_probe_bw_cycle_stock_gain_stays_in_cruise():
    state, peak_v = _run(CYCLE, 1.0, 1200)  # agent picks stock 1.0 the whole time
    assert state.probe_phase == 0            # never leaves ProbeBW_CRUISE
    assert peak_v <= 0.02 * P.bdp_bytes      # no autonomous probe -> queue stays near empty


def test_probe_bw_cycle_timestep_stability_through_env():
    # v7 numerics: mean RTT is substep-size stable, and -- what the acceptance
    # gates actually use -- the RTT p95 delta of a policy vs matched stock is
    # dt-stable even though the absolute p95 tail still carries an O(dt) bias
    # (common-mode, so it cancels in the delta). Regression guard for both.
    from qbbr.env.fluid_env import FluidSimEnv
    calibration = {'c': {'downlink': {'B_max_mbps': 200., 'B_median_mbps': 120., 'RTT_min_ms': 30., 'RTT_max_ms': 90.}}}
    overrides = dict(consistent_transport=True, cwnd_gain=2.0, probe_bw_cycle=True,
                     probe_rtt_interval_s=5.0, loss_thresh=.02, bandwidth_estimate_recovery_s=3.0)

    def run(dt, act_req):
        env = FluidSimEnv('c', 'downlink', calibration, episode_s=60.0, substep_s=dt,
                          reward_mode='throughput_only', dynamics_overrides=overrides, probe_bw_phase_gate=True)
        env.reset(7001)
        rtts, gp, rx, done = [], [], [], False
        while not done:
            act = act_req if act_req in env.allowed_action_indices() else 2
            _s, _r, done, info = env.step(act)
            rtts.append(info['rtt_ms'])
            gp.append(info['delivered_bytes'] * 8.0 / info['t_dec_s'] / 1e6)
            rx.append(info['retransmits'] / info['t_dec_s'])
        return dict(mean=float(np.mean(rtts)), p95=float(np.percentile(rtts, 95)),
                    gp=float(np.mean(gp)), rx=float(np.mean(rx)))

    pol_a, stock_a = run(.02, 4), run(.02, 2)
    pol_b, stock_b = run(.01, 4), run(.01, 2)
    # Mean RTT: substep-size stable.
    assert pol_a['mean'] == pytest.approx(pol_b['mean'], rel=.05)
    # Gate-critical deltas vs stock: substep-size stable to a tight bound.
    gp_delta_a = pol_a['gp'] / stock_a['gp'] - 1.0
    gp_delta_b = pol_b['gp'] / stock_b['gp'] - 1.0
    assert abs(gp_delta_a - gp_delta_b) <= 0.01
    assert abs((pol_a['rx'] - stock_a['rx']) - (pol_b['rx'] - stock_b['rx'])) <= 0.05
    # RTT p95 delta vs stock still carries an O(dt) residual in the spiky tail
    # (documented in docs/transport_v6_review.md, v7 section) -- bound it loosely.
    assert abs((pol_a['p95'] - stock_a['p95']) - (pol_b['p95'] - stock_b['p95'])) <= 4.0


def test_probe_max_queue_delay_ms_bounds_probe_rtt_cost():
    # On a low-capacity path, an unbounded ProbeBW pulse builds ~0.5 BDP of
    # queue = tens of ms of RTT. probe_max_queue_delay_ms caps that.
    from qbbr.env.fluid_env import FluidSimEnv
    calibration = {'c': {'uplink': {'B_max_mbps': 50., 'B_median_mbps': 3., 'RTT_min_ms': 250., 'RTT_max_ms': 400.}}}
    base_ov = dict(consistent_transport=True, cwnd_gain=2.0, probe_bw_cycle=True,
                   bandwidth_estimate_recovery_s=3.0, utilization_fraction=0.05)

    def worst_rtt(extra):
        env = FluidSimEnv('c', 'uplink', calibration, episode_s=60.0, substep_s=.02,
                          reward_mode='throughput_only', dynamics_overrides={**base_ov, **extra},
                          probe_bw_phase_gate=True)
        env.reset(7001)
        peak, done = 0.0, False
        while not done:
            act = 4 if 4 in env.allowed_action_indices() else 2
            _s, _r, done, info = env.step(act)
            peak = max(peak, info['rtt_ms'] - 250.0)  # RTT_min_ms
        return peak

    unbounded = worst_rtt({})
    bounded = worst_rtt({'probe_max_queue_delay_ms': 15.0})
    assert bounded < unbounded
    assert bounded <= 22.0  # ~15 ms budget + one substep of overshoot


def test_overflow_retransmit_frac_charges_loss_only_past_buffer():
    bdp = P.bdp_bytes
    base = replace(P, base_retransmit_rate_pps=50.0)
    over = replace(base, overflow_retransmit_frac=.3)
    # backlog below the 2*BDP buffer ceiling: identical to the legacy rate
    _s2, _d2, r_under = step_fluid_state(FluidState(v_bytes=1.5 * bdp, startup_done=True), 1.0, .01, over, 0.0, capacity_bps_override=1e6)
    _s3, _d3, r_base = step_fluid_state(FluidState(v_bytes=1.5 * bdp, startup_done=True), 1.0, .01, base, 0.0, capacity_bps_override=1e6)
    assert r_under == pytest.approx(r_base)
    # backlog well past the ceiling: extra retransmissions on top of the base rate
    _s4, _d4, r_over = step_fluid_state(FluidState(v_bytes=3.0 * bdp, startup_done=True), 1.0, .01, over, 0.0, capacity_bps_override=1e6)
    _s5, _d5, r_base_hi = step_fluid_state(FluidState(v_bytes=3.0 * bdp, startup_done=True), 1.0, .01, base, 0.0, capacity_bps_override=1e6)
    assert r_over > r_base_hi * 1.5
