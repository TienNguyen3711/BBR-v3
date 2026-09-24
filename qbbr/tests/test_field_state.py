"""Does the field state pipeline reproduce the simulator's state vector?

The policy was trained on features FluidSimEnv produced. In the field the same
features are rebuilt from kernel telemetry by FieldStateBuilder. If the two
disagree, the deployed policy is reading a different world than the one it
learned, and nothing measured in the field would be attributable.

This is the one part of the field path testable without a Linux kernel: drive
the simulator, capture what the kernel WOULD have reported at each decision,
feed that through FieldStateBuilder, and require the seven features to match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from qbbr.env.calibration import load_calibration
from qbbr.env.fluid_env import FluidSimEnv
from qbbr.field.telemetry import FieldStateBuilder, ReplayTelemetrySource, TcpSample

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PACKAGE_ROOT / "configs" / "tier1_v14_fidelity.yaml"
CALIBRATION = PACKAGE_ROOT / "data" / "calibrated" / "per_location_constants_v14.json"
STOCK_ACTION = 2


def _simulate(location="Sydney", direction="downlink", seed=0, decisions=40):
    """Run the simulator and record, per decision, both the state it produced
    and the TcpSample a kernel would have reported for the same interval."""
    config = yaml.safe_load(CONFIG.read_text())
    simulator = config["simulator"]
    calibration = load_calibration(CALIBRATION)
    env = FluidSimEnv(
        location, direction, calibration, risk_mode=simulator["risk_mode"],
        episode_s=300.0, reward_mode="difference", reward_kwargs=simulator["reward_kwargs"],
        dynamics_overrides=simulator.get("dynamics_overrides"),
        probe_bw_phase_gate=bool(simulator.get("probe_bw_phase_gate", False)),
    )
    env.reset(seed=seed)
    params = env.params
    sim_states, samples = [], []
    for _ in range(decisions):
        allowed = env.allowed_action_indices()
        state, _reward, done, info = env.step(STOCK_ACTION if STOCK_ACTION in allowed else allowed[0])
        bdp_bytes = params.bdp_bytes
        inflight = info["queue_bytes"] + params.steady_inflight_bdp_frac * bdp_bytes \
            if not params.consistent_transport else info["queue_bytes"] + bdp_bytes
        samples.append(TcpSample(
            # The simulator's t_start is its own clock plus the phase offset;
            # the field reads UTC. Both enter compute_state_vector as `t_start`,
            # so replaying the simulator's value here tests the SAME arithmetic
            # the field will run on UTC seconds.
            now_s=info["t_start"],
            delivered_bytes=info["delivered_bytes"],
            interval_s=info["t_dec_s"],
            rtt_ms=info["rtt_ms"],
            min_rtt_ms=params.rtt_rtp_s * 1000.0,
            inflight_bytes=inflight,
            delivery_rate_bps=params.sustained_x_btl_bps * 8.0,
            retransmits=info["retransmits"],
        ))
        sim_states.append(np.asarray(state, dtype=float))
        if done:
            break
    return sim_states, samples, calibration[location][direction], simulator["risk_mode"]


def test_reconfig_phase_matches_the_simulator_exactly():
    """s7 is pure arithmetic on the timestamp, so it must match to float error.

    This is the feature most likely to be wired up wrong in the field, because
    it depends on the time BASE: the 15 s cycle is anchored to UTC (see
    validate_handover_cadence.py, per-run Rayleigh R_bar=0.7368 at mean phase
    10.5 s over 60 runs, built from start.timestamp.timesecs + interval.start).
    """
    sim_states, samples, calibration, risk_mode = _simulate()
    builder = FieldStateBuilder(calibration, risk_mode=risk_mode)
    field_states = [builder.push(sample) for sample in samples]
    sim_s7 = np.array([s[6] for s in sim_states])
    field_s7 = np.array([s[6] for s in field_states])
    assert np.allclose(sim_s7, field_s7, atol=1e-9)


def test_risk_features_match_the_simulator():
    """s5 and s6 are functions of the timestamp alone, so they must match too."""
    sim_states, samples, calibration, risk_mode = _simulate()
    builder = FieldStateBuilder(calibration, risk_mode=risk_mode)
    field_states = [builder.push(sample) for sample in samples]
    for index in (4, 5):
        assert np.allclose([s[index] for s in sim_states],
                           [s[index] for s in field_states], atol=1e-9)


def test_rtt_ratio_matches_the_simulator():
    """s2 is a pure normalisation of the reported RTT against calibration."""
    sim_states, samples, calibration, risk_mode = _simulate()
    builder = FieldStateBuilder(calibration, risk_mode=risk_mode)
    field_states = [builder.push(sample) for sample in samples]
    assert np.allclose([s[1] for s in sim_states],
                       [s[1] for s in field_states], atol=1e-9)


def test_every_feature_stays_in_the_unit_box():
    """The policy was trained on clipped [0,1] features; a field feature outside
    that range would be extrapolation, not inference."""
    _sim, samples, calibration, risk_mode = _simulate()
    builder = FieldStateBuilder(calibration, risk_mode=risk_mode)
    for sample in samples:
        state = builder.push(sample)
        assert state.shape == (7,)
        assert np.all(state >= 0.0) and np.all(state <= 1.0)


def test_utc_time_base_changes_s7_materially():
    """Guard against the silent failure this module's docstring warns about.

    Feeding seconds-since-connection-start instead of UTC leaves every other
    feature untouched and randomises s7. If this ever stops holding, s7 has
    stopped depending on the time base and the field wiring needs re-checking.
    """
    _sim, samples, calibration, risk_mode = _simulate()

    utc = FieldStateBuilder(calibration, risk_mode=risk_mode)
    utc_s7 = [utc.push(s)[6] for s in samples]

    origin = samples[0].now_s
    relative = FieldStateBuilder(calibration, risk_mode=risk_mode)
    rel_s7 = [relative.push(
        TcpSample(**{**s.__dict__, "now_s": s.now_s - origin})
    )[6] for s in samples]

    assert not np.allclose(utc_s7, rel_s7, atol=1e-3)


def test_replay_source_drives_the_builder():
    _sim, samples, calibration, risk_mode = _simulate(decisions=10)
    source = ReplayTelemetrySource(samples)
    builder = FieldStateBuilder(calibration, risk_mode=risk_mode)
    seen = 0
    while True:
        try:
            builder.push(source.sample())
        except StopIteration:
            break
        seen += 1
    assert seen == len(samples)
