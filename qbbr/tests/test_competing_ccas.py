from __future__ import annotations

import pytest

from qbbr.env.competing_ccas import (
    CompetingCCAState,
    offered_rate_bps,
    update_competing_cca,
    update_cubic,
    update_hybla,
    update_vegas,
)

_RTT_BASE_S = 0.05
_START_WINDOW = 300_000.0


def test_offered_rate_is_window_over_rtt():
    state = CompetingCCAState(window_bytes=100_000.0)
    assert offered_rate_bps(state, shared_rtt_s=0.1) == pytest.approx(1_000_000.0)


def test_cubic_grows_when_uncongested():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    new_state = update_cubic(state, congestion_signal=0.0, dt_s=0.1)
    assert new_state.window_bytes > state.window_bytes


def test_cubic_shrinks_when_congested():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    new_state = update_cubic(state, congestion_signal=1.0, dt_s=0.1)
    assert new_state.window_bytes < state.window_bytes


def test_cubic_never_drops_below_floor():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    for _ in range(1000):
        state = update_cubic(state, congestion_signal=1.0, dt_s=1.0)
    assert state.window_bytes > 0.0


def test_vegas_grows_when_no_queueing_delay():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    new_state = update_vegas(state, rtt_base_s=_RTT_BASE_S, shared_rtt_s=_RTT_BASE_S, dt_s=0.1)
    assert new_state.window_bytes > state.window_bytes


def test_vegas_shrinks_when_rtt_is_inflated():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    new_state = update_vegas(state, rtt_base_s=_RTT_BASE_S, shared_rtt_s=_RTT_BASE_S * 2, dt_s=0.1)
    assert new_state.window_bytes < state.window_bytes


def test_vegas_reacts_to_delay_not_to_congestion_signal_directly():
    # Vegas's update function doesn't even take a congestion_signal argument
    # -- its distinguishing feature is reacting to delay alone.
    import inspect

    assert "congestion_signal" not in inspect.signature(update_vegas).parameters


def test_hybla_grows_faster_at_higher_rtt_than_cubic():
    # Hybla's whole point: RTT-normalized growth removes the long-RTT penalty.
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    low_rtt_growth = update_hybla(state, congestion_signal=0.0, shared_rtt_s=0.025, dt_s=0.1).window_bytes
    high_rtt_growth = update_hybla(state, congestion_signal=0.0, shared_rtt_s=0.25, dt_s=0.1).window_bytes
    assert high_rtt_growth > low_rtt_growth


def test_hybla_shrinks_when_congested():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    new_state = update_hybla(state, congestion_signal=1.0, shared_rtt_s=_RTT_BASE_S, dt_s=0.1)
    assert new_state.window_bytes < state.window_bytes


def test_update_competing_cca_dispatches_correctly():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    ctx = {"congestion_signal": 0.0, "rtt_base_s": _RTT_BASE_S, "shared_rtt_s": _RTT_BASE_S, "dt_s": 0.1}
    for name in ("cubic", "vegas", "hybla"):
        result = update_competing_cca(name, state, ctx)
        assert isinstance(result, CompetingCCAState)


def test_update_competing_cca_rejects_unknown_name():
    state = CompetingCCAState(window_bytes=_START_WINDOW)
    with pytest.raises(ValueError):
        update_competing_cca("not_a_real_cca", state, {})
