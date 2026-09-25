"""Coexistence throughput must be measured inside the window where every flow
of a run is active; averaging whole files mixes contended and solo periods."""
from __future__ import annotations

import math

import pytest

from qbbr.scripts.analyze_measured_coexistence import (
    ALPHAS, MIN_WINDOW_S, bytes_in_window, competitive_run,
)


def _flow(rows):
    return {"rows": rows, "t0": rows[0][0], "t1": rows[-1][1]}


def test_window_splits_partially_covered_intervals_pro_rata():
    # One 10 s interval carrying 1000 bytes; half of it lies inside the window.
    flow = _flow([(0.0, 10.0, 1000.0, 20.0)])
    size, retransmits = bytes_in_window(flow, 5.0, 10.0)
    assert size == pytest.approx(500.0)
    assert retransmits == pytest.approx(10.0)


def test_window_excludes_intervals_outside_it_entirely():
    flow = _flow([(0.0, 10.0, 1000.0, 0.0), (10.0, 20.0, 4000.0, 0.0)])
    size, _ = bytes_in_window(flow, 10.0, 20.0)
    assert size == pytest.approx(4000.0)


def test_a_solo_head_is_not_counted_as_contended_throughput(monkeypatch):
    """A flow that starts early must not bank its uncontended head.

    This is the Mumbai downlink case: one flow starts up to 136 s before the
    others on a 300 s test. Counting its whole file would credit it with
    throughput it achieved while it had the link to itself.
    """
    early = _flow([(0.0, 200.0, 200_000.0, 0.0)])   # 8 kbit/s over its own run
    late = _flow([(100.0, 200.0, 50_000.0, 0.0)])   # joins halfway
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.load_flow",
                        lambda path: {"early": early, "late": late}[path])
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.MIN_WINDOW_S", 10.0)

    result = competitive_run({"early": "early", "late": "late"})
    assert result["excluded"] is False
    assert result["window_s"] == pytest.approx(100.0)
    # Only the second half of `early` counts: 100 000 bytes, not 200 000.
    assert result["throughput_bps"]["early"] == pytest.approx(100_000 * 8 / 100)
    assert result["throughput_bps"]["late"] == pytest.approx(50_000 * 8 / 100)


def test_run_with_too_short_an_overlap_is_excluded_not_silently_used(monkeypatch):
    a = _flow([(0.0, 300.0, 1000.0, 0.0)])
    b = _flow([(295.0, 595.0, 1000.0, 0.0)])
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.load_flow",
                        lambda path: {"a": a, "b": b}[path])
    result = competitive_run({"a": "a", "b": "b"})
    assert result["excluded"] is True
    assert result["window_s"] == pytest.approx(5.0)
    assert "throughput_bps" not in result


def test_alpha_sweep_avoids_the_unbounded_region():
    """alpha>1 makes the efficiency ratio unbounded (see the metric docstring),
    so a near-starved flow would report a ratio far outside (0, 1]."""
    assert all(a <= 1.0 or math.isinf(a) for a in ALPHAS)


def test_equal_flows_are_maxmin_fair_and_starvation_is_visible(monkeypatch):
    equal = {name: _flow([(0.0, 100.0, 1000.0, 0.0)]) for name in "abc"}
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.load_flow",
                        lambda path: equal[path])
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.MIN_WINDOW_S", 10.0)
    balanced = competitive_run({name: name for name in "abc"})
    assert balanced["rho_alpha"]["inf"] == pytest.approx(1.0)

    starved = dict(equal, c=_flow([(0.0, 100.0, 1.0, 0.0)]))
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.load_flow",
                        lambda path: starved[path])
    result = competitive_run({name: name for name in "abc"})
    assert result["rho_alpha"]["inf"] < 0.01  # max-min ratio collapses
    assert result["rho_alpha"]["0.0"] == pytest.approx(1.0)  # total is unaffected


def test_a_run_needs_at_least_two_flows_to_be_a_contention_measurement(monkeypatch):
    monkeypatch.setattr("qbbr.scripts.analyze_measured_coexistence.load_flow",
                        lambda path: _flow([(0.0, 300.0, 1000.0, 0.0)]))
    assert competitive_run({"only": "only"}) is None


def test_min_window_guard_is_a_meaningful_fraction_of_a_run():
    assert MIN_WINDOW_S >= 100.0  # runs are 300 s
