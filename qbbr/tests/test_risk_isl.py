from __future__ import annotations

import pytest
from skyfield.api import load

from qbbr.risk.isl import (
    build_plane_pair_satellites,
    compute_isl_failure_probability,
    compute_isl_link_up_fraction,
)

_FAST_KWARGS = dict(duration_hours=1.0, step_s=10.0)


def test_coorbital_and_adjacent_plane_links_are_always_up():
    for sep in (0, 1):
        p = compute_isl_failure_probability("shell1", plane_separation=sep, **_FAST_KWARGS)
        assert p == pytest.approx(0.0, abs=1e-6)


def test_opposite_side_of_shell_is_always_down():
    # shell1 has 72 planes; half-way around (36 planes) is the far side.
    p = compute_isl_failure_probability("shell1", plane_separation=36, **_FAST_KWARGS)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_intermediate_plane_separation_is_intermittent():
    p = compute_isl_failure_probability("shell1", plane_separation=10, **_FAST_KWARGS)
    assert 0.0 < p < 1.0


def test_failure_probability_increases_with_plane_separation():
    separations = [0, 5, 10, 20, 36]
    failures = [
        compute_isl_failure_probability("shell1", plane_separation=s, **_FAST_KWARGS) for s in separations
    ]
    assert failures == sorted(failures)


def test_unknown_shell_raises():
    with pytest.raises(ValueError):
        compute_isl_failure_probability("not_a_shell", plane_separation=1, **_FAST_KWARGS)


def test_build_plane_pair_satellites_returns_two_distinct_satellites():
    ts = load.timescale()
    sat_a, sat_b = build_plane_pair_satellites("shell1", plane_separation=1, epoch_time=ts.now())
    assert sat_a is not sat_b


def test_link_up_fraction_is_bounded():
    ts = load.timescale()
    sat_a, sat_b = build_plane_pair_satellites("shell3", plane_separation=2, epoch_time=ts.now())
    frac = compute_isl_link_up_fraction(sat_a, sat_b, **_FAST_KWARGS)
    assert 0.0 <= frac <= 1.0
