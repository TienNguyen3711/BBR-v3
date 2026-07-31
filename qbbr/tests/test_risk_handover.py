from __future__ import annotations

from skyfield.api import load, wgs84

from qbbr.risk.handover import (
    BURWOOD_LATLON,
    STARLINK_SHELLS,
    build_shell_satellites,
    compute_pass_durations_s,
    estimate_handover_cadence_s,
)

# Short/coarse propagation window purely for test speed; the real-scale
# computation (24h, 8 sats/shell) is what estimate_handover_cadence_s's
# module-docstring numbers (~240-255s median) are based on.
_FAST_KWARGS = dict(n_sats_per_shell=6, duration_hours=8.0, step_s=15.0)


def test_build_shell_satellites_returns_all_five_shells():
    ts = load.timescale()
    sats = build_shell_satellites(ts.now(), n_sats_per_shell=3)
    assert set(sats.keys()) == {name for name, *_ in STARLINK_SHELLS}
    for shell_sats in sats.values():
        assert len(shell_sats) == 3


def test_orbital_period_is_realistic_leo_range():
    # Starlink's real orbital period is ~95-96 minutes; sgp4init's mean
    # motion should reproduce that from the paper's own altitude figures.
    import numpy as np

    from qbbr.risk.handover import _MU_EARTH, _R_EARTH

    for _name, alt_km, _incl, _planes, _per_plane in STARLINK_SHELLS:
        a_km = _R_EARTH + alt_km
        period_min = 2 * np.pi / (np.sqrt(_MU_EARTH / a_km**3) * 60.0)
        assert 90.0 < period_min < 100.0


def test_pass_durations_are_positive_and_bounded():
    ts = load.timescale()
    observer = wgs84.latlon(*BURWOOD_LATLON)
    sats = build_shell_satellites(ts.now(), n_sats_per_shell=_FAST_KWARGS["n_sats_per_shell"])
    durations = compute_pass_durations_s(
        sats["shell1"], observer,
        duration_hours=_FAST_KWARGS["duration_hours"], step_s=_FAST_KWARGS["step_s"],
    )
    assert len(durations) > 0
    for d in durations:
        assert 0.0 < d < 3600.0  # a single LEO pass can't last an hour


def test_estimate_handover_cadence_covers_all_shells_with_sane_medians():
    estimates = estimate_handover_cadence_s(**_FAST_KWARGS)
    assert len(estimates) == len(STARLINK_SHELLS)
    for e in estimates:
        assert e.n_passes > 0
        assert 10.0 < e.median_pass_s < 96 * 60.0
        assert e.min_pass_s <= e.median_pass_s <= e.max_pass_s


def _mean_of_shells_with_passes(estimates):
    # median_pass_s is nan for a shell with zero qualifying passes in the
    # window (real-epoch-dependent at this fixture's coarse settings); drop
    # those rather than let a single nan poison the whole mean.
    values = [e.median_pass_s for e in estimates if e.n_passes > 0]
    assert values, "expected at least one shell with a qualifying pass"
    return sum(values) / len(values)


def test_higher_min_elevation_gives_shorter_passes():

    low_cutoff = estimate_handover_cadence_s(min_elevation_deg=25.0, **_FAST_KWARGS)
    high_cutoff = estimate_handover_cadence_s(min_elevation_deg=60.0, **_FAST_KWARGS)
    low_median = _mean_of_shells_with_passes(low_cutoff)
    high_median = _mean_of_shells_with_passes(high_cutoff)
    assert high_median < low_median
