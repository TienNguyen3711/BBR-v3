from __future__ import annotations

import warnings

import numpy as np

from qbbr.risk.handover import (
    BURWOOD_LATLON,
    STARLINK_SHELLS,
    _DEFAULT_DURATION_HOURS,
    _DEFAULT_MIN_ELEVATION_DEG,
    _DEFAULT_N_SATS_PER_SHELL,
    _DEFAULT_STEP_S,
    build_shell_satellites,
)

_DEFAULT_FREQUENCY_GHZ = 12.5  # Ku-band downlink; ACMA/ITU allocation 10.7-12.7 GHz (main.tex)
_DEFAULT_ANTENNA_DIAMETER_M = 0.5  # Starlink UTA-232 user terminal, approx.
_P_GRID_PERCENT = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0])
_ELEVATION_BIN_WIDTH_DEG = 5.0


def elevation_time_weights(
    lat: float = BURWOOD_LATLON[0],
    lon: float = BURWOOD_LATLON[1],
    min_elevation_deg: float = _DEFAULT_MIN_ELEVATION_DEG,
    bin_width_deg: float = _ELEVATION_BIN_WIDTH_DEG,
    n_sats_per_shell: int = _DEFAULT_N_SATS_PER_SHELL,
    duration_hours: float = _DEFAULT_DURATION_HOURS,
    step_s: float = _DEFAULT_STEP_S,
) -> list[tuple[float, float]]:
    from skyfield.api import load, wgs84

    ts = load.timescale()
    t0 = ts.now()
    observer = wgs84.latlon(lat, lon)
    satellites_by_shell = build_shell_satellites(t0, n_sats_per_shell)

    n_steps = int(duration_hours * 3600.0 / step_s)
    times = ts.tt_jd(t0.tt + np.arange(n_steps) * step_s / 86400.0)

    all_elev, all_weight = [], []
    for name, _alt, _incl, planes, per_plane in STARLINK_SHELLS:
        n_shell_sats = planes * per_plane
        for sat in satellites_by_shell[name]:
            elev_deg, _az, _dist = (sat - observer).at(times).altaz()
            mask = elev_deg.degrees >= min_elevation_deg
            all_elev.append(elev_deg.degrees[mask])
            all_weight.append(np.full(mask.sum(), float(n_shell_sats)))

    elev = np.concatenate(all_elev)
    weight = np.concatenate(all_weight)
    bin_edges = np.arange(min_elevation_deg, 90.0 + bin_width_deg, bin_width_deg)
    hist_w, _ = np.histogram(elev, bins=bin_edges, weights=weight)
    hist_w = hist_w / hist_w.sum()
    bin_centers = bin_edges[:-1] + bin_width_deg / 2.0
    return [(float(c), float(w)) for c, w in zip(bin_centers, hist_w) if w > 0]


def _attenuation_vs_p_db(
    lat: float, lon: float, elevation_deg: float, f_ghz: float, antenna_diameter_m: float
) -> np.ndarray:
    """A(p) in dB over _P_GRID_PERCENT, at a fixed elevation angle."""
    import itur

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # itur warns outside its p in [0.001, 5]% validity range
        return np.array(
            [
                float(
                    itur.atmospheric_attenuation_slant_path(
                        lat, lon, f_ghz, elevation_deg, p, antenna_diameter_m
                    ).value
                )
                for p in _P_GRID_PERCENT
            ]
        )


def p_at_threshold_single_elevation(
    attenuation_threshold_db: float,
    elevation_deg: float,
    lat: float = BURWOOD_LATLON[0],
    lon: float = BURWOOD_LATLON[1],
    f_ghz: float = _DEFAULT_FREQUENCY_GHZ,
    antenna_diameter_m: float = _DEFAULT_ANTENNA_DIAMETER_M,
) -> float:
    """p_i(M): exceedance probability (%) for one elevation angle, by inverting itur's A(p)."""
    a_db = _attenuation_vs_p_db(lat, lon, elevation_deg, f_ghz, antenna_diameter_m)
    # A(p) is monotonically decreasing in p; np.interp needs increasing xp, so reverse both.
    p_grid_desc = _P_GRID_PERCENT[::-1]
    a_db_desc = a_db[::-1]
    if attenuation_threshold_db >= a_db_desc[-1]:
        return float(_P_GRID_PERCENT[0])  # threshold exceeds even the rarest tabulated event
    if attenuation_threshold_db <= a_db_desc[0]:
        return float(_P_GRID_PERCENT[-1])  # threshold is below the most common tabulated event
    return float(np.interp(attenuation_threshold_db, a_db_desc, p_grid_desc))


def compute_atmospheric_failure_probability(
    attenuation_threshold_db: float,
    lat: float = BURWOOD_LATLON[0],
    lon: float = BURWOOD_LATLON[1],
    f_ghz: float = _DEFAULT_FREQUENCY_GHZ,
    antenna_diameter_m: float = _DEFAULT_ANTENNA_DIAMETER_M,
    elevation_weights: list[tuple[float, float]] | None = None,
) -> float:
    """p^at(M) (Eq. 9): elevation-weighted attenuation-exceedance probability, as a fraction in [0, 1].

    elevation_weights defaults to elevation_time_weights(lat, lon) (a real
    orbital-mechanics computation, not instantaneous); pass a cached result
    to avoid recomputing it on every call.
    """
    if elevation_weights is None:
        elevation_weights = elevation_time_weights(lat, lon)

    p_percent = sum(
        w_i * p_at_threshold_single_elevation(attenuation_threshold_db, el_i, lat, lon, f_ghz, antenna_diameter_m)
        for el_i, w_i in elevation_weights
    )
    return p_percent / 100.0
