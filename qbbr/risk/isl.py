from __future__ import annotations

import numpy as np
from sgp4.api import WGS72, Satrec
from skyfield.api import Time, load
from skyfield.sgp4lib import EarthSatellite

from qbbr.risk.handover import STARLINK_SHELLS, _MU_EARTH, _R_EARTH

_MAX_ISL_RANGE_KM = 5000.0  # commonly-cited approximate Starlink optical-ISL range; not an
# authoritative published spec, documented as an assumption like main.tex's other unstated constants.
_DEFAULT_DURATION_HOURS = 2.0
_DEFAULT_STEP_S = 5.0


def _shell_params(shell: str) -> tuple[float, float, int]:
    for name, alt_km, incl_deg, planes, _per_plane in STARLINK_SHELLS:
        if name == shell:
            return alt_km, incl_deg, planes
    raise ValueError(f"unknown shell: {shell!r}; expected one of {[s[0] for s in STARLINK_SHELLS]}")


def build_plane_pair_satellites(
    shell: str, plane_separation: int, epoch_time: Time
) -> tuple[EarthSatellite, EarthSatellite]:
    alt_km, incl_deg, n_planes = _shell_params(shell)
    a_km = _R_EARTH + alt_km
    no_kozai = np.sqrt(_MU_EARTH / a_km**3) * 60.0  # rad/min
    epoch_days = epoch_time.tt - 2433281.5
    ts = load.timescale()

    raan_spacing_deg = 360.0 / n_planes

    def make_sat(satnum: int, raan_deg: float, mo_deg: float) -> EarthSatellite:
        satrec = Satrec()
        satrec.sgp4init(
            WGS72, "i", satnum, epoch_days, 0.0, 0.0, 0.0, 0.0001, 0.0,
            np.radians(incl_deg), np.radians(mo_deg), no_kozai, np.radians(raan_deg),
        )
        return EarthSatellite.from_satrec(satrec, ts)

    sat_a = make_sat(80000, 0.0, 0.0)
    mean_anomaly_offset = 15.0 if plane_separation == 0 else 0.0  # co-orbital: offset along-track
    sat_b = make_sat(80001, plane_separation * raan_spacing_deg, mean_anomaly_offset)
    return sat_a, sat_b


def compute_isl_link_up_fraction(
    sat_a: EarthSatellite,
    sat_b: EarthSatellite,
    max_range_km: float = _MAX_ISL_RANGE_KM,
    duration_hours: float = _DEFAULT_DURATION_HOURS,
    step_s: float = _DEFAULT_STEP_S,
) -> float:
    ts = load.timescale()
    t0 = ts.now()
    n_steps = int(duration_hours * 3600.0 / step_s)
    times = ts.tt_jd(t0.tt + np.arange(n_steps) * step_s / 86400.0)

    pos_a = sat_a.at(times).position.km
    pos_b = sat_b.at(times).position.km
    seg = pos_b - pos_a
    dist_km = np.linalg.norm(seg, axis=0)

    seg_len_sq = np.sum(seg**2, axis=0)
    u_star = np.clip(-np.sum(pos_a * seg, axis=0) / seg_len_sq, 0.0, 1.0)
    closest_point = pos_a + u_star * seg
    closest_dist_km = np.linalg.norm(closest_point, axis=0)

    in_range = dist_km <= max_range_km
    unobstructed = closest_dist_km >= _R_EARTH
    return float((in_range & unobstructed).mean())


def compute_isl_failure_probability(
    shell: str = "shell1",
    plane_separation: int = 1,
    max_range_km: float = _MAX_ISL_RANGE_KM,
    duration_hours: float = _DEFAULT_DURATION_HOURS,
    step_s: float = _DEFAULT_STEP_S,
) -> float:
    ts = load.timescale()
    sat_a, sat_b = build_plane_pair_satellites(shell, plane_separation, ts.now())
    up_fraction = compute_isl_link_up_fraction(sat_a, sat_b, max_range_km, duration_hours, step_s)
    return 1.0 - up_fraction
