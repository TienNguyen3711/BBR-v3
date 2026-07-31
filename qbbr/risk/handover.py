from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import WGS72, Satrec
from skyfield.api import Time, load, wgs84
from skyfield.sgp4lib import EarthSatellite
from skyfield.toposlib import GeographicPosition

# Deakin University, Burwood campus -- the fixed Starlink ground terminal
# location for every trace in this dataset (see module docstring point 1).
BURWOOD_LATLON = (-37.8477, 145.1131)

# TMC base paper Table I: (name, altitude_km, inclination_deg, planes, sats_per_plane).
STARLINK_SHELLS = [
    ("shell1", 540.0, 53.2, 72, 22),
    ("shell2", 550.0, 53.0, 72, 22),
    ("shell3", 560.0, 97.6, 6, 58),
    ("shell4", 560.0, 97.6, 4, 43),
    ("shell5", 570.0, 70.0, 56, 20),
]

_MU_EARTH = 398600.4418  # km^3/s^2
_R_EARTH = 6378.137  # km, WGS72 equatorial radius
_DEFAULT_MIN_ELEVATION_DEG = 25.0  # TMC paper: "LEOs visible above 25 deg elevation angle"
_DEFAULT_N_SATS_PER_SHELL = 8
_DEFAULT_DURATION_HOURS = 24.0
_DEFAULT_STEP_S = 10.0


@dataclass(frozen=True)
class HandoverCadenceEstimate:
    shell: str
    n_passes: int
    median_pass_s: float
    mean_pass_s: float
    min_pass_s: float
    max_pass_s: float


def build_shell_satellites(
    epoch_time: Time, n_sats_per_shell: int = _DEFAULT_N_SATS_PER_SHELL
) -> dict[str, list[EarthSatellite]]:
    ts = load.timescale()
    epoch_days = epoch_time.tt - 2433281.5  # sgp4 epoch: days since 1949-12-31 00:00 UT

    satellites: dict[str, list[EarthSatellite]] = {}
    for name, alt_km, incl_deg, _planes, _per_plane in STARLINK_SHELLS:
        a_km = _R_EARTH + alt_km
        no_kozai = np.sqrt(_MU_EARTH / a_km**3) * 60.0  # rad/min

        sats = []
        for i in range(n_sats_per_shell):
            mo = 2 * np.pi * i / n_sats_per_shell
            satrec = Satrec()
            satrec.sgp4init(
                WGS72, "i", 90000 + i, epoch_days,
                0.0, 0.0, 0.0,  # bstar, ndot, nddot
                0.0001,  # ecco: near-circular (exactly 0 can misbehave in SGP4)
                0.0,  # argpo
                np.radians(incl_deg),
                mo,
                no_kozai,
                0.0,  # nodeo
            )
            sats.append(EarthSatellite.from_satrec(satrec, ts))
        satellites[name] = sats
    return satellites


def compute_pass_durations_s(
    satellites: list[EarthSatellite],
    observer: GeographicPosition,
    min_elevation_deg: float = _DEFAULT_MIN_ELEVATION_DEG,
    duration_hours: float = _DEFAULT_DURATION_HOURS,
    step_s: float = _DEFAULT_STEP_S,
) -> list[float]:
    ts = load.timescale()
    t0 = ts.now()
    n_steps = int(duration_hours * 3600.0 / step_s)
    times = ts.tt_jd(t0.tt + np.arange(n_steps) * step_s / 86400.0)

    durations: list[float] = []
    for sat in satellites:
        alt, _az, _dist = (sat - observer).at(times).altaz()
        above = alt.degrees >= min_elevation_deg

        changes = np.diff(above.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if above[0]:
            starts = np.insert(starts, 0, 0)
        if above[-1]:
            ends = np.append(ends, len(above))
        durations.extend((e - s) * step_s for s, e in zip(starts, ends))
    return durations


def estimate_handover_cadence_s(
    lat: float = BURWOOD_LATLON[0],
    lon: float = BURWOOD_LATLON[1],
    min_elevation_deg: float = _DEFAULT_MIN_ELEVATION_DEG,
    n_sats_per_shell: int = _DEFAULT_N_SATS_PER_SHELL,
    duration_hours: float = _DEFAULT_DURATION_HOURS,
    step_s: float = _DEFAULT_STEP_S,
) -> list[HandoverCadenceEstimate]:
    """Per-shell single-satellite pass-duration statistics for a ground station (default: Burwood).

    See module docstring: this is a physical upper bound on how rarely a
    handover could be avoided, derived from the base paper's own published
    constellation geometry -- not Starlink's actual (unpublished, likely
    more frequent) handover schedule.
    """
    ts = load.timescale()
    observer = wgs84.latlon(lat, lon)
    satellites_by_shell = build_shell_satellites(ts.now(), n_sats_per_shell)

    estimates = []
    for name, _alt, _incl, _planes, _per_plane in STARLINK_SHELLS:
        durations = compute_pass_durations_s(
            satellites_by_shell[name], observer, min_elevation_deg, duration_hours, step_s
        )
        arr = np.array(durations, dtype=float)
        estimates.append(
            HandoverCadenceEstimate(
                shell=name,
                n_passes=len(arr),
                median_pass_s=float(np.median(arr)) if len(arr) else float("nan"),
                mean_pass_s=float(arr.mean()) if len(arr) else float("nan"),
                min_pass_s=float(arr.min()) if len(arr) else float("nan"),
                max_pass_s=float(arr.max()) if len(arr) else float("nan"),
            )
        )
    return estimates


def compute_handover_failure_probability(p_rb: float, p_prach: float) -> float:
    """p^ho: probability a single handover attempt fails."""
    raise NotImplementedError(
        "Eq. 19 handover *failure* model needs per-attempt resource-block/PRACH "
        "contention data Starlink doesn't publish -- unlike the ETA/cadence functions "
        "above, this part remains blocked; see qbbr.risk.ptot for the interim proxy."
    )
