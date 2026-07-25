"""Handover failure probability and ETA (Eq. 19, TMC base paper; state feature s5).

p^ho(k,n) = 1 - (1 - p_rb(k,n)) * (1 - p_prach(k,n)): composes resource-block
denial and PRACH preamble-collision probabilities for a single handover
attempt. main.tex's s5 additionally needs the handover ETA (delta_T_ho),
estimated from TLE ephemerides and elevation geometry.

Neither the per-attempt RB/PRACH contention inputs nor TLE ephemerides for
the trace period are available yet. qbbr.risk.ptot's "empirical_proxy" mode
uses a fixed 15s sawtooth cycle as a documented stand-in for delta_T_ho in
the meantime; this module's cross-correlation against real TLE-derived
handover windows is exactly the risk-feature validation gate described in
main.tex ("Risk-Feature Validation (Pre-Training Gate)").
"""
from __future__ import annotations


def compute_handover_failure_probability(p_rb: float, p_prach: float) -> float:
    """p^ho: probability a single handover attempt fails."""
    raise NotImplementedError("Eq. 19 handover failure model not yet implemented; see qbbr.risk.ptot.")


def compute_handover_eta_from_tle(
    tle_lines: tuple[str, str], observer_location: tuple[float, float]
) -> float:
    """delta_T_ho: seconds until the next predicted handover, from TLE + elevation geometry."""
    raise NotImplementedError(
        "TLE-ephemeris-based handover ETA not yet implemented; "
        "qbbr.risk.ptot's empirical_proxy mode uses a fixed 15s sawtooth instead."
    )
