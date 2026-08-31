from __future__ import annotations

from functools import lru_cache
import math

import pandas as pd

from qbbr.features.state_builder import HANDOVER_CYCLE_MIN

# The validated 15-second reconfiguration clock is exposed independently as
# s7_reconfig_phase.  s5/s6 must not duplicate that feature: they represent a
# slower, forecast-only handover proxy.  Its 5.4-minute cycle is a modelling
# assumption, not a claim that every Starlink reconfiguration is a physical
# terminal-to-satellite handover.
_LEGACY_RECONFIG_CYCLE_S = 15.0
_HANDOVER_CYCLE_S = HANDOVER_CYCLE_MIN * 60.0
_HANDOVER_RISK_WIDTH_S = 20.0
_HANDOVER_PEAK_FAILURE = 0.05
_RETRANSMIT_PROXY_WINDOW = 10
_RETRANSMIT_PROXY_CAP = 5.0  # retransmits/interval considered "high risk" for the proxy
_ATTENUATION_THRESHOLD_DB = 3.0  # M in p^at(M): a representative fade-margin threshold; not
_STUB_DELTA_T_HO_MIN = 0.5 * HANDOVER_CYCLE_MIN
_STUB_P_TOT = 0.5

_closed_form_p_tot_cache: float | None = None


def closed_form_p_tot() -> float:
    """Legacy atmospheric-only climatological risk used by archived RQ3 runs.

    Keep this function stable so previously trained checkpoints and their
    reports remain reproducible.  New work should use ``closed_form_dynamic``
    below and retrain its risk-on arm from scratch.
    """
    global _closed_form_p_tot_cache
    if _closed_form_p_tot_cache is None:
        from qbbr.risk.atmospheric import compute_atmospheric_failure_probability

        _closed_form_p_tot_cache = compute_atmospheric_failure_probability(_ATTENUATION_THRESHOLD_DB)
    return _closed_form_p_tot_cache


def _circular_distance_s(t_s: float, period_s: float) -> float:
    phase = t_s % period_s
    return min(phase, period_s - phase)


def handover_failure_probability(
    t_s: float,
    cycle_s: float = _HANDOVER_CYCLE_S,
    width_s: float = _HANDOVER_RISK_WIDTH_S,
    peak_failure: float = _HANDOVER_PEAK_FAILURE,
) -> float:
    """Bounded scheduled-handover failure hazard at time ``t_s``.

    The Gaussian is a smooth forecast feature, not an observed loss signal.
    It peaks at a scheduled boundary and falls with time-to-boundary, which
    lets an agent distinguish an imminent handover from the separate 15-second
    reconfiguration clock.  The cycle, width, and peak are explicit arguments
    so they can be calibrated or sensitivity-swept rather than treated as
    hidden physical constants.
    """
    if cycle_s <= 0.0 or width_s <= 0.0:
        raise ValueError("cycle_s and width_s must be positive")
    if not 0.0 <= peak_failure <= 1.0:
        raise ValueError("peak_failure must lie in [0, 1]")
    distance_s = _circular_distance_s(float(t_s), cycle_s)
    return peak_failure * math.exp(-0.5 * (distance_s / width_s) ** 2)


def compose_failure_probabilities(*probabilities: float) -> float:
    """Compose independent per-cause failure probabilities into ``p_tot``."""
    success_probability = 1.0
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"failure probability must lie in [0, 1], got {probability!r}")
        success_probability *= 1.0 - probability
    return 1.0 - success_probability


@lru_cache(maxsize=1)
def closed_form_isl_failure_probability() -> float:
    """Representative ISL baseline from the repository's orbital geometry.

    This is a long-run route-component probability for a representative
    neighbouring-plane link, not a per-packet measurement of Starlink's
    undisclosed routing.  Caching keeps risk-state construction cheap during
    training.  A trace- or route-calibrated value can be supplied to
    ``dynamic_closed_form_p_tot`` when available.
    """
    from qbbr.risk.isl import compute_isl_failure_probability

    return compute_isl_failure_probability(shell="shell1", plane_separation=1)


def dynamic_closed_form_p_tot(
    t_s: float,
    isl_failure: float | None = None,
    handover_cycle_s: float = _HANDOVER_CYCLE_S,
    handover_width_s: float = _HANDOVER_RISK_WIDTH_S,
    handover_peak_failure: float = _HANDOVER_PEAK_FAILURE,
) -> float:
    """Time-varying composition of atmospheric, ISL, and handover risk.

    Atmospheric and ISL terms are long-run baselines; the handover term is
    forecast from time-to-boundary.  Keeping the three terms explicit makes
    the model auditable and allows a future calibration to replace the
    representative ISL assumption without changing the composition rule.
    """
    p_atmospheric = closed_form_p_tot()
    p_isl = closed_form_isl_failure_probability() if isl_failure is None else isl_failure
    p_handover = handover_failure_probability(
        t_s,
        cycle_s=handover_cycle_s,
        width_s=handover_width_s,
        peak_failure=handover_peak_failure,
    )
    return compose_failure_probabilities(p_atmospheric, p_isl, p_handover)


def compute_risk_features(telemetry: pd.DataFrame, mode: str = "stub_constant") -> pd.DataFrame:
    n = len(telemetry)
    if mode == "stub_constant":
        delta_t_ho_min = pd.Series([_STUB_DELTA_T_HO_MIN] * n, index=telemetry.index)
        p_tot = pd.Series([_STUB_P_TOT] * n, index=telemetry.index)
    elif mode == "empirical_proxy":
        t_start = telemetry["t_start"].astype(float)
        seconds_to_next_handover = _LEGACY_RECONFIG_CYCLE_S - (t_start % _LEGACY_RECONFIG_CYCLE_S)
        delta_t_ho_min = seconds_to_next_handover / 60.0

        rtx_rate = telemetry["retransmits"].rolling(
            window=_RETRANSMIT_PROXY_WINDOW, min_periods=1
        ).mean()
        p_tot = (rtx_rate / _RETRANSMIT_PROXY_CAP).clip(lower=0.0, upper=1.0)
    elif mode == "closed_form":
        t_start = telemetry["t_start"].astype(float)
        # Preserve the legacy risk-on state exactly for archived RQ3 runs.
        # It was tied to the observed 15-second reconfiguration clock.
        seconds_to_next_handover = _LEGACY_RECONFIG_CYCLE_S - (t_start % _LEGACY_RECONFIG_CYCLE_S)
        delta_t_ho_min = seconds_to_next_handover / 60.0

        p_tot = pd.Series([closed_form_p_tot()] * n, index=telemetry.index)
    elif mode == "closed_form_dynamic":
        t_start = telemetry["t_start"].astype(float)
        seconds_to_next_handover = _HANDOVER_CYCLE_S - (t_start % _HANDOVER_CYCLE_S)
        delta_t_ho_min = seconds_to_next_handover / 60.0
        p_tot = t_start.map(dynamic_closed_form_p_tot)
    else:
        raise ValueError(f"unknown risk mode: {mode!r}")

    return pd.DataFrame(
        {
            "delta_t_ho_min": delta_t_ho_min,
            "p_tot": p_tot,
            "mode": mode,
        }
    )
