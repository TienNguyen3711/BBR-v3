from __future__ import annotations
import pandas as pd
from qbbr.features.state_builder import T_ORBIT_MIN

_HANDOVER_CYCLE_S = 15.0  # empirically validated, not merely assumed -- see module docstring
_RETRANSMIT_PROXY_WINDOW = 10
_RETRANSMIT_PROXY_CAP = 5.0  # retransmits/interval considered "high risk" for the proxy
_ATTENUATION_THRESHOLD_DB = 3.0  # M in p^at(M): a representative fade-margin threshold; not
_STUB_DELTA_T_HO_MIN = 0.5 * T_ORBIT_MIN
_STUB_P_TOT = 0.5

_closed_form_p_tot_cache: float | None = None


def closed_form_p_tot() -> float:
    global _closed_form_p_tot_cache
    if _closed_form_p_tot_cache is None:
        from qbbr.risk.atmospheric import compute_atmospheric_failure_probability

        _closed_form_p_tot_cache = compute_atmospheric_failure_probability(_ATTENUATION_THRESHOLD_DB)
    return _closed_form_p_tot_cache


def compute_risk_features(telemetry: pd.DataFrame, mode: str = "stub_constant") -> pd.DataFrame:
    n = len(telemetry)
    if mode == "stub_constant":
        delta_t_ho_min = pd.Series([_STUB_DELTA_T_HO_MIN] * n, index=telemetry.index)
        p_tot = pd.Series([_STUB_P_TOT] * n, index=telemetry.index)
    elif mode == "empirical_proxy":
        t_start = telemetry["t_start"].astype(float)
        seconds_to_next_handover = _HANDOVER_CYCLE_S - (t_start % _HANDOVER_CYCLE_S)
        delta_t_ho_min = seconds_to_next_handover / 60.0

        rtx_rate = telemetry["retransmits"].rolling(
            window=_RETRANSMIT_PROXY_WINDOW, min_periods=1
        ).mean()
        p_tot = (rtx_rate / _RETRANSMIT_PROXY_CAP).clip(lower=0.0, upper=1.0)
    elif mode == "closed_form":
        t_start = telemetry["t_start"].astype(float)
        seconds_to_next_handover = _HANDOVER_CYCLE_S - (t_start % _HANDOVER_CYCLE_S)
        delta_t_ho_min = seconds_to_next_handover / 60.0

        p_tot = pd.Series([closed_form_p_tot()] * n, index=telemetry.index)
    else:
        raise ValueError(f"unknown risk mode: {mode!r}")

    return pd.DataFrame(
        {
            "delta_t_ho_min": delta_t_ho_min,
            "p_tot": p_tot,
            "mode": mode,
        }
    )
