"""Stage 1 risk-model features (s5, s6) -- p_tot and handover ETA.

The base paper's closed-form risk model (atmospheric Eq. 9, ISL Eq. 15,
handover Eq. 19, combined p_tot Eq. 20-21) needs TLE ephemerides and
weather/attenuation data. Neither exists anywhere in the tcp-cc-starlink
dataset or this repo, so this module is a deliberately labeled stand-in,
selected via `mode`, not an implementation of the real risk model:

- "stub_constant" (default): risk-neutral placeholder. Returns values that,
  after Stage 2's Table-1 normalization, are exactly s5 = s6 = 0.5 -- a
  non-informative RY(pi/2) rotation that asserts nothing. This keeps the
  quantum circuit's 6-qubit / 6-input architecture stable now, so it does
  not need to change shape once real risk data is sourced later.
- "empirical_proxy": a shape-only stand-in derived from this same dataset,
  for exercising the pipeline with non-constant risk features before real
  data exists. p_tot is proxied from a rolling retransmit-burst rate (this
  is causally backwards -- it uses *realized* loss to proxy *predicted*
  risk -- and must never be read as an actual risk estimate). Handover ETA
  is proxied by a fixed 15s sawtooth cycle (Starlink's approximate handover
  period), not derived from real orbital ephemerides.

Every call result carries `mode` so a stubbed run can never be silently
mistaken for the real risk-feature contribution (RQ3 in the design doc).
"""
from __future__ import annotations

import pandas as pd

from qbbr.features.normalize import T_ORBIT_MIN

_HANDOVER_CYCLE_S = 15.0
_RETRANSMIT_PROXY_WINDOW = 10
_RETRANSMIT_PROXY_CAP = 5.0  # retransmits/interval considered "high risk" for the proxy

# Backsolved so that Stage 2's Table-1 mapping (s5 = 1 - dT_ho/T_orbit, s6 = p_tot
# identity) yields exactly 0.5 for both features.
_STUB_DELTA_T_HO_MIN = 0.5 * T_ORBIT_MIN
_STUB_P_TOT = 0.5


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
    else:
        raise ValueError(f"unknown risk mode: {mode!r}")

    return pd.DataFrame(
        {
            "delta_t_ho_min": delta_t_ho_min,
            "p_tot": p_tot,
            "mode": mode,
        }
    )
