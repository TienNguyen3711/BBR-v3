"""Atmospheric-attenuation failure probability (Eq. 9, TMC base paper).

p^at(M) = sum_i w_i * p_i(M): an elevation-weighted combination of per-bin
attenuation-exceedance probabilities (rain + cloud + gaseous + scintillation
components, ITU-R P.618/P.840/P.676). Computing this needs rain rate, cloud
liquid-water density, and elevation-angle time series for the exact
trace period and location -- none of which exist in the tcp-cc-starlink
dataset or anywhere else in this repo.

Not yet implemented. qbbr.risk.ptot's "stub_constant"/"empirical_proxy"
modes stand in for this (and qbbr.risk.isl, qbbr.risk.handover) until real
weather data is sourced and validated per main.tex's risk-feature
validation gate.
"""
from __future__ import annotations

import pandas as pd


def compute_atmospheric_failure_probability(
    elevation_deg: pd.Series, weather: dict
) -> pd.Series:
    """p^at(M): probability that total atmospheric attenuation exceeds threshold M."""
    raise NotImplementedError(
        "Eq. 9 atmospheric attenuation model requires ITU rain/cloud/scintillation "
        "inputs not present in this dataset; see qbbr.risk.ptot for the interim proxy."
    )
