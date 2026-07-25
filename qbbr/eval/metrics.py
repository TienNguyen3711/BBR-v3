"""Shared evaluation metrics: throughput, retransmissions, RTT variance, rho_alpha.

rho_alpha = Phi_alpha(x_achieved) / Phi_alpha(x*) (main.tex's fairness
equation) needs the alpha-fair-optimal allocation x* for the measured
aggregate capacity: a convex program, with closed-form water-filling at
alpha -> infinity. Jain's index is deliberately not implemented here --
main.tex's design explicitly drops it as fairness-blind.

Not yet implemented.
"""
from __future__ import annotations

import pandas as pd


def alpha_fair_efficiency_ratio(x_achieved: pd.Series, alpha: float) -> float:
    """rho_alpha: efficiency of the achieved per-flow allocation vs. the alpha-fair optimum."""
    raise NotImplementedError(
        "rho_alpha efficiency ratio not yet implemented; see main.tex Eq. (fairness)."
    )
