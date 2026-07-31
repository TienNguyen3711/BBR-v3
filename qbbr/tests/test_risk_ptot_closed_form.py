from __future__ import annotations

import time

import pandas as pd
import pytest

from qbbr.risk.atmospheric import compute_atmospheric_failure_probability
from qbbr.risk.ptot import _ATTENUATION_THRESHOLD_DB, compute_risk_features

_TELEMETRY = pd.DataFrame({"t_start": [0.0, 5.0, 16.0, 40.0], "retransmits": [0, 1, 0, 2]})


def test_closed_form_mode_label_is_carried():
    risk = compute_risk_features(_TELEMETRY, mode="closed_form")
    assert (risk["mode"] == "closed_form").all()


def test_closed_form_p_tot_matches_real_atmospheric_computation():
    risk = compute_risk_features(_TELEMETRY, mode="closed_form")
    expected = compute_atmospheric_failure_probability(_ATTENUATION_THRESHOLD_DB)
    # NOTE: `pandas_series == pytest.approx(x)` silently returns all-False
    assert risk["p_tot"].tolist() == [pytest.approx(expected, rel=0.05)] * len(risk)


def test_closed_form_p_tot_is_constant_across_rows():
    # a long-run climatological quantity, not resampled per interval.
    risk = compute_risk_features(_TELEMETRY, mode="closed_form")
    assert risk["p_tot"].nunique() == 1


def test_closed_form_handover_eta_matches_empirical_proxy_sawtooth():
    closed_form = compute_risk_features(_TELEMETRY, mode="closed_form")
    proxy = compute_risk_features(_TELEMETRY, mode="empirical_proxy")
    assert (closed_form["delta_t_ho_min"] == proxy["delta_t_ho_min"]).all()


def test_closed_form_p_tot_is_cached_across_calls():
    compute_risk_features(_TELEMETRY, mode="closed_form")  # warm the cache
    t0 = time.time()
    compute_risk_features(_TELEMETRY, mode="closed_form")
    assert time.time() - t0 < 0.1  # a fresh computation takes ~1-2s; cached is near-instant


def test_unknown_mode_still_raises():
    with pytest.raises(ValueError):
        compute_risk_features(_TELEMETRY, mode="not_a_real_mode")
