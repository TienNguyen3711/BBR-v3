from __future__ import annotations

import numpy as np

from qbbr.eval.stats import mann_whitney_test, summarize_median_iqr


def test_summarize_median_iqr_matches_numpy():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    summary = summarize_median_iqr(samples)
    assert summary["n"] == 8
    assert summary["median"] == np.median(samples)
    assert summary["iqr"] == summary["q3"] - summary["q1"]
    assert summary["iqr"] > 0


def test_mann_whitney_detects_clearly_different_arms():
    rng = np.random.default_rng(0)
    low = rng.normal(loc=0.0, scale=0.5, size=30)
    high = rng.normal(loc=5.0, scale=0.5, size=30)
    result = mann_whitney_test(low, high)
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_mann_whitney_does_not_flag_identical_distributions():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, scale=1.0, size=30)
    b = rng.normal(loc=0.0, scale=1.0, size=30)
    result = mann_whitney_test(a, b)
    assert result["significant"] is False


def test_significance_threshold_is_configurable():
    rng = np.random.default_rng(2)
    a = rng.normal(loc=0.0, scale=1.0, size=20)
    b = rng.normal(loc=0.3, scale=1.0, size=20)
    lenient = mann_whitney_test(a, b, alpha=0.99)
    assert lenient["significant"] is True
