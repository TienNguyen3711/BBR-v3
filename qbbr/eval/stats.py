from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats


def summarize_median_iqr(samples: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def mann_whitney_test(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    result = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
    }
